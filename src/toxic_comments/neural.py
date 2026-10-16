"""Repaired original random-initialized RoBERTa experiment.

The real-data path loads a ``roberta-base`` tokenizer, never pretrained model
weights. Neural imports are lazy, so the core pipeline and CLI help work without
the optional neural environment. The smoke path builds its tokenizer locally.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import math
import os
import random
import shutil
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .data import LABELS, create_synthetic_fixture, load_dataset, split_dataset, split_summary
from .metrics import DEFAULT_THRESHOLD_GRID, evaluate_predictions, select_thresholds

ORIGINAL_ARCHITECTURE = {
    "vocab_size": 50265,
    "hidden_size": 512,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "intermediate_size": 2048,
    "max_position_embeddings": 512,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
}
SMOKE_ARCHITECTURE = {
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "max_position_embeddings": 64,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
}


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    epochs: int = 5
    patience: int = 3
    batch_size: int = 8
    accumulation_steps: int = 32
    learning_rate: float = 4e-5
    weight_decay: float = 1e-2
    warmup_fraction: float = 0.1
    max_length: int = 256
    max_gradient_norm: float = 1.0
    num_workers: int = 0
    cpu_threads: int = 4
    device: str = "cpu"
    amp: bool = True
    tokenizer_name_or_path: str = "roberta-base"
    tokenizer_revision: str | None = None
    local_files_only: bool = False
    conflict_policy: str = "error"
    empty_text_policy: str = "error"
    cache_tokenization: bool = False
    dynamic_padding: bool = False

    def __post_init__(self) -> None:
        for name in ("epochs", "patience", "batch_size", "accumulation_steps", "cpu_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        if isinstance(self.max_length, bool) or not isinstance(self.max_length, int) or not 4 <= self.max_length <= 510:
            raise ValueError("max_length must be an integer between 4 and 510")
        for name in ("learning_rate", "max_gradient_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(self.warmup_fraction) or not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be within [0, 1)")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'; GPU use is explicit")
        if self.conflict_policy not in {"error", "keep"}:
            raise ValueError("conflict_policy must be 'error' or 'keep'")
        if self.empty_text_policy not in {"error", "drop"}:
            raise ValueError("empty_text_policy must be 'error' or 'drop'")
        for name in ("cache_tokenization", "dynamic_padding"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")


def _dependencies() -> SimpleNamespace:
    # Configure cuBLAS before any CUDA context is initialized. Determinism still
    # depends on using the same software and hardware; cross-platform equality
    # is not promised.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch
        import transformers
        from transformers import (
            AutoTokenizer,
            PreTrainedTokenizerFast,
            RobertaConfig,
            RobertaForSequenceClassification,
            get_cosine_schedule_with_warmup,
        )
    except ImportError as error:
        raise ImportError(
            "Neural dependencies are unavailable. Install requirements-neural.txt "
            "in the documented Python environment, then run scripts/check_neural_environment.py."
        ) from error
    return SimpleNamespace(
        torch=torch,
        transformers=transformers,
        AutoTokenizer=AutoTokenizer,
        PreTrainedTokenizerFast=PreTrainedTokenizerFast,
        RobertaConfig=RobertaConfig,
        RobertaForSequenceClassification=RobertaForSequenceClassification,
        get_cosine_schedule_with_warmup=get_cosine_schedule_with_warmup,
    )


def _set_determinism(dependencies: SimpleNamespace, config: TrainConfig) -> Any:
    torch = dependencies.torch
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; select --device cpu")
    return torch.device(config.device)


def _seed_worker(worker_id: int) -> None:
    import torch

    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class _TextDataset:
    """Map-style dataset; raw comments never appear in diagnostic output."""

    def __init__(self, texts: np.ndarray, targets: np.ndarray, tokenizer: Any, max_length: int):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Import within the worker instead of storing a module on the dataset;
        # datasets must remain pickleable when multiprocessing uses spawn.
        import torch

        encoded = self.tokenizer(
            str(self.texts[index]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {name: encoded[name].squeeze(0) for name in ("input_ids", "attention_mask")}
        item["labels"] = torch.tensor(self.targets[index], dtype=torch.float32)
        return item


TOKENIZATION_BATCH_SIZE = 256


class _CachedTextDataset:
    """Tokenize once in bounded batches, retaining five bytes per token slot.

    The cache is private, in memory, and contains only the requested partition.
    Token IDs and padding match the original per-row tokenizer calls exactly.
    """

    def __init__(self, texts: np.ndarray, targets: np.ndarray, tokenizer: Any, max_length: int):
        self.texts = texts
        self.targets = targets
        self.input_ids = np.empty((len(texts), max_length), dtype=np.int32)
        self.attention_mask = np.empty((len(texts), max_length), dtype=np.uint8)
        for start in range(0, len(texts), TOKENIZATION_BATCH_SIZE):
            end = min(start + TOKENIZATION_BATCH_SIZE, len(texts))
            encoded = tokenizer(
                [str(text) for text in texts[start:end]],
                truncation=True, padding="max_length", max_length=max_length,
                return_tensors="np",
            )
            ids = np.asarray(encoded["input_ids"])
            mask = np.asarray(encoded["attention_mask"])
            expected_shape = (end - start, max_length)
            if ids.shape != expected_shape or mask.shape != expected_shape:
                raise ValueError("Tokenizer cache requires fixed-length input IDs and attention masks")
            if not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any() or (ids > np.iinfo(np.int32).max).any():
                raise ValueError("Tokenizer input IDs cannot be stored safely as int32")
            if not np.isin(mask, [0, 1]).all():
                raise ValueError("Tokenizer attention mask must contain only 0/1 values")
            self.input_ids[start:end] = ids
            self.attention_mask[start:end] = mask

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        return {
            "input_ids": torch.from_numpy(self.input_ids[index]),
            "attention_mask": torch.from_numpy(self.attention_mask[index]),
            "labels": torch.tensor(self.targets[index], dtype=torch.float32),
        }


class _TokenBatchCollator:
    """Restore model dtypes and optionally remove a batch's right padding."""

    def __init__(self, max_length: int, *, dynamic_padding: bool):
        self.max_length = max_length
        self.dynamic_padding = dynamic_padding

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        from torch.utils.data import default_collate

        batch = default_collate(items)
        batch["input_ids"] = batch["input_ids"].long()
        batch["attention_mask"] = batch["attention_mask"].long()
        if self.dynamic_padding:
            longest = int(batch["attention_mask"].sum(dim=1).max().item())
            length = min(self.max_length, max(8, math.ceil(longest / 8) * 8))
            for name in ("input_ids", "attention_mask"):
                batch[name] = batch[name][:, :length].contiguous()
        return batch


def _text_dataset(dataset: Any, indices: np.ndarray, tokenizer: Any, config: TrainConfig) -> Any:
    if config.dynamic_padding and tokenizer.padding_side != "right":
        raise ValueError("Dynamic padding requires a right-padding tokenizer")
    dataset_type = _CachedTextDataset if config.cache_tokenization else _TextDataset
    return dataset_type(dataset.texts[indices], dataset.targets[indices], tokenizer, config.max_length)


def _collate_fn(config: TrainConfig) -> Any:
    if config.cache_tokenization or config.dynamic_padding:
        return _TokenBatchCollator(config.max_length, dynamic_padding=config.dynamic_padding)
    return None


def _loader(dataset: Any, indices: np.ndarray, tokenizer: Any, config: TrainConfig, dependencies: Any, *, shuffle: bool) -> Any:
    torch = dependencies.torch
    generator = torch.Generator().manual_seed(config.seed)
    return torch.utils.data.DataLoader(
        _text_dataset(dataset, indices, tokenizer, config),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda",
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=False,
        collate_fn=_collate_fn(config),
    )


def validate_training_targets(targets: np.ndarray) -> np.ndarray:
    """Return finite train-only positive weights, rejecting unlearnable labels."""
    labels = np.asarray(targets)
    if labels.ndim != 2 or labels.shape[1] != len(LABELS) or not len(labels):
        raise ValueError(f"Training targets must have nonempty (n, {len(LABELS)}) shape")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Training labels must be finite binary 0/1 values")
    positive = labels.sum(axis=0, dtype=np.float64)
    negative = len(labels) - positive
    missing = [label for column, label in enumerate(LABELS) if not positive[column] or not negative[column]]
    if missing:
        raise ValueError(
            "Every training label needs positive and negative examples for weighted BCE; "
            f"missing coverage: {', '.join(missing)}. Inspect the split without tuning on test results."
        )
    weights = (negative / positive).astype(np.float32)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Training positive weights must be finite and positive")
    return weights


def _synthetic_tokenizer(texts: np.ndarray, dependencies: Any, *, max_length: int) -> Any:
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    special_tokens = ("<s>", "<pad>", "</s>", "<unk>", "<mask>")
    words = sorted({word for text in texts for word in str(text).split()} - set(special_tokens))
    vocabulary = {word: index for index, word in enumerate((*special_tokens, *words))}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", vocabulary["<s>"]), ("</s>", vocabulary["</s>"])],
    )
    return dependencies.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        mask_token="<mask>",
        model_max_length=max_length,
        model_input_names=["input_ids", "attention_mask"],
    )


def _new_model(tokenizer: Any, dependencies: Any, *, smoke: bool) -> Any:
    architecture = dict(SMOKE_ARCHITECTURE if smoke else ORIGINAL_ARCHITECTURE)
    if smoke:
        architecture["vocab_size"] = len(tokenizer)
    elif tokenizer.vocab_size != ORIGINAL_ARCHITECTURE["vocab_size"] or len(tokenizer) != tokenizer.vocab_size:
        raise ValueError("The original experiment requires the unmodified 50,265-token roberta-base tokenizer")
    if not smoke and (tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id) != (0, 1, 2, 3):
        raise ValueError("Tokenizer special-token IDs differ from the original roberta-base contract")
    config = dependencies.RobertaConfig(
        **architecture,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: index for index, label in enumerate(LABELS)},
        problem_type="multi_label_classification",
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # Eager attention avoids hardware-dependent fused attention choices.
        attn_implementation="eager",
    )
    # This constructor initializes random weights. No model.from_pretrained
    # call is permitted here: only a saved local export is reloaded below.
    return dependencies.RobertaForSequenceClassification(config)


def _check_label_mapping(model: Any) -> None:
    if model.config.num_labels != len(LABELS):
        raise ValueError("Saved model has an incompatible number of labels")
    if model.config.id2label != dict(enumerate(LABELS)) or model.config.label2id != {label: index for index, label in enumerate(LABELS)}:
        raise ValueError("Saved model label order differs from the explicit six-label contract")


def _autocast(torch: Any, *, enabled: bool) -> Any:
    return torch.amp.autocast("cuda", dtype=torch.float16) if enabled else nullcontext()


def _device_batch(batch: dict[str, Any], device: Any, model: Any) -> dict[str, Any]:
    # DataLoader tensors are still on CPU here. Checking after transfer forces
    # two GPU synchronizations for every microbatch.
    input_ids = batch["input_ids"]
    if input_ids.min().item() < 0 or input_ids.max().item() >= model.config.vocab_size:
        raise ValueError("Tokenizer produced an input ID outside the model vocabulary")
    return {name: tensor.to(device, non_blocking=device.type == "cuda") for name, tensor in batch.items()}


def _predict(model: Any, loader: Any, dependencies: Any, device: Any, *, criterion: Any = None, amp_enabled: bool = False) -> tuple[float | None, np.ndarray, np.ndarray]:
    torch = dependencies.torch
    model.eval()
    losses = 0.0
    rows = 0
    probabilities, targets = [], []
    with torch.inference_mode():
        for batch in loader:
            values = _device_batch(batch, device, model)
            with _autocast(torch, enabled=amp_enabled):
                logits = model(input_ids=values["input_ids"], attention_mask=values["attention_mask"]).logits
                loss = criterion(logits, values["labels"]) if criterion is not None else None
            if not torch.isfinite(logits).all().item() or (loss is not None and not torch.isfinite(loss).item()):
                raise FloatingPointError("Nonfinite logits or loss during inference")
            count = len(values["labels"])
            if loss is not None:
                losses += float(loss.item()) * count
            rows += count
            probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
            targets.append(values["labels"].cpu().numpy().astype(np.uint8))
    if not rows:
        raise ValueError("Cannot predict an empty partition")
    return losses / rows if criterion is not None else None, np.vstack(targets), np.vstack(probabilities)


def _train(model: Any, train_loader: Any, validation_loader: Any, criterion: Any, config: TrainConfig, dependencies: Any, device: Any, checkpoint_path: Path, *, quiet: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch = dependencies.torch
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    planned_steps = math.ceil(len(train_loader) / config.accumulation_steps) * config.epochs
    scheduler = dependencies.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_fraction * planned_steps),
        num_training_steps=planned_steps,
    )
    amp_enabled = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    # A public optimizer hook records actual updates. GradScaler can skip a
    # step after overflow; scheduler/step counts must stay unchanged then.
    step_counter = {"value": 0}

    def record_optimizer_step(*_: Any) -> None:
        step_counter["value"] += 1

    step_hook = optimizer.register_step_post_hook(record_optimizer_step)
    history: list[dict[str, Any]] = []
    best = {"epoch": 0, "validation_loss": float("inf"), "optimizer_steps": 0}
    epochs_without_improvement = 0
    try:
        for epoch in range(1, config.epochs + 1):
            model.train()
            training_loss = 0.0
            training_rows = 0
            skipped_steps = 0
            epoch_start_steps = step_counter["value"]
            iterator = iter(train_loader)
            while True:
                micro_batches = list(itertools.islice(iterator, config.accumulation_steps))
                if not micro_batches:
                    break
                group_rows = sum(len(batch["labels"]) for batch in micro_batches)
                optimizer.zero_grad(set_to_none=True)
                for batch in micro_batches:
                    values = _device_batch(batch, device, model)
                    with _autocast(torch, enabled=amp_enabled):
                        logits = model(input_ids=values["input_ids"], attention_mask=values["attention_mask"]).logits
                        loss = criterion(logits, values["labels"])
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("Nonfinite training loss; no checkpoint/export will be frozen")
                    count = len(values["labels"])
                    # Sample weighting handles both the last partial batch and
                    # a last accumulation group smaller than the configured one.
                    scaler.scale(loss * (count / group_rows)).backward()
                    training_loss += float(loss.item()) * count
                    training_rows += count
                scaler.unscale_(optimizer)
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                if not gradients:
                    raise RuntimeError("Training produced no gradients")
                finite_gradients = torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all().item()
                if finite_gradients:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm, error_if_nonfinite=True)
                elif not amp_enabled:
                    raise FloatingPointError("Nonfinite training gradients in full precision")
                # Nonfinite AMP gradients were detected by unscale_ and are
                # skipped by GradScaler rather than clipped into NaNs.
                previous_steps = step_counter["value"]
                scaler.step(optimizer)
                scaler.update()
                if step_counter["value"] > previous_steps:
                    scheduler.step()
                else:
                    skipped_steps += 1
            if step_counter["value"] == epoch_start_steps:
                raise FloatingPointError("An epoch completed without any optimizer updates")
            validation_loss, _, _ = _predict(
                model, validation_loader, dependencies, device,
                criterion=criterion, amp_enabled=amp_enabled,
            )
            record = {
                "epoch": epoch,
                "training_loss": training_loss / training_rows,
                "validation_loss": validation_loss,
                "optimizer_steps": step_counter["value"],
                "skipped_amp_steps": skipped_steps,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            history.append(record)
            if not quiet:
                print(f"epoch={epoch} train_loss={record['training_loss']:.6f} validation_loss={validation_loss:.6f} optimizer_steps={step_counter['value']} skipped_amp_steps={skipped_steps}", flush=True)
            if validation_loss < best["validation_loss"]:
                best = {"epoch": epoch, "validation_loss": validation_loss, "optimizer_steps": step_counter["value"]}
                torch.save({**best, "label_order": list(LABELS), "model_state_dict": model.state_dict()}, checkpoint_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    break
    finally:
        step_hook.remove()
    return history, best


def _export_best(model: Any, tokenizer: Any, checkpoint_path: Path, model_path: Path, dependencies: Any, device: Any) -> tuple[Any, Any, dict[str, Any]]:
    torch = dependencies.torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["label_order"] != list(LABELS):
        raise ValueError("Checkpoint label mapping changed")
    # Restore the selected checkpoint BEFORE save_pretrained. The historical
    # notebook exported last-epoch weights before it restored the best model.
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    _check_label_mapping(model)
    model.save_pretrained(model_path, safe_serialization=True)
    tokenizer.save_pretrained(model_path)
    checkpoint_metadata = {name: checkpoint[name] for name in ("epoch", "validation_loss", "optimizer_steps")}
    del checkpoint
    model.cpu()
    exported = dependencies.RobertaForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, use_safetensors=True, attn_implementation="eager",
    )
    exported_tokenizer = dependencies.AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    _check_label_mapping(exported)
    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor.cpu(), exported.state_dict()[name].cpu()):
            raise RuntimeError("Saved model parameters differ from the restored best checkpoint")
    if tokenizer.get_vocab() != exported_tokenizer.get_vocab() or tokenizer.special_tokens_map != exported_tokenizer.special_tokens_map:
        raise RuntimeError("Saved tokenizer differs from the tokenizer used in training")
    return exported.to(device).eval(), exported_tokenizer, checkpoint_metadata


def run_development(data_path: str | Path, run_dir: str | Path, *, config: TrainConfig | None = None, quiet: bool = False, _smoke: bool = False) -> dict[str, Any]:
    """Train/select/freeze using train and validation only, never test scores."""
    from .artifacts import create_run_directory, freeze_run, runtime_metadata

    config = config or TrainConfig()
    initial_metadata = runtime_metadata(config.seed)
    dependencies = _dependencies()
    device = _set_determinism(dependencies, config)
    destination = create_run_directory(run_dir)
    private = destination / "private"
    private.mkdir(exist_ok=True)
    source_snapshot = private / "source.csv"
    shutil.copyfile(Path(data_path).resolve(), source_snapshot)
    dataset = load_dataset(source_snapshot, conflict_policy=config.conflict_policy, empty_text_policy=config.empty_text_policy)
    if _smoke and not dataset.summary["synthetic"]:
        raise ValueError("The tiny smoke configuration is restricted to marked synthetic fixtures")
    splits = split_dataset(dataset, seed=config.seed)
    positive_weights = validate_training_targets(dataset.targets[splits.train])
    if _smoke:
        tokenizer = _synthetic_tokenizer(dataset.texts[splits.train], dependencies, max_length=config.max_length)
    else:
        tokenizer = dependencies.AutoTokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            revision=config.tokenizer_revision,
            local_files_only=config.local_files_only,
            use_fast=True,
        )
    model = _new_model(tokenizer, dependencies, smoke=_smoke).to(device)
    if config.max_length + tokenizer.pad_token_id + 1 > model.config.max_position_embeddings:
        raise ValueError("max_length exceeds the configured RoBERTa position capacity")
    torch = dependencies.torch
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weights, dtype=torch.float32, device=device))
    train_loader = _loader(dataset, splits.train, tokenizer, config, dependencies, shuffle=True)
    validation_loader = _loader(dataset, splits.validation, tokenizer, config, dependencies, shuffle=False)
    checkpoint_path = private / "best_checkpoint.pt"
    history, best = _train(model, train_loader, validation_loader, criterion, config, dependencies, device, checkpoint_path, quiet=quiet)
    model_path = destination / "model"
    model, tokenizer, checkpoint_metadata = _export_best(model, tokenizer, checkpoint_path, model_path, dependencies, device)
    # Discard every epoch's cached predictions. Fresh inference through the
    # reloaded export is the sole source for threshold selection and export.
    validation_loader = _loader(dataset, splits.validation, tokenizer, config, dependencies, shuffle=False)
    fresh_loss, validation_targets, validation_probabilities = _predict(
        model, validation_loader, dependencies, device,
        criterion=criterion, amp_enabled=config.amp and device.type == "cuda",
    )
    if not np.array_equal(validation_targets, dataset.targets[splits.validation]):
        raise RuntimeError("Validation inference changed label or row order")
    if not np.isclose(fresh_loss, best["validation_loss"], rtol=1e-5, atol=1e-6):
        raise RuntimeError("Reloaded export does not reproduce the best checkpoint validation loss")
    thresholds = select_thresholds(validation_targets, validation_probabilities)
    validation_predictions_path = private / "validation_predictions.npz"
    np.savez_compressed(validation_predictions_path, targets=validation_targets, probabilities=validation_probabilities, row_indices=splits.validation)
    history_path = destination / "training_history.json"
    history_path.write_text(json.dumps({"epochs": history, "best_checkpoint": checkpoint_metadata, "fresh_validation_loss": fresh_loss}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    validation_metrics_path = destination / "validation_metrics.json"
    validation_metrics = {
        "split": "validation",
        "synthetic": bool(dataset.summary["synthetic"]),
        "metrics_at_0_5": evaluate_predictions(validation_targets, validation_probabilities, np.full(len(LABELS), 0.5)),
        "metrics_at_validation_thresholds": evaluate_predictions(validation_targets, validation_probabilities, thresholds),
        "selection_note": "Checkpoint and thresholds were selected using this validation partition; these metrics are selection diagnostics.",
    }
    validation_metrics_path.write_text(json.dumps(validation_metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    architecture = {name: getattr(model.config, name) for name in ORIGINAL_ARCHITECTURE}
    metadata = {
        **initial_metadata,
        "model_family": "custom RoBERTa sequence classifier",
        "initialization": "random weights; no pretrained model weights",
        "tokenizer_source": "locally constructed synthetic tokenizer" if _smoke else config.tokenizer_name_or_path,
        "synthetic": bool(dataset.summary["synthetic"]),
        "engineering_smoke": _smoke,
        "checkpoint_selection": "minimum sample-weighted validation BCE loss",
        "best_checkpoint": checkpoint_metadata,
        "train_positive_weights": dict(zip(LABELS, positive_weights.tolist())),
        "threshold_selection": "per-label validation F1; >= comparator; ties nearest 0.5 then larger; missing class keeps 0.5",
        "threshold_grid": list(DEFAULT_THRESHOLD_GRID),
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "attention_implementation": "eager",
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "amp_enabled": config.amp and device.type == "cuda",
        "effective_batch_size": config.batch_size * config.accumulation_steps,
        "effective_batch_size_note": "Nominal maximum; the final accumulation group uses its actual sample count.",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "test_status": "unscored during development",
        "protocol_note": "New internal exploratory holdout; the historical cohort was previously examined, so this is not a pristine external test.",
        "export_matches_best_checkpoint": True,
        "fresh_export_validation_loss_matches_best": True,
    }
    freeze_run(
        destination,
        config={"method": "random_initialized_roberta", "training": asdict(config), "architecture": architecture, "label_order": list(LABELS)},
        data_summary={**dataset.summary, "splits": split_summary(dataset, splits)},
        source_path=source_snapshot,
        model_path=model_path,
        thresholds=thresholds,
        splits=splits,
        metadata=metadata,
        extra_artifacts=[checkpoint_path, validation_predictions_path, history_path, validation_metrics_path],
    )
    return {
        "run_dir": str(destination),
        "stage": "developed_and_frozen",
        "synthetic": bool(dataset.summary["synthetic"]),
        "best_epoch": best["epoch"],
        "label_order": list(LABELS),
        "checks": {"export_matches_best_checkpoint": True, "fresh_validation_predictions": True, "test_unscored_during_development": True},
    }


def evaluate_run(run_dir: str | Path, *, data_path: str | Path | None = None, device: str = "cpu") -> dict[str, Any]:
    """Score a verified frozen run once; marker is acquired before inference."""
    from .artifacts import begin_test_evaluation, complete_test_evaluation, load_splits, load_thresholds, resolve_run_path, verify_frozen_run

    destination = Path(run_dir).resolve()
    manifest = verify_frozen_run(destination, source_path=data_path)
    frozen_config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    if frozen_config["method"] != "random_initialized_roberta" or frozen_config["label_order"] != list(LABELS):
        raise ValueError("Run is not a compatible frozen RoBERTa experiment")
    training_config = dict(frozen_config["training"])
    training_config["device"] = device
    config = TrainConfig(**training_config)
    dependencies = _dependencies()
    torch_device = _set_determinism(dependencies, config)
    source_path = Path(data_path).resolve() if data_path is not None else resolve_run_path(destination, manifest["source_path"])
    dataset = load_dataset(source_path, conflict_policy=config.conflict_policy, empty_text_policy=config.empty_text_policy)
    splits = load_splits(destination)
    expected_splits = split_dataset(dataset, seed=config.seed)
    if any(not np.array_equal(getattr(splits, name), getattr(expected_splits, name)) for name in ("train", "validation", "test")):
        raise ValueError("Frozen splits do not match the verified data and recorded seed")
    model_path = resolve_run_path(destination, manifest["model_path"])
    model = dependencies.RobertaForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, use_safetensors=True, attn_implementation="eager",
    ).to(torch_device).eval()
    tokenizer = dependencies.AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    _check_label_mapping(model)
    thresholds = load_thresholds(destination)
    begin_test_evaluation(destination, source_path=source_path)
    loader = _loader(dataset, splits.test, tokenizer, config, dependencies, shuffle=False)
    _, targets, probabilities = _predict(
        model, loader, dependencies, torch_device,
        amp_enabled=config.amp and torch_device.type == "cuda",
    )
    if not np.array_equal(targets, dataset.targets[splits.test]):
        raise RuntimeError("Test inference changed label or row order")
    results = {
        "method": "random_initialized_roberta",
        "synthetic": bool(dataset.summary["synthetic"]),
        "engineering_smoke": bool(metadata["engineering_smoke"]),
        "split": "internal exploratory test",
        "protocol_note": metadata["protocol_note"],
        "selection_policy": "best validation loss checkpoint; thresholds frozen from validation only",
        "metrics_at_0_5": evaluate_predictions(targets, probabilities, np.full(len(LABELS), 0.5)),
        "metrics_at_frozen_validation_thresholds": evaluate_predictions(targets, probabilities, thresholds),
    }
    complete_test_evaluation(
        destination, public_results=results, targets=targets,
        probabilities=probabilities, row_indices=splits.test,
        source_path=source_path,
    )
    return results


def run_smoke(run_dir: str | Path) -> dict[str, Any]:
    """Run tiny offline CPU train/save/reload/freeze/evaluate mechanics only."""
    config = TrainConfig(
        epochs=2, patience=2, batch_size=8, accumulation_steps=3,
        learning_rate=1e-3, max_length=32, cpu_threads=1, device="cpu",
        amp=False, local_files_only=True,
    )
    with tempfile.TemporaryDirectory(prefix="toxic-neural-fixture-") as temporary:
        source = create_synthetic_fixture(Path(temporary) / "synthetic.csv", rows=120, seed=config.seed)
        developed = run_development(source, run_dir, config=config, quiet=True, _smoke=True)
    results = evaluate_run(run_dir, device="cpu")
    destination = Path(run_dir).resolve()
    with np.load(destination / "private" / "test_predictions.npz", allow_pickle=False) as predictions:
        finite = bool(np.isfinite(predictions["probabilities"]).all())
        six_columns = predictions["probabilities"].shape[1] == len(LABELS)
    if not finite or not six_columns or not results["engineering_smoke"]:
        raise RuntimeError("Offline neural smoke assertions failed")
    return {
        "status": "passed",
        "run_dir": str(destination),
        "synthetic": True,
        "purpose": "engineering mechanics only; no real-data performance claim",
        "tiny_architecture": SMOKE_ARCHITECTURE,
        "checks": {
            **developed["checks"],
            "local_synthetic_tokenizer": True,
            "cpu_full_precision": True,
            "finite_test_probabilities": finite,
            "explicit_six_label_order": six_columns,
            "frozen_test_evaluation_completed": True,
        },
    }


def check_environment() -> dict[str, Any]:
    """Check the optional environment using a tiny CPU forward pass only."""
    dependencies = _dependencies()
    config = TrainConfig(cpu_threads=1, amp=False)
    device = _set_determinism(dependencies, config)
    tokenizer = _synthetic_tokenizer(np.array(["synthetic environment check"]), dependencies, max_length=16)
    model = _new_model(tokenizer, dependencies, smoke=True).to(device).eval()
    encoded = tokenizer("synthetic environment check", return_tensors="pt")
    with dependencies.torch.inference_mode():
        logits = model(**encoded).logits
    if logits.shape != (1, len(LABELS)) or not dependencies.torch.isfinite(logits).all().item():
        raise RuntimeError("Tiny CPU forward pass failed")
    return {
        "status": "passed",
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "safetensors", "huggingface-hub")},
        "cuda_available": dependencies.torch.cuda.is_available(),
        "checked_device": "cpu",
        "check": "tiny local random model forward pass; no downloads or training",
    }


def resource_diagnostic() -> dict[str, Any]:
    """Count the original architecture on the meta device without weights or data.

    The memory figures cover float32 parameters, gradients and AdamW moments
    only. They exclude activations and temporary tensors, so they cannot be
    used as a measured peak-memory or training-time estimate.
    """
    dependencies = _dependencies()
    config = dependencies.RobertaConfig(
        **ORIGINAL_ARCHITECTURE,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: index for index, label in enumerate(LABELS)},
        problem_type="multi_label_classification",
        attn_implementation="eager",
    )
    with dependencies.torch.device("meta"):
        model = dependencies.RobertaForSequenceClassification(config)
    _check_label_mapping(model)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "architecture": ORIGINAL_ARCHITECTURE,
        "max_length": TrainConfig().max_length,
        "parameter_count": parameters,
        "float32_parameter_memory_mib": parameters * 4 / 1024**2,
        "float32_parameters_gradients_adam_moments_lower_bound_mib": parameters * 16 / 1024**2,
        "memory_note": "Analytical lower bound; excludes activations, temporary tensors and input storage. Peak memory and runtime were not measured.",
        "cuda_available": dependencies.torch.cuda.is_available(),
        "model_initialization": "architecture instantiated on meta device; no full weight allocation or downloads",
        "full_model_training_executed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    develop = stages.add_parser("develop", help="Explicitly train/select/freeze the original random RoBERTa on a supplied CSV")
    develop.add_argument("--data", required=True, type=Path)
    develop.add_argument("--run-dir", required=True, type=Path)
    develop.add_argument("--epochs", type=int, default=5)
    develop.add_argument("--patience", type=int, default=3)
    develop.add_argument("--batch-size", type=int, default=8)
    develop.add_argument("--accumulation-steps", type=int, default=32)
    develop.add_argument("--learning-rate", type=float, default=4e-5)
    develop.add_argument("--weight-decay", type=float, default=1e-2)
    develop.add_argument("--max-length", type=int, default=256)
    develop.add_argument("--seed", type=int, default=42)
    develop.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    develop.add_argument("--num-workers", type=int, default=0)
    develop.add_argument("--cpu-threads", type=int, default=4)
    develop.add_argument("--no-amp", action="store_true", help="Disable CUDA-only mixed precision; CPU always uses float32")
    develop.add_argument("--tokenizer", default="roberta-base", help="roberta-base identifier or equivalent local tokenizer directory; tokenizer only")
    develop.add_argument("--tokenizer-revision", default=None)
    develop.add_argument("--local-files-only", action="store_true")
    develop.add_argument("--conflict-policy", choices=("error", "keep"), default="error")
    develop.add_argument("--empty-text-policy", choices=("error", "drop"), default="error", help="Explicitly drop normalized-empty comments; default rejects them")
    evaluate = stages.add_parser("evaluate", help="Evaluate a verified frozen model once on its internal exploratory test split")
    evaluate.add_argument("--run-dir", required=True, type=Path)
    evaluate.add_argument("--data", type=Path, help="Optional byte-identical replacement for the frozen source CSV")
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    smoke = stages.add_parser("smoke", help="Offline tiny CPU engineering check with synthetic data; not a benchmark")
    smoke.add_argument("--run-dir", required=True, type=Path)
    stages.add_parser("check-environment", help="Validate neural imports and tiny local CPU forward pass")
    stages.add_parser("diagnose-resources", help="Count the original architecture without training, downloads or full weight allocation")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.stage == "develop":
            config = TrainConfig(
                seed=args.seed, epochs=args.epochs, patience=args.patience,
                batch_size=args.batch_size, accumulation_steps=args.accumulation_steps,
                learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                max_length=args.max_length, device=args.device,
                num_workers=args.num_workers, cpu_threads=args.cpu_threads,
                amp=not args.no_amp, tokenizer_name_or_path=args.tokenizer,
                tokenizer_revision=args.tokenizer_revision, local_files_only=args.local_files_only,
                conflict_policy=args.conflict_policy,
                empty_text_policy=args.empty_text_policy,
            )
            result = run_development(args.data, args.run_dir, config=config)
        elif args.stage == "evaluate":
            result = evaluate_run(args.run_dir, data_path=args.data, device=args.device)
        elif args.stage == "smoke":
            result = run_smoke(args.run_dir)
        elif args.stage == "diagnose-resources":
            result = resource_diagnostic()
        else:
            result = check_environment()
    except (ImportError, OSError, ValueError, RuntimeError, FloatingPointError) as error:
        parser.exit(2, f"neural: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
