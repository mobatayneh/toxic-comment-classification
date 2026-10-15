"""Resumable Colab runner for the original random eight-layer RoBERTa.

Only the public tokenizer is downloaded. The encoder/classifier is initialized
randomly through toxic_comments.neural._new_model. Private checkpoints and
per-row arrays stay in the chosen run directory; review ZIPs contain aggregates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import random
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Also support ``python scripts/colab_train.py`` without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from toxic_comments import neural as core
from toxic_comments.artifacts import (
    MANIFEST_SCHEMA,
    TEST_MARKER,
    create_run_directory,
    freeze_run,
    read_json,
    runtime_metadata,
    verify_frozen_run,
)
from toxic_comments.data import (
    LABELS,
    create_synthetic_fixture,
    file_sha256,
    load_dataset,
    split_dataset,
    split_summary,
)
from toxic_comments.metrics import DEFAULT_THRESHOLD_GRID, evaluate_predictions, select_thresholds

CHECKPOINT_SCHEMA = 1
LATEST_CHECKPOINT = "private/latest_checkpoint.pt"
BEST_CHECKPOINT = "private/best_checkpoint.pt"
IDENTITY_FILE = "private/run_identity.json"
TOKENIZER_DIRECTORY = "private/source_tokenizer"
FREEZE_TRANSACTION = "private/freeze_transaction.json"
FREEZE_EXTRA_ARTIFACTS = (
    BEST_CHECKPOINT, IDENTITY_FILE, TOKENIZER_DIRECTORY,
    "private/validation_predictions.npz", "training_history.json", "validation_metrics.json",
)
FULL_SOURCE_SHA256 = "bb7a9e1af5282c4c05bf09bf3be3373c4160fe3272a4b26016034ff4bcea5f1e"
FULL_DATA_EXPECTATION = {
    "input_rows": 159571,
    "retained_rows": 159564,
    "excluded_empty_text_rows": 7,
    "conflicting_label_groups": 93,
    "split_rows": {"train": 111731, "validation": 23878, "test": 23955},
}


@dataclass(frozen=True)
class ColabConfig:
    training: core.TrainConfig = field(default_factory=lambda: core.TrainConfig(
        device="cuda", conflict_policy="keep", empty_text_policy="drop",
    ))
    checkpoint_every_updates: int = 100
    gradient_checkpointing: bool = False
    require_full_dataset: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.training, core.TrainConfig):
            raise ValueError("training must be a TrainConfig")
        if isinstance(self.checkpoint_every_updates, bool) or not isinstance(self.checkpoint_every_updates, int) or self.checkpoint_every_updates < 1:
            raise ValueError("checkpoint_every_updates must be a positive integer")
        for name in ("gradient_checkpointing", "require_full_dataset"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        # Worker process RNG/prefetch state is intentionally excluded. Requiring
        # zero workers makes reconstructed iterator offsets fully reviewable.
        if self.training.num_workers != 0:
            raise ValueError("Exact Colab resume requires num_workers=0")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _source_code_hashes() -> dict[str, str]:
    paths = sorted((PROJECT_ROOT / "src" / "toxic_comments").glob("*.py"))
    paths.append(Path(__file__).resolve())
    return {path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(path) for path in paths}


def _directory_hashes(path: Path) -> dict[str, str]:
    return {entry.relative_to(path).as_posix(): file_sha256(entry) for entry in sorted(path.rglob("*")) if entry.is_file()}


def _runtime_identity(dependencies: Any) -> dict[str, Any]:
    torch = dependencies.torch
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "torch", "transformers", "tokenizers", "safetensors", "huggingface-hub")
    hardware = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [
            {"name": torch.cuda.get_device_name(index), "capability": list(torch.cuda.get_device_capability(index)), "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory}
            for index in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
        "cpu_machine": platform.machine(),
    }
    return {
        "python_version": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in packages},
        "hardware": hardware,
    }


def _assert_environment(dependencies: Any, *, smoke: bool, config: ColabConfig) -> None:
    from packaging.version import Version

    if sys.version_info < (3, 12):
        raise RuntimeError("Python 3.12 or newer is required by the pinned, tested core dependencies")
    if Version(dependencies.torch.__version__.split("+")[0]) < Version("2.8.0"):
        raise RuntimeError("torch >=2.8 is required; keep a compatible CUDA-enabled Colab build")
    if not smoke and (config.training.device != "cuda" or not dependencies.torch.cuda.is_available()):
        raise RuntimeError("Full training requires a GPU. In Colab choose Runtime > Change runtime type > GPU, reconnect, and rerun setup")
    if smoke and config.training.device != "cpu":
        raise ValueError("The offline synthetic smoke must use CPU")


def atomic_torch_save(torch: Any, payload: dict[str, Any], path: Path) -> None:
    """Replace a checkpoint only after a complete sibling file is flushed.

    Filesystem durability still depends on the storage provider. A previous
    checkpoint survives a serialization/replace error; incomplete temps are
    removed. Use this only for trusted checkpoints produced by this runner.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _recover_freeze_checksum(run_dir: Path, identity: dict[str, Any], dependencies: Any) -> None:
    """Finish only an intact, transaction-marked freeze missing its checksum.

    A present checksum is never replaced. Truncated manifests and changed
    artifacts remain fail-closed; this is not a general integrity repair tool.
    """
    manifest_path = run_dir / "manifest.json"
    checksum_path = run_dir / "manifest.sha256"
    transaction_path = run_dir / FREEZE_TRANSACTION
    if not manifest_path.exists() or checksum_path.exists():
        return
    if not transaction_path.is_file() or (run_dir / TEST_MARKER).exists():
        raise ValueError("Incomplete freeze cannot be recovered without its transaction or after test access")
    transaction = read_json(transaction_path)
    expected_transaction = {
        "schema_version": 1,
        "identity_sha256": file_sha256(run_dir / IDENTITY_FILE),
        "latest_checkpoint_sha256": file_sha256(run_dir / LATEST_CHECKPOINT),
        "best_checkpoint_sha256": file_sha256(run_dir / BEST_CHECKPOINT),
    }
    if transaction != expected_transaction:
        raise ValueError("Incomplete freeze transaction or checkpoint identity changed")
    checkpoint = dependencies.torch.load(run_dir / LATEST_CHECKPOINT, map_location="cpu", weights_only=True)
    if (checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
            or checkpoint.get("identity_sha256") != transaction["identity_sha256"]
            or checkpoint.get("label_order") != list(LABELS)
            or checkpoint.get("training_complete") is not True):
        raise ValueError("Incomplete freeze requires the intact completed training checkpoint")
    best_metadata = checkpoint["best"]
    del checkpoint
    best = dependencies.torch.load(run_dir / BEST_CHECKPOINT, map_location="cpu", weights_only=True)
    if best.get("label_order") != list(LABELS) or any(best.get(name) != value for name, value in best_metadata.items()):
        raise ValueError("Incomplete freeze best checkpoint differs from completed training")
    del best
    manifest = read_json(manifest_path)
    if (manifest.get("schema_version") != MANIFEST_SCHEMA
            or manifest.get("label_order") != list(LABELS)
            or manifest.get("source_path") != "private/source.csv"
            or manifest.get("source_sha256") != identity["source_sha256"]
            or manifest.get("model_path") != "model"):
        raise ValueError("Incomplete freeze manifest is incompatible with this Colab run")
    artifacts = ("config.json", "data_summary.json", "metadata.json", "thresholds.json", "private/splits.npz", "model", *FREEZE_EXTRA_ARTIFACTS)
    observed = {}
    for relative in artifacts:
        path = run_dir / relative
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if not resolved.is_relative_to(run_dir) or not resolved.is_file():
                raise ValueError("Incomplete freeze contains an invalid artifact path")
            observed[resolved.relative_to(run_dir).as_posix()] = file_sha256(resolved)
    if manifest.get("artifacts_sha256") != observed:
        raise ValueError("Incomplete freeze artifact integrity check failed")
    if not any(name.startswith("model/") for name in observed):
        raise ValueError("Incomplete freeze model export is empty")
    recorded_code = read_json(run_dir / "metadata.json")["code_sha256"]
    expected_code = {name: value for name, value in identity["code_sha256"].items() if name.startswith("src/")}
    if recorded_code != expected_code:
        raise ValueError("Incomplete freeze source code differs from the original run identity")
    descriptor, temporary_name = tempfile.mkstemp(prefix="manifest.sha256.", suffix=".tmp", dir=run_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(file_sha256(manifest_path) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if checksum_path.exists():
            raise FileExistsError("A freeze checksum already exists; it cannot be replaced")
        os.replace(temporary, checksum_path)
    finally:
        temporary.unlink(missing_ok=True)
    verify_frozen_run(run_dir)
    transaction_path.unlink()


def capture_rng_state(torch: Any) -> dict[str, Any]:
    """Serialize NumPy RNG as primitives/tensors, safe for weights_only=True."""
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"algorithm": algorithm, "keys": torch.tensor(keys.astype(np.int64)), "position": int(position), "has_gauss": int(has_gauss), "cached_gaussian": float(cached_gaussian)},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(torch: Any, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["algorithm"], numpy_state["keys"].cpu().numpy().astype(np.uint32), numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG state does not match the current GPU count")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def _epoch_loader(dataset: Any, indices: np.ndarray, tokenizer: Any, config: core.TrainConfig, dependencies: Any, epoch: int) -> Any:
    """Every epoch has an independent deterministic shuffle and loader RNG."""
    generator = dependencies.torch.Generator().manual_seed(config.seed + epoch)
    return dependencies.torch.utils.data.DataLoader(
        core._TextDataset(dataset.texts[indices], dataset.targets[indices], tokenizer, config.max_length),
        batch_size=config.batch_size, shuffle=True, num_workers=0,
        pin_memory=config.device == "cuda", generator=generator,
        worker_init_fn=core._seed_worker, drop_last=False,
    )


def _optimizer_group(model: Any, micro_batches: list[Any], criterion: Any, optimizer: Any, scaler: Any, scheduler: Any, config: core.TrainConfig, dependencies: Any, device: Any, step_counter: dict[str, int]) -> dict[str, Any]:
    """One accumulation boundary; scale by actual samples, including tails."""
    torch = dependencies.torch
    group_rows = sum(len(batch["labels"]) for batch in micro_batches)
    if not group_rows:
        raise ValueError("An optimizer group must have at least one sample")
    amp_enabled = config.amp and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    numerator = 0.0
    for batch in micro_batches:
        values = core._device_batch(batch, device, model)
        with core._autocast(torch, enabled=amp_enabled):
            logits = model(input_ids=values["input_ids"], attention_mask=values["attention_mask"]).logits
            loss = criterion(logits, values["labels"])
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite training loss; this run cannot be frozen")
        rows = len(values["labels"])
        scaler.scale(loss * (rows / group_rows)).backward()
        numerator += float(loss.item()) * rows
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("Training produced no gradients")
    finite = torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all().item()
    if finite:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm, error_if_nonfinite=True)
    elif not amp_enabled:
        raise FloatingPointError("Nonfinite training gradients in full precision")
    previous_updates = step_counter["value"]
    scaler.step(optimizer)
    scaler.update()
    updated = step_counter["value"] > previous_updates
    if updated:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {"loss_numerator": numerator, "sample_count": group_rows, "skipped_amp_steps": int(not updated)}


def _new_tokenizer(dataset: Any, splits: Any, config: ColabConfig, dependencies: Any, tokenizer_dir: Path, *, smoke: bool) -> tuple[Any, str]:
    training = config.training
    if smoke:
        tokenizer = core._synthetic_tokenizer(dataset.texts[splits.train], dependencies, max_length=training.max_length)
        revision = "local-train-only-synthetic"
    else:
        if training.tokenizer_name_or_path != "roberta-base" or training.local_files_only:
            raise ValueError("Full Colab training downloads the public roberta-base tokenizer once; resume uses its saved local snapshot")
        from huggingface_hub import HfApi

        revision = HfApi().model_info("roberta-base", revision=training.tokenizer_revision or "main").sha
        if not revision:
            raise RuntimeError("Unable to resolve an immutable tokenizer revision")
        tokenizer = dependencies.AutoTokenizer.from_pretrained("roberta-base", revision=revision, use_fast=True)
    tokenizer.save_pretrained(tokenizer_dir)
    return tokenizer, revision


def _validate_identity(identity: dict[str, Any], *, config: ColabConfig, source_path: Path, tokenizer_dir: Path, dependencies: Any, smoke: bool) -> None:
    checks = {
        "configuration": (identity["requested_config"], asdict(config)),
        "source dataset": (identity["source_sha256"], file_sha256(source_path)),
        "source code": (identity["code_sha256"], _source_code_hashes()),
        "tokenizer snapshot": (identity["tokenizer_sha256"], _directory_hashes(tokenizer_dir)),
        "runtime/software/hardware": (identity["runtime"], _runtime_identity(dependencies)),
        "smoke mode": (identity["smoke"], smoke),
    }
    for description, (expected, observed) in checks.items():
        if expected != observed:
            raise ValueError(f"Resume/frozen run {description} changed; use the original setting or a new run directory")


def _verify_full_data(dataset: Any, splits: Any, config: ColabConfig, *, smoke: bool) -> None:
    if smoke:
        if not dataset.summary["synthetic"]:
            raise ValueError("Smoke mode is restricted to marked synthetic data")
        return
    if dataset.summary["synthetic"]:
        raise ValueError("Full training refuses synthetic data")
    if config.require_full_dataset:
        if dataset.source_sha256 != FULL_SOURCE_SHA256:
            raise ValueError("Full source SHA-256 differs from the supplied train_preprocessed.csv. Use the original file bytes; do not silently substitute or resave the dataset")
        for key, expected in FULL_DATA_EXPECTATION.items():
            observed = {name: len(indices) for name, indices in splits.as_dict().items()} if key == "split_rows" else dataset.summary[key]
            if observed != expected:
                raise ValueError(f"Full source expectation failed for {key}: expected {expected}, got {observed}. Do not silently reduce rows; confirm the original source or explicitly start a separate experiment")


def _checkpoint_payload(model: Any, optimizer: Any, scheduler: Any, scaler: Any, dependencies: Any, state: dict[str, Any], identity_hash: str) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "identity_sha256": identity_hash,
        "label_order": list(LABELS),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": capture_rng_state(dependencies.torch),
        **state,
    }


def _train_resumable(model: Any, dataset: Any, splits: Any, tokenizer: Any, criterion: Any, config: ColabConfig, dependencies: Any, device: Any, run_dir: Path, *, resume: bool, quiet: bool, stop_after_steps: int | None = None) -> dict[str, Any]:
    torch = dependencies.torch
    training = config.training
    checkpoint_path = run_dir / LATEST_CHECKPOINT
    identity_hash = file_sha256(run_dir / IDENTITY_FILE)
    microbatches = math.ceil(len(splits.train) / training.batch_size)
    planned_updates = math.ceil(microbatches / training.accumulation_steps) * training.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    scheduler = dependencies.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(training.warmup_fraction * planned_updates), num_training_steps=planned_updates)
    amp_enabled = training.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    state = {
        "epoch": 1, "microbatch_offset": 0,
        "training_loss_numerator": 0.0, "training_sample_count": 0,
        "skipped_amp_steps": 0, "epoch_start_optimizer_steps": 0,
        "global_optimizer_steps": 0, "history": [],
        "best": {"epoch": 0, "validation_loss": float("inf"), "optimizer_steps": 0},
        "epochs_without_improvement": 0, "training_complete": False,
    }
    saved_rng = None
    resumed = checkpoint_path.is_file()
    if resumed:
        if not resume:
            raise FileExistsError("An unfinished checkpoint already exists; resume it or choose a new run directory")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA or checkpoint.get("identity_sha256") != identity_hash or checkpoint.get("label_order") != list(LABELS):
            raise ValueError("Checkpoint schema, identity or label order is incompatible")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        state = {key: checkpoint[key] for key in state}
        saved_rng = checkpoint["rng_state"]
        del checkpoint
    if state["training_complete"]:
        return {**state, "resumed": resumed, "interrupted": False}
    step_counter = {"value": state["global_optimizer_steps"]}

    def record_update(*_: Any) -> None:
        step_counter["value"] += 1

    step_hook = optimizer.register_step_post_hook(record_update)

    def save_latest() -> None:
        state["global_optimizer_steps"] = step_counter["value"]
        atomic_torch_save(torch, _checkpoint_payload(model, optimizer, scheduler, scaler, dependencies, state, identity_hash), checkpoint_path)

    if not resumed:
        # Even a preemption before the first update can restart from exact state.
        save_latest()
    last_progress = time.monotonic()
    try:
        while state["epoch"] <= training.epochs:
            epoch = state["epoch"]
            loader = _epoch_loader(dataset, splits.train, tokenizer, training, dependencies, epoch)
            offset = state["microbatch_offset"]
            if not 0 <= offset <= len(loader) or (offset != len(loader) and offset % training.accumulation_steps != 0):
                raise ValueError("Checkpoint microbatch offset is not an accumulation boundary")
            iterator = iter(loader)
            for _ in range(offset):
                next(iterator)
            # Reconstructing/skipping a DataLoader must not perturb the global
            # RNG before the next unseen forward pass (dropout in particular).
            if saved_rng is not None:
                restore_rng_state(torch, saved_rng)
                saved_rng = None
            model.train()
            while True:
                micro_batches = list(itertools.islice(iterator, training.accumulation_steps))
                if not micro_batches:
                    break
                previous_updates = step_counter["value"]
                group = _optimizer_group(model, micro_batches, criterion, optimizer, scaler, scheduler, training, dependencies, device, step_counter)
                state["microbatch_offset"] += len(micro_batches)
                state["training_loss_numerator"] += group["loss_numerator"]
                state["training_sample_count"] += group["sample_count"]
                state["skipped_amp_steps"] += group["skipped_amp_steps"]
                state["global_optimizer_steps"] = step_counter["value"]
                actual_update = step_counter["value"] > previous_updates
                interval_due = actual_update and step_counter["value"] % config.checkpoint_every_updates == 0
                stop_due = stop_after_steps is not None and step_counter["value"] >= stop_after_steps
                if interval_due or stop_due:
                    save_latest()
                if not quiet and (interval_due or time.monotonic() - last_progress >= 30):
                    print(f"epoch={epoch}/{training.epochs} samples={state['training_sample_count']}/{len(splits.train)} optimizer_updates={step_counter['value']} loss={state['training_loss_numerator']/state['training_sample_count']:.6f} checkpoint_saved={interval_due or stop_due}", flush=True)
                    last_progress = time.monotonic()
                if stop_due:
                    return {**state, "resumed": resumed, "interrupted": True}
            if step_counter["value"] == state["epoch_start_optimizer_steps"]:
                raise FloatingPointError("An epoch completed without any optimizer updates")
            validation_loader = core._loader(dataset, splits.validation, tokenizer, training, dependencies, shuffle=False)
            validation_loss, _, _ = core._predict(model, validation_loader, dependencies, device, criterion=criterion, amp_enabled=amp_enabled)
            record = {
                "epoch": epoch,
                "training_loss": state["training_loss_numerator"] / state["training_sample_count"],
                "validation_loss": validation_loss,
                "optimizer_steps": step_counter["value"],
                "skipped_amp_steps": state["skipped_amp_steps"],
                "learning_rate": scheduler.get_last_lr()[0],
            }
            state["history"].append(record)
            if validation_loss < state["best"]["validation_loss"]:
                state["best"] = {"epoch": epoch, "validation_loss": validation_loss, "optimizer_steps": step_counter["value"]}
                atomic_torch_save(torch, {**state["best"], "label_order": list(LABELS), "model_state_dict": model.state_dict()}, run_dir / BEST_CHECKPOINT)
                state["epochs_without_improvement"] = 0
            else:
                state["epochs_without_improvement"] += 1
            if not quiet:
                print(f"epoch={epoch} train_loss={record['training_loss']:.6f} validation_loss={validation_loss:.6f} best_epoch={state['best']['epoch']} optimizer_updates={step_counter['value']}", flush=True)
            state.update({
                "epoch": epoch + 1, "microbatch_offset": 0,
                "training_loss_numerator": 0.0, "training_sample_count": 0,
                "skipped_amp_steps": 0, "epoch_start_optimizer_steps": step_counter["value"],
                "training_complete": epoch >= training.epochs or state["epochs_without_improvement"] >= training.patience,
            })
            save_latest()
            if state["training_complete"]:
                break
    finally:
        step_hook.remove()
    return {**state, "resumed": resumed, "interrupted": False}


def _zip_once(path: Path, entries: dict[str, bytes | Path]) -> Path:
    if path.exists():
        return path
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in entries.items():
                if isinstance(value, Path):
                    archive.write(value, arcname=name)
                else:
                    archive.writestr(name, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _review_plots(run_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = run_dir / "review_plots"
    folder.mkdir(exist_ok=True)
    history = read_json(run_dir / "training_history.json")["epochs"]
    loss_path = folder / "training_losses.png"
    if not loss_path.exists():
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot([item["epoch"] for item in history], [item["training_loss"] for item in history], marker="o", label="Training BCE")
        axis.plot([item["epoch"] for item in history], [item["validation_loss"] for item in history], marker="o", label="Validation BCE")
        axis.set(xlabel="Epoch", ylabel="Sample-weighted BCE loss", title="Random RoBERTa: training and validation")
        axis.legend()
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(loss_path, dpi=160)
        plt.close(figure)
    metrics_path = folder / "validation_per_label.png"
    if not metrics_path.exists():
        selected = read_json(run_dir / "validation_metrics.json")["metrics_at_validation_thresholds"]
        per_label = selected["per_label"]
        figure, axis = plt.subplots(figsize=(10, 4))
        positions = np.arange(len(LABELS))
        for offset, name in enumerate(("precision", "recall", "f1")):
            axis.bar(positions + (offset - 1) * 0.24, [per_label[label][name] for label in LABELS], width=0.24, label=name.title())
        axis.set_xticks(positions, LABELS, rotation=20)
        axis.set(ylim=(0, 1), ylabel="Score", title="Validation selection diagnostics at selected thresholds")
        axis.legend()
        axis.grid(axis="y", alpha=0.2)
        figure.tight_layout()
        figure.savefig(metrics_path, dpi=160)
        plt.close(figure)
    return [loss_path, metrics_path]


def package_results(run_dir: str | Path, *, verification_private: bool = False) -> dict[str, str]:
    """Make a shareable aggregate ZIP and an optional explicitly private ZIP."""
    destination = Path(run_dir).resolve()
    verify_frozen_run(destination)
    metadata = read_json(destination / "metadata.json")
    identity = read_json(destination / IDENTITY_FILE)
    whitelist = (
        "model_family", "initialization", "tokenizer_source", "tokenizer_revision", "synthetic", "engineering_smoke", "checkpoint_selection", "best_checkpoint", "train_positive_weights", "threshold_selection", "threshold_grid", "deterministic_algorithms", "cudnn_benchmark", "attention_implementation", "cublas_workspace_config", "amp_enabled", "effective_batch_size", "effective_batch_size_note", "parameter_count", "test_status", "protocol_note", "export_matches_best_checkpoint", "fresh_export_validation_loss_matches_best", "gradient_checkpointing", "checkpoint_every_updates", "resume_protocol", "training_completed_utc", "tested_environment_limitations",
    )
    test_evaluated = (destination / "public_test_results.json").exists()
    result_metadata = {
        "stage": "developed_and_frozen", "label_order": list(LABELS),
        "source_sha256": identity["source_sha256"],
        "code_sha256": identity["code_sha256"],
        "tokenizer_revision": identity["tokenizer_revision"],
        "test_evaluation_performed": test_evaluated,
        "validation_is_selection_diagnostic": True,
        "raw_texts_included": False, "per_row_arrays_included": False, "model_weights_included": False,
    }
    entries: dict[str, bytes | Path] = {name: destination / name for name in ("config.json", "data_summary.json", "training_history.json", "validation_metrics.json", "thresholds.json")}
    entries["run_metadata.json"] = _json_bytes({name: metadata[name] for name in whitelist if name in metadata})
    entries["versions_hardware.json"] = _json_bytes(identity["runtime"])
    entries["result_metadata.json"] = _json_bytes(result_metadata)
    entries["README.txt"] = (
        "Aggregate review results for the original randomly initialized RoBERTa.\n"
        "Validation selected the checkpoint and thresholds; its scores are selection diagnostics.\n"
        "The internal exploratory test remains unscored unless explicitly requested separately.\n"
        "The original unlabeled competition test CSV is not used.\n"
        "This ZIP has no raw comments, identifiers, per-row arrays, checkpoints, or model weights.\n"
        "Synthetic smoke results, when marked synthetic=true, are engineering checks only.\n"
    ).encode()
    for path in _review_plots(destination):
        entries["plots/" + path.name] = path
    if test_evaluated:
        entries["public_test_results.json"] = destination / "public_test_results.json"
    # Retain the original development bundle after a later explicit test. A new
    # filename prevents silently returning the old validation-only contents.
    review_name = "results_for_review_with_test.zip" if test_evaluated else "results_for_review.zip"
    output = {"results_for_review": str(_zip_once(destination / review_name, entries))}
    if verification_private:
        private_entries: dict[str, bytes | Path] = {
            "validation_predictions.npz": destination / "private" / "validation_predictions.npz",
            "splits.npz": destination / "private" / "splits.npz",
            "thresholds.json": destination / "thresholds.json",
            "config.json": destination / "config.json",
            "data_summary.json": destination / "data_summary.json",
            "validation_metrics.json": destination / "validation_metrics.json",
            "training_history.json": destination / "training_history.json",
            "PRIVATE_README.txt": b"PRIVATE VERIFICATION PACKAGE. Never publish or commit to GitHub.\nContains per-row labels/probabilities/indices, which may be sensitive.\nNo raw comments, original IDs, model weights or optimizer checkpoints.\n",
        }
        if (destination / "private" / "test_predictions.npz").exists():
            private_entries["test_predictions.npz"] = destination / "private" / "test_predictions.npz"
            private_entries["public_test_results.json"] = destination / "public_test_results.json"
        private_entries["verification_manifest.json"] = _json_bytes({
            "label_order": list(LABELS), "source_sha256": identity["source_sha256"],
            "code_sha256": identity["code_sha256"], "tokenizer_revision": identity["tokenizer_revision"],
            "runtime": identity["runtime"],
            "artifacts_sha256": {name: file_sha256(value) for name, value in private_entries.items() if isinstance(value, Path)},
        })
        private_name = "verification_private_with_test.zip" if test_evaluated else "verification_private.zip"
        output["verification_private"] = str(_zip_once(destination / private_name, private_entries))
    return output


def run_colab_training(data_path: str | Path, run_dir: str | Path, *, config: ColabConfig | None = None, resume: bool = True, evaluate: bool = False, verification_private: bool = False, quiet: bool = False, _smoke: bool = False, _stop_after_steps: int | None = None) -> dict[str, Any]:
    """Train/resume/freeze. Test scoring is a separate explicit opt-in."""
    config = config or ColabConfig()
    dependencies = core._dependencies()
    _assert_environment(dependencies, smoke=_smoke, config=config)
    device = core._set_determinism(dependencies, config.training)
    destination = Path(run_dir).resolve()
    supplied_source = Path(data_path).resolve()
    if not supplied_source.is_file():
        raise FileNotFoundError(f"CSV not found: {supplied_source}")
    existed = destination.exists()
    if not existed:
        destination = create_run_directory(destination)
        shutil.copyfile(supplied_source, destination / "private" / "source.csv")
    elif not resume:
        raise FileExistsError("Run directory exists; resume it or choose a new run directory")
    source_path = destination / "private" / "source.csv"
    identity_path = destination / IDENTITY_FILE
    tokenizer_dir = destination / TOKENIZER_DIRECTORY
    if not source_path.is_file() or file_sha256(supplied_source) != file_sha256(source_path):
        raise ValueError("Supplied CSV differs from the original private source snapshot")
    if existed and not identity_path.is_file():
        raise ValueError("Existing directory has no Colab run identity; choose a new run directory")
    identity = read_json(identity_path) if existed else None
    if identity is not None:
        _validate_identity(identity, config=config, source_path=source_path, tokenizer_dir=tokenizer_dir, dependencies=dependencies, smoke=_smoke)
        _recover_freeze_checksum(destination, identity, dependencies)
    if (destination / "manifest.json").exists() or (destination / "manifest.sha256").exists():
        verify_frozen_run(destination)
        if evaluate:
            core.evaluate_run(destination, device=config.training.device)
        history = read_json(destination / "training_history.json")
        return {
            "run_dir": str(destination), "stage": "already_frozen", "resumed": True,
            "history": history["epochs"], "best_checkpoint": history["best_checkpoint"],
            "global_optimizer_steps": history["epochs"][-1]["optimizer_steps"],
            "checkpoint_path": str(destination / LATEST_CHECKPOINT),
            "test_evaluated": (destination / "public_test_results.json").is_file(),
            **package_results(destination, verification_private=verification_private),
        }
    dataset = load_dataset(source_path, conflict_policy=config.training.conflict_policy, empty_text_policy=config.training.empty_text_policy)
    splits = split_dataset(dataset, seed=config.training.seed)
    _verify_full_data(dataset, splits, config, smoke=_smoke)
    split_fingerprint = {name: hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest() for name, indices in splits.as_dict().items()}
    positive_weights = core.validate_training_targets(dataset.targets[splits.train])
    if identity is not None:
        if identity["split_sha256"] != split_fingerprint:
            raise ValueError("Reconstructed split differs from the saved run identity")
        tokenizer = dependencies.AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True)
        revision = identity["tokenizer_revision"]
    else:
        tokenizer, revision = _new_tokenizer(dataset, splits, config, dependencies, tokenizer_dir, smoke=_smoke)
        identity = {
            "schema_version": CHECKPOINT_SCHEMA, "requested_config": asdict(config),
            "source_sha256": dataset.source_sha256, "split_sha256": split_fingerprint,
            "code_sha256": _source_code_hashes(), "tokenizer_sha256": _directory_hashes(tokenizer_dir),
            "tokenizer_revision": revision, "runtime": _runtime_identity(dependencies), "smoke": _smoke,
        }
        _atomic_json(identity_path, identity)
    model = core._new_model(tokenizer, dependencies, smoke=_smoke).to(device)
    if config.training.max_length + tokenizer.pad_token_id + 1 > model.config.max_position_embeddings:
        raise ValueError("max_length exceeds the configured RoBERTa position capacity")
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    criterion = dependencies.torch.nn.BCEWithLogitsLoss(pos_weight=dependencies.torch.tensor(positive_weights, dtype=dependencies.torch.float32, device=device))
    state = _train_resumable(model, dataset, splits, tokenizer, criterion, config, dependencies, device, destination, resume=resume, quiet=quiet, stop_after_steps=_stop_after_steps)
    _validate_identity(identity, config=config, source_path=source_path, tokenizer_dir=tokenizer_dir, dependencies=dependencies, smoke=_smoke)
    if state["interrupted"]:
        return {"run_dir": str(destination), "stage": "interrupted", "checkpoint_path": str(destination / LATEST_CHECKPOINT), "global_optimizer_steps": state["global_optimizer_steps"], "history": state["history"], "best_checkpoint": state["best"], "resumed": state["resumed"]}
    # Recoverable export: only manifest-free derived files are removed. Private
    # source, tokenizer, latest optimizer state and best weights remain intact.
    if (destination / "model").exists():
        shutil.rmtree(destination / "model")
    for name in ("config.json", "data_summary.json", "metadata.json", "thresholds.json", "training_history.json", "validation_metrics.json", "private/splits.npz", "private/validation_predictions.npz"):
        (destination / name).unlink(missing_ok=True)
    model, tokenizer, best = core._export_best(model, tokenizer, destination / BEST_CHECKPOINT, destination / "model", dependencies, device)
    validation_loader = core._loader(dataset, splits.validation, tokenizer, config.training, dependencies, shuffle=False)
    fresh_loss, targets, probabilities = core._predict(model, validation_loader, dependencies, device, criterion=criterion, amp_enabled=config.training.amp and device.type == "cuda")
    if not np.array_equal(targets, dataset.targets[splits.validation]):
        raise RuntimeError("Fresh exported validation inference changed row/label order")
    if not np.isclose(fresh_loss, best["validation_loss"], rtol=1e-5, atol=1e-6):
        raise RuntimeError("Fresh best export does not reproduce the selected validation loss")
    thresholds = select_thresholds(targets, probabilities)
    validation_path = destination / "private" / "validation_predictions.npz"
    np.savez_compressed(validation_path, targets=targets, probabilities=probabilities, row_indices=splits.validation)
    history = {"epochs": state["history"], "best_checkpoint": best, "fresh_validation_loss": fresh_loss}
    _atomic_json(destination / "training_history.json", history)
    _atomic_json(destination / "validation_metrics.json", {
        "split": "validation", "synthetic": bool(dataset.summary["synthetic"]),
        "metrics_at_0_5": evaluate_predictions(targets, probabilities, np.full(len(LABELS), 0.5)),
        "metrics_at_validation_thresholds": evaluate_predictions(targets, probabilities, thresholds),
        "selection_note": "Checkpoint and thresholds were selected using validation. Scores are selection diagnostics; the internal test was not used.",
    })
    metadata = {
        **runtime_metadata(config.training.seed),
        "model_family": "custom RoBERTa sequence classifier",
        "initialization": "random weights; no pretrained model weights",
        "tokenizer_source": "locally constructed train-only synthetic tokenizer" if _smoke else "roberta-base tokenizer only",
        "tokenizer_revision": revision,
        "synthetic": bool(dataset.summary["synthetic"]), "engineering_smoke": _smoke,
        "checkpoint_selection": "minimum sample-weighted validation BCE loss", "best_checkpoint": best,
        "train_positive_weights": dict(zip(LABELS, positive_weights.tolist())),
        "threshold_selection": "per-label validation F1; >= comparator; ties nearest 0.5 then larger; missing class keeps 0.5",
        "threshold_grid": list(DEFAULT_THRESHOLD_GRID), "deterministic_algorithms": True,
        "cudnn_benchmark": False, "attention_implementation": "eager", "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "amp_enabled": config.training.amp and device.type == "cuda",
        "effective_batch_size": config.training.batch_size * config.training.accumulation_steps,
        "effective_batch_size_note": "Nominal maximum; each accumulation group is weighted by its actual sample count.",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "test_status": "unscored during development",
        "protocol_note": "New internal exploratory holdout; the historical cohort was previously examined, so this is not a pristine external test.",
        "export_matches_best_checkpoint": True, "fresh_export_validation_loss_matches_best": True,
        "gradient_checkpointing": config.gradient_checkpointing, "checkpoint_every_updates": config.checkpoint_every_updates,
        "resume_protocol": "Exact code/config/source/tokenizer/software/hardware fingerprints; accumulation boundaries; global RNG and deterministic per-epoch loader restored.",
        "runner_code_sha256": identity["code_sha256"]["scripts/colab_train.py"],
        "training_completed_utc": runtime_metadata(config.training.seed)["created_utc"],
        "tested_environment_limitations": "Python 3.12 and torch 2.8.0+cpu: tiny offline mechanics only. Preserve Colab CUDA torch >=2.8; CUDA execution, memory and full-data model quality must be verified in the user's runtime. Exact continuation requires the recorded versions and GPU family/capability.",
    }
    architecture = {name: getattr(model.config, name) for name in core.ORIGINAL_ARCHITECTURE}
    _validate_identity(identity, config=config, source_path=source_path, tokenizer_dir=tokenizer_dir, dependencies=dependencies, smoke=_smoke)
    if not state["training_complete"] or (destination / TEST_MARKER).exists():
        raise ValueError("Freezing requires completed training before any test access")
    _atomic_json(destination / FREEZE_TRANSACTION, {
        "schema_version": 1,
        "identity_sha256": file_sha256(identity_path),
        "latest_checkpoint_sha256": file_sha256(destination / LATEST_CHECKPOINT),
        "best_checkpoint_sha256": file_sha256(destination / BEST_CHECKPOINT),
    })
    freeze_run(
        destination,
        config={"method": "random_initialized_roberta", "training": asdict(config.training), "architecture": architecture, "label_order": list(LABELS), "colab": {"gradient_checkpointing": config.gradient_checkpointing, "checkpoint_every_updates": config.checkpoint_every_updates, "require_full_dataset": config.require_full_dataset}},
        data_summary={**dataset.summary, "splits": split_summary(dataset, splits)},
        source_path=source_path, model_path=destination / "model", thresholds=thresholds, splits=splits, metadata=metadata,
        extra_artifacts=[destination / relative for relative in FREEZE_EXTRA_ARTIFACTS],
    )
    verify_frozen_run(destination)
    (destination / FREEZE_TRANSACTION).unlink()
    if evaluate:
        core.evaluate_run(destination, device=config.training.device)
    return {
        "run_dir": str(destination), "stage": "developed_and_frozen", "resumed": state["resumed"],
        "global_optimizer_steps": state["global_optimizer_steps"], "history": state["history"], "best_checkpoint": best,
        "checkpoint_path": str(destination / LATEST_CHECKPOINT), "test_evaluated": evaluate,
        **package_results(destination, verification_private=verification_private),
    }


def run_smoke(run_dir: str | Path) -> dict[str, Any]:
    """Offline interrupt/resume check, including an epoch and partial groups."""
    parent = Path(run_dir).resolve()
    parent.mkdir(parents=True, exist_ok=False)
    source = create_synthetic_fixture(parent / "synthetic.csv", rows=240, seed=42)
    config = ColabConfig(training=core.TrainConfig(
        epochs=2, patience=2, batch_size=13, accumulation_steps=5,
        learning_rate=1e-3, max_length=16, cpu_threads=1,
        device="cpu", amp=False, local_files_only=True,
    ), checkpoint_every_updates=1, require_full_dataset=False)
    continuous = run_colab_training(source, parent / "continuous", config=config, quiet=True, _smoke=True)
    interrupted = run_colab_training(source, parent / "resumed", config=config, quiet=True, _smoke=True, _stop_after_steps=2)
    resumed = run_colab_training(source, parent / "resumed", config=config, quiet=True, _smoke=True)
    dependencies = core._dependencies()
    first = dependencies.torch.load(parent / "continuous" / LATEST_CHECKPOINT, map_location="cpu", weights_only=True)
    second = dependencies.torch.load(parent / "resumed" / LATEST_CHECKPOINT, map_location="cpu", weights_only=True)
    exact_weights = all(dependencies.torch.equal(value, second["model_state_dict"][name]) for name, value in first["model_state_dict"].items())
    with np.load(parent / "continuous" / "private" / "validation_predictions.npz", allow_pickle=False) as one, np.load(parent / "resumed" / "private" / "validation_predictions.npz", allow_pickle=False) as two:
        exact_probabilities = np.array_equal(one["probabilities"], two["probabilities"])
    if not exact_weights or not exact_probabilities or continuous["history"] != resumed["history"]:
        raise RuntimeError("Interrupted/resumed smoke differs from uninterrupted training")
    return {
        "status": "passed", "synthetic": True, "purpose": "CPU engineering mechanics only; no real-data or CUDA performance claim",
        "run_dir": str(parent), "interrupted_at_updates": interrupted["global_optimizer_steps"],
        "total_optimizer_updates": resumed["global_optimizer_steps"],
        "checks": {"exact_final_weights": exact_weights, "exact_exported_validation_probabilities": exact_probabilities, "exact_history": True, "epoch_and_partial_accumulation_groups": True, "test_unscored": not resumed["test_evaluated"], "local_train_only_tokenizer": True, "safe_weights_only_checkpoint": True},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    develop = stages.add_parser("develop", help="Train/resume the full original GPU experiment; test stays unscored")
    develop.add_argument("--data", required=True, type=Path)
    develop.add_argument("--run-dir", required=True, type=Path)
    develop.add_argument("--gradient-checkpointing", action="store_true", help="Explicit memory-saving choice recorded in config; cannot change during resume")
    develop.add_argument("--evaluate", action="store_true", help="Opt in to the frozen single-use internal test AFTER development")
    develop.add_argument("--verification-private", action="store_true", help="Also make a private per-row verification ZIP; never publish it")
    evaluate = stages.add_parser("evaluate", help="Explicitly score a frozen run once, using core test guards")
    evaluate.add_argument("--run-dir", required=True, type=Path)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    smoke = stages.add_parser("smoke", help="Tiny offline CPU exact-resume mechanics, never real-data training")
    smoke.add_argument("--run-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.stage == "smoke":
            result = run_smoke(args.run_dir)
        elif args.stage == "evaluate":
            result = {"evaluation": core.evaluate_run(args.run_dir, device=args.device), **package_results(args.run_dir)}
        else:
            result = run_colab_training(args.data, args.run_dir, config=ColabConfig(gradient_checkpointing=args.gradient_checkpointing), evaluate=args.evaluate, verification_private=args.verification_private)
    except (ImportError, OSError, ValueError, RuntimeError, FloatingPointError) as error:
        parser.exit(2, f"colab: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
