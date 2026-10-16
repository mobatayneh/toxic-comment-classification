"""New-run Colab resource calibration and saved-configuration launch commands.

The calibration uses synthetic, full-length inputs and disposable random models.
It selects an execution microbatch, not model parameters from validation/test
scores. Its throughput is not a measured full-dataset training speedup.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts import colab_train as runner

PROFILE = "random-roberta-efficient-v1"
EFFECTIVE_BATCH = 256
CANDIDATES = (8, 16, 32, 64)
MEMORY_FRACTION_LIMIT = 0.85
MINIMUM_FREE_BYTES = 1024 ** 3


class _TokenizerContract:
    """Vocabulary contract for synthetic profiling; downloads no assets."""
    vocab_size = 50265
    bos_token_id = 0
    pad_token_id = 1
    eos_token_id = 2
    unk_token_id = 3

    def __len__(self) -> int:
        return self.vocab_size


def efficient_config(batch_size: int, cpu_threads: int) -> runner.ColabConfig:
    if isinstance(batch_size, bool) or batch_size not in CANDIDATES:
        raise ValueError("Calibration batch size must be one of 8, 16, 32 or 64")
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or not 1 <= cpu_threads <= 4:
        raise ValueError("CPU thread count must be between one and four")
    training = runner.core.TrainConfig(
        device="cuda", conflict_policy="keep", empty_text_policy="drop",
        batch_size=batch_size, accumulation_steps=EFFECTIVE_BATCH // batch_size,
        cpu_threads=cpu_threads, cache_tokenization=True, dynamic_padding=True,
    )
    return runner.ColabConfig(
        training=training, gradient_checkpointing=False,
        checkpoint_every_updates=100, checkpoint_every_seconds=300.0,
    )


def select_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = []
    for record in records:
        throughput = record.get("samples_per_second", 0)
        if (record.get("status") == "ok" and record.get("batch_size") in CANDIDATES
                and isinstance(throughput, (int, float)) and math.isfinite(throughput)
                and throughput > 0 and record.get("within_memory_budget") is True):
            valid.append(record)
    if not valid:
        raise RuntimeError("No candidate completed within the GPU memory budget. No training run was created.")
    # For equal measurements prefer the smaller microbatch and its extra margin.
    return max(valid, key=lambda record: (record["samples_per_second"], -record["batch_size"]))


def _probe(batch_size: int, cpu_threads: int, dependencies: Any) -> dict[str, Any]:
    torch = dependencies.torch
    config = efficient_config(batch_size, cpu_threads)
    device = runner.core._set_determinism(dependencies, config.training)
    model = optimizer = scheduler = scaler = group = criterion = hook = None
    record = {"batch_size": batch_size, "accumulation_steps": EFFECTIVE_BATCH // batch_size}
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = runner.core._new_model(_TokenizerContract(), dependencies, smoke=False).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate,
                                     weight_decay=config.training.weight_decay)
        scheduler = dependencies.get_cosine_schedule_with_warmup(optimizer, 0, 12)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        counter = {"value": 0}
        def updated(*_):
            counter["value"] += 1
        hook = optimizer.register_step_post_hook(updated)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones(len(runner.LABELS), device=device))
        # All inputs have 256 real tokens: a conservative activation-memory case.
        # No text, validation examples or labels from the real dataset are used.
        batch = {
            "input_ids": torch.randint(4, 50265, (batch_size, 256), dtype=torch.long).pin_memory(),
            "attention_mask": torch.ones((batch_size, 256), dtype=torch.long).pin_memory(),
            "labels": (torch.arange(batch_size * len(runner.LABELS)).reshape(batch_size, -1) % 2).float().pin_memory(),
        }
        batch["input_ids"][:, 0] = 0
        batch["input_ids"][:, -1] = 2
        group = [batch] * config.training.accumulation_steps
        durations = []
        successful_groups = 0
        skipped_groups = 0
        for attempt in range(12):
            torch.cuda.synchronize()
            start = time.perf_counter()
            outcome = runner._optimizer_group(model, group, criterion, optimizer, scaler,
                                               scheduler, config.training, dependencies, device, counter)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if outcome["skipped_amp_steps"]:
                skipped_groups += 1
                continue
            successful_groups += 1
            if successful_groups > 1:
                durations.append(elapsed)
            if len(durations) == 2:
                break
        if len(durations) != 2:
            raise RuntimeError("AMP did not stabilize within the bounded probe; no reliable calibration was recorded")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        peak_bytes = torch.cuda.max_memory_reserved()
        within_budget = peak_bytes <= MEMORY_FRACTION_LIMIT * total_bytes and free_bytes >= MINIMUM_FREE_BYTES
        record.update({
            "status": "ok", "within_memory_budget": within_budget,
            "samples_per_second": EFFECTIVE_BATCH / statistics.median(durations),
            "measured_group_seconds": durations, "warmup_groups": 1,
            "skipped_amp_groups": skipped_groups,
            "measured_groups": len(durations), "max_length": 256,
            "peak_reserved_gpu_bytes": peak_bytes,
            "free_gpu_bytes_after_probe": free_bytes, "total_gpu_bytes": total_bytes,
        })
    except torch.cuda.OutOfMemoryError:
        record.update({"status": "cuda_out_of_memory", "within_memory_budget": False})
    finally:
        if hook is not None:
            hook.remove()
        model = optimizer = scheduler = scaler = group = criterion = None
        gc.collect()
        torch.cuda.empty_cache()
    return record


def _calibration_identity(dependencies: Any) -> dict[str, Any]:
    return {"profile": PROFILE, "profile_sha256": runner.file_sha256(Path(__file__)),
            "runner_code_sha256": runner._source_code_hashes(),
            "runtime": runner._runtime_identity(dependencies)}


def _validate_calibration(value: dict[str, Any], dependencies: Any) -> None:
    for key, expected in _calibration_identity(dependencies).items():
        if value.get(key) != expected:
            raise ValueError(f"Saved calibration {key} differs. Use its original environment or a NEW run name.")
    config = efficient_config(value["batch_size"], value["cpu_threads"])
    if value.get("requested_config") != asdict(config):
        raise ValueError("Saved calibration config is inconsistent")
    chosen = select_candidate(value["records"])
    if chosen["batch_size"] != value["batch_size"]:
        raise ValueError("Calibration selection does not match its recorded measurements")


def calibrate(output: Path) -> dict[str, Any]:
    dependencies = runner.core._dependencies()
    threads = max(1, min(4, os.cpu_count() or 1))
    runner._assert_environment(dependencies, smoke=False, config=efficient_config(8, threads))
    if output.is_file():
        value = json.loads(output.read_text())
        _validate_calibration(value, dependencies)
        print("Reusing saved calibration; no benchmark repeated.", flush=True)
        return value
    records = []
    for batch_size in CANDIDATES:
        record = _probe(batch_size, threads, dependencies)
        records.append(record)
        print(json.dumps({"calibration_candidate": record}, allow_nan=False), flush=True)
        if record["status"] == "cuda_out_of_memory":
            # Larger full-length microbatches cannot improve this memory bound.
            break
    chosen = select_candidate(records)
    config = efficient_config(chosen["batch_size"], threads)
    value = {
        **_calibration_identity(dependencies),
        "batch_size": config.training.batch_size,
        "accumulation_steps": config.training.accumulation_steps,
        "effective_batch_size": EFFECTIVE_BATCH, "cpu_threads": threads,
        "gradient_checkpointing": False, "cache_tokenization": True, "dynamic_padding": True,
        "requested_config": asdict(config), "records": records,
        "selection": "Highest measured synthetic full-length throughput within memory budget; ties prefer smaller batch",
        "memory_fraction_limit": MEMORY_FRACTION_LIMIT,
        "minimum_free_gpu_bytes": MINIMUM_FREE_BYTES,
        "limitations": "A short synthetic resource calibration, not an accuracy experiment or an end-to-end speedup measurement. Actual padding mix, data preparation and checkpoint I/O change throughput.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    runner._atomic_json(output, value)
    return value


def saved_config(identity: dict[str, Any]) -> runner.ColabConfig:
    requested = dict(identity["requested_config"])
    training = runner.core.TrainConfig(**requested.pop("training"))
    config = runner.ColabConfig(training=training, **requested)
    if asdict(config) != identity["requested_config"]:
        raise ValueError("Saved config could not be reconstructed exactly")
    if not training.cache_tokenization or not training.dynamic_padding:
        raise ValueError("This is not the efficient profile. Keep the old experiment and use a NEW run name.")
    if asdict(config) != asdict(efficient_config(training.batch_size, training.cpu_threads)):
        raise ValueError("Saved experiment is not the fixed efficient-v1 profile; use its original notebook.")
    return config


def train(data: Path, run_dir: Path, calibration_path: Path) -> dict[str, Any]:
    identity_path = run_dir / runner.IDENTITY_FILE
    if identity_path.is_file():
        config = saved_config(json.loads(identity_path.read_text()))
        print("Resuming the saved configuration. Calibration will not be repeated.", flush=True)
    else:
        if run_dir.exists():
            raise FileExistsError("Run directory exists without a valid identity. Keep it and choose a NEW run name.")
        dependencies = runner.core._dependencies()
        calibration = json.loads(calibration_path.read_text())
        _validate_calibration(calibration, dependencies)
        config = efficient_config(calibration["batch_size"], calibration["cpu_threads"])
        print("Starting a NEW full-data experiment with random model weights.", flush=True)
    print(json.dumps({"saved_training_plan": asdict(config), "test_evaluation_enabled": False}), flush=True)
    return runner.run_colab_training(data, run_dir, config=config, resume=True,
                                     evaluate=False, verification_private=False)


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def status(run_dir: Path) -> dict[str, Any]:
    if not run_dir.is_dir():
        return {"stage": "not_started", "next_step": "Run setup, calibration and training in order."}
    output = {"stage": "unfinished", "final_artifacts_present": all(
        (run_dir / item).is_file() for item in ("manifest.json", "manifest.sha256", "training_history.json"))}
    checkpoint_path = run_dir / runner.LATEST_CHECKPOINT
    if checkpoint_path.is_file():
        dependencies = runner.core._dependencies()
        checkpoint = dependencies.torch.load(str(checkpoint_path), map_location="cpu", weights_only=True, mmap=True)
        if checkpoint.get("identity_sha256") != runner.file_sha256(run_dir / runner.IDENTITY_FILE):
            raise ValueError("Latest checkpoint and identity do not match")
        output["latest_checkpoint"] = {key: checkpoint.get(key) for key in
            ("epoch", "microbatch_offset", "global_optimizer_steps", "training_complete", "best", "history")}
        del checkpoint
    if output["final_artifacts_present"]:
        runner.verify_frozen_run(run_dir)
        output["stage"] = "frozen"
    else:
        output["next_step"] = "Resume training with the SAME notebook and run name; do not package unfinished results."
    return _finite(output)


def package(run_dir: Path) -> dict[str, Any]:
    if not all((run_dir / item).is_file() for item in ("manifest.json", "manifest.sha256", "training_history.json")):
        return {"stage": "unfinished", "next_step": "Resume training before packaging. No result bundle was created."}
    return {"stage": "frozen", **runner.package_results(run_dir, verification_private=False)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--output", type=Path, required=True)
    training = commands.add_parser("train")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--run-dir", type=Path, required=True)
    training.add_argument("--calibration", type=Path, required=True)
    for name in ("status", "package"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.stage == "calibrate":
            value = calibrate(args.output)
        elif args.stage == "train":
            value = train(args.data, args.run_dir, args.calibration)
        elif args.stage == "status":
            value = status(args.run_dir)
        else:
            value = package(args.run_dir)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        parser.exit(2, f"colab-profile: {error}\n")
    print(json.dumps(value, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
