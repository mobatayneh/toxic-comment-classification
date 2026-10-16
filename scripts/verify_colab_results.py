"""Audit returned efficient-v1 Colab bundles and publish aggregate evidence.

``import`` needs the original CSV and the private verification ZIP. It never
extracts or writes per-row arrays and never loads weights, pickle, or a model.
``verify-public`` checks the published aggregates and their checksums without
private data. Public checks cannot repeat the original per-row numerical audit.
Checksums establish consistency, not cryptographic proof of execution.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import statistics
import subprocess
import sys
import tempfile
from typing import Any
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts import colab_profile, colab_train
from toxic_comments.data import LABELS, file_sha256, load_dataset, split_dataset, split_summary
from toxic_comments.metrics import DEFAULT_THRESHOLD_GRID, evaluate_predictions, select_thresholds
from toxic_comments.neural import ORIGINAL_ARCHITECTURE

COMMON_JSON = frozenset({"config.json", "data_summary.json", "training_history.json",
                         "validation_metrics.json", "thresholds.json"})
REVIEW_JSON = COMMON_JSON | {"run_metadata.json", "versions_hardware.json", "result_metadata.json"}
REVIEW_FILES = REVIEW_JSON | {"README.txt", "plots/training_losses.png", "plots/validation_per_label.png"}
PRIVATE_FILES = COMMON_JSON | {"validation_predictions.npz", "splits.npz", "PRIVATE_README.txt", "verification_manifest.json"}
PRIVATE_HASHED = COMMON_JSON | {"validation_predictions.npz", "splits.npz"}
REPORT_JSON = REVIEW_JSON | {"calibration.json", "verification.json", "plot_data.json"}
PLOTS = {"plots/training_losses.png", "plots/validation_per_label.png"}
COMPARISON_FILE = "baseline_validation_comparison.json"
COMPARISON_PLOT = "plots/validation_baseline_comparison.png"
TOLERANCE = 1e-12
FORBIDDEN_KEYS = {"id", "ids", "comment_text", "texts", "comments", "raw_comments", "records",
                  "row_indices", "targets", "predictions", "probabilities", "source_path", "model_path",
                  "run_dir", "tokenizer_sha256", "git_revision"}
LIMITATIONS = [
    "Validation selected the checkpoint and thresholds; scores are selection diagnostics, not an unbiased test estimate.",
    "No model weights, optimizer checkpoint, tokenizer snapshot, or run identity file were returned; model inference, epoch losses, and tokenizer-file hashes were not independently reproduced.",
    "Hardware, software, export-success flags, and training completion time are reported metadata reconciled across supplied artifacts; their execution provenance is not independently authenticated.",
    "The neural internal exploratory test remains unscored in these bundles. The challenge test CSV has no usable labels and was not scored.",
    "No confidence interval, probability calibration, subgroup fairness, robustness, or end-to-end speedup result is established.",
    "SHA-256 manifests detect changes and cross-file inconsistencies; they are not signed attestations or tamper-proof evidence.",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def strict_json(value: bytes | str) -> Any:
    def pairs(items):
        result = {}
        for key, child in items:
            require(key not in result, "Duplicate JSON object key")
            result[key] = child
        return result

    def invalid(_):
        raise ValueError("Non-finite JSON number")

    return json.loads(value, object_pairs_hook=pairs, parse_constant=invalid)


def assert_close(actual: Any, expected: Any, path: str = "value") -> None:
    """Compare every aggregate field; tolerate only floating point rounding."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"{path}: object keys differ")
        for key, value in expected.items():
            assert_close(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{path}: list shape differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_close(left, right, f"{path}[{index}]")
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=0, abs_tol=TOLERANCE),
                f"{path}: numerical value differs")
    else:
        require(type(actual) is type(expected) and actual == expected, f"{path}: value differs")


def read_bundle(path: Path, expected: set[str] | frozenset[str]) -> dict[str, bytes]:
    """Whitelist bounded regular files and read in memory; never extractall."""
    require(path.is_file() and path.stat().st_size <= 16 * 1024**2, "Bundle missing or exceeds 16 MiB")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        require(len(names) == len(set(names)), "Duplicate ZIP entry")
        require(set(names) == set(expected), "Unexpected or missing ZIP entries; only validation-only bundles are accepted")
        require(sum(entry.file_size for entry in entries) <= 32 * 1024**2, "ZIP expands beyond limit")
        result = {}
        for entry in entries:
            name = entry.filename
            parts = PurePosixPath(name).parts
            require(not name.startswith(("/", "\\")) and "\\" not in name
                    and all(part not in {".", ".."} for part in parts)
                    and not re.match(r"^[A-Za-z]:", name), "Unsafe ZIP path")
            mode = entry.external_attr >> 16
            require(not entry.is_dir() and not stat.S_ISLNK(mode)
                    and stat.S_IFMT(mode) in {0, stat.S_IFREG}, "Nonregular ZIP entry")
            require(not entry.flag_bits & 1 and entry.file_size <= 16 * 1024**2, "Encrypted or oversized ZIP entry")
            # ZipFile.read also checks the CRC, truncation, and overlap guards.
            result[name] = archive.read(entry)
        return result


def load_npz(value: bytes, expected: set[str]) -> dict[str, np.ndarray]:
    """Bound nested NPZ contents before numpy reads; object arrays are forbidden."""
    with zipfile.ZipFile(io.BytesIO(value)) as archive:
        entries = archive.infolist()
        require(len(entries) == len(expected)
                and {entry.filename for entry in entries} == {key + ".npy" for key in expected},
                "Unexpected NPZ entries")
        require(sum(entry.file_size for entry in entries) <= 32 * 1024**2, "NPZ expands beyond limit")
    with np.load(io.BytesIO(value), allow_pickle=False) as arrays:
        require(set(arrays.files) == expected, "Unexpected NPZ arrays")
        return {key: arrays[key] for key in arrays.files}


def validate_arrays(targets: np.ndarray, scores: np.ndarray) -> None:
    require(targets.ndim == 2 and targets.shape == scores.shape
            and targets.shape[1] == len(LABELS) and len(targets) > 0, "Prediction array shape mismatch")
    require(np.issubdtype(targets.dtype, np.number) and np.isin(targets, [0, 1]).all(), "Nonbinary targets")
    require(np.issubdtype(scores.dtype, np.floating) and np.isfinite(scores).all()
            and ((scores >= 0) & (scores <= 1)).all(), "Invalid probabilities")


def independent_thresholds(targets: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Reimplement bounded F1 selection using confusion counts, not project code."""
    scores = scores.astype(np.float64)
    result = []
    for column in range(len(LABELS)):
        truth = targets[:, column].astype(bool)
        if not truth.any() or truth.all():
            result.append(0.5)
            continue
        candidates = []
        for threshold in DEFAULT_THRESHOLD_GRID:
            prediction = scores[:, column] >= threshold
            tp = int((prediction & truth).sum())
            denominator = int(prediction.sum()) + int(truth.sum())
            candidates.append((2 * tp / denominator if denominator else 0, threshold))
        best = max(score for score, _ in candidates)
        ties = [threshold for score, threshold in candidates if abs(score - best) <= TOLERANCE]
        result.append(sorted(ties, key=lambda threshold: (round(abs(threshold - 0.5), 12), -threshold))[0])
    return np.asarray(result)


def independent_metrics(targets: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> dict[str, Any]:
    """Use sklearn ranking/confusion primitives independently of evaluate_predictions."""
    validate_arrays(targets, scores)
    require(thresholds.shape == (len(LABELS),) and np.isfinite(thresholds).all()
            and ((thresholds >= 0) & (thresholds <= 1)).all(), "Invalid thresholds")
    # Project scores are evaluated as float64. This also avoids float32 scalar
    # casting changing the >= decision at a decimal threshold boundary.
    scores = scores.astype(np.float64)
    predictions = scores >= thresholds
    per_label = {}
    ranking_labels, f1_labels = [], []
    for column, label in enumerate(LABELS):
        target, prediction = targets[:, column], predictions[:, column]
        tn, fp, fn, tp = (int(x) for x in confusion_matrix(target, prediction, labels=[0, 1]).ravel())
        positives, negatives = tp + fn, tn + fp
        both = positives > 0 and negatives > 0
        has_positive = positives > 0
        if both:
            ranking_labels.append(label)
        if has_positive:
            f1_labels.append(label)
        per_label[label] = {
            "positive_support": positives, "negative_support": negatives, "predicted_positive": tp + fp,
            "average_precision": float(average_precision_score(target, scores[:, column])) if both else None,
            "roc_auc": float(roc_auc_score(target, scores[:, column])) if both else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if has_positive else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / positives if has_positive else None,
            "threshold": float(thresholds[column]),
            "ranking_status": "defined" if both else "no_positives" if not positives else "no_negatives",
            "f1_status": "defined" if has_positive else "no_positives",
            "confusion": {"true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn},
        }
    def mean(key, labels):
        return float(np.mean([per_label[label][key] for label in labels])) if labels else None
    flat = targets.ravel()
    flat_both = bool(flat.any() and (flat == 0).any())
    return {
        "primary_metric": "macro_average_precision", "rows": len(targets),
        "macro_average_precision": mean("average_precision", ranking_labels) if len(ranking_labels) == 6 else None,
        "macro_average_precision_eligible_labels": mean("average_precision", ranking_labels),
        "macro_roc_auc": mean("roc_auc", ranking_labels) if len(ranking_labels) == 6 else None,
        "macro_roc_auc_eligible_labels": mean("roc_auc", ranking_labels),
        "macro_f1": mean("f1", f1_labels) if len(f1_labels) == 6 else None,
        "macro_f1_eligible_labels": mean("f1", f1_labels),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)) if flat.any() else None,
        "micro_average_precision": float(average_precision_score(flat, scores.ravel())) if flat_both else None,
        "coverage": {"total_labels": 6, "ranking_eligible_count": len(ranking_labels), "ranking_eligible_labels": ranking_labels,
                     "f1_eligible_count": len(f1_labels), "f1_eligible_labels": f1_labels,
                     "macro_averaging": "six-label macro scores require full coverage; separately named eligible-label diagnostics use their stated subset"},
        "per_label": per_label,
    }


def audit_predictions(arrays: dict[str, np.ndarray], targets: np.ndarray, indices: np.ndarray,
                      metrics: dict[str, Any], thresholds: np.ndarray) -> None:
    require(np.issubdtype(arrays["row_indices"].dtype, np.integer)
            and np.array_equal(arrays["row_indices"], indices), "Prediction rows/order differ from reconstructed validation split")
    require(np.array_equal(arrays["targets"], targets), "Prediction targets/label order differ from original source")
    validate_arrays(arrays["targets"], arrays["probabilities"])
    require(np.array_equal(independent_thresholds(targets, arrays["probabilities"]), thresholds), "Independent threshold selection differs")
    require(np.array_equal(select_thresholds(targets, arrays["probabilities"]), thresholds), "Project threshold selection differs")
    for key, values in (("metrics_at_0_5", np.full(6, 0.5)), ("metrics_at_validation_thresholds", thresholds)):
        assert_close(metrics[key], independent_metrics(targets, arrays["probabilities"], values), key + ".independent")
        assert_close(metrics[key], evaluate_predictions(targets, arrays["probabilities"], values), key + ".project")


def verify_code_identity(code: dict[str, str], *, head: bool) -> None:
    expected = colab_train._source_code_hashes()
    require(code == expected, "Returned trained-code SHA-256s differ from local source")
    if head:
        for name, value in expected.items():
            content = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=ROOT)
            require(digest(content) == value, "Trained source differs from local HEAD: " + name)


def check_history(config: dict[str, Any], history: dict[str, Any], metadata: dict[str, Any], train_rows: int) -> None:
    epochs = history["epochs"]
    require([row["epoch"] for row in epochs] == list(range(1, config["training"]["epochs"] + 1))
            and len(epochs) == 5, "Expected complete ordered five-epoch history")
    training = config["training"]
    updates = math.ceil(math.ceil(train_rows / training["batch_size"]) / training["accumulation_steps"])
    for index, row in enumerate(epochs, 1):
        require(set(row) == {"epoch", "learning_rate", "optimizer_steps", "skipped_amp_steps", "training_loss", "validation_loss"}, "Unexpected epoch fields")
        require(row["optimizer_steps"] == index * updates and row["skipped_amp_steps"] == 0,
                "Optimizer updates/AMP skips do not reconcile")
        for key in ("training_loss", "validation_loss", "learning_rate"):
            require(isinstance(row[key], (float, int)) and math.isfinite(row[key]) and row[key] >= 0, "Invalid history number")
    best = min(epochs, key=lambda row: row["validation_loss"])
    require(best["epoch"] == 4, "Returned efficient-v1 checkpoint was expected at epoch four")
    expected = {key: best[key] for key in ("epoch", "optimizer_steps", "validation_loss")}
    assert_close(history["best_checkpoint"], expected, "history.best_checkpoint")
    assert_close(metadata["best_checkpoint"], expected, "metadata.best_checkpoint")
    assert_close(history["fresh_validation_loss"], best["validation_loss"], "fresh_validation_loss")
    require(epochs[-1]["learning_rate"] == 0, "Final scheduler learning rate is not zero")


def check_calibration(calibration: dict[str, Any], config: dict[str, Any], runtime: dict[str, Any], code: dict[str, str]) -> None:
    require(set(calibration) == {"accumulation_steps", "batch_size", "cache_tokenization", "cpu_threads", "dynamic_padding", "effective_batch_size", "gradient_checkpointing", "limitations", "memory_fraction_limit", "minimum_free_gpu_bytes", "profile", "profile_sha256", "records", "requested_config", "runner_code_sha256", "runtime", "selection"}, "Unexpected calibration fields")
    require(calibration["profile"] == colab_profile.PROFILE, "Calibration profile differs")
    require(calibration["profile_sha256"] == file_sha256(ROOT / "scripts/colab_profile.py"), "Calibration profile source changed")
    require(calibration["runner_code_sha256"] == code and calibration["runtime"] == runtime, "Calibration code/runtime identity differs")
    expected = asdict(colab_profile.efficient_config(calibration["batch_size"], calibration["cpu_threads"]))
    assert_close(calibration["requested_config"], expected, "calibration.requested_config")
    require(config["training"] == expected["training"] and config["colab"] == {key: value for key, value in expected.items() if key != "training"}, "Calibration/training configurations differ")
    for key in ("accumulation_steps", "batch_size", "cpu_threads", "cache_tokenization", "dynamic_padding"):
        require(calibration[key] == config["training"][key], "Calibration execution setting differs: " + key)
    require(calibration["effective_batch_size"] == 256 and calibration["gradient_checkpointing"] is False
            and calibration["memory_fraction_limit"] == colab_profile.MEMORY_FRACTION_LIMIT
            and calibration["minimum_free_gpu_bytes"] == colab_profile.MINIMUM_FREE_BYTES, "Calibration policy differs")
    records = calibration["records"]
    require([row["batch_size"] for row in records] == list(colab_profile.CANDIDATES), "Expected all four calibration probes")
    total_memory = runtime["hardware"]["gpus"][0]["total_memory_bytes"]
    for record in records:
        require(set(record) == {"accumulation_steps", "batch_size", "free_gpu_bytes_after_probe", "max_length", "measured_group_seconds", "measured_groups", "peak_reserved_gpu_bytes", "samples_per_second", "skipped_amp_groups", "status", "total_gpu_bytes", "warmup_groups", "within_memory_budget"}, "Unexpected calibration probe fields")
        require(record["accumulation_steps"] * record["batch_size"] == 256, "Probe effective batch differs")
        require(record["status"] == "ok" and record["measured_groups"] == 2
                and len(record["measured_group_seconds"]) == 2 and record["warmup_groups"] == 1
                and record["max_length"] == 256 and record["skipped_amp_groups"] == 0, "Incomplete calibration probe")
        require(all(math.isfinite(value) and value > 0 for value in record["measured_group_seconds"]), "Invalid calibration timing")
        assert_close(record["samples_per_second"], 256 / statistics.median(record["measured_group_seconds"]), "probe.samples_per_second")
        require(record["total_gpu_bytes"] == total_memory, "Probe GPU size differs")
        within = record["peak_reserved_gpu_bytes"] <= 0.85 * total_memory and record["free_gpu_bytes_after_probe"] >= 1024**3
        require(record["within_memory_budget"] is within, "Probe memory policy differs")
    require(colab_profile.select_candidate(records)["batch_size"] == calibration["batch_size"], "Calibration chose the wrong candidate")


def check_no_private_fields(value: Any, *, allow_probe_records: bool = False) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            require(key.lower() not in FORBIDDEN_KEYS or (key == "records" and allow_probe_records), "Private per-row/path field in aggregate output")
            check_no_private_fields(child, allow_probe_records=allow_probe_records)
    elif isinstance(value, list):
        for child in value:
            check_no_private_fields(child, allow_probe_records=allow_probe_records)
    elif isinstance(value, str):
        require(not re.search(r"(?<![\w.])[0-9a-f]{16}(?![\w.])", value, re.I), "Potential raw row identifier in output")
        require("/workspace/" not in value and "/content/drive/" not in value, "Private absolute path in output")
    elif isinstance(value, float):
        require(math.isfinite(value), "Nonfinite aggregate number")


def check_aggregate_metrics(metrics: dict[str, Any], thresholds: list[float], support: dict[str, int], rows: int) -> None:
    """Public-only arithmetic checks; rankings still need private probabilities."""
    require(metrics["rows"] == rows and metrics["primary_metric"] == "macro_average_precision", "Aggregate population differs")
    require(set(metrics["per_label"]) == set(LABELS), "Aggregate label set differs")
    sums = {key: 0 for key in ("true_positive", "false_positive", "false_negative", "true_negative")}
    for column, label in enumerate(LABELS):
        row = metrics["per_label"][label]
        c = row["confusion"]
        require(set(c) == set(sums) and all(type(value) is int and value >= 0 for value in c.values()), "Invalid confusion counts")
        tp, fp, fn, tn = [c[key] for key in sums]
        require(tp + fn == row["positive_support"] == support[label]
                and tn + fp == row["negative_support"] == rows - support[label]
                and tp + fp == row["predicted_positive"], "Confusion/support totals differ")
        require(0 < support[label] < rows and row["ranking_status"] == row["f1_status"] == "defined", "Expected full label coverage")
        assert_close(row["threshold"], float(thresholds[column]), label + ".threshold")
        assert_close(row["f1"], 2 * tp / (2 * tp + fp + fn), label + ".f1")
        assert_close(row["precision"], tp / (tp + fp) if tp + fp else None, label + ".precision")
        assert_close(row["recall"], tp / (tp + fn), label + ".recall")
        for key in ("average_precision", "roc_auc"):
            require(isinstance(row[key], (int, float)) and 0 <= row[key] <= 1, "Invalid ranking metric")
        for key in sums:
            sums[key] += c[key]
    for macro_key, per_key in (("macro_average_precision", "average_precision"), ("macro_roc_auc", "roc_auc"), ("macro_f1", "f1")):
        mean = float(np.mean([metrics["per_label"][label][per_key] for label in LABELS]))
        assert_close(metrics[macro_key], mean, macro_key)
        assert_close(metrics[macro_key + "_eligible_labels"], mean, macro_key + "_eligible_labels")
    assert_close(metrics["micro_f1"], 2 * sums["true_positive"] / (2 * sums["true_positive"] + sums["false_positive"] + sums["false_negative"]), "micro_f1")
    require(0 <= metrics["micro_average_precision"] <= 1, "Invalid micro AP")
    coverage = metrics["coverage"]
    require(coverage["total_labels"] == coverage["ranking_eligible_count"] == coverage["f1_eligible_count"] == 6
            and coverage["ranking_eligible_labels"] == coverage["f1_eligible_labels"] == list(LABELS), "Coverage differs")


def check_public_objects(objects: dict[str, Any], *, head: bool = False) -> None:
    for name, value in objects.items():
        check_no_private_fields(value, allow_probe_records=name == "calibration.json")
    config, summary = objects["config.json"], objects["data_summary.json"]
    metadata, result = objects["run_metadata.json"], objects["result_metadata.json"]
    require(config["method"] == "random_initialized_roberta" and config["architecture"] == ORIGINAL_ARCHITECTURE, "Neural architecture differs")
    for item in (config, summary, result, objects["thresholds.json"]):
        require(item["label_order"] == list(LABELS), "Label order differs")
    require(summary["synthetic"] is False and metadata["synthetic"] is False and metadata["engineering_smoke"] is False, "Not real-data results")
    require(result["stage"] == "developed_and_frozen" and result["test_evaluation_performed"] is False
            and result["validation_is_selection_diagnostic"] is True, "Unexpected evaluation stage")
    require(all(result[key] is False for key in ("raw_texts_included", "per_row_arrays_included", "model_weights_included")), "Public bundle declares private artifacts")
    require(re.fullmatch(r"[0-9a-f]{64}", result["source_sha256"]) is not None, "Invalid source SHA-256")
    verify_code_identity(result["code_sha256"], head=head)
    require(re.fullmatch(r"[0-9a-f]{40}", result["tokenizer_revision"]) is not None
            and metadata["tokenizer_revision"] == result["tokenizer_revision"], "Tokenizer revision differs")
    require(metadata["tokenizer_source"] == "roberta-base tokenizer only"
            and metadata["initialization"] == "random weights; no pretrained model weights", "Tokenizer/initialization provenance differs")
    runtime = objects["versions_hardware.json"]
    require(runtime["hardware"]["cuda_available"] is True and runtime["hardware"]["gpu_count"] == 1
            and len(runtime["hardware"]["gpus"]) == 1, "Expected one recorded CUDA GPU")
    check_calibration(objects["calibration.json"], config, runtime, result["code_sha256"])
    if head:
        require(digest(subprocess.check_output(["git", "show", "HEAD:scripts/colab_profile.py"], cwd=ROOT))
                == objects["calibration.json"]["profile_sha256"], "Calibration profile source differs from local HEAD")
    parts = summary["splits"]["partitions"]
    require(set(parts) == {"train", "validation", "test"} and sum(p["rows"] for p in parts.values()) == summary["retained_rows"], "Partition row totals differ")
    for label in LABELS:
        require(sum(part["positive_counts"][label] for part in parts.values()) == summary["positive_counts"][label], "Partition positive totals differ")
    check_history(config, objects["training_history.json"], metadata, parts["train"]["rows"])
    require(metadata["threshold_grid"] == list(DEFAULT_THRESHOLD_GRID) and metadata["effective_batch_size"] == 256
            and metadata["test_status"] == "unscored during development", "Threshold/evaluation policy differs")
    require(metadata["export_matches_best_checkpoint"] is True and metadata["fresh_export_validation_loss_matches_best"] is True, "Export metadata reports failure")
    positive = np.array([parts["train"]["positive_counts"][label] for label in LABELS], dtype=np.float32)
    expected_weights = (parts["train"]["rows"] - positive) / positive
    assert_close(metadata["train_positive_weights"], dict(zip(LABELS, expected_weights.tolist())), "train_positive_weights")
    thresholds = objects["thresholds.json"]["thresholds"]
    require(len(thresholds) == 6 and all(value in DEFAULT_THRESHOLD_GRID for value in thresholds), "Thresholds outside fixed grid")
    value = objects["validation_metrics.json"]
    require(set(value) == {"metrics_at_0_5", "metrics_at_validation_thresholds", "selection_note", "split", "synthetic"}
            and value["split"] == "validation" and value["synthetic"] is False, "Validation result schema differs")
    for key, policy in (("metrics_at_0_5", [0.5] * 6), ("metrics_at_validation_thresholds", thresholds)):
        check_aggregate_metrics(value[key], policy, parts["validation"]["positive_counts"], parts["validation"]["rows"])
    for key in ("macro_average_precision", "macro_roc_auc", "micro_average_precision"):
        assert_close(value["metrics_at_0_5"][key], value["metrics_at_validation_thresholds"][key], key + ".threshold_invariance")


def baseline_comparison(run: Path, dataset: Any, splits: Any, neural: dict[str, Any]) -> dict[str, Any]:
    """Audit saved baseline validation predictions; never deserialize its model."""
    manifest = strict_json((run / "manifest.json").read_bytes())
    require((run / "manifest.sha256").read_text().strip() == file_sha256(run / "manifest.json"), "Baseline manifest checksum differs")
    require(manifest["source_sha256"] == dataset.source_sha256 and manifest["label_order"] == list(LABELS), "Baseline source or labels differ")
    for name in ("config.json", "data_summary.json", "thresholds.json", "validation_metrics.json", "private/splits.npz", "private/validation_predictions.npz"):
        require(file_sha256(run / name) == manifest["artifacts_sha256"][name], "Baseline artifact checksum differs: " + name)
    config = strict_json((run / "config.json").read_bytes())
    require(config["method"] == "tfidf_logistic_regression" and config["seed"] == 42
            and config["conflict_policy"] == "keep" and config["empty_text_policy"] == "drop", "Baseline protocol differs")
    stored = load_npz((run / "private/splits.npz").read_bytes(), {"train", "validation", "test"})
    require(all(np.array_equal(stored[name], value) for name, value in splits.as_dict().items()), "Baseline split/order differs")
    baseline = strict_json((run / "validation_metrics.json").read_bytes())
    require(baseline["split"] == "validation" and baseline["synthetic"] is False, "Not real baseline validation")
    public_baseline = strict_json((ROOT / "reports/baseline_validation_metrics.json").read_bytes())
    assert_close(baseline, public_baseline, "baseline_published_metrics")
    saved_thresholds = strict_json((run / "thresholds.json").read_bytes())
    require(saved_thresholds["label_order"] == list(LABELS), "Baseline threshold label order differs")
    arrays = load_npz((run / "private/validation_predictions.npz").read_bytes(), {"row_indices", "targets", "probabilities"})
    audit_predictions(arrays, dataset.targets[splits.validation], splits.validation, baseline, np.asarray(saved_thresholds["thresholds"]))
    keys = ("macro_average_precision", "macro_roc_auc", "macro_f1", "micro_f1", "micro_average_precision")
    models = {
        "baseline": {"method": "tfidf_logistic_regression", **{key: baseline[key] for key in ("metrics_at_0_5", "metrics_at_validation_thresholds")}},
        "neural": {"method": "random_initialized_roberta", **{key: neural[key] for key in ("metrics_at_0_5", "metrics_at_validation_thresholds")}},
    }
    return {
        "split": "validation", "rows": len(splits.validation), "same_source_same_split_verified": True,
        "label_order": list(LABELS), "source_sha256": dataset.source_sha256,
        "baseline_validation_predictions_sha256": manifest["artifacts_sha256"]["private/validation_predictions.npz"],
        "models": models,
        "numerical_deltas_neural_minus_baseline": {
            policy: {key: models["neural"][policy][key] - models["baseline"][policy][key] for key in keys}
            for policy in ("metrics_at_0_5", "metrics_at_validation_thresholds")},
        "limitations": ["Same source, target order, and grouped validation partition; both models use the same >= comparator and fixed threshold grid.",
                        "Each model selected its own thresholds on these validation rows; the neural checkpoint was also selected on them. This is descriptive selection evidence, not independent generalization evidence.",
                        "Baseline and neural internal-test metrics are not compared here. No confidence interval or statistical significance claim is made.",
                        "Baseline validation metrics were recomputed from saved probabilities; its model was not deserialized or run again."],
    }


def plot_data(objects: dict[str, Any]) -> dict[str, Any]:
    """Exact aggregate series used by each plot, portable across renderers."""
    history = objects["training_history.json"]
    metrics = objects["validation_metrics.json"]
    result = {
        "schema_version": 1,
        "training_losses": {"epoch": [row["epoch"] for row in history["epochs"]],
                            "training_loss": [row["training_loss"] for row in history["epochs"]],
                            "validation_loss": [row["validation_loss"] for row in history["epochs"]],
                            "selected_epoch": history["best_checkpoint"]["epoch"]},
        "validation_per_label": {"label_order": list(LABELS),
                                 "average_precision": [metrics["metrics_at_validation_thresholds"]["per_label"][label]["average_precision"] for label in LABELS],
                                 "f1_at_0_5": [metrics["metrics_at_0_5"]["per_label"][label]["f1"] for label in LABELS],
                                 "f1_at_selected_thresholds": [metrics["metrics_at_validation_thresholds"]["per_label"][label]["f1"] for label in LABELS]},
    }
    if COMPARISON_FILE in objects:
        result["validation_baseline_comparison"] = {
            "label_order": list(LABELS),
            **{model: {key: [objects[COMPARISON_FILE]["models"][model]["metrics_at_validation_thresholds"]["per_label"][label][key] for label in LABELS]
                       for key in ("average_precision", "f1")} for model in ("baseline", "neural")},
        }
    return result


def render_plots(objects: dict[str, Any], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Keep rendering independent of styles applied by another notebook/script.
    plt.rcdefaults()
    folder = destination / "plots"
    folder.mkdir()
    series_data = plot_data(objects)
    history = objects["training_history.json"]
    epochs = history["epochs"]
    figure, axis = plt.subplots(figsize=(9.4, 5.2))
    for key, title, color in (("training_loss", "Training", "#24688e"), ("validation_loss", "Validation", "#c56b36")):
        axis.plot(series_data["training_losses"]["epoch"], series_data["training_losses"][key], marker="o", label=title, color=color)
    axis.axvline(series_data["training_losses"]["selected_epoch"], color="#737b85", linestyle="--", label="Selected checkpoint (epoch 4)")
    axis.set(xlabel="Completed epoch", ylabel="Reported sample-weighted BCE loss", xticks=range(1, 6), title="Efficient-v1: completed training history")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.text(0.03, 0.035, "Returned history aggregates; epoch losses were not independently re-inferred.\nValidation selected the checkpoint. Last epoch and selected epoch are distinct.", fontsize=9, color="#596572")
    figure.tight_layout(rect=(0, 0.13, 1, 1))
    figure.savefig(folder / "training_losses.png", dpi=160)
    plt.close(figure)
    metrics = objects["validation_metrics.json"]
    fixed, selected = metrics["metrics_at_0_5"], metrics["metrics_at_validation_thresholds"]
    positions = np.arange(6)
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.5), sharey=True)
    for axis, values in zip(axes, ((('Average precision', selected, 'average_precision', '#24688e', 0),),
                                  (('F1 at 0.5', fixed, 'f1', '#24688e', -0.12), ('F1 at selected thresholds', selected, 'f1', '#c56b36', 0.12)))):
        for title, source, key, color, offset in values:
            field = "average_precision" if key == "average_precision" else "f1_at_0_5" if source is fixed else "f1_at_selected_thresholds"
            series = series_data["validation_per_label"][field]
            axis.scatter(series, positions + offset, label=title, color=color, s=42)
            for value, y in zip(series, positions + offset):
                axis.text(value + 0.012, y, f"{value:.3f}", va="center", fontsize=9)
        axis.set(yticks=positions, yticklabels=[label.replace('_', ' ') for label in LABELS], xlim=(0, 1.08), xlabel="Score (0–1)")
        axis.grid(axis="x", alpha=0.2)
        axis.legend(loc="lower left", bbox_to_anchor=(0, 1.025), frameon=False, fontsize=9)
    axes[0].invert_yaxis()
    figure.suptitle("Efficient-v1: verified validation selection diagnostics", x=0.04, ha="left", fontsize=14, fontweight="bold")
    figure.text(0.04, 0.035, f"{selected['rows']:,} grouped validation rows; all six labels have positive and negative support.\nCheckpoint and thresholds selected on these rows. AP is sklearn noninterpolated average precision; internal test is unscored.", fontsize=9, color="#596572")
    figure.subplots_adjust(left=0.14, right=0.98, bottom=0.20, top=0.80, wspace=0.15)
    figure.savefig(folder / "validation_per_label.png", dpi=160)
    plt.close(figure)
    if COMPARISON_FILE in objects:
        comparison = objects[COMPARISON_FILE]
        figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.5), sharey=True)
        for axis, key, title in zip(axes, ("average_precision", "f1"), ("Average precision", "F1 at each model's validation-selected thresholds")):
            for model, offset, color in (("baseline", -0.12, "#24688e"), ("neural", 0.12, "#c56b36")):
                values = series_data["validation_baseline_comparison"][model][key]
                axis.scatter(values, positions + offset, label="CPU reference" if model == "baseline" else "Random RoBERTa", color=color, s=42)
            axis.set(yticks=positions, yticklabels=[label.replace('_', ' ') for label in LABELS], xlim=(0, 1), xlabel="Score (0–1)")
            axis.set_title(title, fontsize=10)
            axis.grid(axis="x", alpha=0.2)
        axes[0].invert_yaxis()
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.56, 0.935), ncol=2, frameon=False, fontsize=9)
        figure.suptitle("Same grouped validation rows: descriptive model comparison", x=0.04, ha="left", fontsize=14, fontweight="bold")
        figure.text(0.04, 0.035, "Verified identical source, validation row order and targets; probabilities audited independently.\nValidation informed model selection. These are selection diagnostics; no neural held-out test or significance claim is available.", fontsize=9, color="#596572")
        figure.subplots_adjust(left=0.14, right=0.98, bottom=0.20, top=0.80, wspace=0.15)
        figure.savefig(folder / "validation_baseline_comparison.png", dpi=160)
        plt.close(figure)


def import_results(review: Path, private: Path, calibration_path: Path, source: Path,
                   destination: Path, *, baseline_run: Path | None = None, challenge_test: Path | None = None) -> dict[str, Any]:
    require(not destination.exists(), "Output directory already exists; use verify-public or a new directory")
    for path in (review, private, calibration_path, source, baseline_run, challenge_test):
        if path is not None and path.resolve().is_relative_to(ROOT):
            parts = path.resolve().relative_to(ROOT).parts
            require(parts and parts[0] in {"private", "runs", "data"}, "Private input must stay outside publication paths")
    public, secret = read_bundle(review, REVIEW_FILES), read_bundle(private, PRIVATE_FILES)
    objects = {name: strict_json(public[name]) for name in REVIEW_JSON}
    objects["calibration.json"] = strict_json(calibration_path.read_bytes())
    manifest = strict_json(secret["verification_manifest.json"])
    require(set(manifest["artifacts_sha256"]) == PRIVATE_HASHED, "Private manifest coverage differs")
    for name, expected in manifest["artifacts_sha256"].items():
        require(digest(secret[name]) == expected, "Private manifest checksum differs: " + name)
    for name in COMMON_JSON:
        require(public[name] == secret[name], "Cross-bundle JSON bytes differ: " + name)
        assert_close(objects[name], strict_json(secret[name]), "cross_bundle." + name)
    result = objects["result_metadata.json"]
    for key in ("label_order", "source_sha256", "code_sha256", "tokenizer_revision"):
        require(manifest[key] == result[key], "Private/public identity differs: " + key)
    require(manifest["runtime"] == objects["versions_hardware.json"], "Private/public runtime differs")
    check_public_objects(objects, head=True)
    config = objects["config.json"]["training"]
    dataset = load_dataset(source, conflict_policy=config["conflict_policy"], empty_text_policy=config["empty_text_policy"])
    require(dataset.source_sha256 == result["source_sha256"] == colab_train.FULL_SOURCE_SHA256, "Original source SHA-256 differs")
    splits = split_dataset(dataset, seed=config["seed"])
    assert_close(objects["data_summary.json"], {**dataset.summary, "splits": split_summary(dataset, splits)}, "reconstructed_data_summary")
    stored = load_npz(secret["splits.npz"], {"train", "validation", "test"})
    group_sets = []
    for name, expected in splits.as_dict().items():
        require(np.issubdtype(stored[name].dtype, np.integer) and np.array_equal(stored[name], expected), "Saved split/order differs: " + name)
        group_sets.append(set(dataset.groups[expected]))
    require(not any(left & right for index, left in enumerate(group_sets) for right in group_sets[index + 1:]), "Normalized groups overlap")
    arrays = load_npz(secret["validation_predictions.npz"], {"row_indices", "targets", "probabilities"})
    thresholds = np.asarray(objects["thresholds.json"]["thresholds"])
    audit_predictions(arrays, dataset.targets[splits.validation], splits.validation, objects["validation_metrics.json"], thresholds)
    if baseline_run is not None:
        objects[COMPARISON_FILE] = baseline_comparison(baseline_run, dataset, splits, objects["validation_metrics.json"])
    challenge = {"inspected": False, "scored": False}
    if challenge_test is not None:
        frame = pd.read_csv(challenge_test, usecols=list(LABELS) + ["set"], keep_default_na=False)
        require(frame.loc[:, list(LABELS)].eq("").all().all() and frame["set"].eq("test").all(), "Supplied challenge-test labels were expected to be blank")
        challenge = {"inspected": True, "rows": len(frame), "sha256": file_sha256(challenge_test), "all_label_fields_are_blank": True, "scored": False}
    metadata = objects["run_metadata.json"]
    objects["verification.json"] = {
        "schema_version": 1, "status": "verified_returned_validation_artifacts",
        "scope": "Independent numerical audit of supplied validation probabilities, source rows, grouped splits, thresholds, and aggregates; no model inference.",
        "input_sha256": {"aggregate_bundle": file_sha256(review), "private_verification_bundle": file_sha256(private), "calibration": file_sha256(calibration_path)},
        "audit_environment": {"python_version": platform.python_version(), "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn")}},
        "checks": {key: True for key in ("safe_whitelisted_zip_entries", "private_manifest_all_artifact_hashes", "cross_bundle_json_byte_equality", "cross_bundle_source_code_runtime_tokenizer_revision_identity", "trained_code_matches_local_files_and_HEAD", "calibration_profile_source_matches_local_code", "calibration_configuration_selection_arithmetic", "original_source_byte_hash", "full_eligible_cohort_and_summary", "exact_grouped_splits_and_row_order", "zero_normalized_group_overlap", "validation_targets_match_original_source_in_label_order", "finite_bounded_six_label_probabilities", "thresholds_reselected_independently_and_with_project_code", "all_metrics_recomputed_independently_and_with_project_code", "training_class_weights_from_training_rows_only", "five_epoch_history_and_epoch_four_selection_consistent", "public_output_contains_aggregate_artifacts_only")},
        "numerical_absolute_tolerance": TOLERANCE,
        "eligible_rows": len(dataset.frame), "partition_rows": {name: len(value) for name, value in splits.as_dict().items()},
        "validation_probability_shape": list(arrays["probabilities"].shape),
        "checkpoint": objects["training_history.json"]["best_checkpoint"],
        "training_completed_utc_reported": metadata["training_completed_utc"],
        "neural_internal_test": {"scored_in_supplied_artifacts": False, "evaluation_performed_during_this_audit": False},
        "challenge_test": challenge,
        "baseline_same_validation_comparison_verified": baseline_run is not None,
        "limitations": LIMITATIONS,
    }
    objects["plot_data.json"] = plot_data(objects)
    for name, value in objects.items():
        check_no_private_fields(value, allow_probe_records=name == "calibration.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="toxic-aggregate-import-") as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        for name, value in objects.items():
            (staging / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        render_plots(objects, staging)
        files = {path.relative_to(staging).as_posix(): file_sha256(path) for path in sorted(staging.rglob("*")) if path.is_file()}
        (staging / "public_manifest.json").write_text(json.dumps({"schema_version": 1, "purpose": "Aggregate publication integrity; private audit is not repeatable from these files alone", "artifacts_sha256": files}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        verify_public(staging)
        shutil.copytree(staging, destination)
    return objects["verification.json"]


def verify_public(directory: Path, *, check_plots: bool = False) -> dict[str, Any]:
    manifest = strict_json((directory / "public_manifest.json").read_bytes())
    hashes = manifest["artifacts_sha256"]
    expected = set(REPORT_JSON) | PLOTS
    if COMPARISON_FILE in hashes:
        expected |= {COMPARISON_FILE, COMPARISON_PLOT}
    require(set(hashes) == expected, "Public manifest contains missing or unexpected artifacts")
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    require(actual == expected | {"public_manifest.json"}, "Unmanifested file in report directory")
    for name, expected_hash in hashes.items():
        path = directory / name
        require(not path.is_symlink() and path.resolve().is_relative_to(directory.resolve()), "Unsafe public artifact path")
        require(file_sha256(path) == expected_hash, "Public artifact checksum differs: " + name)
    objects = {name: strict_json((directory / name).read_bytes()) for name in expected if name.endswith(".json")}
    check_public_objects(objects)
    assert_close(objects["plot_data.json"], plot_data(objects), "plot_data")
    report = objects["verification.json"]
    require(report["schema_version"] == 1 and report["status"] == "verified_returned_validation_artifacts"
            and report["checks"] and all(value is True for value in report["checks"].values()), "Private audit did not pass")
    require(report["limitations"] == LIMITATIONS, "Verification limitations changed")
    require(report["baseline_same_validation_comparison_verified"] is (COMPARISON_FILE in objects), "Comparison status differs")
    if COMPARISON_FILE in objects:
        comparison = objects[COMPARISON_FILE]
        require(comparison["split"] == "validation" and comparison["same_source_same_split_verified"] is True
                and comparison["label_order"] == list(LABELS) and comparison["source_sha256"] == objects["result_metadata.json"]["source_sha256"], "Comparison protocol differs")
        baseline = strict_json((ROOT / "reports/baseline_validation_metrics.json").read_bytes())
        for model, value in (("baseline", baseline), ("neural", objects["validation_metrics.json"])):
            for policy in ("metrics_at_0_5", "metrics_at_validation_thresholds"):
                assert_close(comparison["models"][model][policy], value[policy], "comparison." + model + "." + policy)
        for policy, deltas in comparison["numerical_deltas_neural_minus_baseline"].items():
            for metric, difference in deltas.items():
                assert_close(difference, comparison["models"]["neural"][policy][metric] - comparison["models"]["baseline"][policy][metric], "comparison.delta")
    if check_plots:
        import matplotlib.image as mpimg
        with tempfile.TemporaryDirectory(prefix="toxic-plot-check-") as temporary:
            staging = Path(temporary)
            render_plots(objects, staging)
            for name in PLOTS | ({COMPARISON_PLOT} if COMPARISON_FILE in objects else set()):
                require(np.array_equal(mpimg.imread(directory / name), mpimg.imread(staging / name)),
                        "Plot pixels differ from current renderer; portable data-series check passed, but exact rendering requires matching Matplotlib/fonts: " + name)
    return {"status": "public_aggregate_integrity_passed", "files": len(expected) + 1,
            "plot_series_match_aggregate_sources": True, "plot_pixels_rerendered_and_matched": check_plots,
            "private_numerical_audit_repeated": False,
            "note": "Checks aggregate arithmetic, source fingerprints, and published checksums; private per-row evidence and model re-inference are not available in public CI."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    importer = stages.add_parser("import", help="Audit private returned evidence and create a NEW aggregate report directory")
    importer.add_argument("--review", type=Path, required=True)
    importer.add_argument("--private-verification", type=Path, required=True)
    importer.add_argument("--calibration", type=Path, required=True)
    importer.add_argument("--source", type=Path, required=True)
    importer.add_argument("--baseline-run", type=Path)
    importer.add_argument("--challenge-test", type=Path)
    importer.add_argument("--output-dir", type=Path, default=ROOT / "reports/colab_efficient_v1")
    public = stages.add_parser("verify-public", help="Verify published aggregates offline, without repeating the private audit")
    public.add_argument("--report-dir", type=Path, default=ROOT / "reports/colab_efficient_v1")
    public.add_argument("--check-plots", action="store_true", help="Also compare rerendered PNG pixels; requires the original Matplotlib/font environment")
    args = parser.parse_args(argv)
    try:
        if args.stage == "import":
            report = import_results(args.review, args.private_verification, args.calibration, args.source, args.output_dir,
                                    baseline_run=args.baseline_run, challenge_test=args.challenge_test)
            print(json.dumps({"status": report["status"], "partition_rows": report["partition_rows"], "checkpoint": report["checkpoint"], "baseline_same_validation_comparison_verified": report["baseline_same_validation_comparison_verified"]}, sort_keys=True))
        else:
            print(json.dumps(verify_public(args.report_dir, check_plots=args.check_plots), sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        print("Colab evidence verification failed: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
