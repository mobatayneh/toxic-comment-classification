"""Private run artifacts, integrity checks, and a single-use test boundary.

Checksums detect accidental changes; this is not a tamper-proof security system.
Model deserialization is permitted only for an intact run created locally.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data import LABELS, SplitIndices, file_sha256
from .metrics import _validate_arrays, _validate_thresholds

MANIFEST_SCHEMA = 1
TEST_MARKER = ".test_evaluation_started"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: str | Path, value: Any) -> Path:
    """Create a JSON artifact once; NaN/Infinity are always rejected."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_json_default) + "\n"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return destination


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def create_run_directory(path: str | Path) -> Path:
    """Refuse overwrite, and require ignored storage for in-repository runs.

    Temporary directories outside the repository support isolated tests. Normal
    project runs belong under ``runs/`` or ``private/``, both ignored by git.
    """
    destination = Path(path).resolve()
    repository = Path(__file__).resolve().parents[2]
    if destination.is_relative_to(repository):
        relative = destination.relative_to(repository)
        if not relative.parts or relative.parts[0] not in {"runs", "private"} or len(relative.parts) < 2:
            raise ValueError("Run directories inside the repository must be under ignored runs/ or private/")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "private").mkdir()
    return destination


def resolve_run_path(run_dir: str | Path, stored_path: str) -> Path:
    candidate = Path(stored_path)
    return candidate.resolve() if candidate.is_absolute() else (Path(run_dir) / candidate).resolve()


def _relative_private_path(run_dir: Path, path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(run_dir) or resolved == run_dir:
        raise ValueError("Frozen model and artifact files must be inside the run directory")
    return resolved.relative_to(run_dir).as_posix()


def _code_hashes() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[2]
    return {
        path.relative_to(repository).as_posix(): file_sha256(path)
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }


def runtime_metadata(seed: int) -> dict[str, Any]:
    """Capture the executable environment and source state as private evidence."""
    repository = Path(__file__).resolve().parents[2]

    def git(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(repository), *arguments], capture_output=True, text=True, check=False
            )
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "torch", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    git_status = git("status", "--porcelain")
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": versions,
        "seed": int(seed),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "git_revision": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git_status) if git_status is not None else None,
        "code_sha256": _code_hashes(),
    }


def _validated_splits(splits: SplitIndices | Mapping[str, np.ndarray], rows: int) -> dict[str, np.ndarray]:
    mapping = splits.as_dict() if isinstance(splits, SplitIndices) else dict(splits)
    if set(mapping) != {"train", "validation", "test"}:
        raise ValueError("Splits must contain exactly train, validation, and test")
    clean = {}
    for name, indices in mapping.items():
        values = np.asarray(indices)
        if values.ndim != 1 or not len(values) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} split must contain a nonempty integer index array")
        clean[name] = values.astype(np.int64)
    combined = np.concatenate(list(clean.values()))
    if len(combined) != rows or not np.array_equal(np.sort(combined), np.arange(rows)):
        raise ValueError("Splits must partition every retained row exactly once")
    return clean


def freeze_run(
    run_dir: str | Path,
    *,
    config: dict[str, Any],
    data_summary: dict[str, Any],
    source_path: str | Path,
    model_path: str | Path,
    thresholds: Sequence[float],
    splits: SplitIndices | Mapping[str, np.ndarray],
    metadata: dict[str, Any],
    extra_artifacts: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Freeze model, input bytes, split, thresholds, configuration and metadata.

    ``model_path`` and extra artifacts can be files or directories. Paths passed
    by callers should be absolute or relative to their current working directory.
    The manifest stores paths inside the run relative to it.
    """
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ValueError("Create a new run directory before freezing it")
    if (root / "manifest.json").exists() or (root / TEST_MARKER).exists():
        raise FileExistsError("Run is already frozen or test evaluation has started")
    threshold_values = _validate_thresholds(thresholds)
    indices = _validated_splits(splits, int(data_summary["retained_rows"]))
    model = Path(model_path).resolve()
    model_relative = _relative_private_path(root, model)
    if not model.exists():
        raise FileNotFoundError(model)
    source = Path(source_path).resolve()
    source_hash = file_sha256(source)
    source_stored = source.relative_to(root).as_posix() if source.is_relative_to(root) else str(source)
    code_hashes = _code_hashes()
    if "code_sha256" in metadata and metadata["code_sha256"] != code_hashes:
        raise ValueError("Source code changed during development; start a new run after completing code changes")
    metadata = {
        **metadata,
        "code_sha256": code_hashes,
        "code_integrity_policy": "Changes to the recorded source files or source-file set invalidate this run",
    }
    initial = [
        write_json(root / "config.json", config),
        write_json(root / "data_summary.json", data_summary),
        write_json(root / "metadata.json", metadata),
        write_json(root / "thresholds.json", {"label_order": list(LABELS), "thresholds": threshold_values}),
    ]
    split_path = root / "private" / "splits.npz"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("xb") as stream:
        np.savez_compressed(stream, **indices)
    initial.append(split_path)
    files: dict[str, str] = {}
    for artifact in [*initial, model, *map(Path, extra_artifacts)]:
        if not artifact.exists():
            raise FileNotFoundError(artifact)
        candidates = sorted(artifact.rglob("*")) if artifact.is_dir() else [artifact]
        for candidate in candidates:
            if candidate.is_file():
                relative = _relative_private_path(root, candidate)
                files[relative] = file_sha256(candidate)
    if model.is_dir() and not any(name.startswith(model_relative + "/") for name in files):
        raise ValueError("Model directory must contain at least one file")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "label_order": list(LABELS),
        "source_path": source_stored,
        "source_sha256": source_hash,
        "model_path": model_relative,
        "artifacts_sha256": files,
        "test_policy": "single evaluation attempt; thresholds and model frozen before test",
    }
    manifest_path = write_json(root / "manifest.json", manifest)
    with (root / "manifest.sha256").open("x", encoding="ascii") as stream:
        stream.write(file_sha256(manifest_path) + "\n")
    return manifest


def verify_frozen_run(run_dir: str | Path, *, source_path: str | Path | None = None) -> dict[str, Any]:
    """Verify the frozen manifest, every artifact and the selected input bytes."""
    root = Path(run_dir).resolve()
    manifest_path = root / "manifest.json"
    expected_manifest_hash = (root / "manifest.sha256").read_text(encoding="ascii").strip()
    if file_sha256(manifest_path) != expected_manifest_hash:
        raise ValueError("Frozen manifest integrity check failed")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("label_order") != list(LABELS):
        raise ValueError("Frozen run schema or label order is incompatible")
    for name, expected in manifest["artifacts_sha256"].items():
        candidate = resolve_run_path(root, name)
        if not candidate.is_relative_to(root) or not candidate.is_file() or file_sha256(candidate) != expected:
            raise ValueError(f"Frozen artifact integrity check failed: {name}")
    model = resolve_run_path(root, manifest["model_path"])
    if not model.is_relative_to(root) or not model.exists():
        raise ValueError("Frozen model must remain inside the verified run directory")
    model_relative = model.relative_to(root).as_posix()
    if model.is_file():
        recorded_model = model_relative in manifest["artifacts_sha256"]
    else:
        recorded_files = {
            name for name in manifest["artifacts_sha256"] if name.startswith(model_relative + "/")
        }
        actual_files = {
            _relative_private_path(root, candidate)
            for candidate in model.rglob("*") if candidate.is_file()
        }
        recorded_model = bool(recorded_files) and recorded_files == actual_files
    if not recorded_model:
        raise ValueError("Frozen model file set is not covered by the artifact checksums")
    recorded_code = read_json(root / "metadata.json")["code_sha256"]
    actual_code = _code_hashes()
    if recorded_code != actual_code:
        changed = sorted(
            name for name in set(recorded_code) | set(actual_code)
            if recorded_code.get(name) != actual_code.get(name)
        )
        raise ValueError(f"Source code changed after development; frozen run is invalid: {', '.join(changed)}")
    source = Path(source_path).resolve() if source_path is not None else resolve_run_path(root, manifest["source_path"])
    if not source.is_file() or file_sha256(source) != manifest["source_sha256"]:
        raise ValueError("Frozen source dataset integrity check failed")
    return manifest


def load_splits(run_dir: str | Path) -> SplitIndices:
    root = Path(run_dir)
    rows = int(read_json(root / "data_summary.json")["retained_rows"])
    with np.load(root / "private" / "splits.npz", allow_pickle=False) as archive:
        arrays = _validated_splits({name: archive[name] for name in archive.files}, rows)
    return SplitIndices(arrays["train"], arrays["validation"], arrays["test"])


def load_thresholds(run_dir: str | Path) -> np.ndarray:
    value = read_json(Path(run_dir) / "thresholds.json")
    if value["label_order"] != list(LABELS):
        raise ValueError("Threshold label order is incompatible")
    return _validate_thresholds(value["thresholds"])


def begin_test_evaluation(run_dir: str | Path, *, source_path: str | Path | None = None) -> Path:
    """Claim the single test attempt before any test inference or metric access.

    A failed attempt remains claimed, so rerunning cannot become silent test
    tuning. No reset or override is provided by the CLI.
    """
    root = Path(run_dir)
    if not (root / "manifest.sha256").is_file():
        raise ValueError("Test evaluation requires a frozen run")
    verify_frozen_run(root, source_path=source_path)
    if (root / "public_test_results.json").exists():
        raise FileExistsError("Test results already exist; evaluation cannot be repeated")
    return write_json(root / TEST_MARKER, {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": file_sha256(root / "manifest.json"),
    })


def _check_public_aggregates(value: Any) -> None:
    forbidden = {
        "id", "ids", "row_id", "row_ids", "row_indices", "comment_text", "text",
        "texts", "comments", "raw_comments", "rawcomments", "raw_text", "raw_texts",
        "records", "targets", "y_true", "predictions", "y_pred", "probabilities",
        "scores", "y_score", "logits", "features", "embeddings", "source_path", "model_path",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in forbidden or "sha256" in lowered or "hash" in lowered:
                raise ValueError(f"Private field cannot be included in public results: {key}")
            _check_public_aggregates(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _check_public_aggregates(child)


def complete_test_evaluation(
    run_dir: str | Path,
    *,
    public_results: dict[str, Any],
    targets: np.ndarray,
    probabilities: np.ndarray,
    row_indices: np.ndarray,
    source_path: str | Path | None = None,
) -> Path:
    root = Path(run_dir)
    if not (root / TEST_MARKER).exists():
        raise ValueError("Begin the single test evaluation before writing its results")
    if (root / "public_test_results.json").exists():
        raise FileExistsError("Test results already exist; evaluation cannot be repeated")
    # Recheck after inference as well as before it. A code/model/input change
    # during an attempt must not produce a result attributed to frozen bytes.
    verify_frozen_run(root, source_path=source_path)
    claim = read_json(root / TEST_MARKER)
    if claim.get("manifest_sha256") != file_sha256(root / "manifest.json"):
        raise ValueError("Test attempt does not match the frozen manifest")
    clean_targets, clean_scores = _validate_arrays(targets, probabilities)
    indices = np.asarray(row_indices)
    if not np.array_equal(indices, load_splits(root).test) or len(indices) != len(clean_targets):
        raise ValueError("Test predictions must match the frozen test indices in order")
    _check_public_aggregates(public_results)
    with (root / "private" / "test_predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, row_indices=indices, targets=clean_targets, probabilities=clean_scores)
    return write_json(root / "public_test_results.json", public_results)
