"""Strict CSV loading and seeded splits of normalized exact-text groups.

Grouping catches exact text after Unicode/case/whitespace normalization. It does
not detect paraphrases or other near duplicates. Splitting is not stratified.
"""

from __future__ import annotations

import csv
import hashlib
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LABELS = ("identity_hate", "insult", "obscene", "severe_toxic", "threat", "toxic")
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)


def file_sha256(path: str | Path) -> str:
    """Hash bytes without loading an entire dataset into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    """Normalize only for grouping; model inputs retain their original text."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


@dataclass(frozen=True)
class LoadedDataset:
    frame: pd.DataFrame
    groups: np.ndarray
    source_sha256: str
    summary: dict[str, Any]

    @property
    def texts(self) -> np.ndarray:
        # A fixed-width Unicode array allocates every row at the longest
        # comment's length. Keep references to the validated Python strings.
        return self.frame["comment_text"].to_numpy(dtype=object, copy=False)

    @property
    def targets(self) -> np.ndarray:
        return self.frame.loc[:, list(LABELS)].to_numpy(dtype=np.uint8)


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in ("train", "validation", "test")}


def load_dataset(
    path: str | Path,
    *,
    conflict_policy: str = "error",
    empty_text_policy: str = "error",
) -> LoadedDataset:
    """Load official train.csv or the historical cleaned mirror's train rows.

    ``keep`` retains differently labeled normalized duplicates as separate rows
    in the same group. The default ``error`` requires the caller to resolve or
    explicitly accept conflicting annotations. ``empty_text_policy='drop'``
    excludes only rows whose normalized text is empty; its row count is reported
    separately from the mirror's excluded nontraining rows. IDs and extra
    columns, including the label-derived ``toxicity`` field, are not model inputs.
    """
    if conflict_policy not in {"error", "keep"}:
        raise ValueError("conflict_policy must be 'error' or 'keep'")
    if empty_text_policy not in {"error", "drop"}:
        raise ValueError("empty_text_policy must be 'error' or 'drop'")
    source = Path(path).resolve()
    source_hash = file_sha256(source)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(set(header)) != len(header):
        raise ValueError("CSV must have a header with unique column names")
    # Do not infer numeric-looking comments as numbers, or treat 'NA' as missing.
    frame = pd.read_csv(source, dtype={"comment_text": str, "id": str}, keep_default_na=False)
    input_rows = len(frame)
    if "set" in frame.columns:
        frame = frame.loc[frame["set"] == "train"].copy()
        source_format = "preprocessed_mirror_schema"
    else:
        source_format = "jigsaw_train_schema"
    excluded_nontrain_rows = input_rows - len(frame)
    if frame.empty:
        raise ValueError("Dataset has no labeled training rows")
    if "severe_tox" in frame.columns:
        if "severe_toxic" in frame.columns:
            canonical = pd.to_numeric(frame["severe_toxic"], errors="coerce")
            alias = pd.to_numeric(frame["severe_tox"], errors="coerce")
            if not np.array_equal(canonical.to_numpy(), alias.to_numpy(), equal_nan=True):
                raise ValueError("severe_tox and severe_toxic columns disagree")
        else:
            frame = frame.rename(columns={"severe_tox": "severe_toxic"})
    missing = [column for column in ("comment_text", *LABELS) if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    texts = frame["comment_text"]
    if texts.isna().any() or not texts.map(lambda value: isinstance(value, str)).all():
        raise ValueError("comment_text must contain nonmissing strings")
    normalized = texts.map(normalize_text)
    empty = normalized.eq("")
    excluded_empty_text_rows = int(empty.sum())
    if excluded_empty_text_rows:
        if empty_text_policy == "error":
            raise ValueError(
                "comment_text must not contain empty or whitespace-only text; "
                "explicitly use empty_text_policy='drop' to exclude these rows"
            )
        frame = frame.loc[~empty].copy()
        normalized = normalized.loc[~empty]
        if frame.empty:
            raise ValueError("Dataset has no nonempty labeled training rows after dropping empty text")
    missing_ids = duplicate_ids = duplicate_id_groups = None
    if "id" in frame.columns:
        identifiers = frame["id"].fillna("").astype(str).str.strip()
        missing_ids = int(identifiers.eq("").sum())
        present = identifiers.ne("")
        identifier_texts = pd.DataFrame({"identifier": identifiers[present], "text": normalized[present]})
        if identifier_texts.groupby("identifier")["text"].nunique().gt(1).any():
            raise ValueError("A repeated nonblank id refers to different normalized texts")
        identifier_sizes = identifiers[present].value_counts()
        duplicate_ids = int((identifier_sizes - 1).clip(lower=0).sum())
        duplicate_id_groups = int(identifier_sizes.gt(1).sum())
    for label in LABELS:
        values = pd.to_numeric(frame[label], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError(f"Label {label} must contain only nonmissing binary 0/1 values")
        frame[label] = values.astype(np.uint8)
    # Hashes stay private. Sorting group hashes later makes assignment independent
    # of row ordering, while every row and original annotation is retained.
    groups = normalized.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    annotations = pd.DataFrame({"group": groups})
    for label in LABELS:
        annotations[label] = frame[label]
    label_variation = annotations.groupby("group")[list(LABELS)].nunique()
    conflict_count = int(label_variation.gt(1).any(axis=1).sum())
    if conflict_count and conflict_policy == "error":
        raise ValueError(
            f"Found {conflict_count} normalized-text groups with conflicting labels; "
            "resolve annotations or explicitly use conflict_policy='keep'"
        )
    sizes = groups.value_counts()
    synthetic = False
    if "_synthetic_fixture" in frame.columns:
        marker = pd.to_numeric(frame["_synthetic_fixture"], errors="coerce")
        if not marker.eq(1).all():
            raise ValueError("_synthetic_fixture must equal 1 for every retained row")
        synthetic = True
    clean = frame.loc[:, ["comment_text", *LABELS]].reset_index(drop=True)
    summary = {
        "source_format": source_format,
        "source_provenance": "synthetic_engineering_fixture" if synthetic else "user_supplied_origin_unverified",
        "synthetic": synthetic,
        "input_rows": input_rows,
        "excluded_nontrain_rows": excluded_nontrain_rows,
        "excluded_empty_text_rows": excluded_empty_text_rows,
        "retained_rows": len(clean),
        "source_id_column_present": "id" in frame.columns,
        "missing_source_ids": missing_ids,
        "duplicate_source_ids": duplicate_ids,
        "duplicate_source_id_groups": duplicate_id_groups,
        "label_order": list(LABELS),
        "positive_counts": {label: int(clean[label].sum()) for label in LABELS},
        "normalized_text_groups": int(len(sizes)),
        "duplicate_groups": int(sizes.gt(1).sum()),
        "rows_in_duplicate_groups": int(sizes.loc[sizes.gt(1)].sum()),
        "conflicting_label_groups": conflict_count,
        "conflict_policy": conflict_policy,
        "empty_text_policy": empty_text_policy,
        "grouping": "NFKC, casefold, Unicode whitespace collapse; exact matches only",
        "near_duplicate_detection": False,
    }
    if file_sha256(source) != source_hash:
        raise ValueError("Dataset changed while it was being loaded")
    return LoadedDataset(clean, groups.to_numpy(dtype=str), source_hash, summary)


def split_dataset(dataset: LoadedDataset, *, seed: int = 42) -> SplitIndices:
    """Seeded 70/15/15 allocation by group count, with nonempty partitions.

    Row fractions can differ when groups have different sizes. No label
    stratification is attempted; downstream metrics expose missing coverage.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    unique = np.unique(dataset.groups)
    if len(unique) < 3:
        raise ValueError("At least three normalized-text groups are required for train/validation/test")
    shuffled = np.random.default_rng(seed).permutation(unique)
    n_train = max(1, min(len(unique) - 2, int(len(unique) * SPLIT_FRACTIONS[0])))
    n_validation = max(1, min(len(unique) - n_train - 1, int(len(unique) * SPLIT_FRACTIONS[1])))
    group_sets = (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_validation],
        shuffled[n_train + n_validation :],
    )
    arrays = [np.flatnonzero(np.isin(dataset.groups, group_set)) for group_set in group_sets]
    return SplitIndices(*arrays)


def split_summary(dataset: LoadedDataset, splits: SplitIndices) -> dict[str, Any]:
    return {
        "strategy": "seeded normalized exact-text group shuffle; not stratified",
        "target_group_fractions": dict(zip(("train", "validation", "test"), SPLIT_FRACTIONS)),
        "partitions": {
            name: {
                "rows": len(indices),
                "row_fraction": len(indices) / len(dataset.frame),
                "groups": int(len(np.unique(dataset.groups[indices]))),
                "positive_counts": {
                    label: int(dataset.targets[indices, column].sum())
                    for column, label in enumerate(LABELS)
                },
            }
            for name, indices in splits.as_dict().items()
        },
    }


def create_synthetic_fixture(path: str | Path, *, rows: int = 240, seed: int = 42) -> Path:
    """Write an easy, deterministic engineering fixture, never benchmark data."""
    if not 60 <= rows <= 10000:
        raise ValueError("Synthetic fixture rows must be between 60 and 10000")
    rng = np.random.default_rng(seed)
    markers = ("identitymarker", "insultmarker", "obscenemarker", "severemarker", "threatmarker", "toxicmarker")
    targets = rng.integers(0, 2, size=(rows, len(LABELS)), dtype=np.uint8)
    texts = [
        "Synthetic fixture comment " + str(row) + " friendly conversation "
        + " ".join(markers[column] for column in range(len(LABELS)) if targets[row, column])
        for row in range(rows)
    ]
    frame = pd.DataFrame(targets, columns=LABELS)
    frame.insert(0, "comment_text", texts)
    frame["_synthetic_fixture"] = 1
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination
