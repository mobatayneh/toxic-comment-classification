"""Average precision, coverage-aware evaluation, and validation-only thresholds."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .data import LABELS

DEFAULT_THRESHOLD_GRID = tuple(value / 20 for value in range(1, 20))


def _validate_arrays(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=float)
    if targets.ndim != 2 or targets.shape != scores.shape or targets.shape[1] != len(LABELS):
        raise ValueError(f"Targets and probabilities must have the same (n, {len(LABELS)}) shape")
    if targets.shape[0] == 0:
        raise ValueError("Cannot evaluate an empty partition")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("Targets must be binary 0/1 values")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    return targets.astype(np.uint8), scores


def _validate_thresholds(thresholds: Sequence[float]) -> np.ndarray:
    values = np.asarray(thresholds, dtype=float)
    if values.shape != (len(LABELS),) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"Thresholds must be {len(LABELS)} finite values within [0, 1]")
    return values


def select_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
) -> np.ndarray:
    """Maximize each label's validation F1 on a fixed, bounded threshold grid.

    Ties prefer the value nearest 0.5, then the larger value. A label missing
    either validation class keeps 0.5 because its threshold cannot be selected
    meaningfully from that partition. Callers must pass validation data only.
    """
    targets, scores = _validate_arrays(y_true, probabilities)
    candidates = np.asarray(grid, dtype=float)
    if candidates.ndim != 1 or not 1 <= len(candidates) <= 101:
        raise ValueError("Threshold grid must contain between 1 and 101 values")
    if not np.isfinite(candidates).all() or ((candidates <= 0) | (candidates >= 1)).any():
        raise ValueError("Threshold grid values must be finite and strictly between 0 and 1")
    if len(np.unique(candidates)) != len(candidates):
        raise ValueError("Threshold grid values must be unique")
    selected = np.full(len(LABELS), 0.5, dtype=float)
    for column in range(len(LABELS)):
        target = targets[:, column]
        if len(np.unique(target)) != 2:
            continue
        scored = [
            (float(f1_score(target, scores[:, column] >= threshold, zero_division=0)), float(threshold))
            for threshold in candidates
        ]
        best_score = max(value for value, _ in scored)
        tied = [threshold for value, threshold in scored if np.isclose(value, best_score, rtol=0, atol=1e-12)]
        # Round the distance so representational noise in 0.45/0.55 cannot
        # defeat the documented higher-threshold tie breaker.
        selected[column] = min(tied, key=lambda threshold: (round(abs(threshold - 0.5), 12), -threshold))
    return selected


def evaluate_predictions(
    y_true: np.ndarray, probabilities: np.ndarray, thresholds: Sequence[float]
) -> dict[str, Any]:
    """Return aggregates safe to share without rows, comment text, or IDs.

    Average precision is sklearn's noninterpolated average_precision_score,
    never a trapezoidal PR integral. Single-class AP/ROC AUC are null: an
    all-positive AP of 1 would convey no ranking information. The six-label
    macro scores are null unless every label is eligible; separately named
    diagnostics average the eligible subset and expose its denominator. F1
    needs positive support but not negative support.
    """
    targets, scores = _validate_arrays(y_true, probabilities)
    threshold_values = _validate_thresholds(thresholds)
    predicted = scores >= threshold_values
    per_label: dict[str, dict[str, Any]] = {}
    ranking_labels: list[str] = []
    f1_labels: list[str] = []
    for column, label in enumerate(LABELS):
        target, prediction = targets[:, column], predicted[:, column]
        positives = int(target.sum())
        negatives = int(len(target) - positives)
        true_positive = int(np.logical_and(target == 1, prediction).sum())
        false_positive = int(np.logical_and(target == 0, prediction).sum())
        false_negative = positives - true_positive
        true_negative = negatives - false_positive
        both_classes = positives > 0 and negatives > 0
        has_positive = positives > 0
        predicted_positive = true_positive + false_positive
        per_label[label] = {
            "positive_support": positives,
            "negative_support": negatives,
            "predicted_positive": predicted_positive,
            "average_precision": float(average_precision_score(target, scores[:, column])) if both_classes else None,
            "roc_auc": float(roc_auc_score(target, scores[:, column])) if both_classes else None,
            "f1": float(f1_score(target, prediction, zero_division=0)) if has_positive else None,
            "precision": true_positive / predicted_positive if predicted_positive else None,
            "recall": true_positive / positives if has_positive else None,
            "threshold": float(threshold_values[column]),
            "ranking_status": "defined" if both_classes else ("no_positives" if not positives else "no_negatives"),
            "f1_status": "defined" if has_positive else "no_positives",
            "confusion": {
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
            },
        }
        if both_classes:
            ranking_labels.append(label)
        if has_positive:
            f1_labels.append(label)

    def macro(metric: str, eligible: list[str]) -> float | None:
        return float(np.mean([per_label[label][metric] for label in eligible])) if eligible else None

    flattened = targets.ravel()
    micro_has_positive = bool(flattened.any())
    micro_both_classes = micro_has_positive and bool((flattened == 0).any())
    return {
        "primary_metric": "macro_average_precision",
        "rows": len(targets),
        "macro_average_precision": macro("average_precision", ranking_labels) if len(ranking_labels) == len(LABELS) else None,
        "macro_average_precision_eligible_labels": macro("average_precision", ranking_labels),
        "macro_roc_auc": macro("roc_auc", ranking_labels) if len(ranking_labels) == len(LABELS) else None,
        "macro_roc_auc_eligible_labels": macro("roc_auc", ranking_labels),
        "macro_f1": macro("f1", f1_labels) if len(f1_labels) == len(LABELS) else None,
        "macro_f1_eligible_labels": macro("f1", f1_labels),
        "micro_f1": float(f1_score(targets, predicted, average="micro", zero_division=0)) if micro_has_positive else None,
        "micro_average_precision": float(average_precision_score(flattened, scores.ravel())) if micro_both_classes else None,
        "coverage": {
            "total_labels": len(LABELS),
            "ranking_eligible_count": len(ranking_labels),
            "ranking_eligible_labels": ranking_labels,
            "f1_eligible_count": len(f1_labels),
            "f1_eligible_labels": f1_labels,
            "macro_averaging": "six-label macro scores require full coverage; separately named eligible-label diagnostics use their stated subset",
        },
        "per_label": per_label,
    }
