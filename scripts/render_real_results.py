"""Render audited source counts and separately identified new-run aggregates.

This script never reads comments, individual predictions, or model files.
The new grouped protocol cannot establish a controlled improvement over the
historical neural validation experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from render_historical_results import BLUE, CHARCOAL, LABELS, ORANGE, configure_style


def load_data_audit(path: str | Path) -> dict[str, Any]:
    """Reconcile aggregate source counts before presenting prevalence."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("status") != "inspected_uploaded_real_data":
        raise ValueError("Expected an inspected real-data source audit")
    training = value["train"]
    if set(training["positive_counts"]) != set(LABELS) or int(training["rows"]) <= 0:
        raise ValueError("Audited source label counts are incomplete")
    for label in LABELS:
        positive = training["positive_counts"][label]
        if not 0 <= positive <= training["rows"]:
            raise ValueError("Audited source support is out of range")
        if not np.isclose(positive / training["rows"], training["label_prevalence"][label]):
            raise ValueError("Audited source prevalence and counts disagree")
    return value


def audit_dataframe(audit: dict[str, Any]) -> pd.DataFrame:
    """Show source support, before the seven empty-text exclusions."""
    training = audit["train"]
    return pd.DataFrame(
        [
            {
                "Label": label,
                "Positive source rows": training["positive_counts"][label],
                "Source prevalence (%)": 100 * training["label_prevalence"][label],
            }
            for label in LABELS
        ]
    )


def plot_label_prevalence(audit: dict[str, Any]) -> plt.Figure:
    """Show imbalance without inventing mutually exclusive class shares."""
    configure_style()
    table = audit_dataframe(audit)
    figure, axis = plt.subplots(figsize=(10.8, 5.3))
    positions = np.arange(len(LABELS))
    values = table["Source prevalence (%)"].to_numpy() / 100
    axis.barh(positions, values, color=BLUE, height=0.58)
    for position, value, count in zip(positions, values, table["Positive source rows"]):
        axis.text(value + 0.0018, position, f"{value:.2%}  ({count:,})", va="center", fontsize=10)
    axis.set(
        yticks=positions,
        yticklabels=[label.replace("_", " ") for label in LABELS],
        xlim=(0, 0.128),
        xlabel="Positive prevalence in supplied training-source rows",
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axis.invert_yaxis()
    axis.set_axisbelow(True)
    axis.grid(axis="x")
    figure.suptitle("Supplied training mirror: label prevalence", x=0.05, y=0.98, ha="left", fontsize=15, fontweight="bold")
    figure.text(
        0.05, 0.035,
        f"Source: reports/data_audit.json; denominator {audit['train']['rows']:,} source rows, before empty-text exclusions.\n"
        "Labels overlap; these percentages do not sum to 100%. No subgroup or temporal independence is established.",
        fontsize=9, color="#596572",
    )
    figure.subplots_adjust(left=0.15, right=0.97, bottom=0.22, top=0.86)
    return figure


def load_run_results(path: str | Path) -> dict[str, Any]:
    """Require aggregate metrics with explicitly real data and full labels."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("method") != "tfidf_logistic_regression" or value.get("synthetic") is not False:
        raise ValueError("New real-data result must explicitly set synthetic=false")
    metric_sets = [key for key in value if key.startswith("metrics_at_")]
    if "metrics_at_0_5" not in metric_sets or len(metric_sets) != 2:
        raise ValueError("Expected fixed and validation-selected threshold reports")
    for key in metric_sets:
        metrics = value[key]
        if list(metrics["per_label"]) != list(LABELS) and set(metrics["per_label"]) != set(LABELS):
            raise ValueError("Aggregate result label set differs from the saved order")
        if metrics["rows"] <= 0 or metrics["coverage"]["total_labels"] != len(LABELS):
            raise ValueError("Aggregate result coverage is invalid")
        for label in LABELS:
            row = metrics["per_label"][label]
            if row["positive_support"] + row["negative_support"] != metrics["rows"]:
                raise ValueError("Per-label supports do not reconcile to partition size")
    return value


def tuned_metric_key(results: dict[str, Any]) -> str:
    keys = [key for key in results if key.startswith("metrics_at_") and key != "metrics_at_0_5"]
    if len(keys) != 1:
        raise ValueError("Expected one validation-selected threshold report")
    return keys[0]


def run_summary_dataframe(results: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for label, key in (("Fixed 0.5", "metrics_at_0_5"), ("Validation-selected", tuned_metric_key(results))):
        metrics = results[key]
        rows.append({
            "Threshold policy": label,
            "Rows": metrics["rows"],
            "Macro AP": metrics["macro_average_precision"],
            "Macro ROC-AUC": metrics["macro_roc_auc"],
            "Macro F1": metrics["macro_f1"],
            "Micro F1": metrics["micro_f1"],
            "Ranking labels": f"{metrics['coverage']['ranking_eligible_count']}/{metrics['coverage']['total_labels']}",
        })
    return pd.DataFrame(rows)


def run_per_label_dataframe(results: dict[str, Any]) -> pd.DataFrame:
    metrics = results[tuned_metric_key(results)]
    rows = []
    for label in LABELS:
        row = metrics["per_label"][label]
        rows.append({
            "Label": label,
            "Positive support": row["positive_support"],
            "Threshold": row["threshold"],
            "AP": row["average_precision"],
            "ROC-AUC": row["roc_auc"],
            "Precision": row["precision"],
            "Recall": row["recall"],
            "F1": row["f1"],
        })
    return pd.DataFrame(rows)


def plot_run_per_label(results: dict[str, Any]) -> plt.Figure:
    """Keep AP ranking quality and threshold-specific F1 visibly distinct."""
    configure_style()
    fixed = results["metrics_at_0_5"]
    tuned = results[tuned_metric_key(results)]
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.5), sharey=True)
    positions = np.arange(len(LABELS))
    for axis, series in zip(axes, (
        (("Average precision", [tuned['per_label'][label]['average_precision'] for label in LABELS], BLUE, "o", 0),),
        (
            ("F1 at 0.5", [fixed['per_label'][label]['f1'] for label in LABELS], BLUE, "o", -0.11),
            ("F1 at selected thresholds", [tuned['per_label'][label]['f1'] for label in LABELS], ORANGE, "s", 0.11),
        ),
    )):
        for title, values, color, marker, offset in series:
            clean = np.array([np.nan if value is None else value for value in values])
            axis.scatter(clean, positions + offset, color=color, marker=marker, s=50, label=title, zorder=3)
            for value, position in zip(values, positions + offset):
                if value is not None:
                    axis.text(value + 0.012, position, f"{value:.3f}", va="center", fontsize=9, color=CHARCOAL)
        axis.set(yticks=positions, yticklabels=[label.replace("_", " ") for label in LABELS], xlim=(0, 1.08), xlabel="Score (0-1)")
        axis.grid(axis="x")
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False, fontsize=9)
    axes[0].invert_yaxis()
    figure.suptitle("New CPU protocol: per-label internal-test metrics", x=0.04, y=0.99, ha="left", fontsize=15, fontweight="bold")
    figure.text(
        0.04, 0.035,
        f"Population: {tuned['rows']:,} rows in a grouped exploratory internal test; original cohort was previously examined.\n"
        "Thresholds selected on validation only. AP is not trapezoidal PR area. No confidence intervals are available.",
        fontsize=9, color="#596572",
    )
    figure.subplots_adjust(left=0.14, right=0.98, bottom=0.20, top=0.79, wspace=0.15)
    return figure


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=root / "reports/data_audit.json")
    parser.add_argument("--internal-test", type=Path, default=root / "reports/baseline_internal_test_metrics.json")
    parser.add_argument("--output-dir", type=Path, default=root / "reports/figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = [("real_source_label_prevalence.png", plot_label_prevalence(load_data_audit(args.audit)))]
    if args.internal_test.is_file():
        results = load_run_results(args.internal_test)
        if "internal" not in results["split"].lower() or "test" not in results["split"].lower():
            raise ValueError("The final-run chart requires an explicitly internal test partition")
        plots.append(("baseline_internal_test_metrics.png", plot_run_per_label(results)))
    for filename, figure in plots:
        destination = args.output_dir / filename
        figure.savefig(destination, bbox_inches="tight")
        plt.close(figure)
        print(destination)
    print("Aggregate review only; historical and new protocols remain distinct.")


if __name__ == "__main__":
    main()
