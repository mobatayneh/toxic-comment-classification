"""Render aggregate historical notebook evidence, never a new model benchmark.

The same functions are used in notebooks/01_results_review.ipynb. Inputs are
rounded records extracted from uploaded notebooks, not data or checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NOTEBOOK_ORDER = (
    "model_FINAL_Report.ipynb",
    "Untitled10.ipynb",
    "modelXXXX(1).ipynb",
)
METRIC_KEYS = (
    "epoch",
    "train_loss",
    "validation_loss",
    "validation_macro_roc_auc",
    "validation_macro_f1_at_threshold_0_5",
)
LABELS = ("identity_hate", "insult", "obscene", "severe_toxic", "threat", "toxic")
BLUE = "#245A81"
ORANGE = "#BA651D"
CHARCOAL = "#313941"
TEAL = "#46786E"


def configure_style() -> None:
    """Apply one readable, explicit, portable style to all evidence figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#65717C",
            "axes.labelcolor": CHARCOAL,
            "text.color": CHARCOAL,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.color": "#E1E5E8",
            "grid.linewidth": 0.7,
            "lines.linewidth": 1.8,
            "figure.dpi": 120,
            "savefig.dpi": 160,
        }
    )


def load_evidence(path: str | Path) -> dict[str, Any]:
    """Load and reconcile the saved evidence before rendering any result."""
    evidence = json.loads(Path(path).read_text(encoding="utf-8"))
    if evidence.get("status") != "historical_notebook_outputs_only":
        raise ValueError("Expected historical notebook outputs only")
    if evidence.get("fresh_training_executed") is not False:
        raise ValueError("Historical evidence must not claim newly executed training")
    if tuple(evidence.get("labels", [])) != LABELS:
        raise ValueError("Historical label order changed")
    records = {record["file"]: record for record in evidence["notebooks"]}
    if set(records) != set(NOTEBOOK_ORDER):
        raise ValueError("Expected exactly the three audited notebooks")
    for record in records.values():
        history = record["metrics"]
        if [row["epoch"] for row in history] != list(range(1, len(history) + 1)):
            raise ValueError("Recorded epoch sequence is incomplete")
        for row in history:
            if not all(np.isfinite(row[key]) for key in METRIC_KEYS):
                raise ValueError("Historical metrics must be finite")
            if row["train_loss"] < 0 or row["validation_loss"] < 0:
                raise ValueError("Historical losses must be nonnegative")
            if not all(0 <= row[key] <= 1 for key in METRIC_KEYS[3:]):
                raise ValueError("Historical ranking/decision metrics must be in [0, 1]")
        selected = min(history, key=lambda row: row["validation_loss"])
        if any(selected[key] != record["best_by_validation_loss"][key] for key in METRIC_KEYS):
            raise ValueError("Selected checkpoint metadata disagrees with stored history")
    final_history = records[NOTEBOOK_ORDER[0]]["metrics"]
    duplicate_history = records[NOTEBOOK_ORDER[2]]["metrics"]
    if [[row[key] for key in METRIC_KEYS] for row in final_history] != [
        [row[key] for key in METRIC_KEYS] for row in duplicate_history
    ]:
        raise ValueError("The documented matching five-epoch histories no longer match")
    return evidence


def records_by_name(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["file"]: record for record in evidence["notebooks"]}


def summary_dataframe(evidence: dict[str, Any]) -> pd.DataFrame:
    """Return separate selected-checkpoint and last-epoch records."""
    records = records_by_name(evidence)
    rows = []
    for name in NOTEBOOK_ORDER:
        record = records[name]
        for checkpoint, metrics in (
            ("Selected by validation loss", record["best_by_validation_loss"]),
            ("Last epoch", record["metrics"][-1]),
        ):
            rows.append(
                {
                    "Notebook": name,
                    "Recorded checkpoint": checkpoint,
                    "Epoch": metrics["epoch"],
                    "Validation loss": metrics["validation_loss"],
                    "Macro ROC-AUC": metrics["validation_macro_roc_auc"],
                    "Macro F1 at > 0.5": metrics["validation_macro_f1_at_threshold_0_5"],
                }
            )
    return pd.DataFrame(rows)


def per_label_dataframe(evidence: dict[str, Any]) -> pd.DataFrame:
    """Keep the stored macro row; do not recalculate from rounded label values."""
    report = records_by_name(evidence)[NOTEBOOK_ORDER[0]]["per_label_report"]
    if report is None or report["uses_last_epoch_prediction_arrays_from_cell_6"] is not True:
        raise ValueError("Expected the audited last-epoch per-label report")
    rows = []
    for label in (*LABELS, "Macro-Avg"):
        metrics = report["metrics"][label]
        rows.append(
            {
                "Label": "Macro mean (stored)" if label == "Macro-Avg" else label,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
            }
        )
    return pd.DataFrame(rows)


def plot_training_history(evidence: dict[str, Any]) -> plt.Figure:
    """Compare recorded loss trajectories, with selection visibly separated."""
    configure_style()
    records = records_by_name(evidence)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for axis, name, subtitle in zip(
        axes,
        NOTEBOOK_ORDER[:2],
        ("FINAL_Report · 5 recorded epochs", "Untitled10 · 7 recorded epochs"),
    ):
        record = records[name]
        history = record["metrics"]
        epochs = [row["epoch"] for row in history]
        axis.plot(epochs, [row["train_loss"] for row in history], color=BLUE, marker="o", label="Training loss")
        axis.plot(
            epochs,
            [row["validation_loss"] for row in history],
            color=ORANGE,
            marker="s",
            linestyle="--",
            label="Validation loss",
        )
        selected = record["best_by_validation_loss"]
        axis.axvline(selected["epoch"], color=CHARCOAL, linestyle=":", linewidth=1.3)
        axis.text(
            selected["epoch"] + 0.1,
            0.98,
            f"Selected epoch {selected['epoch']}",
            va="top",
            transform=axis.get_xaxis_transform(),
            fontsize=10,
        )
        axis.set(title=subtitle, xlabel="Recorded epoch", xticks=epochs, ylim=(0, 1.15))
        axis.grid(axis="y")
    axes[0].set_ylabel("Weighted binary cross-entropy (stored)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False)
    figure.suptitle("Historical training losses", x=0.06, y=1.01, ha="left", fontsize=15, fontweight="bold")
    figure.text(
        0.06,
        0.02,
        "Source: original cell 6 outputs, rounded to four decimals. Validation repeatedly inspected; no rerun or held-out test.",
        fontsize=9,
        color="#596572",
    )
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.78, wspace=0.15)
    return figure


def plot_per_label_report(evidence: dict[str, Any]) -> plt.Figure:
    """Expose the historical threshold's recall/precision tradeoff honestly."""
    configure_style()
    table = per_label_dataframe(evidence).iloc[: len(LABELS)]
    figure, axis = plt.subplots(figsize=(10.8, 5.8))
    label_positions = np.arange(len(table))
    for metric, offset, color, marker in (
        ("Precision", -0.20, BLUE, "o"),
        ("Recall", 0.00, ORANGE, "s"),
        ("F1", 0.20, TEAL, "D"),
    ):
        values = table[metric].to_numpy()
        axis.scatter(values, label_positions + offset, s=50, color=color, marker=marker, label=metric, zorder=3)
        for value, position in zip(values, label_positions + offset):
            axis.text(value + 0.013, position, f"{value:.3f}", va="center", fontsize=9, color=CHARCOAL)
    axis.set(
        yticks=label_positions,
        yticklabels=table["Label"].str.replace("_", " ", regex=False),
        xlim=(0, 1.08),
        xticks=np.linspace(0, 1, 6),
        xlabel="Metric value (0-1); strict score > 0.5",
    )
    axis.invert_yaxis()
    axis.grid(axis="x")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3, frameon=False)
    figure.suptitle("Historical per-label validation report · FINAL_Report", x=0.05, y=0.98, ha="left", fontsize=15, fontweight="bold")
    figure.text(
        0.05,
        0.03,
        "Source: stored cell 9 output (execution count null), rounded to three decimals; code uses last-epoch arrays.\n"
        "No label supports, raw predictions, uncertainty intervals, or calibration evidence were supplied.",
        fontsize=9,
        color="#596572",
    )
    figure.subplots_adjust(left=0.15, right=0.98, bottom=0.20, top=0.78)
    return figure


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "reports/historical_metrics.json")
    parser.add_argument("--output-dir", type=Path, default=root / "reports/figures")
    args = parser.parse_args()
    evidence = load_evidence(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = (
        ("historical_training_losses.png", plot_training_history(evidence)),
        ("historical_per_label_report.png", plot_per_label_report(evidence)),
    )
    for filename, figure in plots:
        destination = args.output_dir / filename
        figure.savefig(destination, bbox_inches="tight")
        plt.close(figure)
        print(destination)
    print("Historical aggregate evidence only; no model training or fresh performance measurement.")


if __name__ == "__main__":
    main()
