#!/usr/bin/env python3
"""Check reviewable publication files without training or network access.

This is a bounded safeguard, not a claim that automated scanning proves privacy.
Ignored local datasets, weights, and run directories are outside publication.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

START = "<!-- historical-metrics:start -->"
END = "<!-- historical-metrics:end -->"
BASELINE_START = "<!-- baseline-metrics:start -->"
BASELINE_END = "<!-- baseline-metrics:end -->"
COLAB_START = "<!-- colab-efficient-metrics:start -->"
COLAB_END = "<!-- colab-efficient-metrics:end -->"
COMPARISON_START = "<!-- validation-comparison:start -->"
COMPARISON_END = "<!-- validation-comparison:end -->"
COLAB_REPORT_DIR = Path("reports/colab_efficient_v1")
SUMMARY_METRICS = ("macro_average_precision", "macro_roc_auc", "macro_f1", "micro_f1")
TEXT_SUFFIXES = {".md", ".py", ".json", ".ipynb", ".toml", ".txt", ".yml", ".yaml", ".cfg", ".ini"}
PRIVATE_PARTS = {".git", ".venv", "__pycache__", ".ipynb_checkpoints", "runs", "private", "build", "dist"}
PRIVATE_ARTIFACT_SUFFIXES = {".pt", ".pth", ".joblib", ".pkl", ".safetensors", ".npz", ".npy"}
DATA_EXPORT_SUFFIXES = {".csv", ".tsv", ".parquet", ".feather", ".arrow"}
HISTORICAL_NOTEBOOKS = ("model_FINAL_Report.ipynb", "Untitled10.ipynb", "modelXXXX(1).ipynb")
SECRET_PATTERNS = (
    re.compile(r"\b(?:ghp_|gho_|ghu_|ghs_|github_pat_)[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{24,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Decimal fractions can have exactly sixteen digits, so exclude adjacent dots.
# Quoted or standalone sixteen-character IDs still match, including all digits.
ROW_IDENTIFIER = re.compile(r"(?<![\w.])[0-9a-f]{16}(?![\w.])", re.IGNORECASE)


def historical_table(root: Path) -> str:
    """One canonical selected-checkpoint / last-epoch table from saved outputs."""
    payload = json.loads((root / "reports/historical_metrics.json").read_text(encoding="utf-8"))
    if payload.get("fresh_training_executed") is not False:
        raise ValueError("Historical evidence must explicitly state that no fresh training was executed")
    entries = {item["file"]: item for item in payload["notebooks"]}
    if set(entries) != set(HISTORICAL_NOTEBOOKS) or len(payload["notebooks"]) != len(entries):
        raise ValueError("Historical evidence must contain exactly the three distinct original notebooks")
    lines = [
        "| Notebook | Recorded checkpoint | Epoch | Validation loss | Macro ROC-AUC | Macro F1 at > 0.5 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name in HISTORICAL_NOTEBOOKS:
        item = entries[name]
        history = item["metrics"]
        if not history:
            raise ValueError(f"No historical epoch records for {name}")
        epochs = []
        for row in history:
            epoch = row["epoch"]
            if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
                raise ValueError(f"Invalid historical epoch for {name}")
            epochs.append(epoch)
            for metric in ("validation_loss", "validation_macro_roc_auc", "validation_macro_f1_at_threshold_0_5"):
                value = row[metric]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid historical {metric} for {name}")
                if metric != "validation_loss" and value > 1:
                    raise ValueError(f"Historical {metric} must be between 0 and 1 for {name}")
        if epochs != sorted(set(epochs)):
            raise ValueError(f"Historical epochs must be unique and ordered for {name}")
        best = min(history, key=lambda row: row["validation_loss"])
        recorded_best = item["best_by_validation_loss"]
        if best != recorded_best:
            raise ValueError(f"Historical selected checkpoint disagrees with history for {name}")
        for role, row in (("Selected by validation loss", best), ("Last epoch", history[-1])):
            lines.append(
                f"| `{name}` | {role} | {row['epoch']} | {row['validation_loss']:.4f} | "
                f"{row['validation_macro_roc_auc']:.4f} | {row['validation_macro_f1_at_threshold_0_5']:.4f} |"
            )
    lines.extend([
        "",
        "These are historical validation outputs, not a new run or held-out test results. "
        "FINAL_Report and modelXXXX share the same saved five-epoch history; they are not independent replications.",
    ])
    return "\n".join(lines)


def baseline_table(root: Path) -> str:
    """A canonical real CPU baseline table derived only from saved aggregates."""
    validation = json.loads((root / "reports/baseline_validation_metrics.json").read_text(encoding="utf-8"))
    test = json.loads((root / "reports/baseline_internal_test_metrics.json").read_text(encoding="utf-8"))
    for role, payload in (("validation", validation), ("internal exploratory test", test)):
        if payload.get("synthetic") is not False or payload.get("method") != "tfidf_logistic_regression" or payload.get("split") != role:
            raise ValueError("Baseline evidence must identify the real CPU method and recorded partition")
    rows = (
        ("Validation", "Fixed 0.5", validation["metrics_at_0_5"]),
        ("Validation", "Validation-selected", validation["metrics_at_validation_thresholds"]),
        ("Internal test", "Fixed 0.5", test["metrics_at_0_5"]),
        ("Internal test", "Validation-selected", test["metrics_at_frozen_validation_thresholds"]),
    )
    for fixed, selected in ((rows[0][2], rows[1][2]), (rows[2][2], rows[3][2])):
        for field in ("rows", "macro_average_precision", "macro_roc_auc"):
            if fixed[field] != selected[field]:
                raise ValueError(f"Threshold policies must use identical rows and ranking metrics: {field}")
    lines = [
        "| Partition | Threshold policy | Rows | Macro AP | Macro ROC-AUC | Macro F1 | Micro F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for partition, policy, metrics in rows:
        count = metrics["rows"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Baseline aggregate row counts must be positive integers")
        values = []
        for name in ("macro_average_precision", "macro_roc_auc", "macro_f1", "micro_f1"):
            value = metrics[name]
            if value is None:
                values.append("undefined")
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid baseline metric: {name}")
            else:
                values.append(f"{value:.4f}")
        lines.append(f"| {partition} | {policy} | {count} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def _validation_summary_rows(payload: dict) -> tuple[dict, dict]:
    """Validate the threshold policies before presenting validation aggregates."""
    if payload.get("synthetic") is not False or payload.get("split") != "validation":
        raise ValueError("Colab evidence must explicitly identify real validation data")
    fixed, selected = payload["metrics_at_0_5"], payload["metrics_at_validation_thresholds"]
    for field in ("rows", "macro_average_precision", "macro_roc_auc"):
        if fixed[field] != selected[field]:
            raise ValueError(f"Threshold policies must use identical rows and ranking metrics: {field}")
    for metrics in (fixed, selected):
        if isinstance(metrics["rows"], bool) or not isinstance(metrics["rows"], int) or metrics["rows"] < 1:
            raise ValueError("Validation aggregate row counts must be positive integers")
        for field in SUMMARY_METRICS:
            value = metrics[field]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"Invalid validation metric: {field}")
    return fixed, selected


def _metric_cells(metrics: dict) -> str:
    return " | ".join("undefined" if metrics[field] is None else f"{metrics[field]:.4f}"
                      for field in SUMMARY_METRICS)


def colab_table(root: Path) -> str:
    """The completed neural run's validation selection diagnostics only."""
    directory = root / COLAB_REPORT_DIR
    validation = json.loads((directory / "validation_metrics.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "result_metadata.json").read_text(encoding="utf-8"))
    if config.get("method") != "random_initialized_roberta":
        raise ValueError("Colab evidence must identify the random-initialized RoBERTa method")
    if metadata.get("test_evaluation_performed") is not False or metadata.get("validation_is_selection_diagnostic") is not True:
        raise ValueError("Colab validation must remain selection diagnostics with the internal test unscored")
    if any(metadata.get(field) is not False for field in
           ("model_weights_included", "per_row_arrays_included", "raw_texts_included")):
        raise ValueError("Colab public evidence must exclude private model and row artifacts")
    rows = _validation_summary_rows(validation)
    lines = [
        "| Partition | Threshold policy | Rows | Macro AP | Macro ROC-AUC | Macro F1 | Micro F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for policy, metrics in zip(("Fixed 0.5", "Validation-selected"), rows):
        lines.append(f"| Validation | {policy} | {metrics['rows']} | {_metric_cells(metrics)} |")
    return "\n".join(lines)


def validation_comparison_table(root: Path) -> str:
    """Compare the two models on the same verified validation partition."""
    directory = root / COLAB_REPORT_DIR
    comparison = json.loads((directory / "baseline_validation_comparison.json").read_text(encoding="utf-8"))
    if comparison.get("split") != "validation" or comparison.get("same_source_same_split_verified") is not True:
        raise ValueError("Model comparison must verify the same source and validation partition")
    inputs = {
        "baseline": json.loads((root / "reports/baseline_validation_metrics.json").read_text(encoding="utf-8")),
        "neural": json.loads((directory / "validation_metrics.json").read_text(encoding="utf-8")),
    }
    rows = {}
    for name, payload in inputs.items():
        expected_method = "tfidf_logistic_regression" if name == "baseline" else "random_initialized_roberta"
        if comparison["models"][name].get("method") != expected_method:
            raise ValueError(f"Validation comparison must identify the recorded {name} method")
        rows[name] = _validation_summary_rows(payload)
        for key in ("metrics_at_0_5", "metrics_at_validation_thresholds"):
            if comparison["models"][name][key] != payload[key]:
                raise ValueError(f"Validation comparison disagrees with saved {name} metrics")
        if rows[name][0]["rows"] != comparison["rows"]:
            raise ValueError("Model comparison row counts must match the validation evidence")
    lines = [
        "| Model | Threshold policy | Validation rows | Macro AP | Macro ROC-AUC | Macro F1 | Micro F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, title in (("baseline", "TF-IDF + logistic regression"), ("neural", "Random RoBERTa")):
        for policy, metrics in zip(("Fixed 0.5", "Validation-selected"), rows[name]):
            lines.append(f"| {title} | {policy} | {metrics['rows']} | {_metric_cells(metrics)} |")
    return "\n".join(lines)


def publication_files(root: Path) -> list[Path]:
    """Use Git's publication candidates, including tracked files now ignored."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        capture_output=True, check=False,
    )
    if result.returncode == 0:
        return sorted({root / name for name in result.stdout.decode().split("\0") if name and (root / name).is_file()})
    return sorted(
        path for path in root.rglob("*") if path.is_file()
        and not any(part in PRIVATE_PARTS for part in path.relative_to(root).parts)
        and not (path.relative_to(root).parts[0] == "data" and path.name != "README.md")
    )


def markdown_targets(text: str):
    """Read inline Markdown targets, including balanced filename parentheses."""
    for match in re.finditer(r"\]\(", text):
        start = match.end()
        depth, escaped, cursor = 1, False, start
        while cursor < len(text) and depth:
            char = text[cursor]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            cursor += 1
        if depth == 0:
            target = text[start : cursor - 1].strip().split(maxsplit=1)[0]
            yield target.strip("<>").replace("\\(", "(").replace("\\)", ")")
    for match in re.finditer(r"^\s*\[[^\]]+\]:\s*(\S+)", text, re.MULTILINE):
        yield match.group(1).strip("<>")


def verify(root: Path) -> tuple[list[str], dict[str, int]]:
    root = root.resolve()
    errors: list[str] = []
    counts = {"publication_files": 0, "local_links": 0, "historical_blocks": 0, "baseline_blocks": 0,
              "colab_blocks": 0, "validation_comparison_blocks": 0, "archival_notebooks": 0,
              "retired_colab_entries": 0}
    try:
        canonical_table = historical_table(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"Historical evidence: {exc}")
        canonical_table = None
    baseline_paths = [root / "reports/baseline_validation_metrics.json", root / "reports/baseline_internal_test_metrics.json"]
    baseline_expected = any(path.exists() for path in baseline_paths)
    canonical_baseline = None
    if baseline_expected:
        try:
            canonical_baseline = baseline_table(root)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"Baseline evidence: {exc}")
    colab_expected = (root / COLAB_REPORT_DIR).is_dir()
    additional_tables = (
        ("colab_blocks", COLAB_START, COLAB_END, "Colab", colab_table),
        ("validation_comparison_blocks", COMPARISON_START, COMPARISON_END,
         "Validation comparison", validation_comparison_table),
    )
    additional_canonical = {}
    readme_additional_blocks = set()
    if colab_expected:
        for key, _, _, title, render in additional_tables:
            try:
                additional_canonical[key] = render(root)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f"{title} evidence: {exc}")
    files = publication_files(root)
    counts["publication_files"] = len(files)
    readme_has_block = False
    readme_has_baseline_block = False
    for path in files:
        relative = path.relative_to(root)
        if any(part in PRIVATE_PARTS for part in relative.parts) or path.name.startswith(".env"):
            errors.append(f"Private local file is eligible for publication: {relative}")
        if path.suffix.lower() in PRIVATE_ARTIFACT_SUFFIXES:
            errors.append(f"Private model or row-array artifact is eligible for publication: {relative}")
        if path.suffix.lower() in DATA_EXPORT_SUFFIXES:
            errors.append(f"Dataset export is eligible for publication: {relative}")
        if path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            errors.append(f"Cannot inspect text file: {relative}")
            continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            errors.append(f"Potential credential material in {relative}; inspect privately")
        if EMAIL.search(text):
            errors.append(f"Potential contact email in {relative}; inspect privately")
        if ROW_IDENTIFIER.search(text):
            errors.append(f"Potential raw comment identifier in {relative}; file SHA-256 values are allowed")
        notebook = None
        markdown = text if path.suffix == ".md" else None
        if path.suffix == ".ipynb":
            try:
                notebook = json.loads(text)
                markdown = "\n".join(
                    "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
                )
            except (ValueError, KeyError, TypeError):
                errors.append(f"Malformed notebook: {relative}")
            if relative == Path("notebooks/03_colab_full_training.ipynb") and notebook is not None:
                counts["retired_colab_entries"] += 1
                workflow = notebook.get("metadata", {}).get("training_workflow", {})
                if any(cell.get("cell_type") == "code" for cell in notebook["cells"]):
                    errors.append("Retired notebook 03 must contain no executable training cells")
                if workflow.get("status") != "retired" or workflow.get("replacement") != "04_colab_efficient_training.ipynb":
                    errors.append("Retired notebook 03 must identify notebook 04 as its replacement")
                if "04_colab_efficient_training.ipynb" not in tuple(markdown_targets(markdown or "")):
                    errors.append("Retired notebook 03 must link to the canonical notebook 04")
        if markdown is not None:
            for target in markdown_targets(markdown):
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or target.startswith("#"):
                    if parsed.scheme == "sandbox":
                        errors.append(f"Nonportable sandbox link in {relative}")
                    continue
                local = unquote(parsed.path)
                if not local:
                    continue
                counts["local_links"] += 1
                candidate = (path.parent / local).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError:
                    errors.append(f"Local link leaves repository in {relative}: {target}")
                    continue
                if not candidate.exists():
                    errors.append(f"Broken local link in {relative}: {target}")
        if path.suffix == ".md":
            blocks = re.findall(re.escape(START) + r"(.*?)" + re.escape(END), text, re.DOTALL)
            if text.count(START) != text.count(END) or len(blocks) != text.count(START) or any(START in block for block in blocks):
                errors.append(f"Unbalanced or nested historical table markers in {relative}")
            for block in blocks:
                counts["historical_blocks"] += 1
                if relative == Path("README.md"):
                    readme_has_block = True
                if canonical_table is not None and block.strip() != canonical_table:
                    errors.append(f"Historical table differs from saved evidence in {relative}")
            baseline_blocks = re.findall(re.escape(BASELINE_START) + r"(.*?)" + re.escape(BASELINE_END), text, re.DOTALL)
            if text.count(BASELINE_START) != text.count(BASELINE_END) or len(baseline_blocks) != text.count(BASELINE_START) or any(BASELINE_START in block for block in baseline_blocks):
                errors.append(f"Unbalanced or nested baseline table markers in {relative}")
            for block in baseline_blocks:
                counts["baseline_blocks"] += 1
                if relative == Path("README.md"):
                    readme_has_baseline_block = True
                if canonical_baseline is None:
                    errors.append(f"Baseline table has no valid saved aggregate evidence in {relative}")
                elif block.strip() != canonical_baseline:
                    errors.append(f"Baseline table differs from saved evidence in {relative}")
            for key, start, end, title, _ in additional_tables:
                blocks = re.findall(re.escape(start) + r"(.*?)" + re.escape(end), text, re.DOTALL)
                if text.count(start) != text.count(end) or len(blocks) != text.count(start) or any(start in block for block in blocks):
                    errors.append(f"Unbalanced or nested {title.lower()} table markers in {relative}")
                for block in blocks:
                    counts[key] += 1
                    if relative == Path("README.md"):
                        readme_additional_blocks.add(key)
                    if key not in additional_canonical:
                        errors.append(f"{title} table has no valid saved aggregate evidence in {relative}")
                    elif block.strip() != additional_canonical[key]:
                        errors.append(f"{title} table differs from saved evidence in {relative}")
        if relative.parts[:2] == ("archive", "notebooks") and path.suffix == ".ipynb":
            counts["archival_notebooks"] += 1
            try:
                notebook = json.loads(text)
                for index, cell in enumerate(notebook["cells"]):
                    if cell.get("cell_type") == "code" and (cell.get("outputs") or cell.get("execution_count") is not None):
                        errors.append(f"Archive {relative}, cell {index}: saved outputs/counts must be cleared")
            except (ValueError, KeyError, TypeError):
                errors.append(f"Malformed archival notebook: {relative}")
    if not readme_has_block:
        errors.append("README.md must include the canonical historical table between the agreed markers")
    if baseline_expected and not readme_has_baseline_block:
        errors.append("README.md must include the canonical baseline table between the agreed markers")
    if colab_expected:
        for key, _, _, title, _ in additional_tables:
            if key not in readme_additional_blocks:
                errors.append(f"README.md must include the canonical {title.lower()} table between the agreed markers")
    if (root / "notebooks/04_colab_efficient_training.ipynb").is_file() and counts["retired_colab_entries"] != 1:
        errors.append("The canonical Colab workflow must retain one retired notebook 03 entry for existing links")
    if counts["archival_notebooks"] != 3:
        errors.append("Expected three sanitized historical notebook derivatives in archive/notebooks/")
    try:
        manifest = json.loads((root / "docs/source_manifest.json").read_text(encoding="utf-8"))
        known = {item["file"]: item["sha256"] for item in manifest}
        if len(known) != len(manifest) or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in known.values()):
            errors.append("Source manifest must have distinct files and lowercase SHA-256 hashes")
        payload = json.loads((root / "reports/historical_metrics.json").read_text(encoding="utf-8"))
        for notebook in payload["notebooks"]:
            if known.get(notebook["file"]) != notebook["sha256"]:
                errors.append(f"Historical source hash disagrees with source manifest: {notebook['file']}")
    except (OSError, ValueError, KeyError, TypeError):
        errors.append("Source manifest and historical source hashes must be available and well formed")
    return errors, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--print-historical-table", action="store_true")
    parser.add_argument("--print-baseline-table", action="store_true")
    parser.add_argument("--print-colab-table", action="store_true")
    parser.add_argument("--print-validation-comparison-table", action="store_true")
    args = parser.parse_args()
    if args.print_historical_table:
        print(historical_table(args.root))
        return 0
    if args.print_baseline_table:
        print(baseline_table(args.root))
        return 0
    if args.print_colab_table:
        print(colab_table(args.root))
        return 0
    if args.print_validation_comparison_table:
        print(validation_comparison_table(args.root))
        return 0
    errors, counts = verify(args.root)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("Repository verification passed: " + ", ".join(f"{name}={value}" for name, value in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
