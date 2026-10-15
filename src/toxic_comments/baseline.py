"""CPU reference: train-only word TF-IDF and six balanced logistic regressions.

Development uses training and validation data, then freezes the model and
validation-selected thresholds. Test evaluation is a separate, single-use
operation. The offline smoke command uses an explicitly synthetic fixture.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from .artifacts import (
    begin_test_evaluation,
    complete_test_evaluation,
    create_run_directory,
    freeze_run,
    load_splits,
    load_thresholds,
    read_json,
    resolve_run_path,
    runtime_metadata,
    verify_frozen_run,
    write_json,
)
from .data import (
    LABELS,
    create_synthetic_fixture,
    file_sha256,
    load_dataset,
    split_dataset,
    split_summary,
)
from .metrics import DEFAULT_THRESHOLD_GRID, _validate_arrays, evaluate_predictions, select_thresholds

METHOD = "tfidf_logistic_regression"
PROTOCOL_NOTE = (
    "New internal exploratory holdout; the historical cohort was previously examined, "
    "so this is not a pristine external test."
)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    return int(seed)


def build_pipeline(*, seed: int = 42) -> Pipeline:
    """Construct the fixed reference model; fitting never happens here."""
    seed = _validate_seed(seed)
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_features=50_000,
            sublinear_tf=True, lowercase=True, norm="l2", dtype=np.float64,
        )),
        ("classifier", OneVsRestClassifier(
            LogisticRegression(
                solver="liblinear", C=4.0, class_weight="balanced", max_iter=1000,
                tol=1e-4, random_state=seed,
            ),
            n_jobs=1,
        )),
    ])


def validate_training_targets(targets: np.ndarray) -> None:
    """Reject unsupported train labels instead of fitting constant predictors."""
    values = np.asarray(targets)
    if values.ndim != 2 or values.shape[1] != len(LABELS) or not len(values):
        raise ValueError(f"Training targets must have a nonempty (n, {len(LABELS)}) shape")
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Training targets must be binary 0/1 values")
    missing = [label for column, label in enumerate(LABELS) if len(np.unique(values[:, column])) != 2]
    if missing:
        raise ValueError(f"Training requires both positive and negative examples for every label: {', '.join(missing)}")


def _validate_model(model: Any) -> Pipeline:
    if not isinstance(model, Pipeline) or set(model.named_steps) != {"tfidf", "classifier"}:
        raise ValueError("Frozen model is not the expected TF-IDF/logistic-regression pipeline")
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    if not isinstance(vectorizer, TfidfVectorizer) or not isinstance(classifier, OneVsRestClassifier):
        raise ValueError("Frozen model contains incompatible estimator types")
    if tuple(getattr(model, "toxic_label_order_", ())) != LABELS:
        raise ValueError("Frozen model label order is incompatible")
    if not getattr(classifier, "multilabel_", False) or len(classifier.estimators_) != len(LABELS):
        raise ValueError("Frozen classifier must contain six fitted multilabel estimators")
    if not np.array_equal(classifier.classes_, np.arange(len(LABELS))):
        raise ValueError("Frozen classifier column order is incompatible")
    if any(not np.array_equal(estimator.classes_, [0, 1]) for estimator in classifier.estimators_):
        raise ValueError("Each frozen binary estimator must expose class-1 probabilities")
    return model


def run_development(
    data_path: str | Path,
    run_dir: str | Path,
    *,
    seed: int = 42,
    conflict_policy: str = "error",
    empty_text_policy: str = "error",
    quiet: bool = False,
) -> dict[str, Any]:
    """Fit on train, select thresholds on validation, export and freeze once.

    A private, byte-identical source snapshot makes subsequent evaluation
    independent of the caller's original file location. A failed development
    attempt leaves its directory in place and requires a new directory to retry.
    """
    del quiet  # The public API matches neural development; this fit has no epoch log.
    seed = _validate_seed(seed)
    if conflict_policy not in {"error", "keep"}:
        raise ValueError("conflict_policy must be 'error' or 'keep'")
    if empty_text_policy not in {"error", "drop"}:
        raise ValueError("empty_text_policy must be 'error' or 'drop'")
    source = Path(data_path).resolve()
    source_hash = file_sha256(source)
    destination = create_run_directory(run_dir)
    metadata = runtime_metadata(seed)
    source_snapshot = destination / "private" / "source.csv"
    shutil.copyfile(source, source_snapshot)
    dataset = load_dataset(
        source_snapshot, conflict_policy=conflict_policy, empty_text_policy=empty_text_policy,
    )
    if dataset.source_sha256 != source_hash or file_sha256(source) != source_hash:
        raise ValueError("Source dataset changed while its private snapshot was being created")
    splits = split_dataset(dataset, seed=seed)
    training_targets = dataset.targets[splits.train]
    validate_training_targets(training_targets)
    model = build_pipeline(seed=seed)
    # Balanced weights are calculated independently by each estimator using
    # only its training targets. Validation and test rows never enter fit().
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(dataset.texts[splits.train], training_targets)
    model.toxic_label_order_ = list(LABELS)
    _validate_model(model)
    validation_targets = dataset.targets[splits.validation]
    validation_texts = dataset.texts[splits.validation]
    original_validation_probabilities = model.predict_proba(validation_texts)
    _validate_arrays(validation_targets, original_validation_probabilities)
    model_path = destination / "model.joblib"
    with model_path.open("xb") as stream:
        joblib.dump(model, stream, compress=3)
    # Thresholds and metrics are based exclusively on fresh inference through
    # the reloaded model, rather than arrays from a previous fit/checkpoint.
    exported_model = _validate_model(joblib.load(model_path))
    validation_probabilities = exported_model.predict_proba(validation_texts)
    _validate_arrays(validation_targets, validation_probabilities)
    if not np.array_equal(original_validation_probabilities, validation_probabilities):
        raise RuntimeError("Reloaded model does not reproduce the fitted model's validation probabilities")
    thresholds = select_thresholds(validation_targets, validation_probabilities)
    validation_predictions_path = destination / "private" / "validation_predictions.npz"
    with validation_predictions_path.open("xb") as stream:
        np.savez_compressed(
            stream, row_indices=splits.validation, targets=validation_targets,
            probabilities=validation_probabilities,
        )
    validation_metrics_path = write_json(destination / "validation_metrics.json", {
        "method": METHOD,
        "split": "validation",
        "synthetic": bool(dataset.summary["synthetic"]),
        "metrics_at_0_5": evaluate_predictions(validation_targets, validation_probabilities, np.full(len(LABELS), 0.5)),
        "metrics_at_validation_thresholds": evaluate_predictions(validation_targets, validation_probabilities, thresholds),
        "selection_note": "Thresholds were selected using this validation partition; these metrics are selection diagnostics.",
    })
    positives = training_targets.sum(axis=0)
    negatives = len(training_targets) - positives
    metadata.update({
        "model_family": "word TF-IDF and one-vs-rest logistic regression",
        "synthetic": bool(dataset.summary["synthetic"]),
        "engineering_smoke": bool(dataset.summary["synthetic"]),
        "vocabulary_fit_partition": "train",
        "class_weight_fit_partition": "train",
        "training_class_weights": {
            label: {
                "negative": float(len(training_targets) / (2 * negatives[column])),
                "positive": float(len(training_targets) / (2 * positives[column])),
            }
            for column, label in enumerate(LABELS)
        },
        "threshold_selection": "per-label validation F1; >= comparator; ties nearest 0.5 then larger; missing class keeps 0.5",
        "threshold_grid": list(DEFAULT_THRESHOLD_GRID),
        "export_matches_fitted_validation_predictions": True,
        "fresh_validation_predictions": True,
        "test_status": "unscored during development",
        "protocol_note": PROTOCOL_NOTE,
    })
    freeze_run(
        destination,
        config={
            "method": METHOD, "seed": seed, "label_order": list(LABELS),
            "conflict_policy": conflict_policy, "empty_text_policy": empty_text_policy,
            "tfidf": {
                "analyzer": "word", "ngram_range": [1, 2], "min_df": 2,
                "max_features": 50_000, "sublinear_tf": True, "lowercase": True,
                "norm": "l2", "dtype": "float64",
            },
            "classifier": {
                "wrapper": "OneVsRestClassifier", "solver": "liblinear", "C": 4.0,
                "class_weight": "balanced", "max_iter": 1000, "tol": 1e-4,
                "random_state": seed, "n_jobs": 1,
            },
        },
        data_summary={**dataset.summary, "splits": split_summary(dataset, splits)},
        source_path=source_snapshot, model_path=model_path, thresholds=thresholds,
        splits=splits, metadata=metadata,
        extra_artifacts=[validation_predictions_path, validation_metrics_path],
    )
    return {
        "run_dir": str(destination), "stage": "developed_and_frozen",
        "synthetic": bool(dataset.summary["synthetic"]), "label_order": list(LABELS),
        "checks": {
            "export_matches_fitted_validation_predictions": True,
            "fresh_validation_predictions": True,
            "test_unscored_during_development": True,
        },
    }


def evaluate_run(run_dir: str | Path, *, data_path: str | Path | None = None) -> dict[str, Any]:
    """Evaluate frozen thresholds and model once on the internal test split."""
    destination = Path(run_dir).resolve()
    manifest = verify_frozen_run(destination, source_path=data_path)
    config = read_json(destination / "config.json")
    metadata = read_json(destination / "metadata.json")
    if config.get("method") != METHOD or config.get("label_order") != list(LABELS):
        raise ValueError("Run is not a compatible frozen TF-IDF/logistic-regression experiment")
    seed = _validate_seed(config["seed"])
    source = Path(data_path).resolve() if data_path is not None else resolve_run_path(destination, manifest["source_path"])
    dataset = load_dataset(
        source, conflict_policy=config["conflict_policy"], empty_text_policy=config["empty_text_policy"],
    )
    splits = load_splits(destination)
    expected_splits = split_dataset(dataset, seed=seed)
    if any(
        not np.array_equal(getattr(splits, name), getattr(expected_splits, name))
        for name in ("train", "validation", "test")
    ):
        raise ValueError("Frozen splits do not match the verified data and recorded seed")
    model = _validate_model(joblib.load(resolve_run_path(destination, manifest["model_path"])))
    thresholds = load_thresholds(destination)
    # The claim is exclusive and persistent even if subsequent inference fails.
    # It is acquired before reading test targets or computing test predictions.
    begin_test_evaluation(destination, source_path=source)
    targets = dataset.targets[splits.test]
    probabilities = model.predict_proba(dataset.texts[splits.test])
    results = {
        "method": METHOD,
        "synthetic": bool(dataset.summary["synthetic"]),
        "engineering_smoke": bool(metadata["engineering_smoke"]),
        "split": "internal exploratory test",
        "protocol_note": metadata["protocol_note"],
        "selection_policy": "fixed reference settings; thresholds frozen from validation only",
        "metrics_at_0_5": evaluate_predictions(targets, probabilities, np.full(len(LABELS), 0.5)),
        "metrics_at_frozen_validation_thresholds": evaluate_predictions(targets, probabilities, thresholds),
    }
    complete_test_evaluation(
        destination, public_results=results, targets=targets,
        probabilities=probabilities, row_indices=splits.test, source_path=source,
    )
    return results


def run_smoke(run_dir: str | Path, *, rows: int = 240, seed: int = 42) -> dict[str, Any]:
    """Exercise train/save/reload/freeze/evaluate offline on synthetic text."""
    seed = _validate_seed(seed)
    with tempfile.TemporaryDirectory(prefix="toxic-baseline-fixture-") as temporary:
        source = create_synthetic_fixture(Path(temporary) / "synthetic.csv", rows=rows, seed=seed)
        developed = run_development(source, run_dir, seed=seed, quiet=True)
    results = evaluate_run(run_dir)
    destination = Path(run_dir).resolve()
    with np.load(destination / "private" / "test_predictions.npz", allow_pickle=False) as predictions:
        finite = bool(np.isfinite(predictions["probabilities"]).all())
        six_columns = predictions["probabilities"].shape[1] == len(LABELS)
    if not finite or not six_columns or not results["engineering_smoke"]:
        raise RuntimeError("Offline baseline smoke assertions failed")
    return {
        "status": "passed", "run_dir": str(destination), "synthetic": True,
        "purpose": "engineering mechanics only; no real-data performance claim",
        "checks": {
            **developed["checks"], "train_only_tfidf_and_balanced_weights": True,
            "finite_test_probabilities": finite, "explicit_six_label_order": six_columns,
            "frozen_test_evaluation_completed": True,
        },
    }


# Concise programmatic names are kept alongside the CLI-aligned names.
develop = run_development
evaluate = evaluate_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    development = stages.add_parser("develop", help="Fit train/validation reference pipeline and freeze a new run")
    development.add_argument("--data", required=True, type=Path)
    development.add_argument("--run-dir", required=True, type=Path)
    development.add_argument("--seed", default=42, type=int)
    development.add_argument("--conflict-policy", choices=("error", "keep"), default="error")
    development.add_argument("--empty-text-policy", choices=("error", "drop"), default="error")
    evaluation = stages.add_parser("evaluate", help="Evaluate a frozen reference run once on its internal test split")
    evaluation.add_argument("--run-dir", required=True, type=Path)
    evaluation.add_argument("--data", type=Path, help="Optional byte-identical replacement for the frozen source CSV")
    smoke = stages.add_parser("smoke", help="Offline synthetic engineering mechanics check; not a benchmark")
    smoke.add_argument("--run-dir", required=True, type=Path)
    smoke.add_argument("--rows", default=240, type=int)
    smoke.add_argument("--seed", default=42, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.stage == "develop":
            result = run_development(
                arguments.data, arguments.run_dir, seed=arguments.seed,
                conflict_policy=arguments.conflict_policy, empty_text_policy=arguments.empty_text_policy,
            )
        elif arguments.stage == "evaluate":
            result = evaluate_run(arguments.run_dir, data_path=arguments.data)
        else:
            result = run_smoke(arguments.run_dir, rows=arguments.rows, seed=arguments.seed)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, ConvergenceWarning) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
