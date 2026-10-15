from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from toxic_comments.artifacts import TEST_MARKER, load_splits, load_thresholds, read_json, verify_frozen_run
from toxic_comments.data import LABELS, create_synthetic_fixture, load_dataset, split_dataset
from toxic_comments.metrics import evaluate_predictions, select_thresholds


class BaselineInvariants(unittest.TestCase):
    def test_unlearnable_training_labels_require_explicit_coverage(self):
        from toxic_comments.baseline import validate_training_targets

        targets = np.tile(np.array([0, 1])[:, None], (1, len(LABELS)))
        validate_training_targets(targets)
        for value in (0, 1):
            invalid = targets.copy()
            invalid[:, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "both positive and negative"):
                validate_training_targets(invalid)

    def test_develop_and_evaluate_in_separate_processes_preserve_frozen_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = create_synthetic_fixture(parent / "synthetic.csv", rows=120, seed=2718)
            with source.open(encoding="utf-8", newline="") as stream:
                fixture_rows = list(csv.DictReader(stream))
            for index, fixture_row in enumerate(fixture_rows):
                fixture_row["id"] = f"synthetic-id-{index}"
                if index < 2:
                    fixture_row["comment_text"] += " heldoutcanary"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(fixture_rows[0]))
                writer.writeheader()
                writer.writerows(fixture_rows)
            dataset = load_dataset(source)
            # Put the repeated canary in both held-out partitions. min_df=2
            # would admit it immediately if development fitted on all rows.
            for seed in range(1000):
                candidate_split = split_dataset(dataset, seed=seed)
                if 0 in candidate_split.validation and 1 in candidate_split.test:
                    break
            else:
                self.fail("Could not construct the bounded held-out vocabulary fixture")
            run = parent / "baseline"
            repository = Path(__file__).resolve().parents[1]
            environment = {
                **os.environ,
                "PYTHONPATH": str(repository / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
                "PYTHONHASHSEED": "42", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            }

            def cli(*arguments):
                return subprocess.run([sys.executable, "-m", "toxic_comments.baseline", *arguments],
                                      cwd=parent, env=environment, capture_output=True, text=True, timeout=120)

            developed = cli("develop", "--data", str(source), "--run-dir", str(run), "--seed", str(seed))
            self.assertEqual(developed.returncode, 0, developed.stderr)
            development_output = json.loads(developed.stdout)
            self.assertEqual(development_output["stage"], "developed_and_frozen")
            self.assertTrue(development_output["synthetic"])
            self.assertFalse((run / TEST_MARKER).exists())
            self.assertFalse((run / "public_test_results.json").exists())
            manifest = verify_frozen_run(run)
            thresholds_before = (run / "thresholds.json").read_bytes()
            model_before = (run / manifest["model_path"]).read_bytes()
            model = joblib.load(run / manifest["model_path"])
            splits = load_splits(run)
            vectorizer = model.named_steps["tfidf"]
            self.assertNotIn("heldoutcanary", vectorizer.vocabulary_)
            markers = ("identitymarker", "insultmarker", "obscenemarker", "severemarker", "threatmarker", "toxicmarker")
            for column, term in enumerate(markers):
                train_document_frequency = int(dataset.targets[splits.train, column].sum())
                expected_idf = np.log((1 + len(splits.train)) / (1 + train_document_frequency)) + 1
                self.assertAlmostEqual(vectorizer.idf_[vectorizer.vocabulary_[term]], expected_idf)
            with np.load(run / "private" / "validation_predictions.npz", allow_pickle=False) as predictions:
                np.testing.assert_array_equal(predictions["row_indices"], splits.validation)
                np.testing.assert_array_equal(predictions["targets"], dataset.targets[splits.validation])
                np.testing.assert_allclose(predictions["probabilities"], model.predict_proba(dataset.texts[splits.validation]))
                np.testing.assert_allclose(load_thresholds(run), select_thresholds(predictions["targets"], predictions["probabilities"]))

            evaluated = cli("evaluate", "--run-dir", str(run))
            self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
            public = json.loads(evaluated.stdout)
            self.assertEqual(read_json(run / "public_test_results.json"), public)
            self.assertTrue(public["synthetic"])
            self.assertEqual((run / "thresholds.json").read_bytes(), thresholds_before)
            self.assertEqual((run / manifest["model_path"]).read_bytes(), model_before)
            expected_metrics = evaluate_predictions(dataset.targets[splits.test], model.predict_proba(dataset.texts[splits.test]), load_thresholds(run))
            self.assertEqual(public["metrics_at_frozen_validation_thresholds"], expected_metrics)
            forbidden = {"id", "ids", "comment_text", "texts", "comments", "raw_comments", "records",
                         "row_indices", "targets", "predictions", "probabilities", "source_path", "model_path"}

            def check_aggregate(value):
                if isinstance(value, dict):
                    for key, child in value.items():
                        self.assertNotIn(key.lower(), forbidden)
                        self.assertNotIn("sha256", key.lower())
                        check_aggregate(child)
                elif isinstance(value, list):
                    for child in value:
                        check_aggregate(child)

            check_aggregate(public)
            self.assertNotIn("Synthetic fixture comment", evaluated.stdout)
            self.assertNotIn("synthetic-id-", evaluated.stdout)
            with np.load(run / "private" / "test_predictions.npz", allow_pickle=False) as predictions:
                np.testing.assert_array_equal(predictions["row_indices"], splits.test)
                np.testing.assert_array_equal(predictions["targets"], dataset.targets[splits.test])
            public_before = (run / "public_test_results.json").read_bytes()
            repeated_evaluation = cli("evaluate", "--run-dir", str(run))
            self.assertNotEqual(repeated_evaluation.returncode, 0)
            repeated_development = cli("develop", "--data", str(source), "--run-dir", str(run), "--seed", str(seed))
            self.assertNotEqual(repeated_development.returncode, 0)
            self.assertEqual((run / "public_test_results.json").read_bytes(), public_before)
            self.assertEqual((run / "thresholds.json").read_bytes(), thresholds_before)
            self.assertEqual((run / manifest["model_path"]).read_bytes(), model_before)


if __name__ == "__main__":
    unittest.main()
