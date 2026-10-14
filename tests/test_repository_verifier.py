from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.verify_repository import (BASELINE_END, BASELINE_START, END, HISTORICAL_NOTEBOOKS,
                                       ROW_IDENTIFIER, START, baseline_table, historical_table, verify)


class RepositoryVerifierInvariants(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, capture_output=True)
        (self.root / ".gitignore").write_text("private/\nruns/\ndata/*\n!data/README.md\n", encoding="utf-8")
        for name in ("reports", "docs", "archive/notebooks"):
            (self.root / name).mkdir(parents=True)
        notebooks = []
        for index, name in enumerate(HISTORICAL_NOTEBOOKS, 1):
            history = [
                {"epoch": 1, "validation_loss": 0.2, "validation_macro_roc_auc": 0.6,
                 "validation_macro_f1_at_threshold_0_5": 0.4},
                {"epoch": 2, "validation_loss": 0.4, "validation_macro_roc_auc": 0.7,
                 "validation_macro_f1_at_threshold_0_5": 0.5},
            ]
            notebooks.append({"file": name, "sha256": str(index) * 64, "metrics": history,
                              "best_by_validation_loss": history[0]})
            archived = {"cells": [{"cell_type": "code", "execution_count": None, "outputs": [],
                                   "source": ["# Synthetic cleared historical cell"], "metadata": {}}],
                        "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
            (self.root / "archive/notebooks" / name).write_text(json.dumps(archived), encoding="utf-8")
        self.payload = {"fresh_training_executed": False, "notebooks": notebooks}
        self.history = self.root / "reports/historical_metrics.json"
        self.history.write_text(json.dumps(self.payload), encoding="utf-8")
        manifest = [{"file": item["file"], "sha256": item["sha256"]} for item in notebooks]
        (self.root / "docs/source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.readme = self.root / "README.md"
        self.readme.write_text(START + "\n" + historical_table(self.root) + "\n" + END
                               + "\n[Archive](archive/notebooks/modelXXXX(1).ipynb)\n", encoding="utf-8")

    def test_canonical_table_uses_validation_selection_and_last_epoch_separately(self):
        table = historical_table(self.root)
        self.assertEqual(table.count("Selected by validation loss | 1 | 0.2000"), 3)
        self.assertEqual(table.count("Last epoch | 2 | 0.4000"), 3)
        errors, counts = verify(self.root)
        self.assertEqual(errors, [])
        self.assertEqual(counts["historical_blocks"], 1)
        self.assertEqual(counts["archival_notebooks"], 3)
        self.assertEqual(counts["local_links"], 1)
        self.payload["notebooks"][0]["best_by_validation_loss"] = self.payload["notebooks"][0]["metrics"][1]
        self.history.write_text(json.dumps(self.payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "selected checkpoint disagrees"):
            historical_table(self.root)

    def test_stale_table_broken_link_and_archive_outputs_are_detected(self):
        self.readme.write_text(self.readme.read_text(encoding="utf-8").replace("0.2000", "0.9000", 1)
                               + "[Missing](docs/missing.md)\n", encoding="utf-8")
        notebook_path = self.root / "archive/notebooks" / HISTORICAL_NOTEBOOKS[0]
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        notebook["cells"][0]["execution_count"] = 1
        notebook["cells"][0]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["Synthetic output"]}]
        notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
        errors, _ = verify(self.root)
        self.assertTrue(any("Historical table differs" in error for error in errors))
        self.assertTrue(any("Broken local link" in error for error in errors))
        self.assertTrue(any("saved outputs/counts must be cleared" in error for error in errors))

    def test_ignored_local_inputs_stay_private_but_force_tracked_files_are_rejected(self):
        ignored = self.root / "private/input.csv"
        ignored.parent.mkdir()
        ignored.write_text("Synthetic private input\n", encoding="utf-8")
        self.assertEqual(verify(self.root)[0], [])
        subprocess.run(["git", "-C", str(self.root), "add", "-f", "private/input.csv"], check=True, capture_output=True)
        array_file = self.root / "predictions.npz"
        array_file.write_bytes(b"Synthetic private row-array fixture")
        errors, _ = verify(self.root)
        self.assertTrue(any("Private local file is eligible" in error for error in errors))
        self.assertTrue(any("Dataset export is eligible" in error for error in errors))
        self.assertTrue(any("row-array artifact is eligible" in error for error in errors))

    def test_malformed_evidence_and_out_of_order_markers_are_rejected(self):
        self.payload["notebooks"][0]["metrics"][0]["validation_macro_roc_auc"] = 1.1
        self.history.write_text(json.dumps(self.payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            historical_table(self.root)
        (self.root / "docs/invalid_markers.md").write_text(END + "\n" + START, encoding="utf-8")
        errors, _ = verify(self.root)
        self.assertTrue(any("historical table markers" in error for error in errors))

    def test_canonical_real_baseline_and_threshold_independent_ranking_are_checked(self):
        fixed_validation = {"rows": 100, "macro_average_precision": 0.7, "macro_roc_auc": 0.8,
                            "macro_f1": 0.5, "micro_f1": 0.6}
        fixed_test = {"rows": 25, "macro_average_precision": 0.6, "macro_roc_auc": 0.7,
                      "macro_f1": 0.4, "micro_f1": 0.5}
        validation = {"method": "tfidf_logistic_regression", "split": "validation", "synthetic": False,
                      "metrics_at_0_5": fixed_validation,
                      "metrics_at_validation_thresholds": {**fixed_validation, "macro_f1": 0.65, "micro_f1": 0.7}}
        test = {"method": "tfidf_logistic_regression", "split": "internal exploratory test", "synthetic": False,
                "metrics_at_0_5": fixed_test,
                "metrics_at_frozen_validation_thresholds": {**fixed_test, "macro_f1": 0.55, "micro_f1": 0.6}}
        (self.root / "reports/baseline_validation_metrics.json").write_text(json.dumps(validation), encoding="utf-8")
        test_path = self.root / "reports/baseline_internal_test_metrics.json"
        test_path.write_text(json.dumps(test), encoding="utf-8")
        table = baseline_table(self.root)
        self.assertIn("| Internal test | Validation-selected | 25 | 0.6000 | 0.7000 | 0.5500 | 0.6000 |", table)
        self.readme.write_text(self.readme.read_text(encoding="utf-8") + BASELINE_START + "\n" + table + "\n" + BASELINE_END, encoding="utf-8")
        errors, counts = verify(self.root)
        self.assertEqual(errors, [])
        self.assertEqual(counts["baseline_blocks"], 1)
        self.readme.write_text(self.readme.read_text(encoding="utf-8").replace("| Internal test | Fixed 0.5 | 25 | 0.6000", "| Internal test | Fixed 0.5 | 25 | 0.9000"), encoding="utf-8")
        self.assertTrue(any("Baseline table differs" in error for error in verify(self.root)[0]))
        test["metrics_at_frozen_validation_thresholds"]["macro_average_precision"] = 0.61
        test_path.write_text(json.dumps(test), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identical rows and ranking metrics"):
            baseline_table(self.root)

    def test_decimal_metric_precision_is_not_mistaken_for_a_row_identifier(self):
        self.assertIsNone(ROW_IDENTIFIER.search('{"metric": 0.1234567890123456}'))
        self.assertIsNotNone(ROW_IDENTIFIER.search('"' + "abcdef" * 2 + "1234" + '"'))
        self.assertIsNotNone(ROW_IDENTIFIER.search('"' + "12" * 8 + '"'))


if __name__ == "__main__":
    unittest.main()
