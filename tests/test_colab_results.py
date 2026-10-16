"""Bounded offline checks for importing returned evidence without neural imports."""
from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile

import numpy as np

from scripts import verify_colab_results as verifier
from toxic_comments.metrics import evaluate_predictions, select_thresholds


class ReturnedArrayChecks(unittest.TestCase):
    def setUp(self):
        self.targets = np.array([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0],
                                 [0, 0, 1, 1, 0, 1], [1, 1, 0, 0, 1, 0]], dtype=np.uint8)
        self.scores = np.array([[.1, .8, .45, .9, .2, .7], [.7, .3, .8, .4, .6, .2],
                                [.45, .5, .9, .6, .4, .8], [.9, .7, .2, .3, .8, .1]], dtype=np.float32)

    def test_independent_metrics_cover_every_field_and_missing_classes(self):
        for targets in (self.targets, np.column_stack([np.zeros(4), np.ones(4), self.targets[:, 2:]]).astype(np.uint8)):
            thresholds = verifier.independent_thresholds(targets, self.scores)
            np.testing.assert_array_equal(thresholds, select_thresholds(targets, self.scores))
            for policy in (np.full(6, 0.5), thresholds):
                verifier.assert_close(verifier.independent_metrics(targets, self.scores, policy),
                                      evaluate_predictions(targets, self.scores, policy))

    def test_float32_boundary_and_threshold_tie_policy(self):
        scores = np.tile(np.array([np.float32(.45), np.float32(.55)]), (6, 1)).T
        targets = np.tile(np.array([0, 1], dtype=np.uint8), (6, 1)).T
        np.testing.assert_array_equal(verifier.independent_thresholds(targets, scores), select_thresholds(targets, scores))
        verifier.assert_close(verifier.independent_metrics(targets, scores, np.full(6, .45)),
                              evaluate_predictions(targets, scores, np.full(6, .45)))
        no_positive = np.zeros((4, 6), dtype=np.uint8)
        np.testing.assert_array_equal(verifier.independent_thresholds(no_positive, self.scores), np.full(6, .5))

    def test_rejects_changed_row_order_targets_and_probabilities(self):
        thresholds = select_thresholds(self.targets, self.scores)
        metrics = {"metrics_at_0_5": evaluate_predictions(self.targets, self.scores, np.full(6, .5)),
                   "metrics_at_validation_thresholds": evaluate_predictions(self.targets, self.scores, thresholds)}
        indices = np.arange(4)
        arrays = {"row_indices": indices, "targets": self.targets, "probabilities": self.scores}
        verifier.audit_predictions(arrays, self.targets, indices, metrics, thresholds)
        for key, replacement in (("row_indices", indices[::-1]), ("targets", 1 - self.targets),
                                 ("probabilities", np.full((4, 6), np.nan)),
                                 ("probabilities", np.full((4, 6), 1.01))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                verifier.audit_predictions({**arrays, key: replacement}, self.targets, indices, metrics, thresholds)

    def test_changed_ranking_metric_is_detected_against_saved_probabilities(self):
        thresholds = select_thresholds(self.targets, self.scores)
        metrics = {"metrics_at_0_5": evaluate_predictions(self.targets, self.scores, np.full(6, .5)),
                   "metrics_at_validation_thresholds": evaluate_predictions(self.targets, self.scores, thresholds)}
        metrics["metrics_at_0_5"]["per_label"]["threat"]["average_precision"] = .25
        with self.assertRaisesRegex(ValueError, "average_precision"):
            verifier.audit_predictions({"row_indices": np.arange(4), "targets": self.targets, "probabilities": self.scores},
                                       self.targets, np.arange(4), metrics, thresholds)


class ArchiveChecks(unittest.TestCase):
    def test_rejects_duplicate_traversal_symlink_and_unexpected_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bundle.zip"
            for variant in ("duplicate", "traversal", "symlink", "unexpected"):
                with self.subTest(variant=variant):
                    name = "../safe.json" if variant == "traversal" else "safe.json"
                    with zipfile.ZipFile(path, "w") as archive:
                        if variant == "symlink":
                            entry = zipfile.ZipInfo(name)
                            entry.create_system = 3
                            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                            archive.writestr(entry, "destination")
                        else:
                            archive.writestr(name, "{}")
                        if variant == "duplicate":
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", UserWarning)
                                archive.writestr(name, "{}")
                        if variant == "unexpected":
                            archive.writestr("private.npy", b"not public")
                    with self.assertRaises(ValueError):
                        verifier.read_bundle(path, {name})

    def test_npz_object_arrays_and_duplicate_json_fields_are_rejected(self):
        stream = io.BytesIO()
        np.savez(stream, values=np.array([object()], dtype=object))
        with self.assertRaises(ValueError):
            verifier.load_npz(stream.getvalue(), {"values"})
        for value in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                verifier.strict_json(value)

    def test_per_row_fields_cannot_be_published_as_aggregates(self):
        for key in verifier.FORBIDDEN_KEYS:
            with self.subTest(key=key), self.assertRaises(ValueError):
                verifier.check_no_private_fields({key: []})


class PublicEvidenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = verifier.ROOT / "reports/colab_efficient_v1"

    def test_published_report_passes_portable_aggregate_and_plot_series_checks(self):
        result = verifier.verify_public(self.report)
        self.assertEqual(result["status"], "public_aggregate_integrity_passed")
        self.assertTrue(result["plot_series_match_aggregate_sources"])
        self.assertFalse(result["private_numerical_audit_repeated"])

    def test_public_check_has_no_neural_dependency(self):
        code = r'''
import importlib.abc
import sys
class BlockNeural(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'tokenizers', 'safetensors', 'huggingface_hub'}:
            raise ModuleNotFoundError('Optional neural dependency unavailable: ' + fullname)
sys.meta_path.insert(0, BlockNeural())
from scripts.verify_colab_results import ROOT, verify_public
assert verify_public(ROOT / 'reports/colab_efficient_v1')['status'] == 'public_aggregate_integrity_passed'
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=verifier.ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_checksums_arithmetic_and_plot_data_detect_changed_public_artifacts(self):
        for variant in ("checksum", "arithmetic", "plot_data", "comparison"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                report = Path(temporary) / "report"
                shutil.copytree(self.report, report)
                filename = ("plot_data.json" if variant == "plot_data" else
                            "baseline_validation_comparison.json" if variant == "comparison" else "validation_metrics.json")
                path = report / filename
                value = json.loads(path.read_text())
                if variant == "plot_data":
                    value["training_losses"]["validation_loss"][0] += .1
                elif variant == "comparison":
                    value["numerical_deltas_neural_minus_baseline"]["metrics_at_0_5"]["macro_f1"] = .1
                else:
                    value["metrics_at_0_5"]["macro_f1"] = .1
                path.write_text(json.dumps(value))
                if variant != "checksum":
                    manifest_path = report / "public_manifest.json"
                    manifest = json.loads(manifest_path.read_text())
                    manifest["artifacts_sha256"][filename] = verifier.file_sha256(path)
                    manifest_path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    verifier.verify_public(report)


if __name__ == "__main__":
    unittest.main()
