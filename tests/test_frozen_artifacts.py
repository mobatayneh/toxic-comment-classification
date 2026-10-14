from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from toxic_comments import artifacts
from toxic_comments.data import LABELS, SplitIndices


class FrozenArtifactInvariants(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.source = self.parent / "synthetic_source.csv"
        self.source.write_text("Synthetic private input bytes\n", encoding="utf-8")
        self.run = artifacts.create_run_directory(self.parent / "run")
        self.model = self.run / "private" / "model.bin"
        self.model.write_bytes(b"Synthetic model bytes\n")
        self.splits = SplitIndices(np.array([0, 1]), np.array([2, 3]), np.array([4, 5]))
        self.freeze_arguments = dict(
            config={"method": "synthetic-test", "seed": 7},
            data_summary={"retained_rows": 6},
            source_path=self.source,
            model_path=self.model,
            thresholds=np.full(len(LABELS), 0.5),
            splits=self.splits,
            metadata={"seed": 7},
        )
        artifacts.freeze_run(self.run, **self.freeze_arguments)

    def test_frozen_artifacts_and_source_override_detect_byte_changes(self):
        manifest = artifacts.verify_frozen_run(self.run)
        self.assertEqual(manifest["label_order"], list(LABELS))
        for name in ("train", "validation", "test"):
            np.testing.assert_array_equal(getattr(artifacts.load_splits(self.run), name), getattr(self.splits, name))
        np.testing.assert_allclose(artifacts.load_thresholds(self.run), 0.5)
        copy = self.parent / "source_copy.csv"
        copy.write_bytes(self.source.read_bytes())
        artifacts.verify_frozen_run(self.run, source_path=copy)
        changed = [self.model, self.source, self.run / "config.json", self.run / "thresholds.json",
                   self.run / "metadata.json", self.run / "private" / "splits.npz", self.run / "manifest.json"]
        for path in changed:
            with self.subTest(file=path.name):
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"Changed bytes\n")
                    with self.assertRaisesRegex(ValueError, "integrity check failed"):
                        artifacts.verify_frozen_run(self.run)
                finally:
                    path.write_bytes(original)
        copy.write_bytes(b"Different source bytes\n")
        with self.assertRaisesRegex(ValueError, "source dataset integrity"):
            artifacts.verify_frozen_run(self.run, source_path=copy)

    def test_source_code_identity_invalidates_evaluation_before_claim(self):
        recorded = artifacts.read_json(self.run / "metadata.json")["code_sha256"]
        self.assertTrue(recorded)
        repository = Path(artifacts.__file__).resolve().parents[2]
        changed_source = (repository / next(iter(recorded))).resolve()
        actual_hash = artifacts.file_sha256

        def simulated_changed_source(path):
            return "changed-source-identity" if Path(path).resolve() == changed_source else actual_hash(path)

        # Simulate another executable revision without editing shared project files.
        with patch.object(artifacts, "file_sha256", side_effect=simulated_changed_source):
            with self.assertRaisesRegex(ValueError, "Source code changed after development"):
                artifacts.begin_test_evaluation(self.run)
        self.assertFalse((self.run / artifacts.TEST_MARKER).exists())
        with patch.object(artifacts, "_code_hashes", return_value={**recorded, "src/toxic_comments/new_module.py": "new-source"}):
            with self.assertRaisesRegex(ValueError, "Source code changed after development"):
                artifacts.verify_frozen_run(self.run)

    def test_model_directory_rejects_added_or_missing_files(self):
        run = artifacts.create_run_directory(self.parent / "directory_model_run")
        model = run / "model"
        model.mkdir()
        recorded = model / "config.json"
        recorded.write_text('{"synthetic": true}\n', encoding="utf-8")
        artifacts.freeze_run(run, **{**self.freeze_arguments, "model_path": model})
        artifacts.verify_frozen_run(run)
        extra = model / "new_tokenizer.json"
        extra.write_text('{"synthetic": true}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Frozen model file set"):
            artifacts.verify_frozen_run(run)
        extra.unlink()
        artifacts.verify_frozen_run(run)
        recorded.unlink()
        with self.assertRaisesRegex(ValueError, "integrity check failed"):
            artifacts.verify_frozen_run(run)

    def test_new_run_and_frozen_files_cannot_be_overwritten(self):
        with self.assertRaises(FileExistsError):
            artifacts.create_run_directory(self.run)
        manifest_before = (self.run / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            artifacts.freeze_run(self.run, **self.freeze_arguments)
        self.assertEqual((self.run / "manifest.json").read_bytes(), manifest_before)
        config_before = (self.run / "config.json").read_bytes()
        with self.assertRaises(FileExistsError):
            artifacts.write_json(self.run / "config.json", {"replacement": True})
        self.assertEqual((self.run / "config.json").read_bytes(), config_before)
        repository = Path(artifacts.__file__).resolve().parents[2]
        with self.assertRaisesRegex(ValueError, "ignored runs/"):
            artifacts.create_run_directory(repository / "unsafe-public-run")

    def test_failed_attempt_cannot_be_repeated_and_inference_order_is_frozen(self):
        targets = np.tile(np.array([0, 1])[:, None], (1, len(LABELS)))
        probabilities = np.full(targets.shape, 0.5)
        arguments = dict(public_results={"split": "test", "rows": 2}, targets=targets,
                         probabilities=probabilities, row_indices=self.splits.test)
        with self.assertRaisesRegex(ValueError, "Begin the single test"):
            artifacts.complete_test_evaluation(self.run, **arguments)
        artifacts.begin_test_evaluation(self.run)
        with self.assertRaisesRegex(ValueError, "frozen test indices"):
            artifacts.complete_test_evaluation(self.run, **{**arguments, "row_indices": self.splits.test[::-1]})
        with self.assertRaises(FileExistsError):
            artifacts.begin_test_evaluation(self.run)
        self.assertFalse((self.run / "public_test_results.json").exists())
        self.assertFalse((self.run / "private" / "test_predictions.npz").exists())

    def test_public_results_reject_row_fields_and_success_is_single_use(self):
        targets = np.tile(np.array([0, 1])[:, None], (1, len(LABELS)))
        probabilities = np.full(targets.shape, 0.5)
        arguments = dict(targets=targets, probabilities=probabilities, row_indices=self.splits.test)
        artifacts.begin_test_evaluation(self.run)
        for key in ("id", "comment_text", "comments", "raw_comments", "probabilities", "row_indices", "source_sha256"):
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, "Private field"):
                artifacts.complete_test_evaluation(self.run, public_results={"nested": {key: ["synthetic-private-value"]}}, **arguments)
        public = {"split": "test", "rows": 2, "macro_f1": 0.5, "label_order": list(LABELS)}
        destination = artifacts.complete_test_evaluation(self.run, public_results=public, **arguments)
        self.assertEqual(artifacts.read_json(destination), public)
        with np.load(self.run / "private" / "test_predictions.npz", allow_pickle=False) as stored:
            np.testing.assert_array_equal(stored["row_indices"], self.splits.test)
            np.testing.assert_array_equal(stored["targets"], targets)
            np.testing.assert_allclose(stored["probabilities"], probabilities)
        with self.assertRaises(FileExistsError):
            artifacts.complete_test_evaluation(self.run, public_results=public, **arguments)
        with self.assertRaises(FileExistsError):
            artifacts.begin_test_evaluation(self.run)

    def test_completion_rechecks_integrity_after_inference_and_preserves_failed_claim(self):
        targets = np.tile(np.array([0, 1])[:, None], (1, len(LABELS)))
        arguments = dict(public_results={"split": "test", "rows": 2}, targets=targets,
                         probabilities=np.full(targets.shape, 0.5), row_indices=self.splits.test)
        artifacts.begin_test_evaluation(self.run)
        original = self.model.read_bytes()
        self.model.write_bytes(original + b"Changed during inference\n")
        with self.assertRaisesRegex(ValueError, "integrity check failed"):
            artifacts.complete_test_evaluation(self.run, **arguments)
        self.assertFalse((self.run / "public_test_results.json").exists())
        self.assertFalse((self.run / "private" / "test_predictions.npz").exists())
        self.assertTrue((self.run / artifacts.TEST_MARKER).exists())
        self.model.write_bytes(original)
        with self.assertRaises(FileExistsError):
            artifacts.begin_test_evaluation(self.run)


if __name__ == "__main__":
    unittest.main()
