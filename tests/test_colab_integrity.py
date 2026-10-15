"""Offline source-provenance and interrupted-freeze checks; no real-data work."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from toxic_comments.artifacts import TEST_MARKER, read_json, verify_frozen_run
from toxic_comments.data import create_synthetic_fixture, file_sha256


@unittest.skipUnless(os.environ.get("TOXIC_RUN_NEURAL_TESTS") == "1",
                     "Optional tiny CPU Colab checks: set TOXIC_RUN_NEURAL_TESTS=1")
class ColabCompletionIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import colab_train
        from toxic_comments import neural

        cls.colab, cls.neural = colab_train, neural
        cls.dependencies = neural._dependencies()
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.parent = Path(cls.temp.name)
        cls.source = create_synthetic_fixture(cls.parent / "synthetic.csv", rows=240, seed=42)
        cls.config = colab_train.ColabConfig(training=neural.TrainConfig(
            epochs=1, patience=1, batch_size=13, accumulation_steps=5,
            learning_rate=1e-3, max_length=16, cpu_threads=1,
            device="cpu", amp=False, local_files_only=True,
        ), checkpoint_every_updates=1, require_full_dataset=False)
        cls.snapshot = cls.parent / "interrupted_freeze"
        real_open = Path.open

        def disconnect_before_checksum(path, *args, **kwargs):
            if path == cls.snapshot / "manifest.sha256" and args and args[0] == "x":
                raise OSError("simulated disconnect after manifest before checksum")
            return real_open(path, *args, **kwargs)

        with patch.object(Path, "open", new=disconnect_before_checksum):
            with unittest.TestCase().assertRaises(OSError):
                colab_train.run_colab_training(cls.source, cls.snapshot, config=cls.config,
                                              quiet=True, _smoke=True)
        assert (cls.snapshot / "manifest.json").is_file()
        assert not (cls.snapshot / "manifest.sha256").exists()
        assert (cls.snapshot / colab_train.FREEZE_TRANSACTION).is_file()

    def copy_snapshot(self, directory):
        run = Path(directory) / "run"
        shutil.copytree(self.snapshot, run)
        return run

    def run_colab(self, run):
        return self.colab.run_colab_training(self.source, run, config=self.config,
                                             quiet=True, _smoke=True)

    def tree_hashes(self, run):
        return {path.relative_to(run).as_posix(): file_sha256(path)
                for path in run.rglob("*") if path.is_file()}

    def test_missing_checksum_with_transaction_recovers_without_training_or_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.copy_snapshot(directory)
            before = self.tree_hashes(run)
            with patch.object(self.neural, "_new_model", side_effect=AssertionError("new model")), \
                    patch.object(self.neural, "_predict", side_effect=AssertionError("inference")), \
                    patch.object(self.colab, "_train_resumable", side_effect=AssertionError("training")):
                result = self.run_colab(run)
            self.assertEqual(result["stage"], "already_frozen")
            self.assertFalse((run / self.colab.FREEZE_TRANSACTION).exists())
            verify_frozen_run(run)
            for name, value in before.items():
                if name != self.colab.FREEZE_TRANSACTION:
                    self.assertEqual(file_sha256(run / name), value, name)
            identity = read_json(run / self.colab.IDENTITY_FILE)
            with zipfile.ZipFile(result["results_for_review"]) as archive:
                metadata = json.loads(archive.read("result_metadata.json"))
                self.assertEqual(metadata["source_sha256"], identity["source_sha256"])
                self.assertEqual(metadata["code_sha256"], identity["code_sha256"])
                self.assertEqual(metadata["tokenizer_revision"], identity["tokenizer_revision"])
                self.assertIn("scripts/colab_train.py", metadata["code_sha256"])

    def test_present_corrupt_checksum_is_never_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.copy_snapshot(directory)
            (run / "manifest.sha256").write_text("0" * 64 + "\n")
            before = self.tree_hashes(run)
            with self.assertRaisesRegex(ValueError, "manifest integrity"):
                self.run_colab(run)
            self.assertEqual(self.tree_hashes(run), before)

    def test_changed_artifact_missing_transaction_and_test_boundary_fail_without_mutation(self):
        for scenario in ("artifact", "transaction", "test_boundary", "truncated_manifest"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                run = self.copy_snapshot(directory)
                if scenario == "artifact":
                    path = run / "validation_metrics.json"
                    path.write_bytes(path.read_bytes() + b"\n")
                elif scenario == "transaction":
                    (run / self.colab.FREEZE_TRANSACTION).unlink()
                elif scenario == "test_boundary":
                    (run / TEST_MARKER).write_text("{}\n")
                else:
                    (run / "manifest.json").write_text('{"schema_version":')
                before = self.tree_hashes(run)
                with self.assertRaises(ValueError):
                    self.run_colab(run)
                self.assertEqual(self.tree_hashes(run), before)
                self.assertFalse((run / "manifest.sha256").exists())

    def test_changed_or_incomplete_checkpoint_cannot_complete_freeze(self):
        for incomplete in (False, True):
            with self.subTest(incomplete=incomplete), tempfile.TemporaryDirectory() as directory:
                run = self.copy_snapshot(directory)
                path = run / self.colab.LATEST_CHECKPOINT
                state = self.dependencies.torch.load(path, map_location="cpu", weights_only=True)
                state["training_complete"] = False
                self.colab.atomic_torch_save(self.dependencies.torch, state, path)
                if incomplete:
                    transaction_path = run / self.colab.FREEZE_TRANSACTION
                    transaction = read_json(transaction_path)
                    transaction["latest_checkpoint_sha256"] = file_sha256(path)
                    transaction_path.write_text(json.dumps(transaction))
                before = self.tree_hashes(run)
                with self.assertRaisesRegex(ValueError, "transaction|completed training checkpoint"):
                    self.run_colab(run)
                self.assertEqual(self.tree_hashes(run), before)

    def test_source_change_after_training_stops_before_export(self):
        original_hashes = self.colab._source_code_hashes()
        changed = False

        def hashes():
            return {**original_hashes, "scripts/colab_train.py": "0" * 64} if changed else original_hashes

        def finish_training(*args, **kwargs):
            nonlocal changed
            changed = True
            return {"interrupted": False, "training_complete": True}

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            with patch.object(self.colab, "_source_code_hashes", side_effect=hashes), \
                    patch.object(self.colab, "_train_resumable", side_effect=finish_training), \
                    patch.object(self.neural, "_export_best", side_effect=AssertionError("export")):
                with self.assertRaisesRegex(ValueError, "source code changed"):
                    self.run_colab(run)
            self.assertFalse((run / "model").exists())
            self.assertFalse((run / "manifest.json").exists())
            self.assertFalse((run / self.colab.FREEZE_TRANSACTION).exists())

    def test_source_change_after_fresh_validation_stops_before_freeze(self):
        original_hashes = self.colab._source_code_hashes()
        changed = False
        evaluate = self.colab.evaluate_predictions

        def hashes():
            return {**original_hashes, "scripts/colab_train.py": "0" * 64} if changed else original_hashes

        def changed_after_validation(*args, **kwargs):
            nonlocal changed
            result = evaluate(*args, **kwargs)
            changed = True
            return result

        with tempfile.TemporaryDirectory() as directory:
            run = self.copy_snapshot(directory)
            (run / "manifest.json").unlink()
            (run / self.colab.FREEZE_TRANSACTION).unlink()
            checkpoint_hash = file_sha256(run / self.colab.LATEST_CHECKPOINT)
            with patch.object(self.colab, "_source_code_hashes", side_effect=hashes), \
                    patch.object(self.colab, "evaluate_predictions", side_effect=changed_after_validation), \
                    patch.object(self.colab, "freeze_run", side_effect=AssertionError("freeze")), \
                    patch.object(self.colab, "_optimizer_group", side_effect=AssertionError("training")):
                with self.assertRaisesRegex(ValueError, "source code changed"):
                    self.run_colab(run)
            self.assertEqual(file_sha256(run / self.colab.LATEST_CHECKPOINT), checkpoint_hash)
            self.assertFalse((run / "manifest.json").exists())
            self.assertFalse((run / self.colab.FREEZE_TRANSACTION).exists())


if __name__ == "__main__":
    unittest.main()
