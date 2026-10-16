"""Resource-selection and lifecycle guards; CUDA probe itself needs Colab."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import colab_profile as profile


class ColabProfileTests(unittest.TestCase):
    def test_select_fastest_safe_candidate_not_largest(self):
        records = [
            {"batch_size": 8, "status": "ok", "samples_per_second": 20, "within_memory_budget": True},
            {"batch_size": 16, "status": "ok", "samples_per_second": 28, "within_memory_budget": True},
            {"batch_size": 32, "status": "ok", "samples_per_second": 25, "within_memory_budget": True},
            {"batch_size": 64, "status": "ok", "samples_per_second": 40, "within_memory_budget": False},
        ]
        self.assertEqual(profile.select_candidate(records)["batch_size"], 16)

    def test_reject_missing_safe_candidate(self):
        for records in ([], [{"batch_size": 8, "status": "cuda_out_of_memory"}],
                        [{"batch_size": 16, "status": "ok", "samples_per_second": float("nan"), "within_memory_budget": True}]):
            with self.assertRaisesRegex(RuntimeError, "No candidate"):
                profile.select_candidate(records)

    def test_every_candidate_preserves_effective_batch_and_protocol(self):
        for batch in profile.CANDIDATES:
            config = profile.efficient_config(batch, 2)
            self.assertEqual(config.training.batch_size * config.training.accumulation_steps, 256)
            self.assertEqual(config.training.epochs, 5)
            self.assertEqual(config.training.max_length, 256)
            self.assertTrue(config.require_full_dataset)
            self.assertTrue(config.training.cache_tokenization)
            self.assertTrue(config.training.dynamic_padding)
            self.assertFalse(config.gradient_checkpointing)

    def test_saved_config_requires_complete_fixed_profile(self):
        config = profile.efficient_config(16, 2)
        self.assertEqual(profile.saved_config({"requested_config": asdict(config)}), config)
        changed = replace(config, require_full_dataset=False)
        with self.assertRaisesRegex(ValueError, "fixed efficient"):
            profile.saved_config({"requested_config": asdict(changed)})

    def test_unfinished_packaging_does_not_create_run_or_call_freeze_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "not_started"
            with patch.object(profile.runner, "package_results", side_effect=AssertionError("must not package")):
                result = profile.package(destination)
            self.assertEqual(result["stage"], "unfinished")
            self.assertFalse(destination.exists())

    def test_status_missing_run_does_not_import_ml_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(profile.runner.core, "_dependencies", side_effect=AssertionError("must not import")):
                result = profile.status(Path(directory) / "missing")
            self.assertEqual(result["stage"], "not_started")

    def test_resume_uses_saved_config_without_reading_calibration_or_evaluating_test(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "existing"
            (run / "private").mkdir(parents=True)
            config = profile.efficient_config(32, 2)
            (run / profile.runner.IDENTITY_FILE).write_text(json.dumps({"requested_config": asdict(config)}))
            with patch.object(profile.runner, "run_colab_training", return_value={"stub": True}) as train:
                profile.train(run / "private/source.csv", run, Path(directory) / "absent-calibration.json")
            self.assertEqual(train.call_args.kwargs["config"], config)
            self.assertFalse(train.call_args.kwargs["evaluate"])
            self.assertFalse(train.call_args.kwargs["verification_private"])
            self.assertTrue(train.call_args.kwargs["resume"])

    def test_existing_unidentified_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            note = run / "keep.txt"
            note.write_text("existing work")
            with self.assertRaisesRegex(FileExistsError, "choose a NEW"):
                profile.train(run / "source.csv", run, run / "absent-calibration.json")
            self.assertEqual(note.read_text(), "existing work")


if __name__ == "__main__":
    unittest.main()
