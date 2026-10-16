"""Bounded preprocessing, reduced padding and durable checkpoint regressions."""

from __future__ import annotations

import contextlib
import csv
import io
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import colab_train
from toxic_comments import neural
from toxic_comments.artifacts import read_json, verify_frozen_run
from toxic_comments.data import LABELS, create_synthetic_fixture, load_dataset, split_dataset


class EfficientConfigurationAndStorage(unittest.TestCase):
    def test_loaded_texts_retain_python_strings_without_longest_row_allocation(self):
        comments = ["123", "NA", "café 😀", "one very long comment " * 5000] + ["short"] * 64
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "comments.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["comment_text", *LABELS])
                writer.writeheader()
                for comment in comments:
                    writer.writerow({"comment_text": comment, **dict.fromkeys(LABELS, 0)})
            dataset = load_dataset(source)
        texts = dataset.texts
        self.assertEqual(texts.dtype, np.dtype(object))
        self.assertEqual(texts.tolist(), comments)
        self.assertTrue(all(type(value) is str for value in texts))
        self.assertEqual(texts.nbytes, len(comments) * np.dtype(object).itemsize)
        self.assertTrue(np.shares_memory(texts, dataset.frame["comment_text"].to_numpy()))
        fixed_width_bytes = len(comments) * max(map(len, comments)) * np.dtype("U1").itemsize
        self.assertGreater(fixed_width_bytes, texts.nbytes * 1000)

    def test_new_efficiency_settings_validate_strictly_and_keep_legacy_defaults(self):
        self.assertFalse(neural.TrainConfig().cache_tokenization)
        self.assertFalse(neural.TrainConfig().dynamic_padding)
        self.assertEqual(colab_train.ColabConfig().checkpoint_every_seconds, 0.0)
        for name in ("cache_tokenization", "dynamic_padding"):
            for invalid in (0, 1, "true", None, np.bool_(True)):
                with self.subTest(name=name, invalid=invalid), self.assertRaisesRegex(ValueError, name):
                    neural.TrainConfig(**{name: invalid})
        for invalid in (-1, float("nan"), float("inf"), True, "300"):
            with self.subTest(seconds=invalid), self.assertRaisesRegex(ValueError, "checkpoint_every_seconds"):
                colab_train.ColabConfig(checkpoint_every_seconds=invalid)
        self.assertEqual(colab_train.ColabConfig(checkpoint_every_seconds=300).checkpoint_every_seconds, 300)

    def test_checkpoint_serializes_locally_and_retains_previous_bytes_on_copy_or_flush_failure(self):
        locations = []

        def serialize(payload, path):
            locations.append(Path(path))
            Path(path).write_bytes(payload["bytes"])

        fake_torch = SimpleNamespace(save=serialize)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "latest.pt"
            destination.write_bytes(b"previous valid checkpoint")
            old_bytes = destination.read_bytes()

            def interrupted_copy(source, target, **kwargs):
                self.assertEqual(kwargs["length"], 8 * 1024 * 1024)
                target.write(b"incomplete")
                raise OSError("copy interrupted")

            for target, failure in (("copyfileobj", interrupted_copy), ("fsync", OSError("flush interrupted"))):
                owner = colab_train.shutil if target == "copyfileobj" else colab_train.os
                with patch.object(owner, target, side_effect=failure), self.assertRaises(OSError):
                    colab_train.atomic_torch_save(fake_torch, {"bytes": b"replacement"}, destination)
                self.assertEqual(destination.read_bytes(), old_bytes)
                self.assertEqual(list(Path(directory).iterdir()), [destination])
            colab_train.atomic_torch_save(fake_torch, {"bytes": b"complete replacement"}, destination)
            self.assertEqual(destination.read_bytes(), b"complete replacement")
            self.assertTrue(all(path.parent != destination.parent for path in locations))
            self.assertTrue(all(not path.exists() for path in locations))

    def test_token_id_bounds_are_checked_before_any_device_transfer(self):
        operations = []

        class Tensor:
            def __init__(self, low=0, high=3):
                self.low, self.high = low, high

            def min(self):
                operations.append("min_cpu")
                return SimpleNamespace(item=lambda: self.low)

            def max(self):
                operations.append("max_cpu")
                return SimpleNamespace(item=lambda: self.high)

            def to(self, *args, **kwargs):
                operations.append("transfer")
                return self

        model = SimpleNamespace(config=SimpleNamespace(vocab_size=8))
        device = SimpleNamespace(type="cuda")
        with self.assertRaisesRegex(ValueError, "outside the model vocabulary"):
            neural._device_batch({"input_ids": Tensor(high=8), "labels": Tensor()}, device, model)
        self.assertEqual(operations, ["min_cpu", "max_cpu"])
        operations.clear()
        neural._device_batch({"input_ids": Tensor(), "labels": Tensor()}, device, model)
        self.assertEqual(operations, ["min_cpu", "max_cpu", "transfer", "transfer"])


@unittest.skipUnless(os.environ.get("TOXIC_RUN_NEURAL_TESTS") == "1", "Optional CPU cache/training checks: set TOXIC_RUN_NEURAL_TESTS=1")
class CompactTokenCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dependencies = neural._dependencies()
        cls.torch = cls.dependencies.torch
        cls.device = neural._set_determinism(cls.dependencies, neural.TrainConfig(cpu_threads=1, amp=False))

    def test_bounded_cache_matches_original_inputs_and_bytes_and_is_not_retokenized(self):
        torch = self.torch
        texts = np.array(["hello café " + "alpha " * (index % 25) for index in range(515)], dtype=object)
        targets = np.array([[index % 2] * len(LABELS) for index in range(len(texts))], dtype=np.uint8)
        dataset = SimpleNamespace(texts=texts, targets=targets)
        tokenizer = neural._synthetic_tokenizer(texts, self.dependencies, max_length=16)
        calls = []

        class RecordingTokenizer:
            def __getattr__(self, name):
                return getattr(tokenizer, name)

            def __call__(self, inputs, **kwargs):
                calls.append(len(inputs))
                return tokenizer(inputs, **kwargs)

        config = neural.TrainConfig(batch_size=17, max_length=16, amp=False, cache_tokenization=True)
        loader = neural._loader(dataset, np.arange(len(texts)), RecordingTokenizer(), config, self.dependencies, shuffle=False)
        self.assertEqual(calls, [256, 256, 3])
        cache = loader.dataset
        self.assertEqual(cache.input_ids.dtype, np.dtype(np.int32))
        self.assertEqual(cache.attention_mask.dtype, np.dtype(np.uint8))
        self.assertEqual(cache.input_ids.nbytes + cache.attention_mask.nbytes, len(texts) * 16 * 5)
        original = neural._TextDataset(texts, targets, tokenizer, 16)
        old_rows = [original[index] for index in range(len(texts))]
        old = {name: torch.stack([row[name] for row in old_rows]) for name in old_rows[0]}
        for _ in range(2):
            rows = list(loader)
            for name in old:
                actual = torch.cat([batch[name] for batch in rows])
                self.assertEqual(actual.dtype, old[name].dtype)
                self.assertEqual(actual.numpy().tobytes(), old[name].numpy().tobytes())
        self.assertEqual(calls, [256, 256, 3])

    def test_dynamic_padding_only_crops_right_padding_and_keeps_labels_and_logits(self):
        torch = self.torch
        texts = np.array(["alpha", "alpha beta gamma", "alpha " * 20, "beta gamma"], dtype=object)
        targets = np.arange(len(texts) * len(LABELS), dtype=np.uint8).reshape(len(texts), -1) % 2
        dataset = SimpleNamespace(texts=texts, targets=targets)
        tokenizer = neural._synthetic_tokenizer(texts, self.dependencies, max_length=14)
        fixed_config = neural.TrainConfig(max_length=14, batch_size=2, amp=False, cache_tokenization=True)
        dynamic_config = replace(fixed_config, dynamic_padding=True)
        fixed = list(neural._loader(dataset, np.arange(4), tokenizer, fixed_config, self.dependencies, shuffle=False))
        cropped = list(neural._loader(dataset, np.arange(4), tokenizer, dynamic_config, self.dependencies, shuffle=False))
        self.assertEqual([batch["input_ids"].shape[1] for batch in cropped], [8, 14])
        model = neural._new_model(tokenizer, self.dependencies, smoke=True).eval()
        for padded, batch in zip(fixed, cropped):
            width = batch["input_ids"].shape[1]
            for name in ("input_ids", "attention_mask"):
                self.assertTrue(torch.equal(batch[name], padded[name][:, :width]))
            self.assertFalse(bool(padded["attention_mask"][:, width:].any()))
            self.assertTrue(torch.equal(batch["labels"], padded["labels"]))
            with torch.inference_mode():
                expected = model(input_ids=padded["input_ids"], attention_mask=padded["attention_mask"]).logits
                actual = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
        tokenizer.padding_side = "left"
        with self.assertRaisesRegex(ValueError, "right-padding"):
            neural._loader(dataset, np.arange(4), tokenizer, dynamic_config, self.dependencies, shuffle=False)

    def test_nonfinite_group_loss_stops_before_optimizer_or_scheduler_updates(self):
        torch = self.torch

        class ScalarModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.1))
                self.config = SimpleNamespace(vocab_size=8)

            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(logits=(self.weight * input_ids.float()).expand(-1, len(LABELS)))

        model = ScalarModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        scheduler = SimpleNamespace(step=lambda: self.fail("scheduler must not advance"))
        updates = {"value": 0}
        hook = optimizer.register_step_post_hook(lambda *args: self.fail("optimizer must not update"))
        batch = {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]]),
                 "labels": torch.zeros((1, len(LABELS)))}
        calls = {"value": 0}

        def criterion(logits, labels):
            calls["value"] += 1
            return logits.square().mean() * (float("nan") if calls["value"] == 2 else 1)

        try:
            with self.assertRaisesRegex(FloatingPointError, "Nonfinite training loss"):
                colab_train._optimizer_group(model, [batch, batch], criterion, optimizer, scaler, scheduler,
                                             neural.TrainConfig(amp=False), self.dependencies, self.device, updates)
        finally:
            hook.remove()
        self.assertEqual(calls["value"], 2)
        self.assertEqual(updates["value"], 0)
        self.assertEqual(float(model.weight.detach()), float(torch.tensor(0.1)))


@unittest.skipUnless(os.environ.get("TOXIC_RUN_NEURAL_TESTS") == "1", "Optional CPU cached-resume checks: set TOXIC_RUN_NEURAL_TESTS=1")
class EfficientColabResume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dependencies = neural._dependencies()
        cls.torch = cls.dependencies.torch

    def test_cached_training_reuses_partitions_and_exactly_resumes_with_dynamic_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = create_synthetic_fixture(parent / "source.csv", rows=240)
            config = colab_train.ColabConfig(training=neural.TrainConfig(
                epochs=2, patience=2, batch_size=13, accumulation_steps=5, learning_rate=1e-3,
                max_length=16, cpu_threads=1, device="cpu", amp=False, local_files_only=True,
                cache_tokenization=True, dynamic_padding=True,
            ), checkpoint_every_updates=1000, require_full_dataset=False)
            epoch_datasets, validation_loaders = [], []
            epoch_loader, predict = colab_train._epoch_loader, neural._predict

            def observe_epoch(*args, **kwargs):
                loader = epoch_loader(*args, **kwargs)
                epoch_datasets.append(loader.dataset)
                return loader

            def observe_validation(model, loader, *args, **kwargs):
                validation_loaders.append(loader)
                return predict(model, loader, *args, **kwargs)

            with patch.object(colab_train, "_epoch_loader", side_effect=observe_epoch), \
                    patch.object(neural, "_predict", side_effect=observe_validation):
                continuous = colab_train.run_colab_training(source, parent / "continuous", config=config, quiet=True, _smoke=True)
            self.assertEqual(len(epoch_datasets), 2)
            self.assertIs(epoch_datasets[0], epoch_datasets[1])
            self.assertIsInstance(epoch_datasets[0], neural._CachedTextDataset)
            self.assertEqual(len(validation_loaders), 3)
            self.assertTrue(all(loader is validation_loaders[0] for loader in validation_loaders))
            self.assertIsInstance(validation_loaders[0].dataset, neural._CachedTextDataset)
            splits = split_dataset(load_dataset(source))
            self.assertEqual(len(epoch_datasets[0]), len(splits.train))
            self.assertEqual(len(validation_loaders[0].dataset), len(splits.validation))

            interrupted = colab_train.run_colab_training(source, parent / "resumed", config=config, quiet=True, _smoke=True, _stop_after_steps=2)
            self.assertEqual(interrupted["stage"], "interrupted")
            resumed = colab_train.run_colab_training(source, parent / "resumed", config=config, quiet=True, _smoke=True)
            self.assertEqual(continuous["history"], resumed["history"])
            first = self.torch.load(parent / "continuous" / colab_train.LATEST_CHECKPOINT, map_location="cpu", weights_only=True)
            second = self.torch.load(parent / "resumed" / colab_train.LATEST_CHECKPOINT, map_location="cpu", weights_only=True)
            for name, tensor in first["model_state_dict"].items():
                self.assertTrue(self.torch.equal(tensor, second["model_state_dict"][name]), name)
            with np.load(parent / "continuous/private/validation_predictions.npz", allow_pickle=False) as one, \
                    np.load(parent / "resumed/private/validation_predictions.npz", allow_pickle=False) as two:
                for name in one.files:
                    np.testing.assert_array_equal(one[name], two[name])
            progress = read_json(parent / "resumed" / colab_train.TRAINING_PROGRESS)
            self.assertTrue(progress["training_complete"])
            self.assertEqual(progress["completed_epochs"], 2)
            self.assertEqual(progress["global_optimizer_steps"], resumed["global_optimizer_steps"])
            self.assertFalse(progress["test_evaluated"])
            self.assertFalse((parent / "resumed/public_test_results.json").exists())
            verify_frozen_run(parent / "resumed")
            frozen_config = read_json(parent / "resumed/config.json")
            self.assertTrue(frozen_config["training"]["cache_tokenization"])
            self.assertTrue(frozen_config["training"]["dynamic_padding"])
            self.assertEqual(frozen_config["colab"]["checkpoint_every_seconds"], 0.0)

    def test_wall_time_checkpoint_saves_only_completed_groups_and_reports_after_durable_save(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = create_synthetic_fixture(parent / "source.csv", rows=240)
            config = colab_train.ColabConfig(training=neural.TrainConfig(
                epochs=1, patience=1, batch_size=13, accumulation_steps=5, learning_rate=1e-3,
                max_length=16, cpu_threads=1, device="cpu", amp=False, local_files_only=True,
                cache_tokenization=True, dynamic_padding=True,
            ), checkpoint_every_updates=1000, checkpoint_every_seconds=300.0, require_full_dataset=False)
            clock = {"now": 0.0}
            saves, progress_snapshots = [], []
            optimizer_group, save, atomic_json = colab_train._optimizer_group, colab_train.atomic_torch_save, colab_train._atomic_json
            run = parent / "run"

            def timed_group(*args, **kwargs):
                result = optimizer_group(*args, **kwargs)
                clock["now"] += 300.0
                return result

            def observe_save(torch, payload, path):
                save(torch, payload, path)
                if Path(path).name == "latest_checkpoint.pt":
                    saves.append((payload["global_optimizer_steps"], payload["microbatch_offset"], payload["training_complete"]))

            def observe_progress(path, value):
                if Path(path).name == "training_progress.json":
                    checkpoint = self.torch.load(run / colab_train.LATEST_CHECKPOINT, map_location="cpu", weights_only=True)
                    self.assertEqual(value["global_optimizer_steps"], checkpoint["global_optimizer_steps"])
                    self.assertEqual(value["microbatch_offset"], checkpoint["microbatch_offset"])
                    progress_snapshots.append(value)
                atomic_json(path, value)

            original_print = print

            def observe_print(*args, **kwargs):
                if args and str(args[0]).startswith("epoch=1 train_loss="):
                    self.assertTrue(saves[-1][2], "Epoch completion was printed before the latest checkpoint was saved")
                    self.assertTrue(progress_snapshots[-1]["training_complete"])
                original_print(*args, **kwargs)

            output = io.StringIO()
            with patch.object(colab_train, "_optimizer_group", side_effect=timed_group), \
                    patch.object(colab_train, "atomic_torch_save", side_effect=observe_save), \
                    patch.object(colab_train, "_atomic_json", side_effect=observe_progress), \
                    patch.object(colab_train.time, "monotonic", side_effect=lambda: clock["now"]), \
                    patch("builtins.print", side_effect=observe_print), \
                    contextlib.redirect_stdout(output):
                result = colab_train.run_colab_training(source, run, config=config, quiet=False, _smoke=True)
            self.assertEqual(result["global_optimizer_steps"], 3)
            self.assertEqual(saves, [(0, 0, False), (1, 5, False), (2, 10, False), (3, 13, False), (3, 0, True)])
            self.assertEqual(len(progress_snapshots), len(saves))
            self.assertIn("epoch=1 train_loss=", output.getvalue())
            self.assertIn("checkpoint_saved=True", output.getvalue().splitlines()[-1])
            allowed = {"schema_version", "checkpoint_saved_utc", "epoch", "epochs_planned", "completed_epochs",
                       "microbatch_offset", "microbatches_per_epoch", "samples_in_epoch", "training_rows",
                       "global_optimizer_steps", "planned_optimizer_steps", "training_loss", "last_completed_epoch",
                       "best_checkpoint", "training_complete", "test_evaluated"}
            self.assertEqual(set(progress_snapshots[-1]), allowed)
            self.assertIsNone(progress_snapshots[0]["best_checkpoint"])
            self.assertTrue(progress_snapshots[-1]["training_complete"])


if __name__ == "__main__":
    unittest.main()
