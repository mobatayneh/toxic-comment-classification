"""Real, tiny CPU tests; opt in explicitly, with no model or dataset download."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from toxic_comments.artifacts import TEST_MARKER, load_splits, load_thresholds, read_json, verify_frozen_run
from toxic_comments.data import LABELS, create_synthetic_fixture, load_dataset, split_dataset
from toxic_comments.metrics import select_thresholds


@unittest.skipUnless(os.environ.get("TOXIC_RUN_NEURAL_TESTS") == "1", "Optional neural tests: set TOXIC_RUN_NEURAL_TESTS=1")
class NeuralInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from toxic_comments import neural

        cls.neural = neural
        # An explicitly selected job must fail when its optional imports fail.
        cls.dependencies = neural._dependencies()
        cls.torch = cls.dependencies.torch
        cls.device = neural._set_determinism(cls.dependencies, neural.TrainConfig(cpu_threads=1, amp=False))

    def test_unequal_microbatches_and_final_partial_group_match_sample_mean_gradients(self):
        torch, neural = self.torch, self.neural

        class ScalarModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.1))
                self.config = SimpleNamespace(vocab_size=16)

            def forward(self, input_ids, attention_mask):
                logits = self.weight * input_ids[:, :1].float()
                return SimpleNamespace(logits=logits.expand(-1, len(LABELS)))

        gradients = []

        class RecordingSGD(torch.optim.SGD):
            def __init__(self, params, lr, weight_decay):
                super().__init__(params, lr=lr, weight_decay=weight_decay)

            def step(self, closure=None):
                gradients.append(float(self.param_groups[0]["params"][0].grad))
                return super().step(closure)

        scheduler = SimpleNamespace(steps=0)

        def make_scheduler(optimizer, *, num_warmup_steps, num_training_steps):
            self.assertEqual(num_training_steps, 2)
            self.assertEqual(num_warmup_steps, 0)
            scheduler.get_last_lr = lambda: [optimizer.param_groups[0]["lr"]]

            def step():
                scheduler.steps += 1

            scheduler.step = step
            return scheduler

        def batch(features, labels):
            return {
                "input_ids": torch.tensor(features, dtype=torch.long)[:, None],
                "attention_mask": torch.ones((len(features), 1), dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.float32)[:, None].expand(-1, len(LABELS)),
            }

        # First update averages three rows across batches of 2 and 1. The final
        # update has one microbatch rather than the configured two.
        loader = [batch([1, 3], [0, 1]), batch([7], [0]), batch([5], [1])]
        model = ScalarModel()
        config = neural.TrainConfig(epochs=1, accumulation_steps=2, learning_rate=0.05,
                                   weight_decay=0, warmup_fraction=0, max_gradient_norm=1000,
                                   cpu_threads=1, amp=False)
        dependencies = SimpleNamespace(**{**vars(self.dependencies), "get_cosine_schedule_with_warmup": make_scheduler})
        with tempfile.TemporaryDirectory() as directory, patch.object(torch.optim, "AdamW", RecordingSGD):
            history, best = neural._train(model, loader, loader, torch.nn.MSELoss(), config,
                                          dependencies, self.device, Path(directory) / "checkpoint.pt", quiet=True)
        first_gradient = sum(2 * (0.1 * x - y) * x for x, y in [(1, 0), (3, 1), (7, 0)]) / 3
        weight_after_first = 0.1 - config.learning_rate * first_gradient
        final_gradient = 2 * (weight_after_first * 5 - 1) * 5
        np.testing.assert_allclose(gradients, [first_gradient, final_gradient], rtol=1e-5, atol=1e-6)
        self.assertAlmostEqual(float(model.weight.detach()), weight_after_first - config.learning_rate * final_gradient, places=6)
        self.assertEqual(scheduler.steps, 2)
        self.assertEqual(history[0]["optimizer_steps"], 2)
        self.assertEqual(best["optimizer_steps"], 2)

    def test_nonfinal_best_checkpoint_is_restored_before_export_and_reload(self):
        torch, neural = self.torch, self.neural
        tokenizer = neural._synthetic_tokenizer(np.array(["Synthetic alpha", "Synthetic beta"]), self.dependencies, max_length=16)
        model = neural._new_model(tokenizer, self.dependencies, smoke=True)
        encoded = tokenizer(["Synthetic alpha", "Synthetic beta"], padding="max_length", max_length=16, return_tensors="pt")
        batch = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"],
                 "labels": torch.tensor([[0] * len(LABELS), [1] * len(LABELS)], dtype=torch.float32)}
        config = neural.TrainConfig(epochs=2, patience=2, accumulation_steps=1, learning_rate=1e-3,
                                   max_length=16, cpu_threads=1, amp=False)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "best.pt"
            # Force the first validation checkpoint to win while both epochs
            # perform real backpropagation and parameter updates.
            with patch.object(neural, "_predict", side_effect=[(0.2, None, None), (0.8, None, None)]):
                history, best = neural._train(model, [batch], [batch], torch.nn.BCEWithLogitsLoss(),
                                              config, self.dependencies, self.device, checkpoint_path, quiet=True)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual(best["epoch"], 1)
            self.assertEqual(history[-1]["epoch"], 2)
            self.assertTrue(any(not torch.equal(value, checkpoint["model_state_dict"][name]) for name, value in model.state_dict().items()))
            exported, exported_tokenizer, selected = neural._export_best(model, tokenizer, checkpoint_path,
                                                                         Path(directory) / "model", self.dependencies, self.device)
            self.assertEqual(selected["epoch"], 1)
            for name, value in exported.state_dict().items():
                self.assertTrue(torch.equal(value.cpu(), checkpoint["model_state_dict"][name]), name)
            self.assertEqual(exported_tokenizer.get_vocab(), tokenizer.get_vocab())

    def test_tiny_development_export_thresholds_and_single_test_are_consistent(self):
        torch, neural = self.torch, self.neural
        config = neural.TrainConfig(epochs=2, patience=2, batch_size=8, accumulation_steps=3,
                                   learning_rate=1e-3, max_length=32, cpu_threads=1, amp=False,
                                   local_files_only=True)
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = create_synthetic_fixture(parent / "synthetic.csv", rows=120, seed=config.seed)
            dataset = load_dataset(source)
            expected_split = split_dataset(dataset, seed=config.seed)
            validation_texts = dataset.texts[expected_split.validation].tolist()
            predict = neural._predict
            observed = []

            def validation_inference_only(model, loader, *args, **kwargs):
                observed.append(loader.dataset.texts.tolist())
                self.assertEqual(observed[-1], validation_texts)
                return predict(model, loader, *args, **kwargs)

            run = parent / "neural"
            with patch.object(neural, "_predict", side_effect=validation_inference_only):
                developed = neural.run_development(source, run, config=config, quiet=True, _smoke=True)
            self.assertTrue(developed["checks"]["test_unscored_during_development"])
            self.assertGreaterEqual(len(observed), 2)
            self.assertFalse((run / TEST_MARKER).exists())
            self.assertFalse((run / "private" / "test_predictions.npz").exists())
            manifest = verify_frozen_run(run)
            model = self.dependencies.RobertaForSequenceClassification.from_pretrained(run / manifest["model_path"],
                                                                                      local_files_only=True, use_safetensors=True,
                                                                                      attn_implementation="eager").eval()
            tokenizer = self.dependencies.AutoTokenizer.from_pretrained(run / manifest["model_path"], local_files_only=True)
            checkpoint = torch.load(run / "private" / "best_checkpoint.pt", map_location="cpu", weights_only=True)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value.cpu(), checkpoint["model_state_dict"][name]), name)
            self.assertEqual(model.config.id2label, dict(enumerate(LABELS)))
            self.assertEqual(model.config.label2id, {label: index for index, label in enumerate(LABELS)})
            splits = load_splits(run)
            loader = neural._loader(dataset, splits.validation, tokenizer, config, self.dependencies, shuffle=False)
            weights = neural.validate_training_targets(dataset.targets[splits.train])
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights))
            loss, targets, probabilities = predict(model, loader, self.dependencies, self.device, criterion=criterion)
            history = read_json(run / "training_history.json")
            best = min(history["epochs"], key=lambda record: record["validation_loss"])
            self.assertEqual(checkpoint["epoch"], best["epoch"])
            self.assertAlmostEqual(loss, best["validation_loss"], places=6)
            self.assertAlmostEqual(loss, history["fresh_validation_loss"], places=6)
            with np.load(run / "private" / "validation_predictions.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["row_indices"], splits.validation)
                np.testing.assert_array_equal(saved["targets"], targets)
                np.testing.assert_allclose(saved["probabilities"], probabilities, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(load_thresholds(run), select_thresholds(targets, probabilities))
            thresholds_before = (run / "thresholds.json").read_bytes()
            public = neural.evaluate_run(run)
            self.assertTrue(public["synthetic"])
            self.assertTrue(public["engineering_smoke"])
            self.assertEqual(read_json(run / "public_test_results.json"), public)
            self.assertNotIn("Synthetic fixture comment", json.dumps(public))
            self.assertEqual((run / "thresholds.json").read_bytes(), thresholds_before)
            with patch.object(neural, "_predict", side_effect=AssertionError("Repeated test inference")):
                with self.assertRaises(FileExistsError):
                    neural.evaluate_run(run)

    def test_train_positive_weights_and_label_mapping_are_explicit(self):
        neural = self.neural
        targets = np.tile(np.array([0, 0, 1, 1])[:, None], (1, len(LABELS)))
        targets[:, 0] = [0, 1, 1, 1]
        np.testing.assert_allclose(neural.validate_training_targets(targets), [1 / 3, 1, 1, 1, 1, 1])
        for value in (0, 1):
            invalid = targets.copy()
            invalid[:, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive and negative"):
                neural.validate_training_targets(invalid)
        wrong_labels = tuple(reversed(LABELS))
        wrong_model = SimpleNamespace(config=SimpleNamespace(num_labels=len(LABELS), id2label=dict(enumerate(wrong_labels)),
                                                             label2id={label: index for index, label in enumerate(wrong_labels)}))
        with self.assertRaisesRegex(ValueError, "label order"):
            neural._check_label_mapping(wrong_model)

    def test_source_code_change_during_training_prevents_freeze(self):
        from toxic_comments import artifacts

        neural = self.neural
        original_hashes = artifacts._code_hashes()
        current_hashes = dict(original_hashes)
        original_train = neural._train

        def train_then_change_source_identity(*args, **kwargs):
            result = original_train(*args, **kwargs)
            # Simulate a disk edit after real training without modifying shared
            # source files or the already imported implementation.
            current_hashes["src/toxic_comments/neural.py"] = "changed-source-identity"
            return result

        config = neural.TrainConfig(epochs=1, patience=1, batch_size=8, accumulation_steps=3,
                                   learning_rate=1e-3, max_length=16, cpu_threads=1, amp=False,
                                   local_files_only=True)
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = create_synthetic_fixture(parent / "synthetic.csv", rows=60)
            run = parent / "neural"
            with patch.object(artifacts, "_code_hashes", side_effect=lambda: dict(current_hashes)), \
                 patch.object(neural, "_train", side_effect=train_then_change_source_identity):
                with self.assertRaisesRegex(ValueError, "Source code changed during development"):
                    neural.run_development(source, run, config=config, quiet=True, _smoke=True)
            self.assertTrue((run / "private" / "best_checkpoint.pt").is_file())
            self.assertTrue((run / "model" / "model.safetensors").is_file())
            self.assertFalse((run / "manifest.json").exists())
            self.assertFalse((run / TEST_MARKER).exists())


if __name__ == "__main__":
    unittest.main()
