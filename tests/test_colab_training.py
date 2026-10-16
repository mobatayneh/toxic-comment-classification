"""Independent offline checks for the portable, resumable Colab workflow.

Notebook/source checks run in the standard suite. Actual tiny CPU training is
explicitly opt-in and never downloads a tokenizer, model or dataset.
"""

from __future__ import annotations

import ast
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from toxic_comments.artifacts import TEST_MARKER, load_splits, load_thresholds, read_json, verify_frozen_run
from toxic_comments.data import LABELS, create_synthetic_fixture, file_sha256, load_dataset, split_dataset
from toxic_comments.metrics import select_thresholds

ROOT = Path(__file__).resolve().parents[1]
COLAB_NOTEBOOK = ROOT / "notebooks/04_colab_efficient_training.ipynb"
COLAB_PREFIX = "/content/toxic-comment-efficient-v1/"
EMBEDDED_PATHS = (
    "src/toxic_comments/__init__.py",
    "src/toxic_comments/artifacts.py",
    "src/toxic_comments/baseline.py",
    "src/toxic_comments/data.py",
    "src/toxic_comments/metrics.py",
    "src/toxic_comments/neural.py",
    "scripts/colab_train.py",
    "scripts/colab_profile.py",
    "requirements-colab.txt",
)


def _notebook_source_cells(notebook):
    sources = {}
    cells = {}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        match = re.fullmatch(r"%%writefile ([^\n]+)\n([\s\S]*)", cell.source)
        if match is None:
            continue
        destination, source = match.groups()
        if not destination.startswith(COLAB_PREFIX):
            raise AssertionError(f"Unexpected embedded destination: {destination}")
        relative = destination[len(COLAB_PREFIX):]
        if relative in sources:
            raise AssertionError(f"Duplicate embedded source: {relative}")
        sources[relative] = source
        cells[relative] = cell.source
    return sources, cells


def _literal_notebook_assignments(notebook):
    values = {}
    for cell in notebook.cells:
        if cell.cell_type != "code" or cell.source.startswith("%"):
            continue
        try:
            tree = ast.parse(cell.source)
        except SyntaxError:
            continue
        for assignment in ast.walk(tree):
            if isinstance(assignment, ast.Assign):
                for target in assignment.targets:
                    if isinstance(target, ast.Name) and target.id in {"EMBEDDED_FILES", "EMBEDDED_SHA256"}:
                        if target.id in values:
                            raise AssertionError(f"Duplicate notebook manifest: {target.id}")
                        values[target.id] = ast.literal_eval(assignment.value)
    return values


def _tree_hashes(directory: Path):
    return {path.relative_to(directory).as_posix(): file_sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


class ColabNotebookInvariants(unittest.TestCase):
    def test_retired_03_is_a_generated_notice_without_executable_cells(self):
        import nbformat
        from scripts import build_colab_notebook

        path = build_colab_notebook.OUTPUT
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        self.assertEqual(path.read_text(encoding="utf-8"), nbformat.writes(build_colab_notebook.build_notebook()) + "\n")
        self.assertTrue(notebook.cells)
        self.assertTrue(all(cell.cell_type == "markdown" for cell in notebook.cells))
        self.assertEqual(notebook.metadata.training_workflow.status, "retired")
        self.assertEqual(notebook.metadata.training_workflow.replacement, COLAB_NOTEBOOK.name)
        text = "\n".join(cell.source for cell in notebook.cells)
        self.assertIn(f"]({COLAB_NOTEBOOK.name})", text)
        self.assertIn("original notebook version", text)
        self.assertIn("Git history", text)
        self.assertIn(build_colab_notebook.LEGACY_REVISION, text)
        self.assertIn("cannot resume", text)

    def test_full_notebook_is_valid_clear_and_embeds_exact_repository_sources(self):
        import nbformat
        from IPython.core.inputtransformer2 import TransformerManager

        notebook = nbformat.read(COLAB_NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
        target_rows = re.findall(r"^\|\s*Targets, in(?: runtime)? order\s*\|\s*(.*?)\s*\|$", markdown, re.MULTILINE)
        self.assertEqual(len(target_rows), 1)
        self.assertEqual(tuple(re.findall(r"`([^`]+)`", target_rows[0])), tuple(LABELS))
        sources, _ = _notebook_source_cells(notebook)
        self.assertEqual(set(sources), set(EMBEDDED_PATHS))
        manifests = _literal_notebook_assignments(notebook)
        self.assertEqual(set(manifests["EMBEDDED_FILES"]), set(EMBEDDED_PATHS))
        expected_hashes = {}
        for relative in EMBEDDED_PATHS:
            with self.subTest(source=relative):
                expected = (ROOT / relative).read_bytes()
                self.assertEqual(sources[relative].encode("utf-8"), expected)
                expected_hashes[relative] = hashlib.sha256(expected).hexdigest()
                if relative.endswith(".py"):
                    compile(sources[relative], relative, "exec")
        self.assertEqual(manifests["EMBEDDED_SHA256"], expected_hashes)
        transformer = TransformerManager()
        for cell in notebook.cells:
            if cell.cell_type == "code":
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])
                compile(transformer.transform_cell(cell.source), "colab-notebook-cell", "exec")

    def test_visible_writefile_cells_materialize_importable_files_without_colab_services(self):
        import nbformat
        from IPython.core.interactiveshell import InteractiveShell
        from traitlets.config import Config

        notebook = nbformat.read(COLAB_NOTEBOOK, as_version=4)
        sources, cells = _notebook_source_cells(notebook)
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            project = parent / "project"
            for relative in EMBEDDED_PATHS:
                (project / relative).parent.mkdir(parents=True, exist_ok=True)
            shell = InteractiveShell(config=Config({"HistoryManager": {"enabled": False}}),
                                     ipython_dir=str(parent / "ipython"))
            for relative in EMBEDDED_PATHS:
                code = cells[relative].replace(COLAB_PREFIX, str(project) + "/", 1)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = shell.run_cell(code, store_history=False)
                if result.error_before_exec is not None:
                    raise result.error_before_exec
                if result.error_in_exec is not None:
                    raise result.error_in_exec
                self.assertEqual((project / relative).read_text(encoding="utf-8"), sources[relative])
            # The embedded project must import outside this checkout. No cell
            # installs packages, contacts Drive, uploads data, or starts training.
            environment = {**os.environ, "PYTHONPATH": str(project / "src"), "HF_HUB_OFFLINE": "1",
                           "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
            imports_without_neural = """\
import importlib.abc
import importlib.metadata
import importlib.util
import sys
from pathlib import Path
optional = {'torch', 'transformers', 'tokenizers', 'safetensors', 'huggingface_hub'}
assert optional.isdisjoint(sys.modules)
blocked_imports = []
blocked_metadata = []
optional_distributions = {name.replace('_', '-') for name in optional}
real_distribution = importlib.metadata.distribution
def missing_optional_distribution(name):
    if name.lower().replace('_', '-') in optional_distributions:
        blocked_metadata.append(name)
        raise importlib.metadata.PackageNotFoundError(name)
    return real_distribution(name)
importlib.metadata.distribution = missing_optional_distribution
class MissingOptionalNeuralDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in optional:
            blocked_imports.append(fullname)
            raise ModuleNotFoundError('Optional neural dependency unavailable: ' + fullname, name=fullname)
sys.meta_path.insert(0, MissingOptionalNeuralDependencies())
# The surrounding test environment may have the neural extras installed. Prove
# both discovery and actual imports are unavailable inside this child process.
for name in sorted(optional):
    for probe in (importlib.util.find_spec, importlib.import_module):
        try:
            probe(name)
        except ModuleNotFoundError as error:
            assert error.name == name
        else:
            raise AssertionError('Optional dependency unexpectedly accessible: ' + name)
    try:
        importlib.metadata.version(name.replace('_', '-'))
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError('Optional distribution unexpectedly accessible: ' + name)
blocked_imports.clear()
blocked_metadata.clear()
import toxic_comments
import toxic_comments.artifacts, toxic_comments.baseline, toxic_comments.data
import toxic_comments.metrics, toxic_comments.neural, scripts.colab_train, scripts.colab_profile
assert len(toxic_comments.data.LABELS) == 6
assert not blocked_imports, blocked_imports
assert not blocked_metadata, blocked_metadata
assert optional.isdisjoint(sys.modules)
project = Path.cwd()
assert Path(scripts.colab_train.__file__).is_relative_to(project)
assert Path(toxic_comments.__file__).is_relative_to(project)
print('Portable core and Colab imports passed with all five optional neural dependencies unavailable.')
"""
            result = subprocess.run([sys.executable, "-c", imports_without_neural],
                                    cwd=project, env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("all five optional neural dependencies unavailable", result.stdout)


@unittest.skipUnless(os.environ.get("TOXIC_RUN_NEURAL_TESTS") == "1",
                     "Optional tiny CPU Colab checks: set TOXIC_RUN_NEURAL_TESTS=1")
class ColabResumeInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import colab_train
        from toxic_comments import neural

        cls.colab, cls.neural = colab_train, neural
        cls.dependencies = neural._dependencies()
        cls.torch = cls.dependencies.torch
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.parent = Path(cls.temp.name)
        training = neural.TrainConfig(epochs=2, patience=2, batch_size=13, accumulation_steps=5,
                                     learning_rate=1e-3, warmup_fraction=0.2, max_length=16,
                                     cpu_threads=1, device="cpu", amp=False, local_files_only=True)
        cls.config = colab_train.ColabConfig(training=training, checkpoint_every_updates=1,
                                            require_full_dataset=False)
        cls.source = create_synthetic_fixture(cls.parent / "synthetic.csv", rows=240, seed=training.seed)
        with cls.source.open(encoding="utf-8", newline="") as stream:
            fixture_rows = list(csv.DictReader(stream))
        for index, row in enumerate(fixture_rows):
            row["id"] = f"synthetic-colab-id-{index:06d}"
        with cls.source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fixture_rows[0]))
            writer.writeheader()
            writer.writerows(fixture_rows)
        cls.dataset = load_dataset(cls.source)
        cls.splits = split_dataset(cls.dataset, seed=training.seed)
        cls.validation_texts = cls.dataset.texts[cls.splits.validation].tolist()
        cls.tokenizer_inputs, cls.prediction_inputs = [], []
        tokenizer = neural._synthetic_tokenizer
        predict = neural._predict
        getitem = neural._TextDataset.__getitem__

        def observe_tokenizer(texts, *args, **kwargs):
            observed = np.asarray(texts).tolist()
            cls.tokenizer_inputs.append(observed)
            if observed != cls.dataset.texts[cls.splits.train].tolist():
                raise AssertionError("Synthetic tokenizer must be constructed from training texts only")
            return tokenizer(texts, *args, **kwargs)

        def observe_validation(model, loader, *args, **kwargs):
            observed = loader.dataset.texts.tolist()
            cls.prediction_inputs.append(observed)
            if observed != cls.validation_texts:
                raise AssertionError("Development inference must use validation texts only")
            return predict(model, loader, *args, **kwargs)

        def consume_preprocessing_rng(text_dataset, index):
            # Real preprocessing can consume global RNG. Rebuilding a loader
            # and skipping completed groups must not advance continuation RNG.
            random.random()
            np.random.random()
            cls.torch.rand(())
            return getitem(text_dataset, index)

        cls.continuous, cls.resumed = cls.parent / "continuous", cls.parent / "resumed"
        cls.interrupted_snapshot = cls.parent / "interrupted_snapshot"
        with patch.object(neural, "_synthetic_tokenizer", side_effect=observe_tokenizer), \
                patch.object(neural, "_predict", side_effect=observe_validation), \
                patch.object(neural._TextDataset, "__getitem__", new=consume_preprocessing_rng):
            cls.interrupted_result = colab_train.run_colab_training(
                cls.source, cls.resumed, config=cls.config, quiet=True, _smoke=True, _stop_after_steps=2)
            cls.interrupted_state = cls.torch.load(cls.resumed / "private/latest_checkpoint.pt",
                                                 map_location="cpu", weights_only=True)
            shutil.copytree(cls.resumed, cls.interrupted_snapshot)
            cls.continuous_result = colab_train.run_colab_training(
                cls.source, cls.continuous, config=cls.config, quiet=True, _smoke=True)
            cls.resumed_result = colab_train.run_colab_training(
                cls.source, cls.resumed, config=cls.config, quiet=True, _smoke=True)

    def assert_nested_equal(self, left, right, path="state"):
        torch = self.torch
        self.assertEqual(type(left), type(right), path)
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right), path)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right, err_msg=path)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right), path)
            for name in left:
                self.assert_nested_equal(left[name], right[name], f"{path}.{name}")
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right), path)
            for index, (left_value, right_value) in enumerate(zip(left, right)):
                self.assert_nested_equal(left_value, right_value, f"{path}[{index}]")
        else:
            self.assertEqual(left, right, path)

    def test_two_update_mid_epoch_resume_matches_continuous_model_optimizer_scheduler_and_predictions(self):
        torch = self.torch
        self.assertEqual(self.interrupted_result["stage"], "interrupted")
        interrupted = self.interrupted_state
        self.assertEqual(interrupted["global_optimizer_steps"], 2)
        self.assertEqual(interrupted["epoch"], 1)
        self.assertEqual(interrupted["microbatch_offset"], 10)
        self.assertEqual(interrupted["training_sample_count"], 130)
        self.assertLess(interrupted["training_sample_count"], len(self.splits.train))
        self.assertEqual(interrupted["history"], [])
        self.assertFalse(interrupted["training_complete"])
        self.assertTrue(self.resumed_result["resumed"])
        expected_updates = math.ceil(math.ceil(len(self.splits.train) / 13) / 5) * 2
        self.assertEqual(self.resumed_result["global_optimizer_steps"], expected_updates)
        continuous = torch.load(self.continuous / "private/latest_checkpoint.pt", map_location="cpu", weights_only=True)
        resumed = torch.load(self.resumed / "private/latest_checkpoint.pt", map_location="cpu", weights_only=True)
        for field in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "rng_state",
                      "global_optimizer_steps", "history", "best", "epochs_without_improvement", "training_complete"):
            with self.subTest(field=field):
                self.assert_nested_equal(continuous[field], resumed[field], field)
        self.assertEqual(read_json(self.continuous / "training_history.json"),
                         read_json(self.resumed / "training_history.json"))
        self.assertEqual((self.continuous / "model/model.safetensors").read_bytes(),
                         (self.resumed / "model/model.safetensors").read_bytes())
        for name in ("validation_predictions.npz",):
            with np.load(self.continuous / "private" / name, allow_pickle=False) as left, \
                    np.load(self.resumed / "private" / name, allow_pickle=False) as right:
                self.assertEqual(set(left.files), set(right.files))
                for key in left.files:
                    np.testing.assert_array_equal(left[key], right[key])
        np.testing.assert_array_equal(load_thresholds(self.continuous), load_thresholds(self.resumed))

    def test_development_uses_train_only_data_and_result_bundles_exclude_private_rows(self):
        self.assertGreaterEqual(len(self.tokenizer_inputs), 2)
        self.assertGreaterEqual(len(self.prediction_inputs), 6)
        for run in (self.continuous, self.resumed):
            verify_frozen_run(run)
            self.assertFalse((run / TEST_MARKER).exists())
            self.assertFalse((run / "private/test_predictions.npz").exists())
            self.assertFalse((run / "public_test_results.json").exists())
            stored_splits = load_splits(run)
            for name in ("train", "validation", "test"):
                np.testing.assert_array_equal(getattr(stored_splits, name), getattr(self.splits, name))

        aggregate_files = {
            "config.json", "data_summary.json", "training_history.json", "validation_metrics.json",
            "thresholds.json", "run_metadata.json", "versions_hardware.json", "result_metadata.json",
            "README.txt", "plots/training_losses.png", "plots/validation_per_label.png",
        }
        private_files = {
            "validation_predictions.npz", "splits.npz", "thresholds.json", "config.json",
            "data_summary.json", "validation_metrics.json", "training_history.json",
            "PRIVATE_README.txt", "verification_manifest.json",
        }
        forbidden_fields = {
            "id", "ids", "comment_text", "texts", "comments", "raw_comments", "records",
            "row_indices", "targets", "predictions", "probabilities", "source_path", "model_path",
            "source_sha256", "code_sha256", "tokenizer_sha256", "git_revision", "run_dir",
        }

        def check_aggregate(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    self.assertNotIn(key.lower(), forbidden_fields)
                    check_aggregate(child)
            elif isinstance(value, list):
                for child in value:
                    check_aggregate(child)

        def inspect_bundle(path, *, private, evaluated):
            expected = private_files if private else aggregate_files
            expected = expected | ({"test_predictions.npz", "public_test_results.json"} if private and evaluated
                                   else {"public_test_results.json"} if evaluated else set())
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(set(archive.namelist()), expected)
                self.assertEqual(len(archive.namelist()), len(expected))
                for name in archive.namelist():
                    if name.endswith((".json", ".txt")):
                        content = archive.read(name).decode("utf-8")
                        self.assertNotIn("Synthetic fixture comment", content)
                        self.assertNotIn("synthetic-colab-id-", content)
                        self.assertNotIn(str(self.parent), content)
                        if name.endswith(".json") and not private:
                            value = json.loads(content)
                            if name == "result_metadata.json":
                                self.assertEqual(value.pop("source_sha256"), file_sha256(self.source))
                                self.assertEqual(value.pop("code_sha256"), self.colab._source_code_hashes())
                            check_aggregate(value)
                if private:
                    self.assertIn("Never publish", archive.read("PRIVATE_README.txt").decode())
                    manifest = json.loads(archive.read("verification_manifest.json"))
                    self.assertEqual(manifest["source_sha256"], file_sha256(self.source))
                    self.assertEqual(manifest["label_order"], list(LABELS))
                    for name, expected_hash in manifest["artifacts_sha256"].items():
                        self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected_hash)
                    for split in ("validation", "test") if evaluated else ("validation",):
                        with np.load(io.BytesIO(archive.read(f"{split}_predictions.npz")), allow_pickle=False) as arrays:
                            self.assertEqual(set(arrays.files), {"row_indices", "targets", "probabilities"})
                            np.testing.assert_array_equal(arrays["row_indices"], getattr(self.splits, split))
                            np.testing.assert_array_equal(arrays["targets"], self.dataset.targets[getattr(self.splits, split)])
                else:
                    metadata = json.loads(archive.read("result_metadata.json"))
                    self.assertEqual(metadata["test_evaluation_performed"], evaluated)
                    self.assertTrue(metadata["validation_is_selection_diagnostic"])
                    for name in ("raw_texts_included", "per_row_arrays_included", "model_weights_included"):
                        self.assertFalse(metadata[name])

        # Copy an existing frozen fixture: scoring/bundling adds no extra train
        # run and leaves the common continuous/resumed fixtures development-only.
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "bundles"
            shutil.copytree(self.continuous, run)
            self.assertFalse((run / "verification_private.zip").exists())
            bundles = self.colab.package_results(run, verification_private=True)
            development = {name: Path(path) for name, path in bundles.items()}
            self.assertEqual(development["results_for_review"].name, "results_for_review.zip")
            self.assertEqual(development["verification_private"].name, "verification_private.zip")
            inspect_bundle(development["results_for_review"], private=False, evaluated=False)
            inspect_bundle(development["verification_private"], private=True, evaluated=False)
            development_bytes = {name: path.read_bytes() for name, path in development.items()}
            thresholds_before = (run / "thresholds.json").read_bytes()
            test_results = self.neural.evaluate_run(run, device="cpu")
            self.assertEqual((run / "thresholds.json").read_bytes(), thresholds_before)
            evaluated = self.colab.package_results(run, verification_private=True)
            for name, path in development.items():
                self.assertEqual(path.read_bytes(), development_bytes[name])
                self.assertNotEqual(Path(evaluated[name]), path)
            self.assertEqual(Path(evaluated["results_for_review"]).name, "results_for_review_with_test.zip")
            self.assertEqual(Path(evaluated["verification_private"]).name, "verification_private_with_test.zip")
            inspect_bundle(evaluated["results_for_review"], private=False, evaluated=True)
            inspect_bundle(evaluated["verification_private"], private=True, evaluated=True)
            with zipfile.ZipFile(evaluated["results_for_review"]) as archive:
                self.assertEqual(json.loads(archive.read("public_test_results.json")), test_results)
            before = _tree_hashes(run)
            self.assertEqual(self.colab.package_results(run, verification_private=True), evaluated)
            self.assertEqual(_tree_hashes(run), before)

    def test_changed_source_config_and_training_code_are_rejected_before_continuation(self):
        original_hashes = self.colab._source_code_hashes()
        self.assertIn("scripts/colab_train.py", original_hashes)
        changed_config = replace(self.config, training=replace(self.config.training, learning_rate=2e-3))
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            changed_source = parent / "changed.csv"
            changed_source.write_bytes(self.source.read_bytes() + b"\n")
            def edit_identity(run, change):
                path = run / self.colab.IDENTITY_FILE
                identity = read_json(path)
                change(identity)
                path.write_text(json.dumps(identity), encoding="utf-8")

            def edit_checkpoint(run, field, value):
                path = run / self.colab.LATEST_CHECKPOINT
                state = self.torch.load(path, map_location="cpu", weights_only=True)
                state[field] = value
                self.colab.atomic_torch_save(self.torch, state, path)

            def change_tokenizer(run):
                path = run / self.colab.TOKENIZER_DIRECTORY / "tokenizer.json"
                path.write_bytes(path.read_bytes() + b"\n")

            scenarios = (
                ("source", changed_source, self.config, None),
                ("config", self.source, changed_config, None),
                ("code", self.source, self.config,
                 {**original_hashes, "scripts/colab_train.py": "0" * 64}),
            )
            for name, source, config, hashes in scenarios:
                with self.subTest(changed=name):
                    run = parent / name
                    shutil.copytree(self.interrupted_snapshot, run)
                    before = _tree_hashes(run)
                    code_patch = patch.object(self.colab, "_source_code_hashes", return_value=hashes) \
                        if hashes is not None else contextlib.nullcontext()
                    with code_patch, patch.object(self.neural, "_new_model", side_effect=AssertionError("model created")):
                        with self.assertRaisesRegex(ValueError, "(?i)(identity|integrity|changed|mismatch|fingerprint|differs)"):
                            self.colab.run_colab_training(source, run, config=config, quiet=True, _smoke=True)
                    self.assertEqual(_tree_hashes(run), before)

            tampering = {
                "tokenizer": change_tokenizer,
                "runtime": lambda run: edit_identity(run, lambda identity: identity["runtime"]["dependencies"].update(torch="0.0.0")),
                "split": lambda run: edit_identity(run, lambda identity: identity["split_sha256"].update(train="0" * 64)),
                "source_fingerprint": lambda run: edit_identity(run, lambda identity: identity.update(source_sha256="0" * 64)),
                "smoke_mode": lambda run: edit_identity(run, lambda identity: identity.update(smoke=False)),
                "checkpoint_identity": lambda run: edit_checkpoint(run, "identity_sha256", "0" * 64),
                "checkpoint_schema": lambda run: edit_checkpoint(run, "schema_version", -1),
                "checkpoint_labels": lambda run: edit_checkpoint(run, "label_order", list(reversed(LABELS))),
            }
            for name, mutate in tampering.items():
                with self.subTest(tampered=name):
                    run = parent / name
                    shutil.copytree(self.interrupted_snapshot, run)
                    mutate(run)
                    before = _tree_hashes(run)
                    model_patch = contextlib.nullcontext() if name.startswith("checkpoint_") else \
                        patch.object(self.neural, "_new_model", side_effect=AssertionError("model created"))
                    with model_patch, patch.object(self.colab, "_optimizer_group", side_effect=AssertionError("training update")):
                        with self.assertRaisesRegex(ValueError, "(?i)(identity|changed|differs|incompatible)"):
                            self.colab.run_colab_training(self.source, run, config=self.config, quiet=True, _smoke=True)
                    self.assertEqual(_tree_hashes(run), before)

    def test_atomic_latest_checkpoint_preserves_previous_bytes_on_failed_write_or_replace(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            checkpoint = parent / "latest_checkpoint.pt"
            self.colab.atomic_torch_save(torch, {"step": 1, "weight": torch.tensor([1.0])}, checkpoint)
            old_bytes = checkpoint.read_bytes()
            with patch.object(self.colab.os, "replace", side_effect=OSError("simulated failed replace")):
                with self.assertRaisesRegex(OSError, "simulated failed replace"):
                    self.colab.atomic_torch_save(torch, {"step": 2, "weight": torch.tensor([2.0])}, checkpoint)
            self.assertEqual(checkpoint.read_bytes(), old_bytes)
            self.assertEqual([path.name for path in parent.iterdir()], [checkpoint.name])
            self.assertEqual(torch.load(checkpoint, weights_only=True)["step"], 1)
            with patch.object(torch, "save", side_effect=OSError("simulated serialization failure")):
                with self.assertRaisesRegex(OSError, "simulated serialization failure"):
                    self.colab.atomic_torch_save(torch, {"step": 2}, checkpoint)
            self.assertEqual(checkpoint.read_bytes(), old_bytes)
            self.assertEqual([path.name for path in parent.iterdir()], [checkpoint.name])
            self.colab.atomic_torch_save(torch, {"step": 2, "weight": torch.tensor([2.0])}, checkpoint)
            self.assertEqual(torch.load(checkpoint, weights_only=True)["step"], 2)

    def test_frozen_rerun_is_read_only_and_explicit_restart_refuses_existing_run(self):
        before = _tree_hashes(self.resumed)
        with patch.object(self.neural, "_new_model", side_effect=AssertionError("new training")), \
                patch.object(self.neural, "_predict", side_effect=AssertionError("new inference")):
            result = self.colab.run_colab_training(self.source, self.resumed, config=self.config,
                                                 quiet=True, _smoke=True)
            self.assertEqual(result["stage"], "already_frozen")
            with self.assertRaises(FileExistsError):
                self.colab.run_colab_training(self.source, self.resumed, config=self.config,
                                             resume=False, quiet=True, _smoke=True)
        self.assertEqual(_tree_hashes(self.resumed), before)

    def test_python_numpy_torch_rng_roundtrip_is_weights_only_safe(self):
        torch = self.torch
        before = self.colab.capture_rng_state(torch)
        try:
            random.seed(281)
            np.random.seed(281)
            torch.manual_seed(281)
            state = self.colab.capture_rng_state(torch)
            expected = (random.random(), np.random.random(8), torch.rand(8))
            random.random()
            np.random.random(5)
            torch.rand(5)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "rng.pt"
                torch.save(state, path)
                loaded = torch.load(path, map_location="cpu", weights_only=True)
            self.colab.restore_rng_state(torch, loaded)
            actual = (random.random(), np.random.random(8), torch.rand(8))
            self.assertEqual(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
            self.assertTrue(torch.equal(actual[2], expected[2]))
        finally:
            self.colab.restore_rng_state(torch, before)

    def test_accumulation_uses_actual_sample_mean_for_unequal_batches_and_partial_group(self):
        torch = self.torch

        class ScalarModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.1))
                self.config = SimpleNamespace(vocab_size=16)

            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(logits=(self.weight * input_ids[:, :1].float()).expand(-1, len(LABELS)))

        def batch(features, labels):
            return {"input_ids": torch.tensor(features, dtype=torch.long)[:, None],
                    "attention_mask": torch.ones((len(features), 1), dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.float32)[:, None].expand(-1, len(LABELS))}

        class IsolatedSGD(torch.optim.SGD):
            # Optimizer construction can decorate the concrete class's step.
            # Keep that wrapper off the shared SGD base: another test's custom
            # step may call super().step and otherwise double-fire its hooks.
            pass

        model = ScalarModel()
        config = replace(self.config.training, epochs=1, accumulation_steps=2, learning_rate=0.05,
                         weight_decay=0, warmup_fraction=0, max_gradient_norm=1000)
        original_sgd_step = torch.optim.SGD.step
        optimizer = IsolatedSGD(model.parameters(), lr=config.learning_rate)
        self.assertIs(torch.optim.SGD.step, original_sgd_step)
        gradients = []
        optimizer.register_step_pre_hook(lambda optimizer, args, kwargs: gradients.append(float(model.weight.grad)))
        steps = {"value": 0}

        def count_update(*args):
            steps["value"] += 1

        optimizer.register_step_post_hook(count_update)
        scheduler = SimpleNamespace(steps=0)

        def schedule_update():
            scheduler.steps += 1

        scheduler.step = schedule_update
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        criterion = torch.nn.MSELoss()
        groups = [[batch([1, 3], [0, 1]), batch([7], [0])], [batch([5], [1])]]
        results = [self.colab._optimizer_group(model, group, criterion, optimizer, scaler, scheduler,
                                              config, self.dependencies, torch.device("cpu"), steps)
                   for group in groups]
        first_gradient = sum(2 * (0.1 * feature - target) * feature
                             for feature, target in ((1, 0), (3, 1), (7, 0))) / 3
        first_weight = 0.1 - config.learning_rate * first_gradient
        second_gradient = 2 * (first_weight * 5 - 1) * 5
        np.testing.assert_allclose(gradients, [first_gradient, second_gradient], rtol=1e-5, atol=1e-6)
        self.assertAlmostEqual(float(model.weight.detach()), first_weight - config.learning_rate * second_gradient, places=6)
        self.assertEqual([result["sample_count"] for result in results], [3, 1])
        self.assertEqual([result["skipped_amp_steps"] for result in results], [0, 0])
        self.assertEqual(steps["value"], 2)
        self.assertEqual(scheduler.steps, 2)

    def test_resumed_patience_restores_nonfinal_best_before_fresh_validation_and_thresholds(self):
        torch, neural = self.torch, self.neural
        config = replace(self.config, training=replace(self.config.training, epochs=5, patience=2))
        predict = neural._predict
        observations = []
        first_loss = None

        def first_epoch_wins(model, loader, *args, **kwargs):
            nonlocal first_loss
            observed = predict(model, loader, *args, **kwargs)
            self.assertEqual(loader.dataset.texts.tolist(), self.validation_texts)
            observations.append(model)
            if len(observations) == 1:
                first_loss = observed[0]
            elif len(observations) in (2, 3):
                return first_loss + 0.1 * (len(observations) - 1), observed[1], observed[2]
            return observed

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "early_stop"
            with patch.object(neural, "_predict", side_effect=first_epoch_wins):
                interrupted = self.colab.run_colab_training(self.source, run, config=config, quiet=True,
                                                           _smoke=True, _stop_after_steps=7)
                self.assertEqual(interrupted["stage"], "interrupted")
                state = torch.load(run / "private/latest_checkpoint.pt", map_location="cpu", weights_only=True)
                self.assertEqual(state["epochs_without_improvement"], 1)
                self.assertEqual(len(state["history"]), 2)
                result = self.colab.run_colab_training(self.source, run, config=config, quiet=True, _smoke=True)
            self.assertEqual(result["global_optimizer_steps"], 9)
            history = read_json(run / "training_history.json")
            self.assertEqual(history["best_checkpoint"]["epoch"], 1)
            self.assertEqual(history["epochs"][-1]["epoch"], 3)
            self.assertAlmostEqual(history["fresh_validation_loss"], first_loss, places=7)
            self.assertEqual(len(observations), 4)
            self.assertIsNot(observations[0], observations[-1])
            best = torch.load(run / "private/best_checkpoint.pt", map_location="cpu", weights_only=True)
            latest = torch.load(run / "private/latest_checkpoint.pt", map_location="cpu", weights_only=True)
            exported = self.dependencies.RobertaForSequenceClassification.from_pretrained(
                run / "model", local_files_only=True, use_safetensors=True, attn_implementation="eager")
            self.assertEqual(exported.config.id2label, dict(enumerate(LABELS)))
            self.assertEqual(exported.config.label2id, {label: index for index, label in enumerate(LABELS)})
            self.assertEqual(best["epoch"], 1)
            self.assertTrue(any(not torch.equal(value, latest["model_state_dict"][name])
                                for name, value in best["model_state_dict"].items()))
            for name, value in exported.state_dict().items():
                self.assertTrue(torch.equal(value, best["model_state_dict"][name]), name)
            with np.load(run / "private/validation_predictions.npz", allow_pickle=False) as predictions:
                np.testing.assert_array_equal(predictions["row_indices"], self.splits.validation)
                np.testing.assert_array_equal(predictions["targets"], self.dataset.targets[self.splits.validation])
                np.testing.assert_array_equal(load_thresholds(run),
                                              select_thresholds(predictions["targets"], predictions["probabilities"]))
            verify_frozen_run(run)


if __name__ == "__main__":
    unittest.main()
