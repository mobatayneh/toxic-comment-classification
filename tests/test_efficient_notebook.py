"""Offline notebook/source and mocked Colab lifecycle checks; no GPU claims."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat

from scripts import build_efficient_colab_notebook as builder

ROOT = Path(__file__).resolve().parents[1]
PREFIX = builder.COLAB_ROOT + "/"
EXPECTED_FILES = (
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
FIXTURE_BYTES = b"synthetic notebook lifecycle fixture; never used for model training\n"
FIXTURE_HASH = hashlib.sha256(FIXTURE_BYTES).hexdigest()
ORIGINAL_HASH = "bb7a9e1af5282c4c05bf09bf3be3373c4160fe3272a4b26016034ff4bcea5f1e"


def source_cells(notebook):
    sources = {}
    cells = {}
    for cell in notebook.cells:
        if cell.cell_type != "code" or not cell.source.startswith("%%writefile "):
            continue
        match = re.fullmatch(r"%%writefile ([^\n]+)\n([\s\S]*)", cell.source)
        if match is None or not match[1].startswith(PREFIX):
            raise AssertionError("Unexpected embedded source destination")
        relative = match[1][len(PREFIX):]
        if relative in sources:
            raise AssertionError("Duplicate embedded source")
        sources[relative] = match[2]
        cells[relative] = cell.source
    return sources, cells


def literal_manifest(notebook):
    values = {}
    for cell in notebook.cells:
        if cell.cell_type != "code" or cell.source.startswith("%"):
            continue
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"EMBEDDED_FILES", "EMBEDDED_SHA256"}:
                        if target.id in values:
                            raise AssertionError("Duplicate manifest assignment")
                        values[target.id] = ast.literal_eval(node.value)
    return values


class _FakeProcess:
    def __init__(self, text, return_code=0):
        self.stdout = io.StringIO(text)
        self.return_code = return_code
        self.completed = False

    def wait(self, timeout=None):
        self.completed = True
        return self.return_code

    def poll(self):
        return self.return_code if self.completed else None

    def terminate(self):
        self.completed = True

    def kill(self):
        self.completed = True


class EfficientNotebookSources(unittest.TestCase):
    def test_notebook_is_current_clear_valid_and_compiles_every_cell(self):
        from IPython.core.inputtransformer2 import TransformerManager

        notebook = nbformat.read(builder.OUTPUT, as_version=4)
        nbformat.validate(notebook)
        self.assertEqual(builder.OUTPUT.read_text(encoding="utf-8"), nbformat.writes(builder.build_notebook()) + "\n")
        sources, _ = source_cells(notebook)
        self.assertEqual(tuple(sources), EXPECTED_FILES)
        manifest = literal_manifest(notebook)
        self.assertEqual(tuple(manifest["EMBEDDED_FILES"]), EXPECTED_FILES)
        expected_hashes = {}
        for relative in EXPECTED_FILES:
            expected = (ROOT / relative).read_bytes()
            self.assertEqual(sources[relative].encode("utf-8"), expected, relative)
            expected_hashes[relative] = hashlib.sha256(expected).hexdigest()
            if relative.endswith(".py"):
                compile(sources[relative], relative, "exec")
        self.assertEqual(manifest["EMBEDDED_SHA256"], expected_hashes)
        transformer = TransformerManager()
        for cell in notebook.cells:
            if cell.cell_type == "code":
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])
                compile(transformer.transform_cell(cell.source), "efficient-notebook-cell", "exec")

    def test_visible_writefiles_recreate_exact_bytes_without_colab_services(self):
        from IPython.core.interactiveshell import InteractiveShell
        from traitlets.config import Config

        notebook = nbformat.read(builder.OUTPUT, as_version=4)
        sources, cells = source_cells(notebook)
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            for relative in EXPECTED_FILES:
                (project / relative).parent.mkdir(parents=True, exist_ok=True)
            shell = InteractiveShell(config=Config({"HistoryManager": {"enabled": False}}),
                                     ipython_dir=str(Path(directory) / "ipython"))
            for relative in EXPECTED_FILES:
                source = cells[relative].replace(PREFIX, str(project) + "/", 1)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = shell.run_cell(source, store_history=False)
                if result.error_before_exec is not None:
                    raise result.error_before_exec
                if result.error_in_exec is not None:
                    raise result.error_in_exec
                self.assertEqual((project / relative).read_bytes(), sources[relative].encode("utf-8"))

    def test_operational_kernel_imports_and_commands_keep_ml_in_children(self):
        notebook = nbformat.read(builder.OUTPUT, as_version=4)
        allowed = {"collections", "datetime", "gc", "hashlib", "importlib", "json", "os",
                   "pathlib", "shutil", "subprocess", "sys", "uuid", "google", "IPython"}
        operational = []
        for cell in notebook.cells:
            if cell.cell_type != "code" or cell.source.startswith("%%writefile"):
                continue
            operational.append(cell.source)
            for node in ast.walk(ast.parse(cell.source)):
                if isinstance(node, ast.Import):
                    roots = {alias.name.split(".")[0] for alias in node.names}
                    self.assertLessEqual(roots, allowed)
                elif isinstance(node, ast.ImportFrom):
                    self.assertIn(node.module.split(".")[0], allowed)
        source = "\n".join(operational)
        self.assertNotIn("--evaluate", source)
        self.assertNotIn("verification_private", source)
        self.assertNotIn("load_dataset", source)
        self.assertNotIn("split_dataset", source)
        self.assertNotIn("DATA_FILE.read_bytes", source)
        self.assertIn("import torch", builder.VERIFY_CHILD_SOURCE)
        self.assertIn("--constraint", source)
        self.assertIn('importlib.metadata.version("torch")', builder.BOOTSTRAP_SOURCE)
        self.assertIn('"USE_TF": "0"', builder.BOOTSTRAP_SOURCE)
        self.assertIn('"USE_FLAX": "0"', builder.BOOTSTRAP_SOURCE)
        self.assertEqual(builder.COLAB_ROOT, "/content/toxic-comment-efficient-v1")
        markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
        self.assertIn("separate numerical experiment", markdown.replace("\n", " "))
        self.assertIn("have **not** been validated", markdown)

    def test_dependency_setup_preserves_distribution_pin_and_checks_a_child(self):
        notebook = nbformat.read(builder.OUTPUT, as_version=4)
        setup_source = next(cell.source for cell in notebook.cells if cell.cell_type == "code"
                            and "VERIFY_CHILD_SOURCE = " in cell.source)
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for relative in EXPECTED_FILES:
                target = project / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / relative).read_bytes())
            constraint = project / "torch-constraint.txt"
            constraint.write_text("torch==2.8.0+cu128\n", encoding="utf-8")
            child_env = {"USE_TF": "0", "USE_FLAX": "0"}
            state = {"PROJECT_DIR": project, "TORCH_CONSTRAINT": constraint,
                     "COLAB_TORCH_DISTRIBUTION_VERSION": "2.8.0+cu128", "CHILD_ENV": child_env}
            with patch("importlib.metadata.version", return_value="2.8.0+cu128"), \
                    patch("subprocess.run") as run, contextlib.redirect_stdout(io.StringIO()):
                exec(compile(setup_source, "dependency-cell", "exec"), state)
            self.assertTrue(state["BOOTSTRAP_VERIFIED"])
            self.assertEqual(run.call_count, 2)
            pip_call, check_call = run.call_args_list
            self.assertIn(str(constraint), pip_call.args[0])
            self.assertIn("--constraint", pip_call.args[0])
            self.assertIn("-c", check_call.args[0])
            self.assertEqual(check_call.args[0][-1], "2.8.0+cu128")
            for call in run.call_args_list:
                self.assertEqual(call.kwargs["env"], child_env)


class EfficientNotebookLifecycle(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name)
        self.project = self.parent / "project"
        self.drive_root = self.parent / "drive" / "MyDrive"
        self.persistent = self.drive_root / "toxic-comment-classification"
        self.run_name = "random-roberta-efficient-v1-seed42"
        self.run_dir = self.persistent / "runs" / self.run_name
        self.calibration = self.persistent / "diagnostics" / self.run_name / "calibration.json"
        self.hashes = {}
        for relative in EXPECTED_FILES:
            target = self.project / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
            self.hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
        self.commands = []
        self.uploaded = {}
        self.downloads = []
        self.mounts = []
        google = types.ModuleType("google")
        google.__path__ = []
        colab = types.ModuleType("google.colab")
        self.files = types.SimpleNamespace(upload=self._upload, download=self.downloads.append)
        colab.drive = types.SimpleNamespace(mount=self.mounts.append)
        colab.files = self.files
        google.colab = colab
        self.modules = {"google": google, "google.colab": colab}
        self.child_env = {"USE_TF": "0", "USE_FLAX": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                          "TOKENIZERS_PARALLELISM": "false"}
        self.state = {"PROJECT_DIR": self.project, "BOOTSTRAP_VERIFIED": True,
                      "EMBEDDED_SHA256": self.hashes, "CHILD_ENV": self.child_env}
        self.failure_text = None
        self.saved_config = {"training": {"batch_size": 32, "accumulation_steps": 8,
                                          "cache_tokenization": True, "dynamic_padding": True,
                                          "epochs": 5}, "checkpoint_every_seconds": 300.0}

    def _upload(self):
        if not self.uploaded:
            raise AssertionError("Unexpected upload request")
        return self.uploaded

    def _source(self, location="remembered"):
        paths = {
            "remembered": self.persistent / "inputs" / "train_preprocessed.csv",
            "original": self.persistent / "runs" / "random-roberta-seed42" / "private" / "source.csv",
            "run": self.run_dir / "private" / "source.csv",
            "manual": self.drive_root / "manual" / "train_preprocessed.csv",
        }
        path = paths[location]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(FIXTURE_BYTES)
        return path

    def _saved_identity(self):
        path = self.run_dir / "private" / "run_identity.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"requested_config": self.saved_config}), encoding="utf-8")
        return path

    def _launch_source(self, *, run_name=None, manual=""):
        return (builder.LAUNCH_SOURCE
                .replace('"/content/drive/MyDrive/toxic-comment-classification" # @param', repr(str(self.persistent)) + " # @param")
                .replace('Path("/content/drive/MyDrive")', f"Path({str(self.drive_root)!r})")
                .replace('RUN_NAME = "random-roberta-efficient-v1-seed42"', f"RUN_NAME = {run_name or self.run_name!r}")
                .replace('CSV_ON_DRIVE = "" # @param', f"CSV_ON_DRIVE = {str(manual)!r} # @param")
                .replace(ORIGINAL_HASH, FIXTURE_HASH))

    def _popen(self, command, **kwargs):
        self.commands.append((command, kwargs))
        self.assertEqual(kwargs["env"], self.child_env)
        self.assertIs(kwargs["stderr"], subprocess.STDOUT)
        self.assertTrue(kwargs["text"])
        if "calibrate" in command:
            self.assertFalse(self.uploaded, "Upload bytes must be released before ML work")
            path = Path(command[command.index("--output") + 1])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"batch_size": 32, "accumulation_steps": 8}), encoding="utf-8")
            return _FakeProcess("synthetic calibration complete\n")
        if "train" in command:
            self.assertFalse(self.uploaded, "Upload bytes must be released before ML work")
            if not (self.run_dir / "private" / "run_identity.json").exists():
                self._saved_identity()
            if self.failure_text:
                return _FakeProcess(self.failure_text, 7)
            return _FakeProcess("synthetic child progress\nsynthetic child finished\n")
        raise AssertionError(f"Unexpected child command: {command}")

    def launch(self, **kwargs):
        with patch.dict(sys.modules, self.modules), patch("subprocess.Popen", side_effect=self._popen), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            exec(compile(self._launch_source(**kwargs), "launch-cell", "exec"), self.state)
        return output.getvalue()

    def _run_status_package(self, command, **kwargs):
        self.commands.append((command, kwargs))
        self.assertEqual(kwargs["env"], self.child_env)
        self.assertEqual(kwargs["cwd"], self.project)
        if "package" in command:
            (self.run_dir / "results_for_review.zip").write_bytes(b"synthetic mock package")
        return types.SimpleNamespace(returncode=0)

    def results(self, state=None):
        with patch.dict(sys.modules, self.modules), patch("subprocess.run", side_effect=self._run_status_package), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            exec(compile(builder.RESULTS_SOURCE, "results-cell", "exec"), self.state if state is None else state)
        return output.getvalue()

    def test_new_run_calibrates_once_then_resume_uses_saved_configuration(self):
        self._source()
        self.launch()
        self.assertEqual([command[3] for command, _ in self.commands], ["calibrate", "train"])
        identity = self.run_dir / "private" / "run_identity.json"
        saved_identity = identity.read_bytes()
        self.calibration.unlink()
        self.commands.clear()
        output = self.launch()
        self.assertEqual([command[3] for command, _ in self.commands], ["train"])
        self.assertIn("'calibration_rerun': False", output)
        self.assertEqual(identity.read_bytes(), saved_identity)
        logs = list((self.persistent / "diagnostics" / self.run_name / "logs").glob("*.log"))
        self.assertEqual(len(logs), 3)
        self.assertEqual(len({path.name for path in logs}), 3)
        self.assertTrue(all(not path.is_relative_to(self.run_dir) for path in logs))
        snapshot = self.persistent / "diagnostics" / self.run_name / "source_snapshot"
        self.assertEqual(json.loads((snapshot / "embedded_sha256.json").read_text()), self.hashes)
        for relative, expected in self.hashes.items():
            self.assertEqual(hashlib.sha256((snapshot / relative).read_bytes()).hexdigest(), expected)

    def test_original_source_is_read_only_fallback_and_remembered_input_wins(self):
        original = self._source("original")
        before = {path.relative_to(original.parent): path.read_bytes() for path in original.parent.rglob("*") if path.is_file()}
        self.launch()
        self.assertEqual(self.state["DATA_FILE"], original)
        self.assertEqual({path.relative_to(original.parent): path.read_bytes() for path in original.parent.rglob("*") if path.is_file()}, before)
        remembered = self._source()
        self.launch()
        self.assertEqual(self.state["DATA_FILE"], remembered)

    def test_manual_input_and_saved_run_snapshot_are_supported(self):
        self._saved_identity()
        run_source = self._source("run")
        self.launch()
        self.assertEqual(self.state["DATA_FILE"], run_source)
        manual_source = self._source("manual")
        self.launch(manual=manual_source)
        self.assertEqual(self.state["DATA_FILE"], manual_source)
        self.assertTrue(all("calibrate" not in command for command, _ in self.commands))

    def test_upload_is_hashed_and_released_before_child_launch(self):
        self.uploaded["train_preprocessed.csv"] = FIXTURE_BYTES
        self.launch()
        self.assertEqual(self.uploaded, {})
        remembered = self.persistent / "inputs" / "train_preprocessed.csv"
        self.assertEqual(remembered.read_bytes(), FIXTURE_BYTES)
        self.assertFalse(any((self.project / "uploads").iterdir()))

    def test_wrong_upload_name_or_hash_stops_before_any_ml_child(self):
        self.uploaded["test_preprocessed.csv"] = FIXTURE_BYTES
        with self.assertRaisesRegex(ValueError, "Upload only the original"):
            self.launch()
        self.assertEqual(self.uploaded, {})
        self.assertEqual(self.commands, [])
        self.uploaded["train_preprocessed.csv"] = b"wrong source"
        with self.assertRaisesRegex(ValueError, "Uploaded bytes differ"):
            self.launch()
        self.assertEqual(self.uploaded, {})
        self.assertEqual(self.commands, [])
        self.assertFalse((self.persistent / "inputs" / "train_preprocessed.csv").exists())

    def test_old_or_unidentified_run_cannot_be_overwritten(self):
        self._source()
        with self.assertRaisesRegex(ValueError, "original random-roberta-seed42 run is reserved"):
            self.launch(run_name="random-roberta-seed42")
        self.run_dir.mkdir(parents=True)
        marker = self.run_dir / "keep.txt"
        marker.write_text("unchanged")
        with self.assertRaisesRegex(RuntimeError, "without a complete saved identity"):
            self.launch()
        self.assertEqual(marker.read_text(), "unchanged")
        self.assertEqual(self.commands, [])

    def test_drive_root_inside_original_run_is_rejected_before_writes(self):
        original = self._source("original")
        self.persistent = original.parent.parent
        before = {path.relative_to(self.persistent): path.read_bytes()
                  for path in self.persistent.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(ValueError, "must be outside the original"):
            self.launch()
        after = {path.relative_to(self.persistent): path.read_bytes()
                 for path in self.persistent.rglob("*") if path.is_file()}
        self.assertEqual(after, before)
        self.assertFalse((self.persistent / "diagnostics").exists())
        self.assertEqual(self.commands, [])

    def test_changed_source_snapshot_refuses_resume_before_child_execution(self):
        self._source()
        self.launch()
        self.commands.clear()
        saved_source = self.persistent / "diagnostics" / self.run_name / "source_snapshot" / EXPECTED_FILES[0]
        saved_source.write_bytes(saved_source.read_bytes() + b"\n# changed\n")
        with self.assertRaisesRegex(RuntimeError, "Saved source snapshot differs"):
            self.launch()
        self.assertEqual(self.commands, [])

    def test_child_failure_keeps_actual_tail_and_persistent_log_without_retry(self):
        self._source()
        self._saved_identity()
        self.failure_text = "Traceback (most recent call last):\nRuntimeError: synthetic optimizer failure\n"
        with self.assertRaisesRegex(RuntimeError, "synthetic optimizer failure") as caught:
            self.launch()
        self.assertIn("code 7", str(caught.exception))
        self.assertEqual([command[3] for command, _ in self.commands], ["train"])
        logs = list((self.persistent / "diagnostics" / self.run_name / "logs").glob("train-*.log"))
        self.assertEqual(len(logs), 1)
        log_text = logs[0].read_text(encoding="utf-8")
        self.assertIn(self.failure_text, log_text)
        completion = json.loads(log_text.splitlines()[-1])
        self.assertEqual(completion["child_returncode"], 7)
        self.assertIn("finished_utc", completion)

    def test_results_before_launch_or_without_complete_freeze_are_friendly(self):
        output = self.results(state={})
        self.assertIn("Run steps 1–4 first", output)
        self.assertEqual(self.commands, [])
        self._source()
        self.launch()
        self.commands.clear()
        (self.run_dir / "manifest.json").write_text("{}")
        output = self.results()
        self.assertIn("Training is unfinished", output)
        self.assertEqual([command[3] for command, _ in self.commands], ["status"])
        self.assertIsNone(self.state["review_zip"])
        self.assertEqual(self.downloads, [])

    def test_results_package_only_after_freeze_guard_and_download_aggregate_zip(self):
        self._source()
        self.launch()
        self.commands.clear()
        for relative in ("manifest.json", "manifest.sha256", "training_history.json"):
            (self.run_dir / relative).write_text("{}", encoding="utf-8")
        (self.run_dir / "validation_metrics.json").write_text(
            json.dumps({"metrics_at_validation_thresholds": {"micro_f1": 0.5}}), encoding="utf-8")
        self.results()
        self.assertEqual([command[3] for command, _ in self.commands], ["status", "package"])
        expected = self.run_dir / "results_for_review.zip"
        self.assertEqual(self.state["review_zip"], expected)
        self.assertEqual(self.downloads, [str(expected), str(self.calibration)])
        self.assertTrue(all("--evaluate" not in command for command, _ in self.commands))


if __name__ == "__main__":
    unittest.main()
