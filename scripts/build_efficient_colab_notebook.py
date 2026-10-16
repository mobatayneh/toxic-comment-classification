"""Build the isolated, self-contained efficient-profile Colab notebook.

The notebook kernel uses only the standard library and Colab/display helpers.
Pinned ML imports, GPU calibration, training, status, and packaging run in fresh
Python subprocesses. Visible writefile cells preserve the reviewed source bytes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "04_colab_efficient_training.ipynb"
COLAB_ROOT = "/content/toxic-comment-efficient-v1"
EMBEDDED_FILES = (
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


BOOTSTRAP_SOURCE = '''import importlib.metadata
import os
import sys
from pathlib import Path

if sys.version_info < (3, 12):
    raise RuntimeError("Select a Python 3.12+ Colab runtime, then rerun this notebook.")

# Do not import torch, NumPy, pandas, or project modules in this kernel.
# Every child receives these values before any ML imports or CUDA initialization.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
PROJECT_DIR = Path("/content/toxic-comment-efficient-v1")
for relative in ("src/toxic_comments", "scripts"):
    (PROJECT_DIR / relative).mkdir(parents=True, exist_ok=True)
try:
    COLAB_TORCH_DISTRIBUTION_VERSION = importlib.metadata.version("torch")
except importlib.metadata.PackageNotFoundError as error:
    raise RuntimeError("Choose a Colab GPU runtime with CUDA-enabled torch >=2.8 already installed.") from error
TORCH_CONSTRAINT = PROJECT_DIR / "colab-torch-constraint.txt"
TORCH_CONSTRAINT.write_text("torch==" + COLAB_TORCH_DISTRIBUTION_VERSION + "\\n", encoding="utf-8")
CHILD_ENV = dict(os.environ)
CHILD_ENV.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TOKENIZERS_PARALLELISM": "false",
                  "USE_TF": "0", "USE_FLAX": "0", "PYTHONUNBUFFERED": "1"})
CHILD_ENV["PYTHONPATH"] = os.pathsep.join(
    [str(PROJECT_DIR / "src"), str(PROJECT_DIR)]
    + ([CHILD_ENV["PYTHONPATH"]] if CHILD_ENV.get("PYTHONPATH") else [])
)
BOOTSTRAP_VERIFIED = False
print({"python": sys.version.split()[0], "installed_torch_distribution": COLAB_TORCH_DISTRIBUTION_VERSION,
       "temporary_project": str(PROJECT_DIR)})
'''


VERIFY_CHILD_SOURCE = '''import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from packaging.version import Version

if sys.version_info < (3, 12):
    raise RuntimeError("Python 3.12+ is required.")
expected_torch = sys.argv[1]
expected_versions = {}
for line in Path("requirements-colab.txt").read_text(encoding="utf-8").splitlines():
    if "==" in line and not line.lstrip().startswith("#"):
        name, version = line.strip().split("==", 1)
        expected_versions[name] = version
for name, version in expected_versions.items():
    observed = importlib.metadata.version(name)
    if observed != version:
        raise RuntimeError(f"Installed {name} {observed}; expected {version}.")
module_names = {"scikit-learn": "sklearn", "huggingface-hub": "huggingface_hub"}
for name in expected_versions:
    importlib.import_module(module_names.get(name, name))
import torch
if importlib.metadata.version("torch") != expected_torch:
    raise RuntimeError("The installed torch distribution changed during dependency installation.")
if Version(torch.__version__.split("+")[0]) < Version("2.8.0") or not torch.cuda.is_available():
    raise RuntimeError("Select Runtime > Change runtime type > GPU with CUDA-enabled torch >=2.8, then rerun setup.")
from scripts import colab_train
from toxic_comments import neural
colab_train._assert_environment(neural._dependencies(), smoke=False, config=colab_train.ColabConfig())
print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__,
                  "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                  "pinned_versions": expected_versions,
                  "ml_imports_verified_in_fresh_child": True}, indent=2))
'''


LAUNCH_SOURCE = '''import collections
import datetime
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from google.colab import drive, files

if not globals().get("BOOTSTRAP_VERIFIED", False):
    raise RuntimeError("Run steps 1–3, including every source-writing cell and the dependency check, before this launch cell.")

DRIVE_PROJECT_DIR = "/content/drive/MyDrive/toxic-comment-classification" # @param {type:"string"}
RUN_NAME = "random-roberta-efficient-v1-seed42" # @param {type:"string"}
CSV_ON_DRIVE = "" # @param {type:"string"}

def stream_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if not RUN_NAME or RUN_NAME in {".", "..", "random-roberta-seed42"} or "/" in RUN_NAME or "\\\\" in RUN_NAME:
    raise ValueError("Choose a plain new-profile RUN_NAME; the original random-roberta-seed42 run is reserved.")
drive.mount("/content/drive")
PERSISTENT_ROOT = Path(DRIVE_PROJECT_DIR).expanduser().resolve()
if not PERSISTENT_ROOT.is_relative_to(Path("/content/drive/MyDrive")):
    raise ValueError("DRIVE_PROJECT_DIR must be under /content/drive/MyDrive.")
RUN_DIR = PERSISTENT_ROOT / "runs" / RUN_NAME
ORIGINAL_RUN_DIR = PERSISTENT_ROOT / "runs" / "random-roberta-seed42"
if RUN_DIR.resolve() == ORIGINAL_RUN_DIR.resolve():
    raise ValueError("The efficient profile cannot write into the original run directory.")
for protected_path in (PERSISTENT_ROOT, RUN_DIR.resolve()):
    parts = protected_path.parts
    if any(parts[index:index + 2] == ("runs", "random-roberta-seed42") for index in range(len(parts) - 1)):
        raise ValueError("The project and new run must be outside the original runs/random-roberta-seed42 directory.")
identity_file = RUN_DIR / "private" / "run_identity.json"
if RUN_DIR.exists() and not identity_file.is_file():
    raise RuntimeError("This run directory exists without a complete saved identity. Choose a new RUN_NAME; do not overwrite it.")

# Diagnostics and logs stay outside the run protected by the freeze manifest.
DIAGNOSTICS_DIR = PERSISTENT_ROOT / "diagnostics" / RUN_NAME
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_FILE = DIAGNOSTICS_DIR / "calibration.json"
SOURCE_SNAPSHOT_DIR = DIAGNOSTICS_DIR / "source_snapshot"
SOURCE_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

def copy_exact_source(source, destination, expected_sha256):
    if stream_sha256(source) != expected_sha256:
        raise RuntimeError(f"Embedded source changed before launch: {source}")
    if destination.exists():
        if stream_sha256(destination) != expected_sha256:
            raise RuntimeError("Saved source snapshot differs. Reopen the original notebook used for this run; do not replace saved code.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp-" + uuid.uuid4().hex)
    try:
        shutil.copyfile(source, temporary)
        if stream_sha256(temporary) != expected_sha256:
            raise RuntimeError("Source snapshot copy failed its SHA-256 check.")
        if destination.exists():
            if stream_sha256(destination) != expected_sha256:
                raise RuntimeError("A different source snapshot already exists; refusing to replace it.")
        else:
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

for relative, expected in EMBEDDED_SHA256.items():
    copy_exact_source(PROJECT_DIR / relative, SOURCE_SNAPSHOT_DIR / relative, expected)
snapshot_manifest = SOURCE_SNAPSHOT_DIR / "embedded_sha256.json"
if snapshot_manifest.exists():
    if json.loads(snapshot_manifest.read_text(encoding="utf-8")) != EMBEDDED_SHA256:
        raise RuntimeError("Saved source manifest differs from this notebook. Use the original notebook to resume.")
else:
    temporary = snapshot_manifest.with_name(snapshot_manifest.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(EMBEDDED_SHA256, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
    os.replace(temporary, snapshot_manifest)

FULL_SOURCE_SHA256 = "bb7a9e1af5282c4c05bf09bf3be3373c4160fe3272a4b26016034ff4bcea5f1e"
remembered_source = PERSISTENT_ROOT / "inputs" / "train_preprocessed.csv"
run_source = RUN_DIR / "private" / "source.csv"
original_source = ORIGINAL_RUN_DIR / "private" / "source.csv"
if CSV_ON_DRIVE.strip():
    DATA_FILE = Path(CSV_ON_DRIVE).expanduser().resolve()
elif remembered_source.is_file():
    DATA_FILE = remembered_source
elif run_source.is_file():
    DATA_FILE = run_source
elif original_source.is_file():
    DATA_FILE = original_source  # Read-only fallback; the original run is never modified.
else:
    staging_dir = PROJECT_DIR / "uploads"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_file = staging_dir / (uuid.uuid4().hex + "-train_preprocessed.csv")
    uploaded = files.upload()
    try:
        if len(uploaded) != 1 or "train_preprocessed.csv" not in uploaded:
            raise ValueError("Upload only the original train_preprocessed.csv with its original filename.")
        with staging_file.open("xb") as handle:
            handle.write(uploaded["train_preprocessed.csv"])
    finally:
        uploaded.clear()
        del uploaded
        gc.collect()
    try:
        if stream_sha256(staging_file) != FULL_SOURCE_SHA256:
            raise ValueError("Uploaded bytes differ from the original supplied train_preprocessed.csv.")
        remembered_source.parent.mkdir(parents=True, exist_ok=True)
        if remembered_source.exists():
            if stream_sha256(remembered_source) != FULL_SOURCE_SHA256:
                raise ValueError("An existing remembered input differs; refusing to overwrite it.")
        else:
            persistent_temporary = remembered_source.with_name(remembered_source.name + ".tmp-" + uuid.uuid4().hex)
            try:
                shutil.copyfile(staging_file, persistent_temporary)
                if stream_sha256(persistent_temporary) != FULL_SOURCE_SHA256:
                    raise ValueError("The Drive input copy failed its SHA-256 check.")
                if remembered_source.exists():
                    if stream_sha256(remembered_source) != FULL_SOURCE_SHA256:
                        raise ValueError("An existing remembered input differs; refusing to overwrite it.")
                else:
                    os.replace(persistent_temporary, remembered_source)
            finally:
                persistent_temporary.unlink(missing_ok=True)
        DATA_FILE = remembered_source
    finally:
        staging_file.unlink(missing_ok=True)
if not DATA_FILE.is_file():
    raise FileNotFoundError("Training CSV not found at the chosen Drive path.")
if stream_sha256(DATA_FILE) != FULL_SOURCE_SHA256:
    raise ValueError("Use the original supplied train_preprocessed.csv bytes; this file has a different SHA-256.")

def stream_child(command, stage):
    log_dir = DIAGNOSTICS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log_path = log_dir / f"{stage}-{timestamp}-{uuid.uuid4().hex[:8]}.log"
    tail = collections.deque(maxlen=80)
    print(f"{stage.capitalize()} log: {log_path}", flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "started_utc": timestamp}) + "\\n")
        log.flush()
        process = subprocess.Popen(command, cwd=PROJECT_DIR, env=CHILD_ENV,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                tail.append(line)
            return_code = process.wait()
            log.write(json.dumps({"child_returncode": return_code,
                                  "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}) + "\\n")
            log.flush()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()
    if return_code:
        raise RuntimeError(f"{stage.capitalize()} child exited with code {return_code}. Full log: {log_path}\\n"
                           + "Final child output:\\n" + "".join(tail))
    return log_path

profile_script = str(PROJECT_DIR / "scripts" / "colab_profile.py")
if identity_file.is_file():
    saved_identity = json.loads(identity_file.read_text(encoding="utf-8"))
    print({"mode": "resume", "run_name": RUN_NAME,
           "saved_configuration": saved_identity.get("requested_config"),
           "calibration_rerun": False})
else:
    print({"mode": "new efficient-profile experiment", "run_name": RUN_NAME,
           "source_verified": True, "calibration_file": str(CALIBRATION_FILE)})
    stream_child([sys.executable, "-u", profile_script, "calibrate", "--output", str(CALIBRATION_FILE)], "calibration")
    if not CALIBRATION_FILE.is_file():
        raise RuntimeError("Calibration did not save its decision. Read the calibration log before continuing.")
    calibration = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    print({"batch_size": calibration["batch_size"],
           "accumulation_steps": calibration["accumulation_steps"],
           "effective_batch_size": calibration["batch_size"] * calibration["accumulation_steps"]})

# One child attempt. A failure is reported with its actual output tail; no automatic retry.
TRAIN_LOG = stream_child([sys.executable, "-u", profile_script, "train", "--data", str(DATA_FILE),
                          "--run-dir", str(RUN_DIR), "--calibration", str(CALIBRATION_FILE)], "train")
print("Training child finished. The next cell checks saved status and only packages a completed frozen run.")
'''


RESULTS_SOURCE = '''import json
import subprocess
import sys
from pathlib import Path

review_zip = None
if not all(name in globals() for name in ("PROJECT_DIR", "RUN_DIR", "CHILD_ENV")):
    print("Run steps 1–4 first. This cell can also be rerun after an interrupted launch to inspect saved status.")
else:
    profile_script = str(PROJECT_DIR / "scripts" / "colab_profile.py")
    subprocess.run([sys.executable, "-u", profile_script, "status", "--run-dir", str(RUN_DIR)],
                   cwd=PROJECT_DIR, env=CHILD_ENV, check=True)
    completion_files = ("manifest.json", "manifest.sha256", "training_history.json")
    missing = [relative for relative in completion_files if not (RUN_DIR / relative).is_file()]
    if missing:
        print("Training is unfinished: no completed freeze is available yet. Keep the same RUN_NAME and rerun the launch cell to resume.")
        print({"missing_completion_files": missing, "download_ready": False})
    else:
        subprocess.run([sys.executable, "-u", profile_script, "package", "--run-dir", str(RUN_DIR)],
                       cwd=PROJECT_DIR, env=CHILD_ENV, check=True)
        expected_zip = RUN_DIR / "results_for_review.zip"
        if not expected_zip.is_file():
            print("The completed aggregate review ZIP is not available. Read the package status above before downloading.")
        else:
            from IPython.display import Image, display
            from google.colab import files
            review_zip = expected_zip
            validation = json.loads((RUN_DIR / "validation_metrics.json").read_text(encoding="utf-8"))
            print(json.dumps({"split": "validation selection diagnostics",
                              "metrics": validation["metrics_at_validation_thresholds"]}, indent=2))
            for plot in sorted((RUN_DIR / "review_plots").glob("*.png")):
                display(Image(filename=str(plot)))
            print("Return results_for_review.zip and calibration.json to the conversation for review.")
            files.download(str(review_zip))
            if "CALIBRATION_FILE" in globals() and CALIBRATION_FILE.is_file():
                files.download(str(CALIBRATION_FILE))
            else:
                print("The separate calibration.json is unavailable. Do not rerun calibration for this saved run.")
'''


def _label_order() -> tuple[str, ...]:
    tree = ast.parse((ROOT / "src/toxic_comments/data.py").read_text(encoding="utf-8"))
    return tuple(next(ast.literal_eval(node.value) for node in tree.body
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "LABELS"
                              for target in node.targets)))


def build_notebook() -> nbformat.NotebookNode:
    cells: list[nbformat.NotebookNode] = []

    def markdown(source: str) -> None:
        cell = nbformat.v4.new_markdown_cell(source.strip())
        cell.id = f"efficient-{len(cells):02d}"
        cells.append(cell)

    def code(source: str) -> None:
        cell = nbformat.v4.new_code_cell(source, execution_count=None, outputs=[])
        cell.id = f"efficient-{len(cells):02d}"
        cells.append(cell)

    markdown('''
# New Colab experiment: efficient full-cohort random RoBERTa

**Author: Mohammad Albatayneh.** Mohammad wrote the original individual project
code, with AI assistance correcting errors. This notebook prepares a **new
experiment** with cached tokenization, dynamic padding, and a GPU-calibrated
microbatch. A completed five-epoch run was returned on 16 October 2026; its
[reviewed validation evidence](../docs/review_20261016.md) is published separately.
Validation selected the checkpoint and thresholds. No end-to-end speedup or
independent neural test result is established.

Open this notebook in [Google Colab](https://colab.research.google.com/), select
**Runtime → Change runtime type → GPU**, then choose **Runtime → Run all**.
Use Python **3.12+** and Colab's installed CUDA-enabled torch **2.8+**. The normal
path mounts Drive, finds or uploads the original `train_preprocessed.csv`,
calibrates once for a new run, trains/resumes, and downloads the aggregate
`results_for_review.zip`. GPU availability and session duration are not guaranteed.

This notebook writes only `/content/toxic-comment-efficient-v1` and a separate
Drive run, initially `random-roberta-efficient-v1-seed42`. The original
`/content/toxic-comment-classification` source folder and `random-roberta-seed42`
run are not reused for training. Their private CSV may be read as an input fallback.
All project code is visible below; no Git checkout, API token, or pretrained
encoder weights are required. Only the public `roberta-base` tokenizer is downloaded.
''')
    markdown('''
## Experiment boundary

| Setting | This new experiment |
|---|---|
| Source | Original `train_preprocessed.csv`; streaming SHA-256 verification |
| Input / eligible rows | 159,571 / 159,564; drop seven empty texts; retain all 93 conflicting duplicate groups |
| Split | Seed 42; normalized-text groups kept intact; train / validation / internal test: 111,731 / 23,878 / 23,955 |
| Targets, in order | __LABEL_ORDER__ |
| Features | Text only; derived target-sum `toxicity` excluded |
| Random model | 8 layers; hidden 512; 8 heads; intermediate 2,048; 51,484,678 parameters |
| Tokenizer / maximum length | Unmodified `roberta-base` tokenizer / 256 tokens |
| Optimizer / schedule | AdamW; learning rate 4e-5; weight decay 0.01; cosine; 10% warmup |
| Training / selection | Up to five epochs; patience three; train-only weighted BCE; lowest validation loss; fresh best-export validation |
| Effective batch / precision | 256; sample-weighted accumulation including partial tail groups; CUDA float16 AMP |
| New execution choices | Cache truncated tokenization; pad each microbatch to its longest sequence; calibrate microbatch 8 / 16 / 32 / 64 with accumulation 32 / 16 / 8 / 4 |
| Other execution settings | Gradient checkpointing False; zero workers; deterministic algorithms; clip gradient norm at 1.0 |
| Saved boundaries | Initial state; each epoch; every 100 optimizer updates or about 300 seconds at completed accumulation boundaries |
| Thresholds / internal test | Six validation-selected F1 thresholds; internal test remains unscored in this notebook |

The cohort, architecture, learning schedule, objective, and validation selection
rules are unchanged. Dynamic padding and a different microbatch can change
floating-point results and random-number consumption. This is a separate
numerical experiment and **cannot resume the old run**. Calibration records the
current GPU observations; it is not evidence of a full-training speedup.

Validation scores are selection diagnostics. The internal split was historically
examined and is exploratory; the unlabeled competition CSV cannot be scored.
'''.replace("__LABEL_ORDER__", ", ".join(f"`{label}`" for label in _label_order())))
    markdown('''
## 1. Set up an isolated source folder

The notebook kernel keeps only paths, small metadata, and standard-library
helpers. It never imports the ML stack or loads a dataset into memory. Child
processes receive the deterministic CUDA and tokenizer settings before imports.
The exact installed torch distribution is constrained during pip installation.
''')
    code(BOOTSTRAP_SOURCE)
    markdown('''
## 2. Restore the exact embedded source

Run all nine visible `%%writefile` cells. They recreate this notebook's code after
every new Colab session. The next cell verifies all bytes against the embedded
SHA-256 manifest. A separate source snapshot is also saved on Drive at launch.
''')
    hashes: dict[str, str] = {}
    for relative in EMBEDDED_FILES:
        source = (ROOT / relative).read_bytes()
        hashes[relative] = hashlib.sha256(source).hexdigest()
        code(f"%%writefile {COLAB_ROOT}/{relative}\n" + source.decode("utf-8"))
    markdown('''
## 3. Install pins and check the GPU in a fresh process

Torch is absent from `requirements-colab.txt`; the constraint preserves the exact
installed CUDA distribution. All pinned imports and the GPU check run in a fresh
child, so previously loaded notebook packages cannot contaminate training.
There is no need to import or repeatedly audit the full CSV in the parent kernel.
''')
    code("import hashlib\nimport importlib.metadata\nimport subprocess\nimport sys\n\n"
         "if not all(name in globals() for name in ('PROJECT_DIR', 'CHILD_ENV', 'TORCH_CONSTRAINT', 'COLAB_TORCH_DISTRIBUTION_VERSION')):\n"
         "    raise RuntimeError('Run step 1 and every step 2 source-writing cell before this dependency cell.')\n"
         "BOOTSTRAP_VERIFIED = False\n"
         f"EMBEDDED_FILES = {EMBEDDED_FILES!r}\n"
         f"EMBEDDED_SHA256 = {hashes!r}\n"
         "for relative in EMBEDDED_FILES:\n"
         "    path = PROJECT_DIR / relative\n"
         "    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != EMBEDDED_SHA256[relative]:\n"
         "        raise RuntimeError(f'Missing or changed source: {relative}. Rerun every source-writing cell in step 2.')\n"
         "print('All nine embedded source files match their manifest.')\n"
         "subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',\n"
         "                '--constraint', str(TORCH_CONSTRAINT), '-r', str(PROJECT_DIR / 'requirements-colab.txt')],\n"
         "               cwd=PROJECT_DIR, env=CHILD_ENV, check=True)\n"
         "if importlib.metadata.version('torch') != COLAB_TORCH_DISTRIBUTION_VERSION:\n"
         "    raise RuntimeError('The installed torch distribution changed. Reconnect to a compatible CUDA runtime and rerun setup.')\n"
         f"VERIFY_CHILD_SOURCE = {VERIFY_CHILD_SOURCE!r}\n"
         "subprocess.run([sys.executable, '-u', '-c', VERIFY_CHILD_SOURCE, COLAB_TORCH_DISTRIBUTION_VERSION],\n"
         "               cwd=PROJECT_DIR, env=CHILD_ENV, check=True)\n"
         "BOOTSTRAP_VERIFIED = True\n"
         "print('Fresh-child dependency and CUDA checks passed; the notebook kernel has not imported the ML stack.')\n")
    markdown('''
## 4. Mount Drive, select input, and launch new training or resume

Keep the same `DRIVE_PROJECT_DIR` and `RUN_NAME` after a disconnect. `RUN_NAME`
must be a plain folder name and cannot be the original `random-roberta-seed42`.
Leave `CSV_ON_DRIVE` empty to reuse `inputs/train_preprocessed.csv`, this run's
private source snapshot, or the original run's read-only source snapshot, in that
order. Otherwise supply an existing CSV path, or upload only the original file
when prompted. Upload bytes are released before calibration and training.

This is the expensive cell. For a new run it performs a bounded GPU calibration
and stores the decision under `diagnostics/<RUN_NAME>/calibration.json`. An
existing saved run restores its complete original configuration and skips
calibration, even if the calibration file is no longer present. No row cap, CPU
fallback, or automatic retry is used. Only the child loads and audits the cohort.

Live combined stdout/stderr and a unique UTC log are saved outside the frozen run
under `diagnostics/<RUN_NAME>/logs/`. On failure, read the actual final child
output and that log. Reconnect and rerun this notebook to resume the last complete
checkpoint with the same source bytes, embedded code, configuration, software,
tokenizer snapshot, CUDA build, and GPU family/capability/memory/count. Unsaved
work since the last completed boundary can be replayed. Never edit identities or
checkpoints to bypass a mismatch. Load only your own trusted checkpoint files.
''')
    code(LAUNCH_SOURCE)
    markdown('''
## 5. Inspect fresh saved status and return aggregate results

This cell is safe to rerun after an interrupted launch. Status is read in a new
subprocess without loading a model. Missing completion files produce an
unfinished message. Packaging verifies a completed freeze before it can download
`results_for_review.zip`; no raw comments, IDs, row arrays, model weights, or
optimizer checkpoints are included. Keep the private Drive run for continuation.

The saved validation results and plots are selection diagnostics from the best
export. Return the aggregate ZIP and the separate `calibration.json` to this
conversation for review. Calibration is outside the frozen run and records the
synthetic probe, without proving end-to-end training speed. This notebook has no
internal-test scoring step or private numerical ZIP option. The unlabeled
`test_preprocessed.csv` is not needed.
''')
    code(RESULTS_SOURCE)
    markdown('''
## Recovery and verification limits

- **Runtime disconnected:** reopen this same notebook, select a compatible GPU,
  run all source/setup cells again, and keep the same Drive root and run name.
  The embedded source snapshot must match; calibration is not rerun for a saved run.
- **One active session per run:** stop any older Colab session using this Drive
  run before launching another. Concurrent sessions can race on checkpoint files.
- **Different software or GPU:** use the recorded environment or choose a new
  run name. Do not overwrite saved configuration, calibration, identity, or code.
- **Calibration or training failure:** inspect the printed child traceback/tail
  and the persistent log. No automatic retry changes the experiment settings.
- **Input hash mismatch:** use the original supplied bytes. Resaving a visually
  identical CSV changes its SHA-256. Do not upload the competition test CSV.
- **Incomplete freeze:** rerun the launch cell with the same run name; the runner
  checks its recovery transaction. Do not manually create completion files.

Local checks cover notebook syntax/format, exact source materialization and
hashes, light parent imports, and mocked new/resume/failure/package lifecycle
guards. The notebook is delivered with cleared outputs. Full-data GPU training,
Colab/Drive services, CUDA memory use, calibration results, wall time, and model
quality have **not** been validated by those local checks.
''')
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "colab": {"name": OUTPUT.name, "provenance": []},
            "accelerator": "GPU",
        },
    )
    nbformat.validate(notebook)
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the saved notebook differs from the current exact sources")
    args = parser.parse_args()
    rendered = nbformat.writes(build_notebook()) + "\n"
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != rendered:
            parser.exit(1, "Efficient Colab notebook is stale; regenerate with scripts/build_efficient_colab_notebook.py.\n")
        print("Efficient Colab notebook matches the current source bytes and builder.")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(rendered, encoding="utf-8")
        print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
