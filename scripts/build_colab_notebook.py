"""Build the self-contained Colab notebook from the reviewed source bytes.

Visible %%writefile cells embed the six frozen core modules, the Colab runner,
and standalone requirements. No GitHub checkout or encoded source is needed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "03_colab_full_training.ipynb"
COLAB_ROOT = "/content/toxic-comment-classification"
EMBEDDED_FILES = (
    "src/toxic_comments/__init__.py",
    "src/toxic_comments/artifacts.py",
    "src/toxic_comments/baseline.py",
    "src/toxic_comments/data.py",
    "src/toxic_comments/metrics.py",
    "src/toxic_comments/neural.py",
    "scripts/colab_train.py",
    "requirements-colab.txt",
)


def build_notebook() -> nbformat.NotebookNode:
    cells = []
    data_tree = ast.parse((ROOT / "src/toxic_comments/data.py").read_text(encoding="utf-8"))
    label_order = next(ast.literal_eval(node.value) for node in data_tree.body
                       if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "LABELS"
                               for target in node.targets))

    def markdown(source: str) -> None:
        cell = nbformat.v4.new_markdown_cell(source.strip())
        cell.id = f"cell-{len(cells):02d}"
        cells.append(cell)

    def code(source: str) -> None:
        cell = nbformat.v4.new_code_cell(source, execution_count=None, outputs=[])
        cell.id = f"cell-{len(cells):02d}"
        cells.append(cell)

    markdown("""
# Full training: original random eight-layer RoBERTa

**Author: Mohammad Albatayneh.** The original individual project code was written
by Mohammad, with AI assistance correcting errors. This notebook packages the
repaired experiment for Google Colab; it does not claim a completed full GPU run.

Open this `.ipynb` in [Google Colab](https://colab.research.google.com/), select
**Runtime → Change runtime type → GPU**, then run the cells in order. You need the
original **`train_preprocessed.csv`**, Python **3.12+**, CUDA-enabled torch **2.8+**,
and enough Google Drive space for private checkpoints and model exports.

All executable project source is visibly embedded below. No published repository,
Git clone, API token, pretrained encoder weights, or paid service is required.
Only the public `roberta-base` **tokenizer** is downloaded. A free Colab GPU's
availability, memory, and session duration are not guaranteed.

The normal workflow trains the full eligible cohort and returns
**`results_for_review.zip`**. Keep the optional internal-test switch **False** for
the first results review. Do not upload the unlabeled competition test CSV.
""")
    markdown("""
## Fixed experiment and data boundary

| Setting | Value |
|---|---|
| Source | Original supplied `train_preprocessed.csv`, SHA-256 checked |
| Input / eligible rows | 159,571 / 159,564; explicitly drop seven empty texts |
| Conflicting annotations | Retain all 93 conflicting duplicate groups |
| Split | Seed 42; normalized text groups kept intact; nominal 70/15/15 |
| Actual train / validation / internal test | 111,731 / 23,878 / 23,955 |
| Targets, in order | __LABEL_ORDER__ |
| Excluded derived column | `toxicity` is a target sum; never an input feature |
| Random encoder | 8 layers, hidden 512, 8 heads, intermediate 2,048 |
| Tokenizer / sequence length | Unmodified `roberta-base` / 256 tokens |
| Parameters | 51,484,678; model weights initialized randomly |
| Optimizer / schedule | AdamW, learning rate 4e-5, weight decay 0.01; cosine, 10% warmup |
| Training | At most five epochs; patience three; weighted BCE, train-only weights |
| Batches / precision | Microbatch 8 × accumulation 32; CUDA float16 AMP |
| Stability | Zero workers; deterministic algorithms; gradient norm clip 1.0 |
| Selection | Minimum validation loss; reload/export best checkpoint; fresh validation |
| Thresholds | Six validation-selected F1 thresholds, then frozen |

The CSV column order need not match this runtime order: labels are selected by
column name and every artifact records the explicit order above.

Validation scores are **selection diagnostics**. The internal exploratory test
stays unscored by default and is not a pristine external test: this cohort was
examined historically. The competition test CSV has no labels and cannot be scored.
""".replace("__LABEL_ORDER__", ", ".join(f"`{label}`" for label in label_order)))
    markdown("""
## 1. Check the GPU runtime and create the temporary source folder

This cell refuses a CPU-only runtime. It preserves the torch CUDA build already
installed by Colab. Environment variables are set before CUDA is initialized.
""")
    code("""import importlib.metadata
import os
import sys
from pathlib import Path
from packaging.version import Version

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if sys.version_info < (3, 12):
    raise RuntimeError("Select a Python 3.12+ Colab runtime before installing these pins.")
import torch
if Version(torch.__version__.split("+")[0]) < Version("2.8.0"):
    raise RuntimeError("Use a Colab runtime with CUDA-enabled torch >=2.8; do not install a CPU wheel.")
if not torch.cuda.is_available():
    raise RuntimeError("Select Runtime > Change runtime type > GPU, reconnect, and rerun this cell.")
COLAB_TORCH_VERSION = torch.__version__
COLAB_TORCH_DISTRIBUTION_VERSION = importlib.metadata.version("torch")
PROJECT_DIR = Path("/content/toxic-comment-classification")
for relative in ("src/toxic_comments", "scripts"):
    (PROJECT_DIR / relative).mkdir(parents=True, exist_ok=True)
print({"python": sys.version.split()[0], "torch": COLAB_TORCH_VERSION,
       "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
""")
    markdown("""
## 2. Materialize the reviewed source

Run all eight visible `%%writefile` cells. They reproduce the exact repository
source bytes; the following SHA-256 manifest checks every embedded file.
""")
    hashes = {}
    for relative in EMBEDDED_FILES:
        source = (ROOT / relative).read_bytes()
        hashes[relative] = hashlib.sha256(source).hexdigest()
        code(f"%%writefile {COLAB_ROOT}/{relative}\n" + source.decode("utf-8"))
    markdown("""
## 3. Verify source and install the other pinned dependencies

Torch is deliberately absent from this requirements file. Installation changes
only the listed non-torch dependencies and their dependencies. If Colab reports
that already imported packages need a session restart, restart the session and
rerun from cell 1. Persisted Drive files and checkpoints remain available.
""")
    code("import hashlib\n"
         f"EMBEDDED_FILES = {EMBEDDED_FILES!r}\n"
         f"EMBEDDED_SHA256 = {hashes!r}\n"
         "for relative in EMBEDDED_FILES:\n"
         "    observed = hashlib.sha256((PROJECT_DIR / relative).read_bytes()).hexdigest()\n"
         "    if observed != EMBEDDED_SHA256[relative]:\n"
         "        raise RuntimeError(f'Embedded source changed: {relative}')\n"
         "print('All eight embedded source files match their manifest.')\n")
    code("""import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                "-r", str(PROJECT_DIR / "requirements-colab.txt")], check=True)
""")
    code("""import importlib.metadata

expected_versions = {}
for line in (PROJECT_DIR / "requirements-colab.txt").read_text().splitlines():
    if "==" in line and not line.lstrip().startswith("#"):
        name, version = line.strip().split("==", 1)
        expected_versions[name] = version
for name, version in expected_versions.items():
    observed = importlib.metadata.version(name)
    if observed != version:
        raise RuntimeError(f"Installed {name} {observed}; expected {version}.")
module_names = {"scikit-learn": "sklearn", "huggingface-hub": "huggingface_hub"}
for name, version in expected_versions.items():
    imported_module = sys.modules.get(module_names.get(name, name))
    loaded_version = getattr(imported_module, "__version__", None)
    if loaded_version is not None and str(loaded_version) != version:
        raise RuntimeError(f"{name} {loaded_version} is still imported after installing {version}. "
                           "Restart the Colab session, then rerun this notebook from cell 1.")
if importlib.metadata.version("torch") != COLAB_TORCH_DISTRIBUTION_VERSION:
    raise RuntimeError("The torch build changed. Reconnect to a compatible CUDA runtime and rerun setup.")

sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))
from scripts import colab_train as colab_runner
from toxic_comments import neural
dependencies = neural._dependencies()
colab_runner._assert_environment(dependencies, smoke=False, config=colab_runner.ColabConfig())
print("Pinned environment imported successfully; Colab's CUDA torch was preserved.")
print(expected_versions)
""")
    markdown("""
## 4. Mount Drive and select the original training CSV

The **run folder must stay on Google Drive** so a runtime disconnect can resume
from its last complete checkpoint. Keep the same `RUN_NAME` to resume; choose a
new name for a separate experiment. Use a plain folder name without slashes.

Leave `CSV_ON_DRIVE` empty to upload the CSV once. Subsequent sessions reuse the
saved run's private source snapshot or the remembered input. To use a file already
on Drive, enter its full `/content/drive/MyDrive/.../train_preprocessed.csv` path.
The CSV is copied privately into the run and never included in the review ZIP.
""")
    code("""from google.colab import drive, files
drive.mount("/content/drive")

DRIVE_PROJECT_DIR = "/content/drive/MyDrive/toxic-comment-classification" # @param {type:"string"}
RUN_NAME = "random-roberta-seed42" # @param {type:"string"}
CSV_ON_DRIVE = "" # @param {type:"string"}
if not RUN_NAME or RUN_NAME in {".", ".."} or "/" in RUN_NAME or "\\\\" in RUN_NAME:
    raise ValueError("RUN_NAME must be one plain folder name.")
PERSISTENT_ROOT = Path(DRIVE_PROJECT_DIR).expanduser().resolve()
if not PERSISTENT_ROOT.is_relative_to(Path("/content/drive/MyDrive")):
    raise ValueError("Use a persistent folder under /content/drive/MyDrive.")
PERSISTENT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_DIR = PERSISTENT_ROOT / "runs" / RUN_NAME
run_source = RUN_DIR / "private" / "source.csv"
remembered_source = PERSISTENT_ROOT / "inputs" / "train_preprocessed.csv"
if CSV_ON_DRIVE.strip():
    DATA_FILE = Path(CSV_ON_DRIVE).expanduser().resolve()
elif run_source.is_file():
    DATA_FILE = run_source
elif remembered_source.is_file():
    DATA_FILE = remembered_source
else:
    uploaded = files.upload()
    if len(uploaded) != 1 or "train_preprocessed.csv" not in uploaded:
        raise ValueError("Upload only the original train_preprocessed.csv, with its original filename.")
    csv_bytes = uploaded["train_preprocessed.csv"]
    if hashlib.sha256(csv_bytes).hexdigest() != colab_runner.FULL_SOURCE_SHA256:
        raise ValueError("Uploaded bytes differ from the original supplied training CSV.")
    remembered_source.parent.mkdir(parents=True, exist_ok=True)
    remembered_source.write_bytes(csv_bytes)
    del uploaded, csv_bytes
    DATA_FILE = remembered_source
if not DATA_FILE.is_file():
    raise FileNotFoundError("Training CSV not found at the chosen Drive path.")
if hashlib.sha256(DATA_FILE.read_bytes()).hexdigest() != colab_runner.FULL_SOURCE_SHA256:
    raise ValueError("The selected CSV is not the original supplied train_preprocessed.csv.")
print({"source_verified": True, "run_name": RUN_NAME, "existing_run": RUN_DIR.exists(),
       "persistent_checkpoints": True})
""")
    markdown("""
## 5. Audit the complete cohort and show the training plan

This reads the complete CSV and prints aggregate counts only. Row counts, source
hash, empty-text policy, conflicting annotations, and split counts must all match
the supplied dataset before training begins. No raw comments or identifiers are
displayed. The runner performs the same checks again at run creation/resume.

Gradient checkpointing is an optional memory-saving implementation choice. The
default is False. To enable it, do so **before a new run**; it cannot change during
resume. The original architecture, sequence length, and cohort remain fixed.
""")
    code("""import math
from dataclasses import asdict
from toxic_comments.data import load_dataset, split_dataset

GRADIENT_CHECKPOINTING = False # @param {type:"boolean"}
CONFIG = colab_runner.ColabConfig(gradient_checkpointing=GRADIENT_CHECKPOINTING)
audit_dataset = load_dataset(DATA_FILE, conflict_policy="keep", empty_text_policy="drop")
audit_splits = split_dataset(audit_dataset, seed=42)
colab_runner._verify_full_data(audit_dataset, audit_splits, CONFIG, smoke=False)
print({"input_rows": audit_dataset.summary["input_rows"],
       "eligible_rows": audit_dataset.summary["retained_rows"],
       "excluded_empty_text_rows": audit_dataset.summary["excluded_empty_text_rows"],
       "conflicting_label_groups": audit_dataset.summary["conflicting_label_groups"],
       "split_rows": {name: len(indices) for name, indices in audit_splits.as_dict().items()},
       "duplicate_group_overlap": 0})
updates_per_epoch = math.ceil(math.ceil(len(audit_splits.train) / CONFIG.training.batch_size)
                              / CONFIG.training.accumulation_steps)
print({"label_order": list(colab_runner.LABELS),
       "random_architecture": neural.ORIGINAL_ARCHITECTURE,
       "expected_parameters": 51484678, "training": asdict(CONFIG.training),
       "planned_updates_per_epoch": updates_per_epoch,
       "maximum_optimizer_updates": updates_per_epoch * CONFIG.training.epochs,
       "checkpoint_every_updates": CONFIG.checkpoint_every_updates,
       "gradient_checkpointing": CONFIG.gradient_checkpointing})
del audit_dataset, audit_splits
""")
    markdown("""
## 6. Train or resume the original model

This is the expensive cell: it trains the full eligible cohort on the GPU.
There is no hidden row cap or CPU fallback. The loop prints progress at least
about every 30 seconds and saves at every 100 actual optimizer updates plus each
epoch boundary. Recovered work starts at the saved accumulation boundary; at
most the unsaved interval must be replayed after a disconnect.

The atomic `private/latest_checkpoint.pt` contains the model, AdamW optimizer,
cosine scheduler, AMP scaler, Python/NumPy/torch CPU/CUDA RNG, deterministic epoch
shuffle offset, history, and patience state. `private/best_checkpoint.pt` retains
the lowest validation-loss model. Exact continuation requires the **same source
bytes, code, configuration, tokenizer snapshot, software versions, torch CUDA
build, GPU family/capability/memory, and GPU count**. A mismatch fails explicitly;
never edit the saved identity to bypass it.

If disconnected, reopen this same notebook, reconnect to a compatible GPU,
rerun setup, mount Drive, retain `RUN_NAME` and configuration, then rerun this
cell. Full training runs in a fresh Python subprocess so dependencies replaced
by pip cannot leave stale notebook imports in the training process. No tokenizer
download is needed when resuming its saved local snapshot.
Torch checkpoints should come only from your own trusted run folder.
""")
    code("""import json

# Fresh process; no --evaluate flag. Internal-test scoring stays disabled.
training_command = [sys.executable, "-u", str(PROJECT_DIR / "scripts/colab_train.py"),
                    "develop", "--data", str(DATA_FILE), "--run-dir", str(RUN_DIR)]
if CONFIG.gradient_checkpointing:
    training_command.append("--gradient-checkpointing")
subprocess.run(training_command, cwd=PROJECT_DIR, check=True)
saved_history = json.loads((RUN_DIR / "training_history.json").read_text())
print({"training_completed": True,
       "global_optimizer_steps": saved_history["epochs"][-1]["optimizer_steps"],
       "best_checkpoint": saved_history["best_checkpoint"],
       "test_evaluated": (RUN_DIR / "public_test_results.json").is_file()})
""")
    markdown("""
## 7. Inspect aggregate results and prepare the return ZIP

The exported model is reloaded from the best checkpoint before fresh validation
predictions and threshold selection. These are newly run validation selection
diagnostics; compare them with the existing CPU baseline only under this repaired
protocol. They are separate from historical rounded neural notebook logs.

`results_for_review.zip` contains aggregate metrics, thresholds, loss history,
plots, configuration, and versions/hardware. It excludes raw comments, IDs,
per-row arrays, and model/optimizer weights. Set the optional private switch
True only if row-level numerical verification is needed; its separate ZIP has
labels/probabilities/split indices, **no raw texts, original IDs, or weights**.
Never commit or publicly share that private ZIP.
""")
    code("""import json
from IPython.display import display, Image

MAKE_PRIVATE_VERIFICATION = False # @param {type:"boolean"}
review_packages = colab_runner.package_results(
    RUN_DIR, verification_private=MAKE_PRIVATE_VERIFICATION,
)
validation = json.loads((RUN_DIR / "validation_metrics.json").read_text())
selected_metrics = validation["metrics_at_validation_thresholds"]
print(json.dumps({"split": "validation selection diagnostics",
                  "metrics": selected_metrics,
                  "test_evaluated": (RUN_DIR / "public_test_results.json").is_file()}, indent=2))
for plot in sorted((RUN_DIR / "review_plots").glob("*.png")):
    display(Image(filename=str(plot)))
print({name: Path(path).name for name, path in review_packages.items()})
""")
    markdown("""
## 8. Optional internal exploratory test — keep False initially

Return the development ZIP first so checkpoint/threshold results can be reviewed
before improvement decisions. Only turn this switch True for a separately
requested frozen evaluation. It scores the 23,955-row internal split once using
the frozen best export and validation thresholds; it never scores the unlabeled
competition CSV. The core records a single-use evaluation marker before access.
An interrupted test must not be silently rerun.

A later explicit test creates `results_for_review_with_test.zip` and, if requested,
`verification_private_with_test.zip`. The original development bundles are kept.
Changing the switch back to False cannot undo an evaluation already performed.
""")
    code("""START_INTERNAL_TEST = False # @param {type:"boolean"}
if START_INTERNAL_TEST:
    if (RUN_DIR / "public_test_results.json").is_file():
        test_result = json.loads((RUN_DIR / "public_test_results.json").read_text())
        print("Internal test already completed; displaying the saved aggregate result.")
    else:
        test_result = neural.evaluate_run(RUN_DIR, device="cuda")
    print(json.dumps(test_result, indent=2))
    review_packages = colab_runner.package_results(
        RUN_DIR, verification_private=MAKE_PRIVATE_VERIFICATION,
    )
else:
    print("No new internal-test evaluation requested. Return the aggregate development ZIP first.")
""")
    markdown("""
## 9. Download the aggregate result and return it for review

Upload **`results_for_review.zip`** back to this conversation after training.
If a test was explicitly requested, return the bundle with `_with_test` in its
name. The optional private numerical bundle is for private review only. Leave
CSV data, checkpoints, and model weights in your Drive run folder.
""")
    code("""files.download(review_packages["results_for_review"])
if MAKE_PRIVATE_VERIFICATION:
    files.download(review_packages["verification_private"])
""")
    markdown("""
## Troubleshooting and verification limits

- **No GPU / old Python or torch:** choose a compatible Colab runtime and rerun
  setup. Do not replace its CUDA torch with the CPU wheel used by local tests.
- **Dependency import failure after installation:** restart the session and
  rerun setup. The notebook verifies all pinned versions before training.
- **Out of memory:** end the failed attempt. Enable gradient checkpointing for
  a **new run name** and keep all original experiment settings. Do not silently
  reduce the cohort, length, architecture, or accumulation settings.
- **Resume identity mismatch:** reconnect to the recorded software/GPU family
  or start a new clearly separate run. Do not rewrite identity/checkpoint files.
- **Drive serialization error or disconnect:** reconnect and resume. Only the
  last complete checkpoint is usable; durability also depends on Google Drive.
- **Source mismatch:** use the original CSV bytes. Resaving a CSV changes its
  SHA-256 even when displayed values appear identical.

Locally verified: exact embedded source and notebook structure, tiny **offline
CPU synthetic** training, mid-epoch interruption/resume, best export and fresh
validation, numerical checkpoint/array consistency, and test/privacy guards.
The full real-data model, Colab services, CUDA execution, memory use, and final
model quality have **not** been exercised locally. Do not treat CPU synthetic
scores as model-quality evidence or guarantee GPU wall time.
""")
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
    parser.add_argument("--check", action="store_true", help="Verify the saved notebook matches current reviewed sources")
    args = parser.parse_args()
    rendered = nbformat.writes(build_notebook()) + "\n"
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != rendered:
            parser.exit(1, "Colab notebook is stale; regenerate with scripts/build_colab_notebook.py.\n")
        print("Colab notebook matches the current source bytes and builder.")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(rendered, encoding="utf-8")
        print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
