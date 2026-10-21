# Canonical Colab training workflow

Open [`notebooks/04_colab_efficient_training.ipynb`](../notebooks/04_colab_efficient_training.ipynb)
in [Google Colab](https://colab.research.google.com/), choose a GPU runtime with
Python 3.12+ and its installed CUDA-enabled torch 2.8+, then **Run all**.
The notebook visibly embeds all six core modules, the resumable runner, the new
resource-profile launcher, and standalone requirements. No Git checkout or
pretrained encoder weights are needed; only the `roberta-base` tokenizer is
downloaded. The publication notebook keeps its outputs cleared; the completed
user run is recorded separately in the [16 October review](review_20261016.md)
and [aggregate evidence](../reports/colab_efficient_v1/validation_metrics.json).
Notebook 04 replaces the retired notebook 03 for training. A completed run does
not guarantee that a future Colab allocation will have the same runtime or finish
within a session.

Mohammad Albatayneh wrote the original individual project code. This execution profile preserves the scientific settings
below and uses a separate numerical experiment because dynamic padding and
microbatch changes can alter floating-point results and random-number consumption.

## Run and resume

1. Run setup and every visible source-writing cell. Source bytes are checked
   against the embedded SHA-256 manifest. The temporary project is always
   `/content/toxic-comment-efficient-v1`; the original
   `/content/toxic-comment-classification` project is never overwritten.
2. Dependencies are installed with a constraint for the actual installed torch
   distribution version, including its CUDA build suffix. A fresh child verifies
   pinned imports, Python, torch, and CUDA. The notebook parent imports only the
   standard library, Google Colab helpers, and IPython display helpers. It does
   not import torch, NumPy, pandas, or project modules, or load the full dataset.
3. Mount Drive in the launch cell. Keep the default Drive project root or choose
   a folder under `/content/drive/MyDrive`. The initial new run is
   `runs/random-roberta-efficient-v1-seed42`. The original
   `random-roberta-seed42` name is explicitly rejected. Existing folders without
   a complete run identity are not overwritten.
4. Leave `CSV_ON_DRIVE` empty to reuse, in order, the project's
   `inputs/train_preprocessed.csv`, this new run's private source snapshot, or
   the original run's `private/source.csv`. The original run is read only.
   A manual `CSV_ON_DRIVE` path takes precedence. If none exists, upload only the
   original `train_preprocessed.csv`. SHA-256 is streamed from disk; upload bytes
   are freed before ML work begins. The required original SHA-256 is
   `bb7a9e1af5282c4c05bf09bf3be3373c4160fe3272a4b26016034ff4bcea5f1e`.
5. A new run calibrates the actual GPU once. Disposable random models receive
   synthetic full-length 256-token inputs; no real validation/test data is used.
   Candidate microbatches are 8, 16, 32, and 64, with corresponding accumulation
   counts 32, 16, 8, and 4. The selected pair keeps effective batch 256. The
   measured records and choice are saved at
   `diagnostics/<RUN_NAME>/calibration.json`. A saved pre-run calibration is
   validated and reused; incompatible records are not overwritten.
6. The training child loads and audits the full eligible cohort, then trains,
   exports the best model, runs fresh validation, selects thresholds, and freezes
   the completed run. Combined stdout/stderr streams live and is copied to a
   unique UTC log under `diagnostics/<RUN_NAME>/logs/`, outside the frozen run.
   Failures show the actual child output tail and never automatically retry.
7. After a disconnect, reopen this same notebook, select a compatible GPU, run
   setup/source cells again, and keep the same Drive root and run name. The
   complete configuration is restored from `private/run_identity.json`.
   Calibration is skipped on resume, even if its separate diagnostics file has
   gone missing. The saved source snapshot under
   `diagnostics/<RUN_NAME>/source_snapshot/` must still match the notebook.
8. The last cell reads fresh status in a subprocess. It packages and downloads
   `results_for_review.zip` only after `manifest.json`, `manifest.sha256`, and
   `training_history.json` exist and the package command verifies the freeze.
   An unfinished run gets a resume message instead of missing-file errors. Return
   the aggregate ZIP and the separate `calibration.json` to the conversation;
   keep private checkpoints and data on Drive. If calibration diagnostics are
   missing on resume, do not rebenchmark the saved run. This notebook offers no
   test-scoring step or private ZIP option. The unlabeled
   `test_preprocessed.csv` is not needed.

The checkpoint saves model, optimizer, scheduler, AMP scaler, RNG states, epoch,
microbatch offset, history, and patience state at completed accumulation
boundaries. It saves the initial state, each epoch boundary, and after 100 actual
optimizer updates or about 300 seconds. Unsaved work can be replayed. Exact resume
requires the original source bytes, source code, entire configuration, tokenizer
snapshot, software/CUDA build, and recorded GPU family/capability/memory/count.
Do not edit saved identities, calibration, or checkpoints to bypass a mismatch.
Only load checkpoints created by your own trusted run.
Use only one active training session for a given Drive run. Stop an older Colab
session before starting another so they cannot race on checkpoint files.

## Completed returned run

The returned run completed all five epochs on 16 October 2026. Minimum validation
loss selected epoch four (0.36014178103214034); the returned history records the
same fresh-export validation loss. Its 23,878-row validation set reports macro AP
0.5259, macro ROC-AUC 0.9761, macro F1 0.5094, and micro F1 0.6520 at the saved
validation-selected thresholds. The internal test remains unscored.

The recorded runtime was Python 3.13.15, torch 2.11.0+cu130, CUDA 13.0, cuDNN
92700, and one Tesla T4 with 15,637,086,208 bytes of GPU memory. Synthetic resource
calibration chose microbatch 64 with accumulation four. The complete installed
dependency versions are in
[`versions_hardware.json`](../reports/colab_efficient_v1/versions_hardware.json);
these are the recorded completed-run versions, distinct from the local Python 3.12 /
torch 2.8.0+cpu engineering environment. Resume requires the completed run's own
recorded identity, rather than substituting the local environment.

Validation selected both checkpoint and thresholds. Five labels selected the
0.95 grid ceiling, while `toxic` selected 0.90; review preserved those decisions.
`threat` remains weak (AP 0.0753, F1 0.1081; 62 validation positives). The
[numerical verification](../reports/colab_efficient_v1/verification.json) records
which results can be recomputed from returned arrays and what remains unverified.
Review did not retrain, run GPU inference, retune thresholds, or score the internal
test. The returned model weights and tokenizer snapshot were not supplied for
independent export/reload verification. No end-to-end training speedup or stronger
moderation claim follows from the short calibration.

## Scientific profile and execution changes

| Item | Efficient profile |
|---|---|
| Full source / retained cohort | 159,571 / 159,564; drop seven empty texts; retain 93 conflicting duplicate groups |
| Split and target order | Seed 42; group-disjoint 111,731 / 23,878 / 23,955; `identity_hate`, `insult`, `obscene`, `severe_toxic`, `threat`, `toxic` |
| Inputs | Text only; exclude the derived target sum `toxicity` |
| Model | Random eight-layer RoBERTa; hidden 512; eight heads; intermediate 2,048; 51,484,678 parameters |
| Tokenizer and truncation | Unmodified `roberta-base`; maximum 256 tokens |
| Loss and optimizer | Training-only class-weighted BCE; AdamW 4e-5; weight decay 0.01 |
| Schedule and selection | Cosine; 10% warmup; at most five epochs; patience three; minimum validation loss; fresh best-export validation |
| Effective batch and precision | 256; sample-weighted accumulation including the last partial group; CUDA float16 AMP |
| Tokenization and padding changes | Cache truncated examples once per process; pad each microbatch to its longest sequence |
| Batch change | Calibrated 8 / 16 / 32 / 64 microbatch; matching accumulation 32 / 16 / 8 / 4 |
| Other execution settings | Gradient checkpointing False; zero workers; deterministic algorithms; gradient clipping 1.0 |
| Thresholds and internal test | Six validation-selected F1 thresholds; internal test unscored |

The old notebook's execution source remains in Git history for recovery of its
own fixed-microbatch run; the current 03 file is a retirement notice. Notebook 04
cannot continue an old optimizer checkpoint. Calibration measures
short synthetic resource probes, not end-to-end full-cohort speed or accuracy.
Its memory margin does not guarantee that later CUDA/Drive operations will fit or
finish within a session. No smaller cohort, sequence limit, architecture, or
silent precision fallback is used.

Validation selected both the checkpoint and thresholds, so its scores are
selection diagnostics. The historically examined internal test is exploratory.
The competition test CSV is unlabeled and is never used for scoring here.

## Local validation boundary

`tests/test_efficient_notebook.py` checks notebook format and transformed cell
syntax, cleared outputs, exact source bytes and SHA-256 manifest, actual
`%%writefile` materialization, parent import restrictions, torch-preserving pip
arguments, and mocked Colab new/resume/input/failure/package lifecycles. Source
and profile tests elsewhere cover the training mechanics separately. These
checks do not execute Colab/Drive, run a real GPU calibration, or train the full
dataset. They establish local engineering behavior. The returned artifacts above
provide separate evidence from the user's GPU experiment; they do not turn these
CPU checks into GPU tests or establish end-to-end wall-time comparisons.

Regenerate after an authorized source change, then check that the saved notebook
matches the current reviewed code:

```bash
python scripts/build_efficient_colab_notebook.py
python scripts/build_efficient_colab_notebook.py --check
python -m unittest discover -s tests -p 'test_efficient_notebook.py' -v
```

Do not regenerate a notebook and use its different code to force resume of an
existing run. Resume with the original notebook/source bytes, or start a new run.

## Implementation references

- [PyTorch performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html): memory transfers, synchronization and resource tuning.
- [Transformers 4.57.1 padding and truncation](https://huggingface.co/docs/transformers/v4.57.1/en/pad_truncation): sequence-length handling.
- [PyTorch 2.8 checkpoint loading](https://docs.pytorch.org/docs/2.8/generated/torch.load.html): CPU mapping, restricted loading and metadata inspection.
- [Colab resource and Drive limitations](https://research.google.com/colaboratory/faq.html): variable resource availability and mounted-storage I/O.
