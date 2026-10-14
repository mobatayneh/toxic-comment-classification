# Train the original model in Google Colab

Use [`notebooks/03_colab_full_training.ipynb`](../notebooks/03_colab_full_training.ipynb)
for the complete original random-initialized eight-layer RoBERTa experiment.
It embeds all six reviewed `src/toxic_comments` modules, the resumable runner,
and standalone pinned requirements in visible `%%writefile` cells. It does not
need a published GitHub repository. Only the public `roberta-base` tokenizer is
downloaded; no pretrained encoder weights are loaded.

Mohammad Albatayneh wrote the original individual project code, with AI
assistance correcting errors. This package repairs execution and reproducibility
while retaining the original model architecture. Full GPU training has not been
performed in the preparation environment.

## Run it

1. Upload the `.ipynb` to [Google Colab](https://colab.research.google.com/).
   Choose **Runtime → Change runtime type → GPU**. Use Python 3.12+ and the
   CUDA-enabled torch 2.8+ supplied by the runtime. The notebook verifies both
   and installs only the other pinned dependencies. If installation requests a
   session restart, restart it and rerun setup. An explicit imported-version
   guard stops if an old package is still loaded after installation.
2. Run the eight source-writing cells and the source manifest check. Every
   embedded file must match its SHA-256. Imports and installed versions are
   checked before data access/training.
3. Mount Google Drive. Keep a persistent `DRIVE_PROJECT_DIR` and choose a plain
   `RUN_NAME`, initially `random-roberta-seed42`. Allow several GB of free space
   for the input, model export, latest optimizer checkpoint, best weights, and
   temporary checkpoint files. Colab GPU availability/session limits vary.
4. Leave `CSV_ON_DRIVE` empty to upload the original `train_preprocessed.csv`
   once, or supply its full Drive path. A reconnect automatically reuses that
   run's private source snapshot or the remembered input. Do not upload
   `test_preprocessed.csv`; its labels are missing.
5. Run the aggregate data-audit cell. It checks the original source SHA-256 and
   exact full-cohort row/split expectations; no raw comments or identifiers are
   printed. Keep `GRADIENT_CHECKPOINTING=False` unless selecting it explicitly
   before a new run for memory reasons.
6. Run the training cell. It trains/resumes on the GPU, prints progress, exports
   the best validation-loss model, reloads the export, computes fresh validation
   probabilities, selects thresholds, and freezes the developed run. Training
   runs in a fresh Python subprocess with the installed packages.
7. Keep `MAKE_PRIVATE_VERIFICATION=False` and `START_INTERNAL_TEST=False` for the
   first review. Download `results_for_review.zip` and return it to the
   conversation. Preserve the private Drive run folder for continuation and
   integrity checks.

The training cell is expensive and runs the full protocol when reached. The
published notebook has no saved outputs and does not claim successful Colab
execution. There is no hidden row cap, alternate architecture, or CPU fallback.

## Experiment contract

| Item | Required setting |
|---|---|
| Original source SHA-256 | `bb7a9e1af5282c4c05bf09bf3be3373c4160fe3272a4b26016034ff4bcea5f1e` |
| Input / eligible rows | 159,571 / 159,564; drop seven whitespace-only comments |
| Duplicate annotations | Retain 93 conflicting normalized duplicate groups; no label merging |
| Split | Seed 42; normalized text groups kept together; nominal 70/15/15 |
| Actual train / validation / internal test | 111,731 / 23,878 / 23,955; zero duplicate-group overlap |
| Target order | `identity_hate`, `insult`, `obscene`, `severe_toxic`, `threat`, `toxic` |
| Features | Text only; derived `toxicity` target sum excluded |
| Initialization | All model weights random; tokenizer only is pretrained |
| Architecture | Eight layers; hidden 512; eight heads; intermediate 2,048; 51,484,678 parameters |
| Token length | 256; unmodified 50,265-token `roberta-base` tokenizer |
| AdamW | Learning rate 4e-5; weight decay 0.01 |
| Schedule / stopping | Cosine; 10% warmup; at most five epochs; patience three |
| Batching | Microbatch eight; accumulation 32; nominal effective batch 256 |
| Tail groups | Sample-weighted loss by actual group size, including the final partial group |
| Precision / workers | CUDA float16 AMP with scaler; zero workers |
| Selection | Lowest sample-weighted validation BCE; fresh best-export validation |
| Class weights / thresholds | Class weights from training only; F1 thresholds from validation only |
| Internal exploratory test | Disabled initially; a separate explicit single-use evaluation |

The CSV column order need not match the runtime target order. The loader selects
labels by column name; model mappings, arrays, thresholds, and result artifacts
all use and record the fixed order above.

The planned five-epoch schedule has 437 optimizer updates per epoch and 2,185
maximum updates before early stopping/AMP overflow skips. Validation selected
the checkpoint and thresholds, so its scores are selection diagnostics. The
internal test is exploratory because this cohort was historically examined;
the unlabeled competition test cannot supply model-quality evidence.

## Resume after interruption

The runner saves `private/latest_checkpoint.pt` after every 100 actual optimizer
updates and at each epoch boundary. It also saves the initial state. Files are
serialized/flushed to a sibling temporary file and atomically replaced; an
incomplete write does not replace the previous complete checkpoint. Storage
durability still depends on Drive. Work since the last saved boundary can be
replayed after a session interruption.

The checkpoint contains model/AdamW/scheduler/AMP-scaler state, Python and NumPy
RNG, torch CPU/CUDA RNG, epoch, consumed microbatch offset, loss accounting,
optimizer update count, best-epoch metadata, history, and patience state. Each
epoch uses its own deterministic shuffle generator. Resume reconstructs the
loader, skips completed microbatches, then restores continuation RNG before the
next unseen forward pass. No partial gradient-accumulation group is resumed.

Reopen the same notebook, reconnect, run setup again, mount Drive, retain the
same run name and configuration, and run the training cell. Resume checks exact
source bytes, all core/runner code hashes, split indices, saved tokenizer files,
software versions, torch CUDA build, CUDA/cuDNN details, and GPU family,
capability, total memory, and GPU count. The GPU serial/UUID is not recorded.
Colab assigning a different GPU family will fail the identity check. Use the
recorded environment or start a clearly separate new run; never edit identities
or checkpoints to force compatibility.

The notebook deliberately enforces the exact supplied preprocessed CSV bytes.
This mirror guard is narrower than the core neural CLI, which can develop a
separate experiment from a schema-valid labeled CSV under its explicit policies.
Do not confuse a different official-source CSV with this unchanged source
snapshot or transfer results between them without a recorded new experiment.

The best model weights are saved separately in `private/best_checkpoint.pt`.
After training, the runner restores that model, exports/reloads it, and verifies
that fresh validation loss reproduces the selected best loss. A freeze manifest
protects source, model export, split, thresholds, and development evidence from
accidental changes. The latest optimizer checkpoint remains private for recovery.
Load only checkpoints produced by your own trusted run.

## What to return

| File | Contents | Sharing boundary |
|---|---|---|
| `results_for_review.zip` | Aggregate validation metrics at 0.5 and selected thresholds, per-label metrics, thresholds, loss history, configuration, data counts, versions/hardware, and plots | Return for review; no raw comments, IDs, row arrays, or weights |
| `verification_private.zip` (optional) | Validation labels/probabilities/row indices, split indices, aggregate evidence, and hashes | Private numerical verification only; never publish or commit |
| `results_for_review_with_test.zip` | Aggregate development evidence plus an explicitly requested saved internal-test result | Later test review only; preserves the original development ZIP |
| `verification_private_with_test.zip` (optional) | Private numerical verification plus internal-test arrays | Private only; preserves the original private development ZIP |
| Run folder | Private source CSV, tokenizer snapshot, checkpoints, model weights/export, manifests, and numerical evidence | Keep on Drive; do not publish or commit |

Private verification ZIPs contain no raw texts, original IDs, model weights, or
optimizer checkpoints. Their per-row labels/probabilities/indices may still be
sensitive. Leave that optional switch False unless numerical verification is
needed. The normal development ZIP is sufficient to start the results review.

The optional internal-test cell defaults to `START_INTERNAL_TEST=False`. When
explicitly requested, it uses the frozen best export and validation thresholds.
The core writes a single-use marker before test access. An interrupted test
cannot be silently retried. A saved successful result can be inspected without
running new inference. Test evaluation never changes the original development
ZIP; packaging creates separate `_with_test` files.

## Local checks and limitations

The bounded offline CPU check uses 240 marked synthetic rows, a train-only local
tokenizer, and a tiny one-layer smoke architecture. It interrupts after two
optimizer updates within epoch one, resumes, and compares with uninterrupted
two-epoch training. Exact final weights, exported validation probabilities, and
history match. It also checks partial accumulation groups, weights-only-safe RNG
serialization, best export, fresh validation, and an unscored internal test.
These are engineering checks, not real-data model-quality results.

Notebook source materialization/manifest checks run without contacting Colab,
Drive, Hugging Face, or any dataset service. Python 3.12 and torch 2.8.0+cpu are
the locally exercised neural environment. Full-data GPU training, Colab services,
CUDA memory/performance, and real neural quality have not been validated here.
Check the final independent test record in `docs/validation.md` for test counts.

To regenerate the cleared notebook after an authorized runner change:

```bash
python scripts/build_colab_notebook.py
python scripts/build_colab_notebook.py --check
```

The six core source modules are frozen evidence for the completed CPU baseline
and must not be changed as part of notebook regeneration. For a bounded offline
resume check, use a fresh ignored run path:

```bash
python scripts/colab_train.py smoke --run-dir runs/new-colab-smoke
```

Do not rerun the completed full real-data CPU baseline. Future neural
improvements require a separate clearly recorded experiment and validation-only
development; do not tune repeatedly against internal-test results.
