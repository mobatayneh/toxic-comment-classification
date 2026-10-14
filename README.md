# Toxic comment multi-label classification

A reproducible study of six overlapping toxicity labels, with a CPU TF–IDF/logistic-regression reference pipeline and a repaired custom RoBERTa training pipeline. This repository also records what can—and cannot—be concluded from the supplied academic notebooks and report.

**Current evidence:** a completed real-data CPU reference experiment on 159,564 eligible rows, historical rounded neural validation results, and engineering checks of both repaired pipelines. The CPU exploratory internal test has macro AP **0.6502**, macro ROC-AUC **0.9811**, and macro F1 **0.6346** at frozen validation-selected thresholds. Its protocol differs from the historical neural experiment; these values are not a controlled before/after improvement. Full-data neural training has not been rerun and no historical trained checkpoint was supplied. The historical transformer uses a pretrained RoBERTa **tokenizer** and a randomly initialized eight-layer encoder. It is not pretrained-encoder fine-tuning.

## Task and labels

Each comment can receive any combination of these labels, in this saved order:

```text
identity_hate, insult, obscene, severe_toxic, threat, toxic
```

The project is a research implementation, not a deployed moderation system. Outputs need contextual human review; no fairness, calibration, robustness, or production latency claim is established.

## Run the CPU pipeline

Use Python 3.12. Commands below run from the repository root.

The complete reviewed source is on [work/portfolio-hardening](https://github.com/mobatayneh/toxic-comment-classification/tree/work/portfolio-hardening). [PR #1](https://github.com/mobatayneh/toxic-comment-classification/pull/1) is open for review; main retains the initial README until the proposed changes are approved and merged.

```bash
git clone --branch work/portfolio-hardening https://github.com/mobatayneh/toxic-comment-classification.git
cd toxic-comment-classification
```

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m pip install --no-deps -e .

# Offline wiring check on explicitly synthetic data; not a benchmark.
python -m toxic_comments.baseline smoke --run-dir runs/synthetic-smoke

# Add notebook/review dependencies before repository checks.
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python scripts/verify_repository.py
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` in place of the shell activation command above.

Obtain a labeled CSV through the official source described in [data/README.md](data/README.md). Then:

```bash
python -m toxic_comments.baseline develop \
  --data data/raw/train.csv --run-dir runs/baseline

# Run only after the configuration and validation decisions are final.
python -m toxic_comments.baseline evaluate --run-dir runs/baseline
```

Use a new run directory for each experiment. `develop` fits TF–IDF and six balanced logistic regressions on training data, sets per-label thresholds using validation F1, and freezes the run. `evaluate` checks that frozen inputs still match and permits one internal test evaluation. Local run artifacts and data are ignored by Git. Public result files contain aggregate metrics, not comment text, row IDs, individual predictions, or normalized-text fingerprints.

The new split allocates normalized exact-comment **groups** to training/validation/internal test in 70/15/15 proportions. Actual row proportions can differ. It is not stratified and does not detect near duplicates or enforce author/time separation. The original cohort was already explored in the historical work, so this internal test is exploratory rather than evidence from a newly collected, untouched cohort. See [methodology](docs/methodology.md).

The default rejects conflicting labels among normalized duplicates. Use `--conflict-policy keep` only after reviewing the conflicts; this retains each annotation and keeps its entire group in one partition. The supplied mirror audit found 93 conflicting groups and seven normalized-empty training comments. The reviewed real run keeps conflicting annotations together and explicitly drops those seven empty comments, retaining 159,564 rows. The loader accepts the original mirror's `severe_tox` alias and selects `set == "train"` when that column exists.

The supplied challenge-test CSV has 153,164 rows and all six label columns are blank, so it cannot supply a test score. The source files share 636 nonempty normalized-text groups. The [data audit](reports/data_audit.json) records these findings without distributing comments.

## New real-data CPU results

The fixed reference fits TF-IDF and six balanced logistic regressions using 111,731 training rows. Validation has 23,878 rows; the exploratory internal test has 23,955. The source's derived `toxicity` target summary, IDs, and other extra columns are excluded from features. All six labels have ranking coverage. The recorded normalized-group overlap between partitions is zero; near-duplicate, author, and time separation remain unestablished.

<!-- baseline-metrics:start -->
| Partition | Threshold policy | Rows | Macro AP | Macro ROC-AUC | Macro F1 | Micro F1 |
|---|---|---:|---:|---:|---:|---:|
| Validation | Fixed 0.5 | 23878 | 0.6549 | 0.9823 | 0.6087 | 0.7257 |
| Validation | Validation-selected | 23878 | 0.6549 | 0.9823 | 0.6414 | 0.7546 |
| Internal test | Fixed 0.5 | 23955 | 0.6502 | 0.9811 | 0.6144 | 0.7330 |
| Internal test | Validation-selected | 23955 | 0.6502 | 0.9811 | 0.6346 | 0.7483 |
<!-- baseline-metrics:end -->

AP and ROC-AUC use continuous scores and do not change with thresholds. Validation rows informed threshold selection. The internal test was evaluated once after freezing; its metrics were independently recomputed from saved private predictions. This cohort was already examined historically, so the result remains exploratory. No confidence interval, calibration, fairness, or production-latency evidence is established. Sources: [validation aggregates](reports/baseline_validation_metrics.json), [internal-test aggregates](reports/baseline_internal_test_metrics.json), [partition supports](reports/baseline_data_summary.json), and [run verification](reports/baseline_verification.json).

## Repaired neural pipeline

The custom encoder retains the original 512 hidden dimensions, eight layers, eight attention heads, and six-label head. Repairs align checkpoint selection, reloaded validation predictions, model export, labels, and thresholds. This corrects the original save/reload/report mismatch; it does not establish a numerical improvement.

```bash
# Optional CPU-only PyTorch installation for the local/offline checks.
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-neural.txt

# Real-data training; tokenizer files must already be cached or downloaded.
python -m toxic_comments.neural develop \
  --data data/raw/train.csv --run-dir runs/roberta --device cuda
python -m toxic_comments.neural evaluate --run-dir runs/roberta --device cuda

# Separate offline synthetic CPU check.
python -m toxic_comments.neural smoke --run-dir runs/neural-synthetic-smoke
```

Training from random initialization can require substantial compute. The portable defaults use physical batch size 8 with accumulation over 32 batches, workers 0, and device `cpu`; GPU training is explicit. The RoBERTa tokenizer retains external provenance. Pin its revision or supply a local tokenizer directory; `--local-files-only` requires its files to be present. [The training notebook](notebooks/02_train_roberta.ipynb) documents the commands and evidence boundaries.

For a GPU run, use a compatible CUDA-enabled PyTorch installation in place of the optional CPU-only command above.

## Full training in Google Colab

Use [03_colab_full_training.ipynb](notebooks/03_colab_full_training.ipynb) with the [step-by-step guide](docs/colab_training.md). The notebook embeds the reviewed project source and standalone dependency pins in visible `%%writefile` cells, so it can be uploaded to Colab before GitHub publication.

1. Upload the notebook to [Google Colab](https://colab.research.google.com/) and select a compatible GPU runtime with Python 3.12+ and CUDA-enabled torch 2.8+. Run setup and all eight source-writing cells.
2. Mount Google Drive, keep a persistent run name, and upload the original `train_preprocessed.csv` once or enter its existing Drive path. The notebook checks the exact source bytes and the full 159,564-row eligible cohort; optimization uses 111,731 training rows.
3. Run the data audit and training cell. Checkpoints preserve model, optimizer, scheduler, AMP and RNG state. After interruption, rerun setup with the same Drive run name and configuration; resume requires the recorded software and GPU fingerprints to match.
4. Keep `MAKE_PRIVATE_VERIFICATION=False` and `START_INTERNAL_TEST=False` for the first review. Download **`results_for_review.zip`** and return it after training. The normal ZIP contains aggregate evidence and plots; source data, row arrays, checkpoints and model weights stay private.

This is a prepared training workflow. Full-data Colab/CUDA training has not been executed here, and no new neural performance is claimed. Validation selects the checkpoint and thresholds; the internal test remains unscored initially. A returned completed run must be reviewed before updating the report or comparing models. [The guide](docs/colab_training.md) explains the exact-source guard, restart limits and optional later test evaluation.

## Historical results

These values are copied from saved notebook outputs; they were **not recomputed from data or weights**. The original split was a random 80/20 row split of the mirror's `set == "train"` rows, without verified duplicate independence. F1 uses a strict score `> 0.5`. All results are validation metrics, repeatedly inspected during training.

<!-- historical-metrics:start -->
| Notebook | Recorded checkpoint | Epoch | Validation loss | Macro ROC-AUC | Macro F1 at > 0.5 |
|---|---|---:|---:|---:|---:|
| `model_FINAL_Report.ipynb` | Selected by validation loss | 3 | 0.3482 | 0.9790 | 0.3900 |
| `model_FINAL_Report.ipynb` | Last epoch | 5 | 0.3824 | 0.9792 | 0.4391 |
| `Untitled10.ipynb` | Selected by validation loss | 4 | 0.3498 | 0.9801 | 0.3818 |
| `Untitled10.ipynb` | Last epoch | 7 | 0.4081 | 0.9810 | 0.4459 |
| `modelXXXX(1).ipynb` | Selected by validation loss | 3 | 0.3482 | 0.9790 | 0.3900 |
| `modelXXXX(1).ipynb` | Last epoch | 5 | 0.3824 | 0.9792 | 0.4391 |

These are historical validation outputs, not a new run or held-out test results. FINAL_Report and modelXXXX share the same saved five-epoch history; they are not independent replications.
<!-- historical-metrics:end -->

The selected checkpoint and last epoch are distinct. `model_FINAL_Report.ipynb` and `modelXXXX(1).ipynb` have identical displayed five-epoch metrics; they are not independent replications. The per-label report in FINAL_Report uses last-epoch arrays, has a null execution count, and reports macro precision 0.321, recall 0.924, and F1 0.439. No historical average precision, held-out test score, accuracy, confidence interval, or calibrated probability evidence is available.

The [executed results-review notebook](notebooks/01_results_review.ipynb) derives historical tables and figures from [historical_metrics.json](reports/historical_metrics.json) and reviews the separately identified real-data audit and new run aggregates. It executes an evidence review, not historical training. The corrected report is available as [Markdown](reports/corrected_report.md) and [PDF](reports/corrected_report.pdf); [claim corrections](docs/report_corrections.md) explain the unsupported claims removed from the original report.

## Reproducibility and repository structure

```text
src/toxic_comments/     CSV validation, grouped splits, metrics, CPU and neural CLIs
notebooks/             Executed evidence review, neural CLI guide, full Colab training/resume
reports/               Historical evidence, real CPU aggregates and corrected report
docs/                  Methodology, model card, provenance, audits and Colab guide
data/                  Local data instructions; no real dataset distributed
archive/notebooks/     Sanitized historical source derivatives, with outputs cleared
scripts/               Colab runner/build, charts, notebook execution, PDF and verification
tests/                 Data, metric, freeze, pipeline, resume and bundle-boundary checks
requirements-*.txt     Separate core, neural, notebook, report and Colab dependencies
```

Run [the evidence-review notebook](notebooks/01_results_review.ipynb) without data or model downloads:

```bash
python -m pip install -r requirements-notebooks.txt
python scripts/render_historical_results.py
python scripts/render_real_results.py
python scripts/run_results_notebook.py
```

In this sandbox, local Jupyter kernel sockets are blocked. The committed review notebook was executed top-to-bottom with `python scripts/run_results_notebook.py --engine inprocess`, which captures rich outputs in a fresh IPython process. Its metadata records this mode. GitHub Actions additionally executed all eleven code cells through a normal Jupyter kernel and retained the executed notebook as a run artifact; see [validation](docs/validation.md) for the recorded run.

Build the report PDF from its reviewed Markdown source:

```bash
python -m pip install -r requirements-report.txt
python scripts/build_report_pdf.py
```

Read [methodology](docs/methodology.md) for the metric protocol, [model card](docs/model_card.md) for intended use and limitations, [validation](docs/validation.md) for the completed local checks, [provenance](docs/provenance.md) for artifact lineage, and [data sources](docs/data_sources.md) for access and third-party terms. [GitHub core CI passed](https://github.com/mobatayneh/toxic-comment-classification/actions/runs/38525910900): 31 checks passed, 12 optional CPU neural checks were explicitly skipped, and the results notebook/publication verifier completed successfully. All 43 checks, including those 12 optional checks, passed in the separate local CPU environment. Full Colab/CUDA training remains unverified.

## Authorship and third-party terms

Mohammad Albatayneh confirmed that the original academic project was an individual project and that he wrote its original code, with AI assistance to correct errors. This portfolio revision adds AI-assisted code repairs, reproducibility tooling, a CPU reference pipeline, evidence review, and Colab preparation. These additions do not establish a new full-data neural result.

No repository-wide license has been selected. The FizzBuzz mirror's license and preprocessing lineage remain unresolved; data, tokenizer, and library terms are separate from authorship of the project code. Real comments and trained weights are not distributed.

