# Artifact provenance

## Supplied materials

The preparation received three notebooks and two PDFs. Exact original byte counts and SHA-256 values are preserved in [source_manifest.json](source_manifest.json). The two PDFs are byte-identical and therefore represent one report artifact.

| Original file | Role in the preparation |
|---|---|
| `model_FINAL_Report.ipynb` | Main implementation reference: common pipeline, per-label reporting code, and architecture diagram code |
| `modelXXXX(1).ipynb` | Historical five-epoch variant; all displayed epoch metrics match FINAL_Report |
| `Untitled10.ipynb` | Historical learning-rate/warmup/epoch variant; seven recorded epochs |
| `Toxic_Comment_Multi_Label_Classification(1).pdf` | Original academic report; audited for unsupported or mismatched claims |
| `Toxic_Comment_Multi_Label_Classification(2).pdf` | Identical copy of the same report |

The original academic report names Mohammad Albatayneh. He subsequently confirmed that this was his individual project and that he wrote the original code, with AI assistance to correct errors. This author statement establishes the declared contribution; third-party libraries, tokenizer assets, and the cleaned dataset mirror retain their own provenance and terms. No repository-wide license has been selected.

## Preservation and public derivatives

Original uploads are preserved unchanged outside the publication tree. This repository includes sanitized historical **source derivatives** under [archive/notebooks](../archive/notebooks/), not the original byte-identical notebooks. Their output arrays are empty and execution counts are null; illustrative comments and demonstration predictions have been omitted. The flawed historical save/reload source remains visible for audit.

The original PDF and its contact details are not redistributed in the publication tree. [corrected_report.md](../reports/corrected_report.md) is a new evidence-bounded report, not a silent replacement of the original academic document. [report_corrections.md](report_corrections.md) maps claims to source pages/cells and correction decisions.

## Historical metric extraction

[historical_metrics.json](../reports/historical_metrics.json) was extracted from the original notebook JSON and stored text outputs, without executing training. Each metric record includes the original filename and SHA-256 through its parent record, zero-based cell index, output index, execution count, and reported decimal precision.

The extraction preserves rounding. No dataset, checkpoint, or prediction vector accompanied the initial notebook/report uploads, so it cannot recompute the historical scores, inspect calibration, recover historical class supports, or identify the exported model weights. Mirror CSVs supplied later are separately fingerprinted in [data_audit.json](../reports/data_audit.json); their relationship to the exact historical download is unverified. Convenience fields such as `best_by_validation_loss` describe the recorded epoch selection, not independently evaluated checkpoints.

The FINAL_Report per-label cell has stored output and a null execution count. Its code reads arrays from the last training epoch. Null execution metadata alone does not prove the cell failed; it also does not establish a clean top-to-bottom run. The two five-epoch notebooks' identical displayed metrics are explicitly marked as shared evidence.

## New runnable work

The repaired CLI and notebook scaffolding are derived from the supplied pipeline and the documented audit. This portfolio preparation used AI coding and writing assistance to add strict CSV checks, group-disjoint exact-duplicate splitting, train-only fitting, explicit metric coverage, validation-only threshold selection, frozen run artifacts, checkpoint/report/export alignment, and bounded synthetic verification. These new additions are distinct from Mohammad Albatayneh's original academic code and recorded neural experiment. They establish inspectable engineering behavior; they do not establish higher neural real-data performance.

The later real-data CPU experiment is recorded in [baseline_validation_metrics.json](../reports/baseline_validation_metrics.json), [baseline_internal_test_metrics.json](../reports/baseline_internal_test_metrics.json), [baseline_data_summary.json](../reports/baseline_data_summary.json), and [baseline_verification.json](../reports/baseline_verification.json). Its source snapshot, environment, configuration, code hashes, model, split membership, and predictions remain in an ignored private run directory. Public files expose only aggregate results and review checks. Root review independently recomputed metrics from the saved predictions and verified zero normalized-group overlap. The historical neural run has not been reproduced.

[01_results_review.ipynb](../notebooks/01_results_review.ipynb) reads committed aggregate JSON and builds editable native plots and exact tables. Historical outputs, the later data audit, and new run aggregates have separate sections. Its saved execution proves that this evidence-review notebook runs with those inputs. It does not rerun the original experiment. The neural training notebook is a guide to the repaired CLI and keeps real training explicitly opt-in.

Local Jupyter socket creation is unavailable in this sandbox. The committed review notebook executes every code cell top-to-bottom in a fresh in-process IPython session using the project interpreter and saves captured native figures/tables. Notebook metadata records this mode. A normal Jupyter-kernel protocol execution remains available through the default runner in an environment that permits local sockets.

The [self-contained Colab notebook](../notebooks/03_colab_full_training.ipynb), [guide](colab_training.md), [runner](../scripts/colab_train.py) and [dependencies](../requirements-colab.txt) are new AI-assisted portfolio additions derived from the revised neural workflow. Visible source-writing cells reproduce the reviewed core and runner files; [the notebook builder](../scripts/build_colab_notebook.py) checks source synchronization. The runner checks the exact supplied mirror and preserves restart state in private checkpoints. After selected-checkpoint export and fresh validation scoring, `results_for_review.zip` contains aggregate metrics, configuration, training history and native plots. An optional `verification_private.zip` contains per-row labels, scores and split indices for private recomputation; it is excluded from publication. Preparing or smoke-checking this workflow does not produce a new real-data neural result.

## Dataset and tokenizer lineage

The historical source code requests KaggleHub `fizzbuzz/cleaned-toxic-comments` without pinning a dataset version. One saved log refers to version 1, but it does not prove that all historical outputs used identical bytes. Upstream cleaning cannot be reconstructed from the supplied code alone.

`AutoTokenizer.from_pretrained("roberta-base")` loads a pretrained tokenizer. `RobertaForSequenceClassification(config)` creates new random weights. Do not describe this as fine-tuning pretrained RoBERTa model weights or as training a tokenizer on the task dataset.

Current input fingerprints and split details are recorded per new local run. These private traces support reproduction without distributing raw comments, row IDs, normalized-text fingerprints, or individual predictions. Source access and unresolved rights are detailed in [data_sources.md](data_sources.md).

## Confirmed publication scope and remaining gaps

Mohammad Albatayneh approved a new public `mobatayneh/toxic-comment-classification` repository, a working branch, and a pull request without merging. Publication has not been verified here. His individual authorship statement is recorded above; no repository-wide code license has been selected.

- Select a repository-wide code license separately; publication approval does not select one.
- Establish terms and preprocessing lineage if the FizzBuzz mirror is used or distributed.
- Review the new real-data aggregate results separately from historical outputs; confirm supplied mirror provenance and permission.
- Reserve or collect an independent evaluation cohort before making stronger deployment or generalization claims.

