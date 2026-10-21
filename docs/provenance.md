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

The original academic report names Mohammad Albatayneh. He subsequently confirmed that this was his individual project and that he wrote the original code. This author statement establishes the declared contribution; third-party libraries, tokenizer assets, and the cleaned dataset mirror retain their own provenance and terms. No repository-wide license has been selected.

## Preservation and public derivatives

Original uploads are preserved unchanged outside the publication tree. This repository includes sanitized historical **source derivatives** under [archive/notebooks](../archive/notebooks/), not the original byte-identical notebooks. Their output arrays are empty and execution counts are null; illustrative comments and demonstration predictions have been omitted. The flawed historical save/reload source remains visible for audit.

The original PDF and its contact details are not redistributed in the publication tree. [corrected_report.md](../reports/corrected_report.md) is a new evidence-bounded report, not a silent replacement of the original academic document. [report_corrections.md](report_corrections.md) maps claims to source pages/cells and correction decisions.

## Historical metric extraction

[historical_metrics.json](../reports/historical_metrics.json) was extracted from the original notebook JSON and stored text outputs, without executing training. Each metric record includes the original filename and SHA-256 through its parent record, zero-based cell index, output index, execution count, and reported decimal precision.

The extraction preserves rounding. No dataset, checkpoint, or prediction vector accompanied the initial notebook/report uploads, so it cannot recompute the historical scores, inspect calibration, recover historical class supports, or identify the exported model weights. Mirror CSVs supplied later are separately fingerprinted in [data_audit.json](../reports/data_audit.json); their relationship to the exact historical download is unverified. Convenience fields such as `best_by_validation_loss` describe the recorded epoch selection, not independently evaluated checkpoints.

The FINAL_Report per-label cell has stored output and a null execution count. Its code reads arrays from the last training epoch. Null execution metadata alone does not prove the cell failed; it also does not establish a clean top-to-bottom run. The two five-epoch notebooks' identical displayed metrics are explicitly marked as shared evidence.

## New runnable work

The repaired CLI and notebook scaffolding are derived from the supplied pipeline and the documented audit. This portfolio revision adds strict CSV checks, group-disjoint exact-duplicate splitting, train-only fitting, explicit metric coverage, validation-only threshold selection, frozen run artifacts, checkpoint/report/export alignment, and bounded synthetic verification. These new additions are distinct from Mohammad Albatayneh's original academic code and recorded neural experiment. They establish inspectable engineering behavior; they do not establish higher neural real-data performance.

The later real-data CPU experiment is recorded in [baseline_validation_metrics.json](../reports/baseline_validation_metrics.json), [baseline_internal_test_metrics.json](../reports/baseline_internal_test_metrics.json), [baseline_data_summary.json](../reports/baseline_data_summary.json), and [baseline_verification.json](../reports/baseline_verification.json). Its source snapshot, environment, configuration, code hashes, model, split membership, and predictions remain in an ignored private run directory. Public files expose only aggregate results and review checks. Root review independently recomputed metrics from the saved predictions and verified zero normalized-group overlap. The historical neural run has not been reproduced.

[01_results_review.ipynb](../notebooks/01_results_review.ipynb) reads committed aggregate JSON and builds editable native plots and exact tables. Historical outputs, the later data audit, and new run aggregates have separate sections. Its saved execution proves that this evidence-review notebook runs with those inputs. It does not rerun the original experiment. The neural training notebook is a guide to the repaired CLI and keeps real training explicitly opt-in.

Local Jupyter socket creation is unavailable in this sandbox. The committed review notebook executes every code cell top-to-bottom in a fresh in-process IPython session using the project interpreter and saves captured native figures/tables. Notebook metadata records this mode. A normal Jupyter-kernel protocol execution remains available through the default runner in an environment that permits local sockets.

The original self-contained Colab workflow was a portfolio addition derived from the revised neural pipeline. Its source and [earlier guide](colab_training.md) remain recoverable through Git history; the current notebook 03 is a retirement notice. The canonical [notebook 04](../notebooks/04_colab_efficient_training.ipynb), [guide](colab_efficient_training.md), [runner](../scripts/colab_train.py) and [dependencies](../requirements-colab.txt) retain visible source-writing cells and source-hash checks. Its [builder](../scripts/build_efficient_colab_notebook.py) checks source synchronization. The efficient profile adds cached tokenization, dynamic padding, GPU resource calibration, and timed checkpoints as a distinct numerical experiment. Preparing or smoke-checking either workflow is engineering evidence; the returned completed run below provides separate real-data neural validation evidence.

## Dataset and tokenizer lineage

The historical source code requests KaggleHub `fizzbuzz/cleaned-toxic-comments` without pinning a dataset version. One saved log refers to version 1, but it does not prove that all historical outputs used identical bytes. Upstream cleaning cannot be reconstructed from the supplied code alone.

`AutoTokenizer.from_pretrained("roberta-base")` loads a pretrained tokenizer. `RobertaForSequenceClassification(config)` creates new random weights. Do not describe this as fine-tuning pretrained RoBERTa model weights or as training a tokenizer on the task dataset.

Current input fingerprints and split details are recorded per new local run. These private traces support reproduction without distributing raw comments, row IDs, normalized-text fingerprints, or individual predictions. Source access and unresolved rights are detailed in [data_sources.md](data_sources.md).

## Confirmed publication scope and remaining gaps

The approved Public [repository](https://github.com/mobatayneh/toxic-comment-classification), [work branch](https://github.com/mobatayneh/toxic-comment-classification/tree/work/portfolio-hardening), and [PR #1](https://github.com/mobatayneh/toxic-comment-classification/pull/1) are published and verified. The initial publication was reviewed in that PR; GitHub records its current merge state and tested commits. The first public snapshot's 63 path/mode/blob identities and root tree were checked against the local preparation snapshot. Remote commit IDs differ from the preserved local preparation history; no existing remote history was rewritten. Mohammad Albatayneh's individual authorship statement is recorded above; no repository-wide code license has been selected.

- Select a repository-wide code license separately; publication approval does not select one.
- Establish terms and preprocessing lineage if the FizzBuzz mirror is used or distributed.
- Review the new real-data aggregate results separately from historical outputs; confirm supplied mirror provenance and permission.
- Reserve or collect an independent evaluation cohort before making stronger deployment or generalization claims.


## Follow-up source revision — 15 October 2026

This dated entry records the status before the completed efficient-profile run
was returned. See the 16 October entry below for the subsequent evidence; earlier
source identities and result artifacts remain unchanged.

The original published Colab notebook has SHA-256 `2f60edc83f6d70a54b194e4e0fd0372782f820675c14584975a488ae245eba55` at commit `b6819a5a2f6e6822b3d58bd0b7e314a623340ace`. A user run already started from that notebook must retain its original embedded files and recorded environment when resuming. Later source revisions are not interchangeable with a saved run identity. No submitted real-data neural training result has yet been verified.

The completed CPU reference was independently rechecked before this revision: saved targets, partitions, thresholds and aggregate metrics matched the private frozen artifacts exactly. Its source hashes belong to the earlier implementation. To inspect that frozen run, use its recorded source version; never rewrite its manifest to match newer code or retrain merely to make an integrity check pass. Commit `b6819a5a2f6e6822b3d58bd0b7e314a623340ace` retains the six core source files matching that evidence.

## Returned efficient-profile evidence — 16 October 2026

The user returned three artifacts: aggregate `results_for_review.zip`, separate
`calibration.json`, and `verification_private.zip`. The original files are kept
outside the publication tree. Returned aggregate values and regenerated evidence plots are
stored under [`reports/colab_efficient_v1`](../reports/colab_efficient_v1/validation_metrics.json),
alongside the derived [verification](../reports/colab_efficient_v1/verification.json),
comparison, and public file manifest. The original aggregate ZIP SHA-256 is
`270317ca85ff3420f3b003768a1b406ccaa9dcf398d2adb27932e6efbd1e089e`;
the separate calibration SHA-256 is
`084648414396419e8657cd86dd0558c516882a10e1933722feb93438489d4bb7`. Private targets, probabilities, and split
indices were used only for numerical review and are not published. No raw comments,
original row IDs, normalized-text fingerprints, checkpoints, or weights are added.

The bundle records source SHA-256
`bb7a9e1af5282c4c05bf09bf3be3373c4160fe3272a4b26016034ff4bcea5f1e`,
the original runner/core source hashes, fixed label order, and tokenizer revision
`e2da8e2f811d1448a5b465c236feacd80ffbac7b`. These result identities are retained;
review does not rewrite them to match a newer notebook or code revision. The
recorded runtime was Python 3.13.15, torch 2.11.0+cu130, CUDA 13.0, cuDNN 92700,
and one Tesla T4. Its [complete runtime record](../reports/colab_efficient_v1/versions_hardware.json)
is distinct from locally recorded CPU test environments.

All five epochs completed. Validation loss selected epoch four at
0.36014178103214034; the returned history reports the same fresh-export loss.
Validation macro AP is 0.5259 and macro F1 is 0.5094 at validation-selected
thresholds. Both model and thresholds used validation, so these are selection
diagnostics. The internal test remains unscored. The [review](review_20261016.md)
states the numerical checks and limits, including the absence of returned weights,
tokenizer files, and full run identity needed to independently reproduce model
export, inference, or resume. No GPU rerun was performed during this review.

This run does not establish which bytes or weights produced the historical
80/20 outputs. Historical evidence, the CPU reference's original frozen metrics,
and dated verification/CI records remain preserved with their own provenance.
No end-to-end GPU speedup, calibrated probability, or deployment claim is added.
