# Legacy Colab notebook status

Use [`04_colab_efficient_training.ipynb`](../notebooks/04_colab_efficient_training.ipynb)
and the [current execution guide](colab_efficient_training.md) for training.
Notebook 04 is the canonical workflow. The former
[`03_colab_full_training.ipynb`](../notebooks/03_colab_full_training.ipynb)
is retired and contains a notice directing readers to 04.

The user completed the separate efficient-profile experiment on 16 October 2026.
Its [review](review_20261016.md) records validation diagnostics, selected epoch,
runtime, numerical verification, and limitations. Its internal test remains
unscored. The returned evidence does not reproduce the historical 80/20 neural
experiment or establish an end-to-end GPU speedup.

## Recovering a run started with notebook 03

A saved optimizer checkpoint must use the exact notebook, embedded source,
configuration, tokenizer, source CSV, and software/GPU fingerprints recorded
when that run started. Notebook 04 uses a separate profile, run directory,
dynamic padding, and calibrated microbatches. It cannot resume a notebook-03
checkpoint. Do not copy newer source-writing cells into an existing run or edit
its identity files to bypass a mismatch.

The original published notebook is preserved at Git commit
`b6819a5a2f6e6822b3d58bd0b7e314a623340ace`, with notebook SHA-256
`2f60edc83f6d70a54b194e4e0fd0372782f820675c14584975a488ae245eba55`.
Use that version only if it matches the saved run's own source identity. A later
03 revision may have a different identity; retrieve the version recorded by that
run from Git history. The previous full guide is also preserved in that history.
The [15 October integrity review](review_20261015.md) documents the earlier
continuation and completion repairs without changing historical result files.

Preserve private Drive data, checkpoints, tokenizer files, exported weights, and
manifests. Diagnose a completed or interrupted run from its recorded evidence
before considering another training run. Use one active session per run folder.

## Evaluation and sharing boundary

Validation chooses the checkpoint and per-label F1 thresholds. A completed
validation report is therefore a selection diagnostic. Any later internal-test
evaluation is a separate explicit action against the frozen model and thresholds;
this review did not perform it. The cohort has already been explored historically,
so such a test would remain exploratory.

The supplied `test_preprocessed.csv` has 153,164 rows and entirely blank target
columns. It cannot provide AP, ROC-AUC, F1, or accuracy without genuine labels.
The notebook's internal test instead comes from the labeled training source.

Publish aggregate evidence only. Private verification bundles may include per-row
targets, probabilities, and partition indices even when they contain no comment
text or original IDs. Keep them, source files, local manifests, and model weights
outside the publication tree. Mohammad Albatayneh's original individual authorship
and the portfolio revisions remain recorded in [provenance](provenance.md).
