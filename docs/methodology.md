# Methodology and evaluation protocol

## Scope of the evidence

Three types of evidence must remain separate:

1. **Historical outputs:** rounded numbers embedded in three uploaded notebooks. They describe an 80/20 validation experiment, not an independently reproduced benchmark. The initial notebook/report uploads contained no dataset, trained weights, or prediction arrays.
2. **Real data audit and new experiment:** mirror CSVs were supplied later and inspected. A completed full-cohort CPU reference run uses the new 70/15/15 grouped protocol. Its results are a distinct experiment; supplying current mirror bytes does not prove which bytes produced the historical neural outputs.
3. **Engineering runs:** synthetic smoke runs exercise the repaired workflow and are never evidence of performance on real comments. Full-data neural training has not been rerun.

The supplied report's two PDF files have identical SHA-256 hashes. Their agreement does not provide a second report or experimental replication.

## Historical experiment

The notebooks download `fizzbuzz/cleaned-toxic-comments`, read `train_preprocessed.csv`, rename `severe_tox` to `severe_toxic`, and retain `set == "train"` rows. Training is a seeded random 80% row sample; validation is the remaining 20%. This is not the multilabel-stratified split claimed in the report. No recorded duplicate, label-coverage, source-hash, or source-version check establishes independent comments or a pinned dataset.

The tokenizer comes from `roberta-base`. The encoder is created through `RobertaForSequenceClassification(RobertaConfig(...))`, which initializes weights randomly. Its configuration is hidden size 512, eight layers, eight attention heads, intermediate size 2,048, maximum position embeddings 512, dropout 0.1, and maximum input length 256. The six logits are converted with sigmoid outside the model. The classifier uses the built-in RoBERTa head, not a separately implemented token-level rationale model.

Training uses weighted binary cross-entropy, with `pos_weight = training negatives / training positives`, AdamW, a cosine schedule, and checkpoint selection by lowest validation loss. FINAL_Report requests five epochs at learning rate 0.00004 and warmup fraction 0.10. Untitled10 requests ten epochs at 0.00003 and warmup 0.15 and records seven epochs before early stopping. The selected epochs are 3 and 4 respectively.

Three mismatched states exist in the historical workflow: model export happens before the best checkpoint is loaded, demonstration inference uses the selected checkpoint, and FINAL_Report's per-label report uses last-epoch prediction arrays. The corrected implementation reloads the selected checkpoint, recomputes validation scores, then reports and exports the same model. The source audit is in [notebook_audit.md](notebook_audit.md).

## New data validation and partitioning

The loader requires nonempty string `comment_text` and six binary labels. Missing, empty, whitespace-only, nonbinary, or disagreeing severe-label aliases fail validation. Extra columns and IDs are excluded from the modeled frame. The mirror's nontraining rows are not treated as labeled evaluation data automatically.

The inspected training mirror contains 159,571 rows, including seven normalized-empty comments. For the reviewed real-data experiment, the explicit empty-text policy drops those seven rows and records the exclusion, retaining 159,564. The 93 conflicting normalized-text groups retain their original annotations under the explicit `keep` policy; labels are not merged or guessed. These choices precede model fitting. The aggregate [data audit](../reports/data_audit.json) also records class counts and duplicate summaries.

The realized CPU partitions contain 111,731 training rows, 23,878 validation rows, and 23,955 exploratory internal-test rows. The saved private predictions independently reproduce the public metric files, and normalized-group overlap between partitions is zero. [Run verification](../reports/baseline_verification.json) records the single evaluation and these checks. The derived `toxicity` target summary is excluded from features along with IDs and all other extra columns.

The separately supplied challenge-test mirror contains 153,164 rows with every cell of the six target columns blank. It cannot be scored without valid authorized labels. There are 636 shared nonempty normalized-text groups between the two source files; separate files and disjoint IDs do not establish text independence. No challenge-test metric is reported.

Text grouping uses Unicode NFKC normalization, case folding, and Unicode whitespace collapse. This normalization is used **only for duplicate grouping**; each model receives the original text. Every normalized exact-match group is allocated to one partition. Differently labeled duplicates fail by default; the explicit `keep` policy retains them together without resolving the disagreement.

Unique group digests are sorted, shuffled with the recorded seed, and allocated approximately 70%/15%/15% by group count. Partitions are nonempty; their actual row counts and label supports are recorded. This is a group-disjoint random split, not a stratified split. It does not detect near duplicates, paraphrases, shared authors, shared discussions, or time dependence.

The third partition is an **exploratory internal test**. It belongs to a cohort already examined in the original work, and a new random split cannot undo that historical exposure. An untouched external cohort or explicitly reserved future collection is needed for stronger generalization claims. Official Kaggle test labels, where known, could support a separate future evaluation, but no such evaluation has been performed here.

## Model fitting and selection

The CPU reference pipeline fits word unigram/bigram TF–IDF on training text only, with sublinear term frequency, minimum document frequency 2, and at most 50,000 features. Six one-vs-rest logistic regressions use `liblinear`, `C = 4`, balanced class weights, and a maximum of 1,000 iterations. Vocabulary, inverse document frequency, and classifier weights never fit validation or test text.

The repaired neural path preserves the custom random-initialized architecture and computes loss weights from training labels only. The portable defaults use smaller physical batches with gradient accumulation. A selected checkpoint is reloaded before scoring validation or exporting the model and tokenizer. The run records configuration, source fingerprint, split membership, labels, and validation thresholds. Seeded execution improves repeatability but does not guarantee identical floating-point results across hardware or library versions.

Per-label decision thresholds maximize validation F1 on the fixed grid 0.05, 0.10, …, 0.95. Ties select the value nearest 0.5, then the higher threshold. A label with only one validation class retains 0.5. Thresholds are fixed before internal test evaluation. Report both threshold-0.5 and validation-tuned results to make the operating-point change visible.

## Metrics

| Metric | Definition and use |
|---|---|
| Macro average precision (primary) | Unweighted mean of six per-label `average_precision_score` values from continuous scores. AP is the recall-weighted average of precision increments, not trapezoidal PR-curve area. |
| Macro ROC-AUC | Unweighted mean of six per-label ROC-AUC values from continuous scores. High ranking quality does not establish precision at a chosen threshold. |
| Per-label precision, recall, and F1 | Calculated from fixed binary decisions, with positive/negative support and the threshold recorded. |
| Macro F1 | Mean of per-label F1 values, not the harmonic mean of macro precision and macro recall. |
| Micro F1 | Pools all label decisions before computing F1. Frequent labels can dominate it. |

An absent positive or negative class makes per-label AP/ROC-AUC undefined under this reporting protocol. Those fields are `null`; the six-label macro is also `null` without all six eligible labels. Separately named eligible-label means are diagnostics and must not be reported as complete six-label macro results. Macro F1 requires positive support for all six labels. Always inspect coverage and supports before interpreting a aggregate value.

There are no historical AP values, score vectors, confidence intervals, or calibration checks. They cannot be reconstructed from rounded ROC-AUC, F1, precision, or recall. Historical F1 uses strict `score > 0.5`; new metric files document their decision rule explicitly.

## Freeze and evaluation

After development, model artifacts, validation decisions, source/split identity, and thresholds are frozen. Evaluation checks the recorded artifacts before using the internal test. A relocated source CSV must have identical bytes. The one-evaluation guard limits accidental repeated test-driven selection; it does not make the historically exposed cohort untouched. Any future model revision requires a new run, and its relationship to prior experiments must be documented.

Training data, comments, split indices, normalized-text digests, scores, checkpoints, and detailed local manifests remain in ignored run directories. Shareable result exports contain aggregate metrics and coverage only. Do not publish individual comment examples or local prediction arrays as a side effect of notebook execution.

## Interpretation limits

The historical FINAL_Report per-label output indicates high recall and low precision at 0.5, especially for `threat` (0.048 precision, 0.853 recall). Weighted-loss sigmoid scores are not established calibrated probabilities. The new CPU internal test reports macro AP 0.6502, macro ROC-AUC 0.9811, macro F1 0.6346, and micro F1 0.7483 at frozen validation-selected thresholds. These concern a different protocol, so they do not establish a controlled gain over the historical neural run. None of the available results establishes fairness, explanations, production latency, compactness relative to a measured comparator, or readiness for automated moderation.

