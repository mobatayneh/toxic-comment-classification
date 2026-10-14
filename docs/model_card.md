# Model card

## Identification and status

This project contains two research pipelines for English, six-label toxic-comment classification:

| Pipeline | Components | Available evidence |
|---|---|---|
| CPU reference | Train-only TF–IDF and six balanced logistic regressions | Completed real-data experiment on 159,564 eligible rows; macro AP 0.6502 and macro F1 0.6346 on an exploratory internal test |
| Custom RoBERTa | Randomly initialized eight-layer encoder and classification head; pretrained byte-level BPE tokenizer | Historical rounded validation outputs plus repaired training/export workflow; no reproduced real-data result or supplied weights |

There is no trained release model in this repository. A code release or successful smoke test must not be described as a validated model release.

## Task, inputs, and outputs

Input is a nonempty comment string. The label order is `identity_hate`, `insult`, `obscene`, `severe_toxic`, `threat`, `toxic`. Labels can overlap; a comment can have none, one, or several positive labels.

The neural encoder truncates to 256 tokenizer tokens by default. Long comments can lose relevant context. Logistic regression uses a learned word unigram/bigram TF–IDF representation. Model scores and saved thresholds define the final decisions. Neural sigmoid values and logistic regression scores must not be presented as calibrated risk probabilities without a separate calibration study.

## Intended use

Suitable uses are methodological study, offline experimentation, reproducibility practice, and evaluation of candidate moderation aids under human review. A prospective deployment needs data representative of its language, community, policy, and operating period.

Unsupported uses include autonomous account penalties, consequential decisions about people, reliable identification of threats, or generalization to other languages based only on this evidence. The label names reflect the source annotation scheme, not universal definitions of harmful speech.

## Training data and provenance

The historical notebooks use the FizzBuzz cleaned mirror of the Jigsaw Toxic Comment Classification Challenge, not a verified byte-identical official `train.csv`. Its preprocessing details, source version, and redistribution rights remain unresolved. Later supplied mirror files verify 159,571 training-source rows; they do not identify the absent historical training bytes. The new run explicitly drops seven normalized-empty comments, retains 159,564 labeled rows, and keeps 93 conflicting groups' original annotations together. See [the aggregate data audit](../reports/data_audit.json).

The current loader accepts official labeled training CSVs and the historical mirror schema. It validates text and binary labels and groups normalized exact duplicates. The new internal split uses 70/15/15 proportions by group count. Near duplicates, author overlap, and time overlap are not controlled. The cohort has already been explored in the historical project.

Raw comments, raw datasets, per-comment predictions, and trained weights are not distributed. See [data_sources.md](data_sources.md).

## Architecture and training

The custom neural configuration has hidden size 512, eight transformer layers, eight attention heads, intermediate size 2,048, maximum position embeddings 512, and dropout 0.1. Only the tokenizer is pretrained. The model uses a first-token sequence-classification head followed by an external sigmoid for scoring.

The neural objective is weighted BCE with per-label weights derived from training positives and negatives. Historical checkpoint selection minimizes validation loss. The repair reports fresh validation predictions after loading that checkpoint and saves model/tokenizer/labels/thresholds consistently. Per-label thresholds are selected from validation F1 before the guarded internal evaluation.

The [Colab notebook](../notebooks/03_colab_full_training.ipynb), [guide](colab_training.md) and [runner](../scripts/colab_train.py) prepare a full eligible-cohort GPU experiment using the exact supplied training mirror and the grouped partitions above. Optimization uses the 111,731 training rows. Persistent private checkpoints save model, optimizer, scheduler, AMP and RNG state; resume requires matching source, code, configuration, tokenizer and runtime fingerprints. Internal-test scoring is disabled by default. Local synthetic CPU checks of this mechanism do not establish successful Colab execution or a real-data neural score.

The CPU pipeline uses TF–IDF features with balanced one-vs-rest logistic regression. Its exact configuration is recorded with each run. Model comparisons require the same source and frozen partitions; historic validation metrics cannot establish the relative performance of this new baseline.

## Evaluation evidence

The selected epoch-3 FINAL_Report output records validation macro ROC-AUC 0.9790 and macro F1 0.3900 at strict threshold `> 0.5`. The last epoch records 0.9792 and 0.4391. The per-label report concerns last-epoch arrays and reports macro precision 0.321, recall 0.924, and F1 0.439. The selected checkpoint, exported last-epoch model, and report were not synchronized in the original code.

`modelXXXX(1).ipynb` matches every displayed five-epoch metric from FINAL_Report. This is shared historical evidence, not independent replication. Untitled10 is a separate recorded hyperparameter variant whose best-loss epoch differs from its highest recorded F1 and ROC-AUC epochs.

The completed CPU run uses 111,731 training, 23,878 validation, and 23,955 internal-test rows. It reports internal-test macro AP 0.6502, macro ROC-AUC 0.9811, macro F1 0.6346, and micro F1 0.7483 at frozen validation-selected thresholds, with ranking coverage 6/6. Fixed-threshold macro F1 is 0.6144. The saved private predictions independently reproduce these aggregates; normalized-group overlap across the three partitions is zero. The new model, split, and threshold protocol differ from the historical experiment, so these scores do not establish a controlled improvement. [Aggregate run verification](../reports/baseline_verification.json) records the evidence.

No independent held-out test result, historical AP, confidence interval, subgroup metric, calibration curve, production latency measurement, or full-data neural rerun is available. The supplied challenge-test file is entirely unlabeled, and 636 nonempty normalized-text groups occur in both source files. New CPU internal-test results remain exploratory because the training cohort was already examined historically. Detailed tables and sources are in [historical_metrics.json](../reports/historical_metrics.json) and [the results-review notebook](../notebooks/01_results_review.ipynb).

## Limitations and risks

Annotation disagreement, class imbalance, platform differences, quoted offensive text, sarcasm, and identity references can affect predictions. Training labels do not establish intent or real-world danger. False positives can disproportionately burden benign discussions; low precision for rare labels is already visible in the historical operating point.

No subgroup fairness study or robustness test was supplied. Mentioning an identity term is not evidence that a comment is hateful. A deployment should evaluate subgroup error rates, policy fit, abstention/review rules, and failure costs on an appropriate independently collected cohort.

The repository does not implement token-level rationales, establish causal explanations, or prove resource-efficiency claims from the original report. A future explainability or latency study must identify its method, hardware, batch size, input length, and benchmark population.

## Ownership and maintenance

Mohammad Albatayneh confirmed that the original academic project was his individual work and that he wrote its code, using AI assistance to correct errors. The portfolio revision adds AI-assisted engineering and documentation work; its completed CPU experiment and synthetic checks remain separate from the historical neural results. No repository-wide license has been selected, and third-party data/tokenizer terms remain applicable. New results should identify their input version, frozen run, evaluation partition, thresholds, metric definitions, and whether the data are synthetic or real.

