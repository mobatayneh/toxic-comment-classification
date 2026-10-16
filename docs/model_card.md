# Model card

## Identification and status

This project contains two research pipelines for English, six-label toxic-comment classification:

| Pipeline | Components | Available evidence |
|---|---|---|
| CPU reference | Train-only TF–IDF and six balanced logistic regressions | Completed real-data experiment on 159,564 eligible rows; macro AP 0.6502 and macro F1 0.6346 on an exploratory internal test |
| Custom RoBERTa | Randomly initialized eight-layer encoder and classification head; pretrained byte-level BPE tokenizer | Historical rounded outputs plus a returned full-cohort five-epoch Colab experiment; selected-epoch validation macro AP 0.5259 and macro F1 0.5094; neural internal test unscored |

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

The canonical [Colab notebook 04](../notebooks/04_colab_efficient_training.ipynb), [guide](colab_efficient_training.md) and [runner](../scripts/colab_train.py) use the exact supplied training mirror and the grouped partitions above. Optimization uses the 111,731 training rows. Its efficient profile caches tokenization, dynamically pads microbatches, and uses synthetic resource probes to select a batch/accumulation pair with nominal effective batch 256. Notebook 03 is retired. Existing old runs require their original source from Git history; 04 cannot continue their optimizer checkpoints.

The user returned a completed five-epoch 04 run with microbatch 64 and accumulation four. It records Python 3.13.15, torch 2.11.0+cu130, CUDA 13.0 and a Tesla T4. Persistent private checkpoints preserve model, optimizer, scheduler, AMP and RNG state; resume requires the original source, code, configuration, tokenizer and runtime fingerprints. These recorded GPU-run versions differ from the local Python 3.12 / torch 2.8.0+cpu synthetic engineering environment. The notebook offers no internal-test scoring step.

The CPU pipeline uses TF–IDF features with balanced one-vs-rest logistic regression. Its exact configuration is recorded with each run. Model comparisons require the same source and frozen partitions; historic validation metrics cannot establish the relative performance of this new baseline.

## Evaluation evidence

The selected epoch-3 FINAL_Report output records validation macro ROC-AUC 0.9790 and macro F1 0.3900 at strict threshold `> 0.5`. The last epoch records 0.9792 and 0.4391. The per-label report concerns last-epoch arrays and reports macro precision 0.321, recall 0.924, and F1 0.439. The selected checkpoint, exported last-epoch model, and report were not synchronized in the original code.

`modelXXXX(1).ipynb` matches every displayed five-epoch metric from FINAL_Report. This is shared historical evidence, not independent replication. Untitled10 is a separate recorded hyperparameter variant whose best-loss epoch differs from its highest recorded F1 and ROC-AUC epochs.

The completed CPU run uses 111,731 training, 23,878 validation, and 23,955 internal-test rows. It reports internal-test macro AP 0.6502, macro ROC-AUC 0.9811, macro F1 0.6346, and micro F1 0.7483 at frozen validation-selected thresholds, with ranking coverage 6/6. Fixed-threshold macro F1 is 0.6144. The saved private predictions independently reproduce these aggregates; normalized-group overlap across the three partitions is zero. The new model, split, and threshold protocol differ from the historical experiment, so these scores do not establish a controlled improvement. [Aggregate run verification](../reports/baseline_verification.json) records the evidence.

The returned neural run selects epoch four by validation loss 0.36014178103214034, and records the same fresh-export loss. At fixed threshold 0.5, validation macro F1 is 0.3928 and micro F1 is 0.4943. Its validation-selected thresholds give macro AP 0.5259, macro ROC-AUC 0.9761, macro F1 0.5094 and micro F1 0.6520, with full six-label coverage. AP and ROC-AUC do not depend on thresholds. Validation selected both checkpoint and thresholds, so all these validation scores are selection diagnostics. [The review](review_20261016.md) and [numerical record](../reports/colab_efficient_v1/verification.json) identify the evidence and remaining limits; the review did not rerun training or neural inference.

A [verified same-validation comparison](../reports/colab_efficient_v1/baseline_validation_comparison.json) recomputes both models' saved validation scores after checking identical source, partitions/order, targets and definitions. The CPU reference records macro AP 0.6549 and selected-threshold macro F1 0.6414 on these same 23,878 rows, exceeding the observed neural aggregate values. This is descriptive selection evidence, without an uncertainty interval or independent generalization claim. It does not use the CPU internal-test score as a neural comparator.

No independent external test result, neural internal-test result, historical AP, confidence interval, subgroup metric, calibration curve, or production latency measurement is available. The supplied challenge-test file is entirely unlabeled, and 636 nonempty normalized-text groups occur in both source files. The CPU internal-test results remain exploratory because the training cohort was already examined historically. The returned run is a new grouped-protocol experiment, not a reproduction of the historical 80/20 metrics. Detailed tables and sources are in [historical_metrics.json](../reports/historical_metrics.json), [returned validation metrics](../reports/colab_efficient_v1/validation_metrics.json), and [the results-review notebook](../notebooks/01_results_review.ipynb).

## Limitations and risks

Annotation disagreement, class imbalance, platform differences, quoted offensive text, sarcasm, and identity references can affect predictions. Training labels do not establish intent or real-world danger. False positives can disproportionately burden benign discussions. The returned run's `threat` validation scores remain weak: AP 0.0753, F1 0.1081 and precision 0.0590 with 62 positives at its selected threshold. Five labels select the threshold-grid ceiling of 0.95; `toxic` uses 0.90. These frozen settings were not retuned and do not establish unconstrained optimality or calibrated probabilities.

No subgroup fairness study or robustness test was supplied. Mentioning an identity term is not evidence that a comment is hateful. A deployment should evaluate subgroup error rates, policy fit, abstention/review rules, and failure costs on an appropriate independently collected cohort.

The repository does not distribute trained weights or the returned private verification arrays. The returned bundle records selected-export consistency but contains no weights or tokenizer snapshot for an independent export/reload forward pass. It does not establish seed stability. The repository does not implement token-level rationales, establish causal explanations, or prove resource-efficiency claims from the original report. A short synthetic resource calibration is not an end-to-end speedup or inference-latency benchmark. A future explainability or latency study must identify its method, hardware, batch size, input length, and benchmark population.

## Ownership and maintenance

Mohammad Albatayneh confirmed that the original academic project was his individual work and that he wrote its code, using AI assistance to correct errors. The portfolio revision adds AI-assisted engineering and documentation work; the completed CPU experiment, returned Colab experiment, and synthetic checks remain separate from the historical neural results. No repository-wide license has been selected, and third-party data/tokenizer terms remain applicable. New results should identify their input version, frozen run, evaluation partition, thresholds, metric definitions, and whether the data are synthetic or real.

