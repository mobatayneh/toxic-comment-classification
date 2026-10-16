# Data sources and access

Source review date: 2026-10-13. The initial notebook/report uploads contained no CSV or trained model. An anonymous request for the official competition archive returned HTTP 401 (`Unauthenticated`). The user later supplied `train_preprocessed.csv` and `test_preprocessed.csv` privately; their aggregate inspection is recorded in [data_audit.json](../reports/data_audit.json). No trained historical checkpoint was supplied.

The training mirror has 159,571 rows with complete binary labels; seven normalized-empty comments are explicitly excluded from the new run. The separate 153,164-row challenge-test mirror has entirely blank target columns and cannot be evaluated. The audit identifies 636 shared nonempty normalized-text groups across the two files. File hashes identify the supplied inputs, without asserting that they match the historical download or establishing redistribution rights.

## Historical input

The supplied code uses [`fizzbuzz/cleaned-toxic-comments`](https://www.kaggle.com/datasets/fizzbuzz/cleaned-toxic-comments), through `kagglehub.dataset_download`, and reads `train_preprocessed.csv`. It keeps only rows whose `set` column equals `train` and accepts `severe_tox` as an alias of `severe_toxic`.

This is a third-party processed mirror. Its preprocessing details, license, and source lineage were not recoverable from the reviewed page or metadata request. The report's statement that it directly used the original official dataset is therefore more specific than the code establishes. Reproducing the historical data preparation requires the exact authorized mirror file, with its source/version and SHA-256 recorded.

## Official Jigsaw source

The relevant competition is the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). Its six labels are distinct from the later multilingual and unintended-bias datasets.

The [official data page](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) describes the dataset as CC0 but expressly excepts Wikipedia comment text, for which it names `CC-SA-3.0`, and also states that competition rules apply. The canonical [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) terms and [CC0](https://creativecommons.org/publicdomain/zero/1.0/) terms should be consulted with the [competition rules](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/rules). Do not label all comment text as unconditionally CC0.

This repository distributes no real comments or mirror data. Attribution, modification notices, and applicable ShareAlike terms need review before distributing raw or processed text. Dataset terms do not automatically establish the license of any future model weights.

Mohammad Albatayneh's confirmed individual authorship of the original project code and approval of public code publication do not resolve the mirror's terms. No repository-wide license has been selected or inferred from a third-party component.

## Obtain an authorized local CSV

Use Kaggle's official access path from an account authorized to download the competition data. If Kaggle requires authentication or acceptance of the rules, complete that step in the user's account. Do not place API tokens in the repository.

After authorized authentication, the official [Kaggle CLI competition documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md) describes:

```bash
kaggle competitions files jigsaw-toxic-comment-classification-challenge
kaggle competitions download jigsaw-toxic-comment-classification-challenge \
  -p data/raw --unzip
```

These are acquisition instructions, not a claim that the download succeeded here. For the runnable reference pipeline, supply official `train.csv` with `comment_text` and all six binary label columns. For faithful mirror-input reproduction, instead supply the original authorized `train_preprocessed.csv`; its rights remain a separate unresolved issue.

Keep the source URL, acquisition date, file version, file SHA-256, column names, row count, and any processing steps in a local provenance record. The run checks the exact source bytes. Moving a file is allowed when its bytes are identical; editing it creates a different source.

The competition also offers `test.csv` and `test_labels.csv`. They must be joined on `id`; unknown labels marked `-1` are not negative examples and must not be scored as such. The current pipeline creates an internal partition from the supplied training cohort and does not silently join or evaluate those external test files.

The canonical [Colab notebook 04](../notebooks/04_colab_efficient_training.ipynb) and [runner](../scripts/colab_train.py) require the exact privately supplied `train_preprocessed.csv`. They check its SHA-256 and expected source/eligible counts and partitions before full training: 159,571 source rows, 159,564 eligible rows, and 111,731/23,878/23,955 training/validation/internal-test rows. The [returned completed run](review_20261016.md) records this same input identity and keeps its internal test unscored. An official CSV or changed mirror file is a different input and will not silently pass that source guard. Use the general neural CLI for a separately documented experiment on a different authorized source. A matching fingerprint supports reproducibility; it does not resolve the mirror's preprocessing lineage or terms. [The current Colab guide](colab_efficient_training.md) covers private Drive storage, upload and resume; [the legacy guide](colab_training.md) records old-run recovery.

## Tokenizer and model assets

The historical tokenizer source is [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base). The model repository advertises MIT, with upstream provenance in [fairseq](https://github.com/facebookresearch/fairseq) and its [MIT license](https://github.com/facebookresearch/fairseq/blob/main/LICENSE). Preserve applicable notices when redistributing tokenizer or pretrained components. No repository-wide license is inferred from that component license.

Only the pretrained tokenizer is used in the historical custom model. No pretrained encoder weights are loaded. A normal real-data neural run still needs tokenizer files; cache them in an authorized connected environment, record the repository revision, and use a local directory with `--local-files-only` for later offline runs. Hugging Face's [offline-mode documentation](https://huggingface.co/docs/transformers/installation#offline-mode) explains that the files must already exist locally.

The neural smoke check uses a tiny synthetic tokenizer and random model entirely offline. Its outputs cannot estimate the historical or deployed model's performance.

## Method references

- [scikit-learn average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html): the primary new ranking metric is AP, not trapezoidal PR area.
- [scikit-learn ROC-AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html): macro averaging and continuous-score evaluation.
- [scikit-learn F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html): macro and micro decision metrics.
- [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage): split before fitting learned preprocessing.
- [Transformers RoBERTa documentation](https://huggingface.co/docs/transformers/en/model_doc/roberta): model initialization and architecture interfaces.

The original report's ten literature references were not verified, including a placeholder-looking `J. Doe et al.` entry. Their benchmark numbers are excluded from the corrected report and repository performance claims.

