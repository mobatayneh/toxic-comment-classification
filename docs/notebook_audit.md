# Audit of the three supplied classification notebooks

Audit method: read the uploaded notebook JSON, source cells, and stored textual outputs. No training, dataset download, model download, or notebook execution was performed. Original files were read only. Cell numbers below are **zero-based notebook cell indexes**. Machine-readable extraction, including output indexes and file SHA-256 values, is in `historical_metrics.json`.

## Recommendation and comparison

Use `model_FINAL_Report.ipynb` as the main implementation reference because it includes the complete common training pipeline, the only per-label reporting code, and architecture-diagram code. Preserve the original files separately. Treat `Untitled10.ipynb` as a historical hyperparameter variant, and `modelXXXX(1).ipynb` as a near-duplicate 5-epoch version. None is a clean, independently reproducible benchmark in its current form.

| Notebook | Contents | Requested / recorded epochs | Learning rate | Warmup fraction | Main distinction |
|---|---|---:|---:|---:|---|
| `model_FINAL_Report.ipynb` | 11 code cells | 5 / 5 | 0.00004 | 0.10 | English comments, per-label report and plotting in cell 9, Graphviz diagram in cell 10 |
| `modelXXXX(1).ipynb` | 9 code cells | 5 / 5 | 0.00004 | 0.10 | Arabic comments; all five rounded epoch metric records match FINAL_Report exactly |
| `Untitled10.ipynb` | 9 code cells | 10 / 7 | 0.00003 | 0.15 | Longer requested run, different learning rate/warmup; early stops after epoch 7 |

All cells are code cells; there are no Markdown explanations in the originals. The notebooks contain illustrative inference examples, which do not form a labeled evaluation dataset. Their text is deliberately omitted from this audit.

## Exact shared model and training configuration

Sources: all notebooks, cells 1 and 3–6.

- Tokenizer: `AutoTokenizer.from_pretrained("roberta-base")`, using its existing byte-level BPE vocabulary. The tokenizer is downloaded, not fitted on the project dataset.
- Initialization: `RobertaForSequenceClassification(RobertaConfig(...))`. **The model weights are initialized from scratch. There is no call loading pretrained RoBERTa model weights.** Calling this pipeline pretrained-model fine-tuning would misdescribe the code.
- Encoder: hidden size 512; 8 transformer layers; 8 attention heads; intermediate feed-forward size 2,048; maximum position embeddings 512; hidden and attention dropout 0.1; vocabulary size taken from `tokenizer.vocab_size`.
- Classification: 6 logits from the built-in RoBERTa sequence-classification head; `problem_type="multi_label_classification"`; explicit sigmoid applied to logits for probabilities. The built-in head uses the first-token representation, dropout, a dense transformation with tanh, and the final label projection. The diagram's single “Dense Head / 6 Sigmoid neurons” is an approximation; sigmoid occurs outside the model in the actual pipeline.
- Label order: `identity_hate`, `insult`, `obscene`, `severe_toxic`, `threat`, `toxic`. This differs from common dataset column orders and must be saved explicitly with every checkpoint and prediction output.
- Inputs: truncation and fixed padding to 256 tokens; tokenization occurs in `Dataset.__getitem__` on every access. Attention masks are passed to the model. Padding/tokenizer identifiers are not explicitly copied into the custom config; standard RoBERTa defaults happen to match the selected tokenizer but should be recorded rather than assumed.
- Physical batch size: 256; no gradient accumulation; `num_workers=max(1, os.cpu_count()-1)`; pinned memory enabled unconditionally.
- Loss: `BCEWithLogitsLoss(pos_weight=negative_count / (positive_count + 1e-10))`; class weights are calculated from the training subset only. No check rejects labels with zero positive examples.
- Optimizer: AdamW, weight decay 0.01. Learning rate and warmup differ as shown above. Cosine scheduling runs for the planned number of batches times requested epochs.
- AMP: deprecated `torch.cuda.amp.autocast` and `GradScaler`, used without explicit device gating. Scheduler stepping is unconditional even when AMP skips an optimizer update.
- Selection: lowest validation weighted BCE loss; patience 3; checkpoint named `best_model.pt` contains epoch, model state, optimizer state, and scheduler state. It omits scaler state, RNG states, split manifest, label mapping, class weights, and training configuration.
- Random seed: 42 for Python, NumPy, and PyTorch. cuDNN benchmarking is enabled, so seeds alone do not establish deterministic runs. `CUDA_LAUNCH_BLOCKING=1` is permanently enabled for debugging, which can slow training.

## Data, splitting, and leakage assessment

Sources: all notebooks, cells 2–3.

The source is KaggleHub dataset `fizzbuzz/cleaned-toxic-comments`, with expected file `train_preprocessed.csv`. If `archive.zip` exists, the code extracts it into the download directory. `severe_tox` is renamed to `severe_toxic`. Only rows where `set == "train"` are selected. That filtered frame is reset to contiguous indexes, sampled at fraction 0.8 with seed 42, and validation is the remaining indexes.

The row-index split is internally correct: the sampled frame retains its original indexes until those indexes are dropped from `train_full`. There is no direct row-index overlap caused by the subsequent validation reset. No tokenizer is fitted on validation data, and loss weights use training data only.

However, the notebooks do not check repeated IDs, exact duplicate comments, duplicates created by upstream cleaning, conflicting duplicate labels, or duplicates across partitions. Therefore they establish disjoint rows but do not establish independent comments. Label prevalence is not stratified or checked; rare labels can disappear from a small split. The preprocessing performed by the upstream “cleaned” dataset is not specified by these notebook cells and cannot be reconstructed from the uploads alone.

No provided notebook output establishes actual row counts, label counts, dataset file hash, dataset version metadata, or a dataset license. FINAL_Report cell 2 records a download URL for dataset version 1, but the executable source does not pin that version. No CSV, trained weights, or prediction arrays accompanied these notebook uploads. Dataset sizes or class balances quoted elsewhere must remain unverified until the dataset is inspected.

**No held-out test evaluation exists.** Rows with other `set` values are ignored. Validation is repeatedly used for checkpoint selection and metric inspection, so the recorded numbers are validation results. The lists called `test_texts` are demonstration inputs without formal ground-truth vectors. They do not support test-set accuracy, robustness, or fairness claims.

Minimal repair: validate required text/label columns; accept the original severe-label alias; reject missing/non-string text and nonbinary labels; record source hash and split IDs; keep all copies of a normalized comment in a single partition; check each partition's positive/negative counts. Preserve the original 80/20 training/validation intent. Evaluate a separate supplied test partition only when it has complete valid labels, and document its provenance.

## Historical metric evidence

The following numbers are **rounded values stored in uploaded notebook outputs**, not freshly computed results. All epoch metrics come from cell 6. AUC is macro mean one-versus-rest ROC-AUC on sigmoid scores. F1 is macro F1 with strict `probability > 0.5`; precision/recall are calculated but not printed by the epoch loop.

| 5-epoch versions: epoch | Training loss | Validation loss | Validation macro ROC-AUC | Validation macro F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.7962 | 0.3941 | 0.9720 | 0.3688 |
| 2 | 0.3543 | 0.3679 | 0.9766 | 0.4357 |
| 3 | 0.2803 | **0.3482** | 0.9790 | 0.3900 |
| 4 | 0.2409 | 0.3735 | **0.9795** | 0.4306 |
| 5 | 0.2197 | 0.3824 | 0.9792 | **0.4391** |

FINAL_Report and modelXXXX share every displayed value in this table, as well as the probabilities of their quick positive-sentiment example. This is matching historical evidence, not two independent replications. The selected checkpoint is epoch **3**, whereas the final epoch is 5.

| Untitled10: epoch | Training loss | Validation loss | Validation macro ROC-AUC | Validation macro F1 |
|---:|---:|---:|---:|---:|
| 1 | 1.0842 | 0.6579 | 0.9570 | 0.4369 |
| 2 | 0.4316 | 0.3910 | 0.9743 | 0.3886 |
| 3 | 0.3214 | 0.3514 | 0.9781 | 0.3716 |
| 4 | 0.2653 | **0.3498** | 0.9801 | 0.3818 |
| 5 | 0.2259 | 0.3610 | **0.9811** | 0.4221 |
| 6 | 0.1984 | 0.4412 | 0.9808 | **0.4689** |
| 7 | 0.1756 | 0.4081 | 0.9810 | 0.4459 |

Untitled10 selects epoch **4** by validation loss and stops after three further non-improving epochs. Its maximum AUC, maximum F1, last epoch, and selected checkpoint are different. A README must not assemble them into a fictitious single “best model” result.

FINAL_Report cell 9 contains the only per-label report. It has `execution_count: null` despite a stored output. This may reflect notebook metadata editing; it is not proof of either a fresh execution or a failed execution. Its macro F1 0.439 matches the final epoch's rounded 0.4391, and its code consumes the final epoch's arrays.

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| identity_hate | 0.124 | 0.911 | 0.218 |
| insult | 0.440 | 0.945 | 0.601 |
| obscene | 0.539 | 0.959 | 0.690 |
| severe_toxic | 0.203 | 0.975 | 0.336 |
| threat | 0.048 | 0.853 | 0.090 |
| toxic | 0.572 | 0.900 | 0.700 |
| Macro mean | 0.321 | 0.924 | 0.439 |

These stored values indicate a high-recall, low-precision operating point, especially for threat and identity_hate. High ROC-AUC is not equivalent to high classification accuracy or useful precision at threshold 0.5. Weighted BCE changes the operating point; its sigmoid outputs should not be presented as calibrated probabilities without calibration evidence. No accuracy, micro F1, PR-AUC, confidence interval, or fairness metric is saved by the originals.

## Material errors and minimum repairs

1. **Checkpoint/export/report mismatch — all notebooks, cells 6–7; FINAL_Report cell 9.** `save_pretrained` runs before loading `best_model.pt`. Exported weights are from the last epoch, while demonstration predictions use the best-loss checkpoint. The report uses prediction arrays retained from the last epoch and never recomputes them after the reload. Load the selected checkpoint first, recompute validation predictions, calculate reports, and export model/tokenizer/config/threshold/labels together. Keep each metric's checkpoint epoch and evaluation partition in its metadata.

2. **Unsafe reusable filenames and incomplete checkpoint provenance — cells 5–7.** Shared names `best_model.pt`, `runs/toxic_custom_roberta`, and export directories can be overwritten or reused between runs. Use a run directory, explicit config and label mapping, guarded checkpoint loading with `map_location`, and a split/source manifest. Keep weights and datasets out of Git. Full training resumption additionally requires AMP scaler and RNG states; do not promise resumption unless implemented.

3. **Multiprocessing cleanup errors — cell 6 outputs in all three files.** Stored streams include repeated `AssertionError: can only test a child process` from DataLoader worker cleanup. These are ignored cleanup tracebacks; epoch metrics continue afterward and there are no notebook `output_type="error"` entries. Do not describe the runs as aborting at their first traceback. Default workers to 0 for notebooks and portable CLI use; make workers configurable; protect CLI entry points. Do not presume the cleanup issue proves training correctness either.

4. **Hardware portability — cells 3–4 and 6.** Batch 256 at sequence length 256 is a demanding physical batch. `os.cpu_count()` can be `None`, and spawning nearly every CPU worker can be excessive. Use a small default physical batch with explicit gradient accumulation if preserving effective batch 256; leave the original physical batch available in a historical config. Enable pinned memory and AMP only on appropriate devices. Replace deprecated AMP APIs and gate scheduler updates to real optimizer steps.

5. **Incorrect loss aggregation and brittle metrics — cell 6 and FINAL_Report cell 9.** Epoch loss is the mean of batch means, which gives the final short batch equal influence. Aggregate by sample count. Explicitly handle an absent positive/negative label for ROC-AUC and use confusion-matrix labels `[0, 1]`; avoid silently turning undefined AUC into a claimed score. Cast logits/probabilities to float32 before collecting metrics. Retain a documented threshold-0.5 baseline. Optional threshold calibration must use validation/calibration data, save its thresholds, and remain separate from held-out test evaluation.

6. **Missing runnable-project infrastructure — cells 1 and 10.** Interactive `!pip install` commands are not ordinary Python. Dependencies are unpinned; pandas and reporting dependencies are assumed. Graphviz diagram generation also requires an external Graphviz executable and writes to hard-coded `/mnt/data`. Move dependencies to a declared environment, expose CLI configuration/data paths, replace hard-coded artifact paths, and make reporting dependencies optional or explicit. A deterministic tiny smoke test can verify train/save/reload/predict/report wiring without presenting synthetic results as a toxic-comment benchmark.

7. **Reproducibility and claims — all notebooks.** Set seeds and document deterministic settings, but do not promise cross-hardware bit-for-bit identity. Preserve original outputs as historical evidence and clear the derived runnable notebook's outputs until genuinely executed. Untitled10's execution counts jump from 3 to 20 before tokenization; FINAL_Report's reporting cell has a null count. Stored outputs alone cannot establish clean top-to-bottom execution, environment versions, or the identity of absent checkpoints.

## Suggested repository evidence boundaries

- Historical metrics JSON and an audit should identify their uploaded source file, source hash, cell/output indexes, evaluation split, threshold, and rounding.
- New runs should write their own configuration, source/split hashes, training history, selected epoch, per-label report, thresholds, and export metadata. Report tables should be derived from those artifacts.
- Use FINAL_Report's 5-epoch architecture/training settings as the faithful baseline; expose Untitled10's 10-epoch variant as an optional historical configuration.
- A smoke-tested runnable project is not an independently reproduced benchmark. No trained model, full-data validation result, or held-out test result should be represented as delivered until a real dataset run supplies that evidence.
