# Report corrections and evidence ledger

The supplied three-page academic report was reviewed against the three uploaded notebooks. The two supplied PDFs are byte-identical; their source hashes are recorded in [the input manifest](source_manifest.json). Original PDFs remain outside the publication tree. This ledger excludes the original contact information and does not reproduce unverified literature references.

Evidence references use **zero-based notebook cell indices**. `FINAL` means `model_FINAL_Report.ipynb`, `XXXX` means `modelXXXX(1).ipynb`, and `UNTITLED` means `Untitled10.ipynb`. The audit examined code and saved outputs; it did not execute training. Historical values are preserved in [the metrics record](../reports/historical_metrics.json), with a longer notebook review in [the notebook audit](notebook_audit.md). The [corrected report](../reports/corrected_report.md) distinguishes these observations from the revised implementation.

| Original claim / location | Evidence | Correction adopted |
|---|---|---|
| Single credited author, page 1 | The title block credits Mohammad Albatayneh. He later confirmed that the project was individual and that he wrote the original code, with AI assistance to correct errors. | Record confirmed original authorship and AI-assisted error correction. Identify the new AI-assisted portfolio work separately; retain third-party source terms and do not infer a repository-wide license. |
| Direct use of the original Jigsaw dataset, pages 1-2 | All notebooks, cell 2: download `fizzbuzz/cleaned-toxic-comments`, read `train_preprocessed.csv`, select `set == "train"`. | Name the upstream cleaned derivative and its original-data lineage separately. Later mirror CSVs verify current counts, without proving the original download bytes or preprocessing lineage. |
| Training on all 159,571 comments, page 1 | Cell 2 holds out 20% of the selected labeled rows. | Describe training on the 80% partition. Do not present the full labeled source count as the optimization-set size. |
| Original challenge test set, page 2 | No test loader, challenge submission, or test evaluation is recorded. | Label all original results as internal validation results. |
| Advanced preprocessing; lowercasing, markup removal, whitespace normalization, pages 1-2 | The notebooks load preprocessed text; no such cleaning function appears. | Describe upstream preprocessing without claiming an implemented cleaning pipeline or unverified transformations. |
| Six-label stratified 80/20 split, page 2 | All cell 2: `sample(frac=0.8, random_state=42)` and complement. | Replace with a fixed-seed random 80/20 split. |
| RoBERTa model provenance, pages 1-2 | Cell 3 loads the pretrained tokenizer; cell 4 constructs `RobertaForSequenceClassification(config)` directly. | Explicitly state that encoder weights were randomly initialized. This was not pretrained encoder fine-tuning. |
| Single fully connected head, page 2 | Cell 4 instantiates the standard Hugging Face sequence-classification model. | Describe first-token representation, dropout, dense/tanh, dropout, and output projection; sigmoid is applied to logits during evaluation. |
| Maximum sequence length 512, page 2 | Cell 4 sets positional table size 512; cell 3 and inference use `max_length=256`. | Report the actual input cap of 256 tokens, separately from positional-table size. |
| Weighted BCE, AMP, AdamW, cosine schedule, page 2 | FINAL/XXXX cells 3-6 implement these settings. | Retain with the original run's learning rate `4e-5`, decay `1e-2`, batch 256, five epochs, 10% warmup. |
| Early stopping in the reported run, page 2 | FINAL/XXXX configure patience three and finish five epochs; UNTITLED actually stops at epoch seven. | Distinguish configured early stopping from activation in a separate experiment. |
| Best validation loss 0.382, page 2 | FINAL/XXXX cell 6: minimum 0.3482 at epoch 3; final loss 0.3824 at epoch 5. | Report both checkpoint identities. |
| Macro ROC-AUC 0.98 and macro F1 0.44, pages 1-3 | FINAL/XXXX cell 6, epoch 5: 0.9792 and 0.4391. | Retain as rounded **final-epoch historical validation** results, never fresh measurements or loss-best-checkpoint results. |
| Per-label results attributed implicitly to best model, page 2 | FINAL cell 9 uses the final arrays from cell 6 after cell 7 loads different weights. Cell 9 has retained output but a null execution count. | Label the table as retained final-epoch output. Do not treat it as a fresh re-evaluation of epoch 3. |
| Insult recall 0.96, Table II | FINAL cell 9 prints 0.945. | Use 0.945, or 0.95 if rounded to two decimals. |
| Modest minority-label recall, page 3 | FINAL cell 9: threat recall 0.853, identity-hate recall 0.911; precision 0.048 and 0.124. | Identify low precision and false positives as the observed rare-label limitation. |
| F1 plateau at 0.44, page 3 | FINAL cell 6 fluctuates across epochs; UNTITLED records 0.4689 in a distinct run. | Say final F1 was 0.4391; the logs do not establish a plateau. |
| LIME/token-level explanations, pages 1-3 | No LIME installation, import, wrapper, explanation, or assessment in any notebook. | Remove from achieved methods/results. Explanations remain potential future work. |
| DataLoader warning mitigation with workers zero, page 3 | Cell 6 contains worker-shutdown assertions; cell 3 still sets workers to at least one. | Preserve the observed warning; describe `num_workers=0` as a proposed remedy, not a documented historical change. Revised code defaults to zero workers. |
| 40% fewer parameters than BERT, page 1 | No named BERT baseline or parameter-count comparison. | Remove the percentage; do not infer an empirical efficiency gain from architecture size alone. |
| Under 100 ms on a consumer GPU; real-time deployment, pages 1-3 | No hardware-specific timing experiment or latency output. | Remove measured-latency and deployment-readiness claims. |
| Near-state-of-the-art performance, pages 1 and 3 | No common-split reference model, verified comparison, test result, or uncertainty estimate. | Report the observed metrics without a superiority claim. |
| Ten literature performance comparisons, page 1 and references on page 3 | Source papers were not supplied or verified; bibliographic entries are incomplete, including placeholder-like authorship in reference 1. | Omit those numerical comparisons and references. Use verified methodological documentation; any future literature review must consult original publications. |

## Interpretation and remaining limits

High ROC-AUC indicates favorable ranking of positive and negative examples across thresholds. It does not establish calibrated scores or high precision at threshold 0.5. The historical macro precision of 0.321 and macro recall of 0.924 explain why rare-label F1 remains low despite high recall. Macro F1 is the average of per-label F1 scores, not the harmonic mean of macro precision and macro recall.

The two five-epoch notebooks retain exactly the same rounded metric sequence. They are not independent replication evidence. UNTITLED changes learning rate, warmup, and planned epoch count, so its higher recorded F1 cannot be attributed to a single change. All historical validation partitions were repeatedly observed during model selection.

The repaired pipeline introduces a different internal split and evaluation protocol. The completed real-data CPU reference and returned efficient-profile neural run report new aggregate scores under that protocol; they are not a controlled before/after comparison with the original 80/20 split. The neural evidence concerns selected validation only, while the CPU reference additionally has a separately identified exploratory internal-test result. The recorded local CPU run times and synthetic GPU calibration do not establish per-comment latency, end-to-end speedup, or production readiness.

## Review status for revised methods

The real-data CPU run is complete, its public metrics were independently recomputed from saved private predictions, and normalized-group partition overlap is zero. The corrected report preserves this evidence and the original historical neural tables. Synthetic neural fixtures remain engineering checks.

The user returned a completed five-epoch efficient-profile run on 16 October 2026.
The canonical [notebook 04](../notebooks/04_colab_efficient_training.ipynb) and
[guide](colab_efficient_training.md) identify its workflow; notebook 03 is retired.
The [returned-run review](review_20261016.md) and
[verification record](../reports/colab_efficient_v1/verification.json) identify
the source, split, epoch-four checkpoint selection, threshold policy, recorded GPU
runtime, numerical checks, and limits. Validation macro AP 0.5259 and macro F1
0.5094 at selected thresholds are selection diagnostics. Its internal test was
not scored, and review neither reran GPU inference nor independently reloaded
weights. Five thresholds at the 0.95 grid ceiling and weak `threat` scores remain
explicit limitations. No historical result, frozen CPU artifact, or source hash
was rewritten to incorporate this new evidence.

Saved baseline validation predictions support a verified comparison on the same
source, grouped validation indices/order, targets, label order, and metric rules.
The CPU reference records validation macro AP 0.6549 and selected-threshold macro
F1 0.6414, above the returned neural values on these rows. This is a descriptive
selection comparison, not a held-out superiority or statistical significance
claim. Its [comparison record](../reports/colab_efficient_v1/baseline_validation_comparison.json)
does not use the CPU internal-test metrics as a validation comparator.

## Primary implementation reference

[Hugging Face's RoBERTa source documentation](https://huggingface.co/transformers/v4.8.0/_modules/transformers/models/roberta/modeling_roberta.html) defines the standard sequence-classification head and distinguishes initialization from a configuration from loading pretrained weights. The historical notebooks do not pin the runtime package version; a reproduced run must record that version.
