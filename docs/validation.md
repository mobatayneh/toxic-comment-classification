# Verification record

Updated on 16 October 2026. The [returned-run review](review_20261016.md) records
the completed efficient-profile Colab experiment and its private numerical
verification. The [15 October review](review_20261015.md) records earlier integrity
fixes, CPU checks and automatic neural CI. The dated entries below preserve the
environments, source revisions, and counts actually checked; earlier passing CI
does not establish the status of a later revision.

## Returned Colab experiment — 16 October 2026

The user returned aggregate results, resource calibration, and a private numerical
bundle after five real-data epochs completed. Epoch four was selected by validation
loss 0.36014178103214034; its reported fresh-export validation loss matches. The
saved validation arrays support review of aggregate ranking/decision metrics and
threshold choices; the public
[verification record](../reports/colab_efficient_v1/verification.json) states the
exact source, partition, target, metadata, and comparison checks performed.

Independent scikit-learn calculations and project code reproduce all returned
validation metrics and thresholds within absolute tolerance 1e-12. The original
source byte hash, all grouped split indices and their row order, validation
targets, full cohort summary, training-only class weights, and zero normalized
group overlap were verified. The baseline's saved validation predictions were
also recomputed, with the same source/partitions/order/targets established before
the [validation-only comparison](../reports/colab_efficient_v1/baseline_validation_comparison.json).
Neither model was deserialized or run for this numerical audit. Its environment
was Python 3.12.14 with NumPy 2.3.5, pandas 2.2.3 and scikit-learn 1.8.0.

At selected thresholds the run reports validation macro AP 0.5259, macro ROC-AUC
0.9761, macro F1 0.5094, and micro F1 0.6520 on 23,878 rows. Both checkpoint and
threshold selection used this partition. Five thresholds reach the 0.95 grid
ceiling; no threshold was changed during review. The neural internal test remains
unscored. No real training, GPU inference, or test evaluation was repeated.

The recorded GPU environment is Python 3.13.15, torch 2.11.0+cu130,
CUDA 13.0, cuDNN 92700, and a Tesla T4. The full dependency versions remain in
[`versions_hardware.json`](../reports/colab_efficient_v1/versions_hardware.json).
These values differ from the local Python 3.12 / torch 2.8.0+cpu engineering
checks below. Weights, tokenizer files, and the full private run identity were not
returned, so model export/reload, fresh forward-pass loss, tokenizer-file hashes,
checkpoint contents, and actual interruption/resume were not independently
reproduced from this bundle. The recorded export flags and matching fresh loss
are internally consistent metadata, rather than a second inference run.

The canonical training workflow is
[notebook 04](../notebooks/04_colab_efficient_training.ipynb) and its
[guide](colab_efficient_training.md). Notebook 03 is retired; older optimizer
checkpoints require their exact original source and environment from Git history.
The calibration selects a resource setting using synthetic probes and does not
establish an end-to-end speedup or a production-latency result.

The expanded [results-review notebook](../notebooks/01_results_review.ipynb)
executed all 18 code cells in a fresh in-process IPython session and saved seven
native figures, including the new neural and same-validation comparison plots.
Its metadata records that execution mode and `kernel_protocol=false`. Normal
Jupyter kernel startup remains blocked locally by socket permissions. The current
18-cell notebook also executed through a normal Jupyter kernel in the successful
GitHub Actions run recorded below.

## Current software and publication checks — 16 October 2026

The complete local suite passed **96 tests**, with **zero failures, errors or
skips**, with every optional neural check enabled. The machine-readable
[per-test record](../reports/review_20261016_checks.json) identifies the tested
source hashes, environment and outcomes. This includes tiny CPU training and
exact-resume mechanics; it does not repeat real-data model training or inference.
After the suite, the PDF builder's stale footer date and new figure caption were
corrected and verified by rebuilding, extracting the text and inspecting the
17-page PDF. That presentation-only follow-up is recorded separately in the JSON.

The results-review notebook executed all 18 code cells and saved seven native
figures. Both Colab builders pass `--check`; notebook 03 is a retirement notice,
and every executable cell in notebook 04 remains byte-identical to the supplied
training notebook. Publication validation checks relative links, historical and
current evidence tables, archive privacy, and the retired entry point. All three
new figures passed exact pixel comparison with the same renderer; their numerical
series and public file hashes are checked by the portable verifier in CI.

The [push workflow](https://github.com/mobatayneh/toxic-comment-classification/actions/runs/38893185310) and
[PR workflow](https://github.com/mobatayneh/toxic-comment-classification/actions/runs/38893249752)
both passed for published commit `946299d43e400af92016da1c8a4f65be553e7291`.
The core job passed 72 tests and deferred 24 neural checks. The neural job passed
31 tests with zero skips, including every deferred check; seven checks overlap
the jobs, giving 96 distinct passing tests. Test identities were reconciled from
the job logs. The expanded 18-cell evidence notebook executed through a normal
Jupyter kernel, and the repository and public aggregate checks passed. The
[publication PR](https://github.com/mobatayneh/toxic-comment-classification/pull/2) records this update.
The neural job requires zero skips. The [live workflow runs](https://github.com/mobatayneh/toxic-comment-classification/actions)
and pull-request checks determine remote success for each exact commit; earlier
passing revisions are not substitutes.

## Software checks recorded on 14 October 2026

The candidate suite recorded on 14 October 2026 passed **43 tests, with zero failures, errors or skips**, in **15.492 seconds** on Python 3.12.14 with torch 2.8.0+cpu. It includes 31 always-on core/review checks and 12 explicitly enabled tiny CPU neural checks. The ten added Colab checks comprise two always-on notebook/portable-import checks and eight CPU training/resume/bundle checks.

| Test module | Passed |
|---|---:|
| Data and metrics | 14 |
| Frozen artifacts | 7 |
| CPU baseline | 2 |
| Neural pipeline | 4 |
| Colab notebook, resume and result bundles | 10 |
| Publication verifier | 6 |
| **Total** | **43** |

The prior base suite passed 33 tests with zero skips in 7.922 seconds on 13 October 2026. That record is retained as prior evidence; that suite covered the Colab additions.

```bash
python -m pip install -r requirements-dev.txt
# Install the optional CPU neural environment from README before enabling it.
PYTHONPATH=src TOXIC_RUN_NEURAL_TESTS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTHONHASHSEED=42 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m unittest discover -s tests -v
python -m compileall -q src scripts
python scripts/verify_repository.py
```

Repository test discovery needs `requirements-dev.txt`, which includes the core and notebook/review dependencies. `requirements-core.txt` alone supports CPU training and its baseline smoke command. Optional neural checks are skipped unless explicitly enabled with their dependencies installed. The tests verify train-only vocabulary and independently calculated TF-IDF weights, held-out-only vocabulary exclusion, duplicate-group isolation, strict CSV policies, validation thresholds, AP definitions, frozen artifacts, one-attempt test evaluation, best-checkpoint export and unequal microbatch sample weighting. CI configuration parses locally. At that date, the core job used synthetic fixtures and the aggregate-results notebook, while the neural job was manually enabled. The 15 October revision removes that job-level gate: CPU neural checks run automatically on push, pull request and manual workflow runs. Missing dependencies and skipped checks fail the neural job.

The added Colab checks execute all eight visible source-writing cells in in-process IPython, verify exact embedded bytes and the manifest, and import the materialized project in a subprocess where all five optional neural modules and their distribution metadata are deliberately inaccessible. The CPU tests independently check interruption/resume model, optimizer, scheduler and RNG state, uneven accumulation math, patience and best-model restoration, identity mismatch rejection, atomic weights-only-safe checkpoints, read-only frozen restart, and aggregate/private ZIP boundaries. The reviewed runner, builder, notebook, requirements, tests, CI file and six core modules remained unchanged during the recorded suite.

The 14 October console transcript is partial. Counts, duration, failure/skip lists and checked-file hashes come from the complete structured runner result; missing per-test console lines were not reconstructed. The machine-readable repository summary records this evidence limit.

## GitHub publication checks

The repaired source at remote commit `1391907f6945069b25831cf7f77b1a38116b4895` passed both [push CI](https://github.com/mobatayneh/toxic-comment-classification/actions/runs/38525905392) and [PR CI](https://github.com/mobatayneh/toxic-comment-classification/actions/runs/38525910900). The clean Python 3.12.15 runner discovered 43 tests, passed 31 core/review checks, and explicitly skipped the 12 optional CPU neural checks; core tests took 8.861 seconds. All eleven results-review code cells executed through a normal Jupyter kernel. The publication verifier passed for 63 files, 92 local links, two historical and two CPU evidence blocks, and three sanitized archive notebooks. The executed notebook was retained as the historical-evidence-review run artifact.

The first push/PR runs failed only at notebook display because pandas Styler required undeclared Jinja2. Adding the locally tested `Jinja2==3.1.6` notebook dependency resolved that clean-environment failure. No model code, recorded metrics or figures changed, and no real training/test inference was repeated. At that publication, full Colab/Drive/CUDA execution remained unverified.

## CPU actual-data experiment — 13–15 October 2026

The full eligible-cohort CPU reference was trained, exported, reloaded, frozen and evaluated once. Source 159,571 rows minus seven explicitly excluded empty texts gives 159,564 eligible rows. Partition counts are 111,731 train, 23,878 validation and 23,955 internal test. Root review independently recomputed the published AP/ROC-AUC/F1 scores from the saved private predictions and found zero normalized duplicate-group overlap. [The machine-readable experiment record](../reports/baseline_verification.json) contains exact results, resource observations and limits.

At the time of these CPU checks, no original eight-layer neural real-data rerun had been completed. Historical neural output values remain [historical evidence](../reports/historical_metrics.json). The successful tiny neural mechanics run verifies training/backward, best-loss restoration, export/reload coherence and a frozen synthetic test; its scores are not portfolio performance. The separate returned real-data neural experiment is recorded above.

## Earlier notebook and artifact checks — 13–15 October 2026

All eleven code cells in the earlier results-review version executed top-to-bottom in a fresh in-process IPython session, saving aggregate tables and four native figures. Local Jupyter/ZeroMQ kernel startup was blocked by this environment; the kernel protocol was not verified. The same interpreter can run a normal kernel in environments that permit local sockets:

```bash
python scripts/run_results_notebook.py
python scripts/run_results_notebook.py --engine inprocess
```

The eight exact default code cells in [the neural tutorial](../notebooks/02_train_roberta.ipynb) passed sequential in-process checking. Expensive real training defaults to disabled; its publication outputs are cleared. The three archival source derivatives have cleared outputs and documented omitted illustrative comments. Corrected report tables are generated from committed aggregate evidence and verified against it. Report/PDF and CV presentation are reviewed separately; career files are excluded from the repository.

The earlier full-training Colab notebook 03 embedded eight visible source-writing cells and had all code outputs/counts cleared. Private verification and internal-test evaluation defaulted to disabled. The current [03 file](../notebooks/03_colab_full_training.ipynb) is a retirement notice; the [legacy guide](colab_training.md) records recovery of old runs. Current notebook 04 and its [guide](colab_efficient_training.md) replace that training handoff. Local source materialization and synthetic CPU continuation checks remain distinct from the returned real-data GPU evidence and do not themselves establish CUDA behavior or model quality.

The publication verifier checks relative links, evidence tables, archived outputs, source-only publication boundaries and likely identifiers/secrets. A focused independent review found no material privacy issue in candidate files or reachable Git history. These checks complement review; they do not prove the absence of every possible sensitive string.

[validation_summary.json](../reports/validation_summary.json) records commands, counts and scope. Dependencies are pinned directly and their installed versions are recorded privately with each frozen run; this is not a complete transitive environment lock. Seeds and deterministic settings do not promise cross-hardware bit-identical neural results.
