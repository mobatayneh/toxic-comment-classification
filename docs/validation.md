# Verification record

Updated on 15 October 2026. The [follow-up review](review_20261015.md) records the new integrity fixes, full enabled CPU suite and automatic neural CI. [PR #1](https://github.com/mobatayneh/toxic-comment-classification/pull/1) is the authoritative record of tested commits and merge state. The dated 13–14 October entries below are retained as historical verification evidence, not current test counts. Full real-data Colab/CUDA results have not yet been reviewed.

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

The first push/PR runs failed only at notebook display because pandas Styler required undeclared Jinja2. Adding the locally tested `Jinja2==3.1.6` notebook dependency resolved that clean-environment failure. No model code, recorded metrics or figures changed, and no real training/test inference was repeated. Full Colab/Drive/CUDA execution remains unverified.

## Actual-data experiment

The full eligible-cohort CPU reference was trained, exported, reloaded, frozen and evaluated once. Source 159,571 rows minus seven explicitly excluded empty texts gives 159,564 eligible rows. Partition counts are 111,731 train, 23,878 validation and 23,955 internal test. Root review independently recomputed the published AP/ROC-AUC/F1 scores from the saved private predictions and found zero normalized duplicate-group overlap. [The machine-readable experiment record](../reports/baseline_verification.json) contains exact results, resource observations and limits.

No original eight-layer neural real-data rerun was completed. Historical neural output values remain [historical evidence](../reports/historical_metrics.json). The successful tiny neural mechanics run verifies training/backward, best-loss restoration, export/reload coherence and a frozen synthetic test; its scores are not portfolio performance.

## Notebook and artifact checks

All eleven code cells in [the results review](../notebooks/01_results_review.ipynb) executed top-to-bottom in a fresh in-process IPython session, saving aggregate tables and four native figures. Local Jupyter/ZeroMQ kernel startup is blocked by this environment; the kernel protocol was not verified. The same interpreter can run a normal kernel in environments that permit local sockets:

```bash
python scripts/run_results_notebook.py
python scripts/run_results_notebook.py --engine inprocess
```

The eight exact default code cells in [the neural tutorial](../notebooks/02_train_roberta.ipynb) passed sequential in-process checking. Expensive real training defaults to disabled; its publication outputs are cleared. The three archival source derivatives have cleared outputs and documented omitted illustrative comments. Corrected report tables are generated from committed aggregate evidence and verified against it. Report/PDF and CV presentation are reviewed separately; career files are excluded from the repository.

The separate [full-training Colab notebook](../notebooks/03_colab_full_training.ipynb) embeds eight visible source-writing cells and has all code outputs/counts cleared. It is a training handoff: executing its main training cell launches the full GPU experiment. Private verification and internal-test evaluation default to disabled. [The guide](colab_training.md) records the exact source, architecture, partitions, resume requirements and result bundles. Local source materialization and synthetic CPU continuation checks do not establish Colab/Drive service execution, CUDA behavior or real-data neural performance.

The publication verifier checks relative links, evidence tables, archived outputs, source-only publication boundaries and likely identifiers/secrets. A focused independent review found no material privacy issue in candidate files or reachable Git history. These checks complement review; they do not prove the absence of every possible sensitive string.

[validation_summary.json](../reports/validation_summary.json) records commands, counts and scope. Dependencies are pinned directly and their installed versions are recorded privately with each frozen run; this is not a complete transitive environment lock. Seeds and deterministic settings do not promise cross-hardware bit-identical neural results.
