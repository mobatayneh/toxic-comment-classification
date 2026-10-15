# Progress and resumption record

## Goal
Prepare Mohammad Albatayneh’s toxic-comment multi-label classification project for reviewed GitHub publication. Code and documentation in English. Preserve original academic artifacts and distinguish historical saved outputs from new runs. Never publish or merge without project-specific approval.

## Current status — follow-up review, 2026-10-15
The user authorized a second review, all neural checks and merge after successful verification. The original 43-test suite passed again with all twelve neural checks enabled. Concrete source-provenance and interrupted-finalization findings are fixed; 50/50 tests pass with zero skips/failures/errors; the latest follow-up section below records completed stages. PR #1 and its checks are the authoritative publication/merge record. The user is training the original Colab notebook separately; do not replace its embedded source while that run is active. Full real-data neural performance remains unverified until the results are returned.

## Initial snapshot — 2026-10-13
- Inputs: three notebooks and two byte-identical reports; original files remain unchanged outside the publication tree in ../upload/.
- No dataset or saved model supplied. No existing repository for this project identified among connected account repositories. Confirmed account: mobatayneh.
- Prior diabetes project read only as documentation reference; no changes or execution.
- Three GPT-6.1 Sol / Max audit agents: notebooks, report, sources/rights.
- Preliminary foundation: model_FINAL_Report.ipynb, pending complete audit.
- Original architecture: random-initialized 8-layer RoBERTa with a pretrained tokenizer, not pretrained encoder fine-tuning.

## Decisions
- Fix correctness and reproducibility before considering new architectures.
- Preserve originals privately; avoid publishing student contact information, raw comments, individual predictions, secrets or obsolete unqualified claims.
- No repository-wide license until ownership and third-party rights are established.
- Public results must identify whether historical, synthetic or newly rerun real-data results.

## Changed files
- .gitignore: excludes local data, artifacts, model weights and credentials.
- docs/source_manifest.json: exact input hashes.
- PROGRESS.md: this checkpoint.

## Verification
- Two PDF SHA-256 hashes match exactly.
- Inputs parsed and source/output text extracted for audit (not training execution).
- No real-data model run or GitHub publication yet.

## Remaining / next step
Finish source/claim audit; implement repaired runnable pipeline and offline mechanics tests; prepare corrected report, documentation and career wording; ask ownership questions and first-publication approval after concrete package is ready.

## Important commands
Run commands from this project directory. Runtime and tests will be documented after implementation.

## Stage 1 completed — audit checkpoint
- Selected model_FINAL_Report.ipynb as the most complete reference; modelXXXX has identical five-epoch logs, not independent replication. Untitled10 is a different hyperparameter variant.
- Confirmed critical export/report mismatch: model saved before best-checkpoint load, and per-label report uses stale epoch-5 arrays. Correct best-loss epoch is 3 (loss 0.3482, validation macro ROC-AUC 0.9790, macro F1 0.3900); final epoch 5 F1 is 0.4391. These are historical rounded logs, not reruns.
- Unsupported PDF claims include stratified split, LIME/token rationale implementation, measured sub-100ms latency, 40% parameter saving against a measured BERT baseline, and near-SOTA comparison. Rare-label issue is low precision despite high recall.
- Copied historical aggregate evidence and audit into repository; preserved all original files in private recovery ZIP. Sanitized historical source notebooks clear outputs and remove demonstration input/prediction blocks.
- Parallel implementation owners: CPU/shared core; neural workflow; tests/CI; docs/results notebook; report correction. All delegated agents explicitly GPT-6.1 Sol, Max.
- Installed notebook execution dependencies plus torch 2.8.0 CPU and transformers 4.57.1; neural GPU unavailable.
- Next: synthetic integration/invariant checks, official-data access attempt, documentation alignment. No GitHub writes yet.

## Resumed with real data — 2026-10-13
- Uploaded train_preprocessed.csv has 159,571 labeled rows; test_preprocessed.csv has 153,164 rows with every label empty. Do not score challenge test or turn missing labels into zero.
- Train audit: seven whitespace-only comments; 737 normalized duplicate groups (2,061 rows), 93 conflicting-label groups (213 rows), no repeated IDs.
- Chosen explicit policy: exclude seven empty training texts; retain all other original annotations, including conflicting labels, and keep entire normalized-text groups in one split. Derived toxicity is exactly the sum of six target labels; it is excluded from features.
- Aggregate audit: reports/data_audit.json. Uploaded CSVs remain private outside publication tree.
- Prior pip-installed packages were absent after suspension. Creating isolated .venv and rechecking dependency availability.
- Four GPT-6.1 Sol / Max agents resume incomplete core, neural, tests/CI and docs; no completed audit is repeated.
- Next: finish existing modules, run full eligible-cohort CPU baseline, run tiny neural mechanics checks, update report/results notebook/CV, save reviewed bundle. GitHub publication still awaits explicit approval.

## Stage 2 completed — executable workflows and real CPU baseline
- Core, repaired neural workflow and pinned Python3.12 packaging complete. Core and neural source files stable; code snapshot commit 1c46df2.
- Real full eligible-cohort CPU TF-IDF/one-vs-rest logistic experiment developed and frozen once, then evaluated once. Original159571 rows -> explicit exclusion7 empty texts ->159564 eligible; conflicting annotations retained without label merging.
- Actual split counts:111731 train,23878 validation,23955 internal exploratory test. Normalized duplicate-group overlap0, independently checked.
- Frozen-validation-threshold internal test: macro AP0.6501947841473626, macro ROC-AUC0.98114015450262, macro F10.6345797586846106, micro F10.7483005695388573. Root independently recomputed aggregates from saved private predictions.
- Development47.104s, evaluation7.133s in this environment; measured developmentpeakRSS6958132KiB. This is a single-run baseline under a new protocol, not evidence of neural improvement over old validation logs.
- Reports: baseline_validation_metrics.json,baseline_internal_test_metrics.json,baseline_data_summary.json,baseline_verification.json. No raw text/IDs/individualpredictions in public result artifacts.
- Private frozen run: runs/real-baseline-20261013. Do not rerun or overwrite it. Restore its omitted private/source.csv from the retained train_preprocessed.csv if recovering from the checkpoint ZIP.
- Tiny offline neural train/backward/best-checkpoint/export/reload/fresh-validation/frozen-evaluation check passed; four independent neural invariants and21 core checks passed. Full discovery/CI/docverification in progress. No full real-data eight-layer neural training.
- Corrected three-bullet master CV compiled cleanly to2pages; only this project's itemize block changed. It remains outside repository.
- Next: finish corrected PDF/results notebook, final30-test/distribution/privacyreview, prepare concrete publication request, persist allfinalfiles. No GitHub writes.

## Current run commands
```bash
python -m pip install -r requirements-core.txt
python -m pip install --no-deps -e .
python -m toxic_comments.baseline develop --data data/raw/train_preprocessed.csv --run-dir runs/new-name --seed 42 --conflict-policy keep --empty-text-policy drop
python -m toxic_comments.baseline evaluate --run-dir runs/new-name
python -m toxic_comments.neural smoke --run-dir runs/neural-new-name
python -m unittest discover -s tests -v
```
New run paths must not already exist. The challenge test CSV is unlabeled and cannot be used for scoring.

## Stage 3 completed — reviewable publication candidate
- Final common suite:33tests passed in7.922s; all4optional CPUneural tests executed, zero skips. No full real neural training.
- Results review:11codecells executed top-to-bottom in a fresh in-process IPython session, four native chart outputs; Jupyter/ZeroMQ kernel sockets are blocked here. Neural tutorial8default codecells checked in-process, expensive training disabled, published outputs cleared.
- Corrected English report prepared in Markdown/PDF with real CPU results separated from historical neural validation logs. Original unsupported claims removed; canonical historical and CPU tables checked against aggregateJSON.
- Independent publication review found no material privacy issue in candidate files or reachableGit history. No rawdata/model/rowpredictions/CV enters publicationtree.
- Verification evidence:docs/validation.md,reports/validation_summary.json,baseline_verification.json. CIpreparedandYAMLchecked; GitHubActions notrun until publication.
- Updated masterCVPDF/TeX and Arabic career/interview notes are under ../career/, outside repository. Only toxic-comment project paragraph changed, two-page layout retained.
- Publication proposal and PRdraft:../Publication_Review.md. Targetmobatayneh/toxic-comment-classification, proposedPublic, branchwork/portfolio-hardening. No remote/repository/PR/merge created. Userapproval required before first publication and separately before mainmerge.
- Proposed publication excludes unchanged originals(private recovery), rawCSVs, weights and per-rowarrays. Ownership/detailcontributions/source-reuse confirmation remainspending; no repo-wideLICENSE asserted.

## Resume next
1. Readthisrecord and gitstatus/log; do notrerun completedmodel experiments.
2. Resolve user's firstpublication privacy/approval and any contribution/sourceattribution details.
3. Create/use approvedrepository, publishreviewedbranch and openpreparedPR; verifyremotecommit and CI. No mainmerge withoutseparateexplicitapproval.
4. Addverifiedliveprojectlink toCV afterpublication; verifywriteback/compilation.
5. If userprovidesGPUresources, planandrunfullrepairedneuralprotocol separately; no prior improvementclaim.

## Recovery notes
The privatecheckpoint preserves code/Githistory, originals, aggregateevidence, fittedbaseline and privatepredictionarrays. OriginalCSVdata is alreadyretained separately and intentionallyomitted from thatZIP. Recovery_Input_Map.json identifiesthe trainCSV needed at runs/real-baseline-20261013/private/source.csv. Testevalalreadycompleted; restoreforintegrity checks/read-onlyresultreview, notnewtesttuning. The public ZIP is a clean publicationcandidate withno privateartifacts.

## Authorized Colab and publication stage — 2026-10-13
- User explicitly approved creating `mobatayneh/toxic-comment-classification` as Public, publishing the work branch and opening a Pull Request, without merging. This approval supersedes the pending-approval statements in earlier historical milestones.
- User confirmed original individual authorship: original code written by Mohammad Albatayneh, with AI assistance correcting errors. Documentation is being aligned; no blanket license is inferred.
- User requested a complete self-contained Google Colab notebook for full random-initialized RoBERTa training, then returning results for further review. Three explicitly configured GPT-6.1 Sol / Max agents own the training notebook/resume implementation, independent tests, and attribution/documentation. The main model setting has not been changed through a tool.
- Colab additions stay outside the six frozen `src/toxic_comments` modules so the completed CPU baseline remains reproducible. Full GPU training is for the user to execute; CPU mechanics tests here must not be described as full Colab training.
- New training will persist model/optimizer/scheduler/scaler/RNG state and input/configuration fingerprints at optimizer boundaries. Validation selects checkpoint and thresholds; optional internal-test evaluation is disabled initially to preserve the development boundary.
- GitHub target did not exist at the last connector check. The available connector can write files/branches/PRs in accessible repositories but has no repository-creation operation; no repository, remote commit or PR has yet been created. Browser fallback requires user approval under the browser capability instructions. Do not modify the previous diabetes repository.
- Next: finish Colab implementation, verify resume and notebook source consistency, persist updated reviewable artifacts, then resolve repository creation and verify the approved publication. No full baseline rerun and no main merge.

## Resumption verification — 2026-10-14
- Resumed from local HEAD `01d7ea36a142c633d2a64f3d6f2b9cf36c411a99`, branch `work/portfolio-hardening`, with the previous uncommitted authorship and partial Colab additions preserved. No local remote is configured.
- Authenticated GitHub owner remains `mobatayneh`; the target repository request returned `404 NOT_FOUND` and the accessible repository list includes only the prior diabetes project. This cannot verify publication of this project; do not claim a remote branch/PR/CI exists.
- Rechecked the frozen CPU run manifest, every recorded artifact and source checksum. Public validation and internal-test JSON exactly equal their frozen-run copies. Core modules, recorded metrics, four prior figures and the aggregate review notebook are unchanged. No training or test inference was repeated.
- Existing Colab runner and tests are partial work to complete, not a finished Colab experiment. Resumed three GPT-6.1 Sol / Max owners for the notebook/runner, independent checks and documentation; prior inactive agents were not assumed to be working.
- Scientific boundary remains: completed real CPU baseline, historical neural validation outputs, and tiny synthetic CPU mechanics checks are separate evidence. No controlled neural improvement or full Colab GPU result is established.
- Next: finish the missing self-contained notebook and guide; check resumability, bundle privacy and source consistency; record final tests and local commit; resolve repository creation and perform the approved publication without merging.

## Colab stage completed — 2026-10-14
- Added `scripts/colab_train.py`, `scripts/build_colab_notebook.py`, `notebooks/03_colab_full_training.ipynb`, `requirements-colab.txt`, `docs/colab_training.md` and `tests/test_colab_training.py`. Notebook: 30 cells/18 code cells, eight visible exact embedded files, cleared outputs; SHA-256 `2f60edc83f6d70a54b194e4e0fd0372782f820675c14584975a488ae245eba55`.
- Tiny offline CPU interrupted/resumed training matches continuous model/optimizer/scheduler/scaler/RNG state, history and exported validation predictions. Checkpoint save is atomic; changed source/config/code/tokenizer/runtime identities are rejected. Existing frozen runs are not retrained or overwritten. No Colab service, Drive persistence or CUDA execution was tested here.
- Fixed stale review ZIP reuse after later test scoring by preserving development bundles and using separate `_with_test` bundles. Corrected displayed label order to unchanged core order; checked Torch preservation using distribution versions; guarded stale imports after pip; full training uses a fresh process. Source CSV SHA and full-cohort counts are enforced.
- Independent final suite: 43 tests/43 passed/zero skips/failures/errors in 15.491686 seconds (14 data/metrics, 7 frozen artifacts, 2 baseline, 4 neural, 2 Colab notebook, 8 Colab resume, 6 publication verifier). A test-only shared-SGD instrumentation interaction was fixed after the first failed suite; the affected pair and final full suite then passed. Original per-case console transcript is partial; complete machine-readable result and checked-file hashes were retained, with a separately labelled aggregate log summary.
- Updated README, provenance/model card/data/correction/validation docs, confirmed-authorship report Markdown/PDF, CI notebook dependency setup and `reports/validation_summary.json`. Core source modules, real baseline metrics and prior four figures/results notebook remain unchanged. Authorship is confirmed; a repository-wide license has not been selected.
- Ready user flow: upload the complete `.ipynb` to Colab, choose GPU, run setup/mount Drive/upload original train CSV, train on the full protocol, leave `START_INTERNAL_TEST=False`, return `results_for_review.zip`. Exact continuation requires matching recorded environment/GPU family; unavailable/mismatched resources are reported explicitly. Optional private per-row verification ZIP must never be published.
- PR draft: `../Publication_Review.md`. Final local commit and saved delivery artifacts will be recorded in `../Recovery_State.json`. Publication and remote CI remain unverified; no remote branch/PR URL can be supplied yet.
- Next authorized action: obtain browser fallback permission to create the approved Public repository (or use an empty repository created by the user); publish `work/portfolio-hardening`, open the prepared PR, verify remote files/commit and CI, add the verified live project link to the private CV. Do not repeat successful experiments or merge without separate explicit approval.

## Browser publication attempt — 2026-10-14
- User explicitly approved browser fallback to create the previously approved Public repository, publish the work branch and open the PR. No separate approval for main merge was requested or established.
- Opened GitHub's repository-creation page; it redirected to the ordinary GitHub sign-in page. Requested credentials/method selection through the secure browser authentication capability only. Its result was `declined` with `user_took_over`; fresh visible evidence still showed the sign-in page, so authentication is not verified.
- No credentials were requested in chat, read, reconstructed or entered through ordinary browser controls. No repository, remote branch, commit, PR or GitHub Actions run was created by these actions.
- Next: user completes authentication in the existing cloud browser handoff. Recheck the resulting signed-in account is `mobatayneh`, create the approved Public repository, publish the reviewed branch and PR, verify remote files/CI, and update the CV link. Preserve the successful 43-test evidence; no experiment or full-suite repeat is needed for this documentation-only status update.

## Public repository created — 2026-10-14
- Authentication completed in the existing browser. Verified the repository owner `mobatayneh` before creation.
- Created and verified Public repository `https://github.com/mobatayneh/toxic-comment-classification`, repository ID 1418490115. Initial main commit `8ac5bb8cc74c53f21df3ab9fc5f0963781bda6d3` contains only the generated README; no feature merge occurred.
- The connector now reports push/admin permissions for this new repository. Publish the reviewed 63-file candidate to `work/portfolio-hardening`, then open the prepared PR and check remote content and CI. Do not upload private datasets, weights, prediction arrays, CV files or the recovery ZIP.
- The initial public snapshot will be transferred through Git data objects. Its content/tree identity can be checked against the local candidate; remote commit IDs differ from the preserved local preparation history. Keep that distinction explicit.

## Work branch and PR published; clean-runner repair — 2026-10-14
- Published all 63 reviewed files to `work/portfolio-hardening` at remote commit `5fc5770d316568bc1bc133422e3c946c15f3a061`. Remote root tree `308cb9c31a0e89260930902cefa6552d739cebd1` exactly matches local snapshot `15c1c8d4d8ec69fd7aaf32d4c6856cdc515d7054`; all path/mode/blob identities were verified.
- Opened PR #1, `https://github.com/mobatayneh/toxic-comment-classification/pull/1`, targeting main. It is open/unmerged; main remains the initial README commit. No reviewers, teams, comments or bot review were requested manually.
- First GitHub push/PR runs executed 31 core checks successfully and explicitly skipped 12 optional CPU neural checks. The native Jupyter results-review step exposed an undeclared Jinja2 dependency used by pandas Styler, so the complete workflows failed. Added the already locally tested `Jinja2==3.1.6` pin to the notebook requirements; the repair does not change model code, metrics, plots or Colab embedded files.
- Initial failing run IDs: push 38525366826, PR 38525422341. Rerun by publishing this dependency repair; no real model training or test inference is repeated.
- An automatic `chatgpt-codex-connector[bot]` comment appeared when the PR opened. This was repository/app automation, not a manual ping or review request. No comments or automation settings were deleted/changed.

## Publication verification completed — 2026-10-14
- Dependency repair published at remote commit `1391907f6945069b25831cf7f77b1a38116b4895`, matching local root tree `06e4c458ae5072189a949650529227064c96c099`. Push run 38525905392 and PR run 38525910900 concluded success.
- GitHub core job verified 31 passing tests/12 explicit optional skips, eleven native Jupyter code cells, repository evidence/link/privacy checks and successful executed-notebook artifact upload. The separate 43-pass local suite still supplies the twelve tiny CPU neural/resume checks; full Colab/Drive/CUDA training remains unverified.
- Published README, provenance, validation and machine-readable status now describe the actual repository/branch/PR/CI state. Model code, historical/CPU result JSON and all four existing plot images are unchanged. The static corrected report retains its 13 October evidence-review date; current publication checks live in docs/validation.md.
- Added the verified repository URL only to the private CV's toxic-project header, preserving all bullets/other source bytes and the two-page layout. Career files remain outside GitHub. Original contribution and third-party rights boundaries remain explicit; no license has been invented.
- Final documentation-only commit/tree will be verified on the remote branch and against its own CI before handoff. Main must still equal `8ac5bb8cc74c53f21df3ab9fc5f0963781bda6d3`; PR #1 must remain open/unmerged. Record final remote/local SHAs and URLs in the private Recovery_State.json and Publication_Review.md.
- No additional model architecture, interface, API, Docker, paid training or real model rerun was added. Remaining next work is the user's full Colab training with the test switch off, review of returned aggregates, and a separate explicit decision about merging PR #1.

## Follow-up review authorized — 2026-10-15
- User explicitly requested a second review, execution of all twelve optional CPU neural/resume checks, correction of discovered errors, publication and merge only after verification succeeds. This supersedes the earlier no-merge authorization status.
- User is currently training the supplied self-contained Colab notebook. Preserve its source/run identities and all completed real baseline results; do not rerun real-data experiments.
- Starting snapshots: local 80f4580fe8f0a8259d021b13fece966990d7ab3c; remote work branch b6819a5a2f6e6822b3d58bd0b7e314a623340ace; PR #1 open/unmerged; main 8ac5bb8cc74c53f21df3ab9fc5f0963781bda6d3.
- Phase 1: independent core and Colab reviews, fresh complete CPU suite, required neural CI, focused data/evidence checks. Phase 2: publish review fixes, verify latest-head CI, merge the approved PR and verify main. Save checkpoints between phases.
- The provided test_preprocessed.csv is the unlabeled challenge set; verify its label-null audit, explain why scored evaluation uses a held-out portion of labeled training data, and never treat missing labels as negatives.
- Next: collect review findings and full test evidence; no merge until resolved material findings and green full CI.

## Phase 1 complete — 15 October 2026
- Reviewed core data/metrics/freeze/neural code and Colab execution/resume/export/bundles independently. Verified actual challenge-test SHA and all 153164 rows lack six-label truth.
- Reran prior43 checks with all12 neural checks enabled: all passed. Fixed standalone/Colab code-identity timing and bounded finalization recovery; added original code/source provenance to aggregate bundles. Seven new regressions bring the full suite to50; all50 passed with zero skips/failures/errors. Public per-test outcomes and checked-file hashes: reports/review_20261015.json; review: docs/review_20261015.md.
- CI now runs its neural job on push/PR/manual with a zero-skip guard. Core31checks and neural21checks cover50distinct tests (two notebook checks overlap); the19optional core skips must execute in the neural job.
- Before code changes, independently revalidated frozenCPU source/artifacts/splits/thresholds and recomputed all metrics from saved predictions; exact agreement, no new model inference/training. Keep original baseline metadata and code identity intact.
- Updated Colab notebook is for new runs. User's active original copy must retain its exact source/runtime for continuation; original commit b6819a5a2f6e6822b3d58bd0b7e314a623340ace and notebook SHA are documented.
- Next authorized stage: publish this reviewed source, verify both GitHub jobs on the latest commit, resolve any failure, then merge PR1 with an expected-head check and verify main. Do not repeat real experiments. Full neural results await the user's returned Colab bundle.
