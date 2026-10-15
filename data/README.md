# Local data

No real dataset is distributed in this repository. Place an authorized CSV in `data/raw/`; that directory is ignored by Git. See [data sources](../docs/data_sources.md) for official access, the unresolved historical mirror provenance, and third-party terms.

## Accepted training schemas

| Column | Requirement |
|---|---|
| `comment_text` | Nonempty string, with at least one non-whitespace character |
| `identity_hate`, `insult`, `obscene`, `severe_toxic`, `threat`, `toxic` | Binary 0/1 labels, without missing values |
| `severe_tox` | Optional historical alias; if both severe columns exist, they must agree |
| `set` | Optional mirror column; only rows whose value equals `train` are retained |
| Other columns, including `id` | Ignored for modeling; never emitted in public metrics |

The canonical saved label order is `identity_hate`, `insult`, `obscene`, `severe_toxic`, `threat`, `toxic`. The loader rejects duplicate CSV headers, empty rows of text, invalid labels, and conflicting normalized duplicate annotations by default. The explicit `--conflict-policy keep` flag preserves conflicting rows in one group; it does not settle annotation disagreements. The explicit `--empty-text-policy drop` option records and removes empty-text rows before splitting; it does not silently invent text.

Grouping uses NFKC Unicode normalization, case folding, and whitespace collapse. Model inputs retain original text. Approximately 70/15/15 of unique exact-text groups are allocated to training/validation/internal test, with actual row proportions recorded. No near-duplicate, author, temporal, or stratification guarantee is made.

```bash
python -m toxic_comments.baseline develop \
  --data data/raw/train.csv --run-dir runs/baseline
```

An authorized `train_preprocessed.csv` can be substituted to reproduce the mirror input format. It will not recreate the historical outputs automatically: the exact historical data bytes, upstream cleaning, environment, and trained weights remain unverified.

For the later supplied mirror, the reviewed real-run choices are explicit:

```bash
python -m toxic_comments.baseline develop \
  --data data/raw/train_preprocessed.csv --run-dir runs/baseline \
  --empty-text-policy drop --conflict-policy keep
```

The source has 159,571 labeled rows; seven normalized-empty comments are excluded and 93 conflicting groups retain their annotations. The later challenge-test mirror has 153,164 rows with entirely blank targets and is not evaluated. The [aggregate audit](../reports/data_audit.json) contains source-level counts and identities; raw comments remain private.

## Synthetic fixture

The smoke command creates explicitly synthetic labeled strings and runs the complete engineering workflow offline. Its marker tokens make it deliberately easy. Do not use its scores as evidence about toxic-comment detection.

```bash
python -m toxic_comments.baseline smoke --run-dir runs/synthetic-smoke
```

Raw comments, fixture data, split indices, fingerprints, checkpoint files, and per-row predictions belong in ignored local paths. Aggregate public result exports can be reviewed for sharing.

