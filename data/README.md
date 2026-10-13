# Data, assets and archived evidence

## Input files

Place the original, unmodified inputs at these paths relative to `main.py`:

| Corpus | Path | Schema / identity |
|---|---|---|
| Primary | `data/raw/primary/beijing.csv` | Body column `正文`; optional identifier `序号`, timestamp `发布时间` |
| Primary | `data/raw/primary/shanghai.csv` | Same configured schema |
| Primary | `data/raw/primary/xiamen.csv` | Same configured schema |
| External | `data/raw/external/Transit Tweet Data.csv` | The inspected public release; exact SHA-256 below |

Primary file matching, encodings, screening and grouped splitting are specified in `configs/data.yaml`. Do not fabricate example rows or substitute a similarly named dataset for reproduction of the archived experiment. Raw files are not present in the Step 1 source-inspection ZIP.

External source: https://github.com/etujnr/Transit-Tweet-Data

Expected external CSV SHA-256:
`3e386a2d6e0ba4023559992415880687326a43c86a7be36d226d960e1a39ec9d`

Download the release CSV from its source or use the original local copy. An edited CSV, a GitHub HTML page saved as CSV, or a re-export can fail the byte check. Use one copy in the input directory. Public availability does not by itself assign redistribution rights; retain source attribution and applicable terms.

## Encoder and upstream model assets

The pipeline resolves hash-pinned author modules into `data/cache/`. Encoder preparation resolves these model repositories and revisions:

| Encoder | Revision | Local directory |
|---|---|---|
| `BAAI/bge-small-zh-v1.5` | `7999e1d3359715c523056ef9478215996d62a620` | `data/assets/bge-small-zh-v1.5/<revision>/` |
| `BAAI/bge-small-en-v1.5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | `data/assets/bge-small-en-v1.5/<revision>/` |

Each directory contains the model weights, tokenizer files, configuration and verified asset manifest. Pinned online downloads are performed by the appropriate preparation/training stage. Frozen inference requires its original saved assets and model checkpoints. `--offline` applies to Step 7's English encoder only; it does not guarantee every dependency/source operation is offline.

## Transportation audit inputs

The original audit package uses four files under `data/transportation_audit_v1/`:

* `case_selection.json`
* `input_manifest.json`
* `presentation_notes.json`
* `topic_words_all_primary_seeds.csv`

The Step 2 `--copy-local-data` option copies these files and the four corpus CSVs locally when present. They remain ignored by Git. If distributing the audit fixture to a colleague, supply the corresponding evidence package separately. Full Chinese posts and English summaries are qualitative examples, not new service labels.

## Saved-result reproduction

Keep the original `state/` and `outputs/` trees in the working project. Their ledgers, predictions, arrays, completion masks and indexes are required by saved-result commands. Do not copy state pointers alone into an empty clone: the files to which they point must also be present.

The primary-only affinity analysis command creates `primary_affinity_reproducibility.zip`, containing the code and exact used frozen inputs for that report. This provides a compact way to reproduce the saved primary analysis on another computer. Full competitor/external report packaging is a separate task.

These local directories are ignored by Git; they are not removed from disk. Dataset permissions and archive contents must be reviewed before distributing a result bundle publicly.
