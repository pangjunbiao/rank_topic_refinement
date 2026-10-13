# Saved-result packages

The source repository and frozen reporting evidence are separate deliverables.

| Package | Purpose |
|---|---|
| `dist/frozen_reporting_inputs.zip` | Matching preparation, primary/external frozen predictions, report exports, ledgers and transportation inputs needed by the repository's saved-report commands |
| `primary_affinity_reproducibility.zip` from the original affinity export | Standalone primary-only English affinity figure and divergence-table reproduction, including its code |
| `compact_paper_package.zip` in the original compact result folder | Compact existing paper exports and numerical affinity evidence; not by itself a complete pipeline restoration |

The frozen input archive has a SHA-256 manifest. The restoration helper checks the experiment source against its archived source hashes, validates every payload and refuses to overwrite different existing files. The archive is ignored by Git; runtime files restored from it remain ignored as well.

## Restore into a matching source clone

After installing the reporting dependencies and obtaining the matching archive:

```powershell
python .\restore_saved_results.py --bundle "PATH\TO\frozen_reporting_inputs.zip" --project-root .
python .\main.py --step 8 --compact-results --device auto
python .\main.py --step 8 --primary-affinity --device auto
python .\reproduce_transportation_analysis.py --compact-dir ".\outputs\steps\step08_compact\171540c39f00cafba837" --device auto
```

Preserve the source bytes and Git checkout attributes; frozen protocols compare exact source hashes. The archive omits only external `.pt` training states under `workspace/data/cache/`, following the reporting interface's existing optional-checkpoint rule. It does not provide the assets required to train or rerun neural inference from scratch.

The archived numerical tables reproduce the saved run that produced them. They do not certify later manuscript revisions. The numerical sensitivity runner remains absent from the inspected source and must be restored separately before its reproduction is claimed.

Saved preparation/results include original post text and provenance. Keep these archives local during release review. No dataset redistribution license, public release, or upload is performed by the preparation tool.
