# Source review and reproduction status

Snapshot: `rank_topic_refinement_github_step1_20261013T052026860722Z.zip`.

The ZIP passed archive validation. It contains 430 collected source/document/state files; it omits raw datasets, trained weights and generated output directories by design. The Step 2 builder keeps runnable modules, configuration files, tests and dependency files, preserves their exact bytes, and replaces the landing-page documentation in a separate destination.

| Item | Source-review finding |
|---|---|
| Main CLI | Steps 1–8 and all documented stage-specific flags are present in `main.py`; `--help` executed successfully |
| Core implementation | Current signed-association selection and two-stage affinity projection are present |
| Saved reporting | Hash-bound to the reviewed original runs; needs corresponding runtime artifacts |
| Primary affinity export | Present; can package its exact frozen inputs without training |
| Transportation audit | Present; needs its four fixture files and optional live compact arrays |
| Numerical parameter sensitivity | Recovered in the sensitivity integration update; 65 support/margin and 15 replacement-budget cells are replayed by --sensitivity-budget |
| New training from an empty clone | Commands and prerequisites documented; complete execution not validated here |
| Public release | Source license, primary-data permission and archive distribution remain to be decided |

## Frozen identities

`configs/step06_frozen_test.yaml` binds Step 6 to a particular preparation and Step 5 recovery index. `step07_external/protocol.py` requires the completed primary freeze. `step08_assembly/integrity.py` and `step08_settings.json` bind assembly to specific primary/external runs. `primary_affinity_analysis/data/reference_inputs.json` lists exact input files and hashes.

These checks are intentional provenance protections. Fresh runs may differ because of numerical libraries, device, upstream availability, or training. A new run is not automatically the historical run; do not edit expected hashes to claim it is.

## Remaining repository tasks

1. Validate the clean source folder using the existing working environment.
2. Package saved artifacts for affinity/report reproduction and document restoration paths.
3. Supply the recovered sensitivity input bundle for full array replay; tracked CSVs permit figure-only regeneration.
4. Validate a clean installation independently before claiming a complete fresh-training reproduction.
5. Review release terms, then initialize and publish through GitHub Desktop.

Historical installation instructions retained at their runtime-required paths apply to the original project and may describe older research stages. `RUN_COMMANDS.md` is the repository's current command index.

## Settings-file correction

The initial preparation list omitted the root-level `step07_settings.json` and `step08_settings.json`. The corrected source manifest includes both (219 reviewed files). Existing repository copies must receive the two original settings files before compact/external reporting. Their values and saved evidence are unchanged.
