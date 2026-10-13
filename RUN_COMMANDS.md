# Run commands: experiments and paper results

All commands below use **PowerShell**, from the directory containing `main.py`, with the same Python interpreter for installation and execution. PyCharm's Terminal is sufficient. Anaconda is optional and is not required for the archived project.

This is a command guide, not a script to execute from top to bottom. The CLI branches were checked against the submitted source. Reporting self-checks were tested locally where noted in `VERIFICATION.json` supplied with the preparation kit. Complete fresh GPU training has not been executed during repository preparation.

**After restoring the archived inputs, use the saved-result commands in section 5. These do not retrain Steps 1–7.** An empty source clone initially contains no saved outputs or state ledgers.

## 1. Interpreter and environment

In the existing working project:

```powershell
cd "C:\Users\Sabir\PycharmProjects\rank_topic_refinement_release"
python -c "import sys; print(sys.executable); print(sys.version)"
python .\main.py --help
python -m pip check
```

The inspected machine uses Python 3.10.9 and PyTorch `2.7.0+cu128`. Keep this environment for the archived frozen model inference. On a separate computer, create a new Python 3.10 environment instead of altering another project's packages:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r .\requirements-step08.txt
.\.venv\Scripts\python.exe -m pip check
```

For the subsequent commands, either select `.venv\Scripts\python.exe` as the PyCharm interpreter or activate it:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

If PowerShell activation is blocked, use `.\.venv\Scripts\python.exe` in place of `python`; no execution-policy change is necessary. The CUDA wheel command follows the official PyTorch 2.7.0 instructions: https://pytorch.org/get-started/previous-versions/

For independent CPU work in a new environment, install the CPU build instead of the CUDA command:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
```

CPU execution does not automatically satisfy archived CUDA inference's device/version replay requirements. `--device auto` selects available CUDA for neural work; preparation, solvers and saved-array report calculations remain on CPU. No CUDA claim follows merely from selecting `auto`.

The nested `requirements-step08.txt` includes the earlier requirements, including the numerical/plotting packages needed by the affinity exporter. In an existing working environment, install it only if dependency checks require it:

```powershell
python -m pip install -r .\requirements-step08.txt
```

For the isolated primary affinity analysis bundle, only these export dependencies are needed:

```powershell
python -m pip install -r .\requirements-primary-affinity.txt
```

Do not treat `requirements-lock.txt` as a complete environment lock: it pins Step 1 packages only.

## 2. Input files and saved assets

Read `data/README.md` before running an experiment. Primary CSVs belong in `data\raw\primary`; the original external CSV belongs in `data\raw\external`. Raw CSVs, trained weights, caches, outputs and state are intentionally ignored by Git. In an empty clone, supply the appropriate data or exact saved-evidence package first.

Check the original external CSV identity:

```powershell
Get-FileHash ".\data\raw\external\Transit Tweet Data.csv" -Algorithm SHA256
```

Expected SHA-256: `3e386a2d6e0ba4023559992415880687326a43c86a7be36d226d960e1a39ec9d`.

Online neural stages retrieve pinned BGE assets and upstream author modules. Saved frozen inference requires its original checkpoints/assets. Keep complete original `data\cache`, `data\assets`, `outputs`, and `state` in the working project.

## 3. Software self-checks

Run checks individually. A self-check validates software/fixtures; it does not establish model performance. Early-stage checks can skip optional neural checks; later neural-stage checks can resolve pinned author modules over the network.

```powershell
python .\main.py --step 1 --self-check --device auto
python .\main.py --step 2 --self-check --device auto
python .\main.py --step 3 --self-check --device auto
python .\main.py --step 3 --repair --self-check --device auto
python .\main.py --step 3 --complete --self-check --device auto
python .\main.py --step 3 --improve --self-check --device auto
python .\main.py --step 4 --self-check --device auto
python .\main.py --step 5 --self-check --device auto
python .\main.py --step 5 --recover --self-check --device auto
python .\main.py --step 6 --self-check --device auto
python .\main.py --step 7 --self-check --device auto
python .\main.py --step 8 --self-check --device auto
python .\main.py --step 8 --compact-results --self-check --device auto
python .\main.py --step 8 --primary-affinity --self-check --device auto
python .\reproduce_transportation_analysis.py --self-check
```

The two final checks are the small checks requested during repository preparation. The transportation audit needs its four fixture files. Stage-specific modes belong only to the indicated stage; do not combine revision flags.

For dependency/hardware diagnostics, use this separately from `--self-check`:

```powershell
python .\main.py --step 1 --doctor-only --device auto
```

## 4. Training and stage execution catalog

The historical progression is 1 → 2 → finalize → 3 → repair → complete → improve → 4 → 5 → recover → 6 → 7 → 8. Later stages validate their accepted source and gates. This catalog does not certify that fresh training will satisfy the archived hashes or reproduce identical results on another device. Stop at a missing prerequisite or failed gate; do not bypass it.

| Stage | Required input | Output directory under `outputs\steps` |
|---|---|---|
| 1 | Original primary CSVs | `step01_prepare` (pointer; prepared payload is in `data\processed`) |
| 2 | Accepted Step 1 preparation | `step02_native_refinement` |
| 2 finalize | Completed Step 2 pilot | `step02_completion` |
| 3 | Accepted finalized pilot | `step03_neural_pilot` |
| 3 repair | Saved neural pilot | `step03_saved_replay` |
| 3 complete | Saved repaired pilot | `step03_completion` |
| 3 improve | Completed R2 source | `step03_coherent_refinement` |
| 4 | Accepted improved pilot | `step04_competitor_pilot` |
| 5 | Completed Step 4 panel | `step05_replication` |
| 5 recover | Original five-seed parent panel | `step05_recovery` |
| 6 | Exact reviewed Step 5 source, preparation, checkpoints and assets | `step06_frozen_test` |
| 7 | Completed primary freeze, external CSV | `step07_external` |
| 8 | Exact completed primary and external runs | `step08_assembly` |

The console and state pointers give each full run path. Run identities vary; directory names are not metric scores. Commands:

```powershell
python .\main.py --step 1 --device auto
python .\main.py --step 2 --device auto
python .\main.py --step 2 --finalize --device auto
python .\main.py --step 3 --device auto
python .\main.py --step 3 --repair --device auto
python .\main.py --step 3 --complete --device auto
python .\main.py --step 3 --improve --device auto
python .\main.py --step 4 --device auto
python .\main.py --step 5 --device auto
python .\main.py --step 5 --recover --device auto
python .\main.py --step 6 --device auto
python .\main.py --step 7 --device auto
python .\main.py --step 8 --device auto
```

Steps 3 repair/complete/improve reuse saved parent models. Recovery checks saved five-seed parents before completing missing competitor fits. Step 6 is frozen inference; if its exposure ledger already exists, the entrypoint uses report-only completion. Step 7 retrains the fixed recipe on the external corpus. Step 8 assembles saved evidence and trains no models.

Optional external preparation, explicit input location, or cached-encoder execution:

```powershell
python .\main.py --step 7 --prepare-only --device auto
python .\main.py --step 7 --external-input ".\data\raw\external\Transit Tweet Data.csv" --device auto
python .\main.py --step 7 --offline --device auto
```

Use one of these as needed; preparation is already included in the full Step 7 run. `--offline` prevents English encoder downloads, not all possible source/dependency access.

## 5. Reproduce paper results from the completed working project

Use these in the original project with its original saved outputs and ledgers. They do not need new model fits. An empty source clone must first receive the matching frozen inputs.

Finish any incomplete frozen-test reports from saved predictions:

```powershell
python .\main.py --step 6 --finish-reports --device auto
```

Assemble the complete primary/external tables, controls, uncertainty and topic inventories:

```powershell
python .\main.py --step 8 --self-check --device auto
python .\main.py --step 8 --device auto
```

Create compact main/supplementary tables and both-corpus divergence/affinity exports:

```powershell
python .\main.py --step 8 --compact-results --self-check --device auto
python .\main.py --step 8 --compact-results --device auto
```

Create the **primary-only** English affinity figure, affinity divergence table and its exact-input reproduction bundle:

```powershell
python .\main.py --step 8 --primary-affinity --self-check --device auto
python .\main.py --step 8 --primary-affinity --device auto
```

Create the matched transportation descriptor-change table from the original compact arrays:

```powershell
python .\reproduce_transportation_analysis.py --self-check
python .\reproduce_transportation_analysis.py --compact-dir ".\outputs\steps\step08_compact\171540c39f00cafba837" --device auto
```

If the verified compact folder differs, supply its actual path. Do not choose an arbitrary folder solely to obtain preferred numbers. To list available folders:

```powershell
Get-ChildItem .\outputs\steps\step08_compact -Directory | Select-Object FullName
```

The audit also supports its supplied historical fixture, without reading live compact arrays:

```powershell
python .\reproduce_transportation_analysis.py --device auto
```

This last command reproduces the bundled historical case evidence. It does not verify subsequent revised experiments or the external benchmark.

## 6. Where tables, figures and bundles are saved

| Export | Location |
|---|---|
| Full assembly | `outputs\steps\step08_assembly\<id>\SUMMARY.md` and `paper_package.zip` |
| Compact tables | `outputs\steps\step08_compact\<id>\MAIN_RESULTS.md`, `SUPPLEMENTARY_RESULTS.md`, `main\tables` |
| Compact figures and arrays | Same compact folder: `main\figures`, `supplementary\figures`, `numerical_evidence` |
| Primary affinity figure | `outputs\steps\step08_primary_affinity_analysis\<run-id>\main\figures\F1_primary_affinity_before_after_English.pdf` |
| Affinity table | Same affinity analysis folder: `main\tables\T6_primary_affinity_analysis.md` and `.tex` |
| Affinity reproduction ZIP | Same affinity analysis folder: `primary_affinity_reproducibility.zip` |
| Transportation table and subsection | `outputs\steps\step08_transportation\<run-id>\transportation_subsection.tex` |
| Transportation evidence | Same transportation folder: `transportation_case_audit.csv`, `all_primary_descriptor_edits.csv`, `provenance.json` |
| Execution records | `logs`, `outputs\runs`, `outputs\inspection`, `run_history.txt` |

The primary affinity exporter uses `LATEST.txt`; most earlier stages use `LATEST.json`. Inspect them without changing them:

```powershell
Get-Content .\outputs\steps\step08_compact\LATEST.json
Get-Content .\outputs\steps\step08_primary_affinity_analysis\LATEST.txt
```

## 7. Numerical parameter sensitivity

The recovered standalone runner replays primary fitting/development arrays for q={3,5,10}, R={2,4,6}, and five scaled rank margins. Frozen model parameters and final-test results stay unchanged.

```powershell
& $repoPython -m pip install -r .\requirements-sensitivity.txt
& $repoPython .\main.py --step 8 --sensitivity-budget --self-check --device auto
& $repoPython .\main.py --step 8 --sensitivity-budget --device auto
```

Read `outputs/steps/step08_sensitivity_budget/LATEST.txt`. In that run directory, use `figures/SF1_parameter_sensitivity.pdf` and `figures/SF2_rank_margin_tradeoff.pdf` for the supplement. Tables and per-seed evidence are saved alongside them.

Figure-only reproduction from the four tracked CSVs:

```powershell
& $repoPython .\reproduce_s5_figures.py --verify-only
& $repoPython .\reproduce_s5_figures.py
```

These figures are written to `outputs/paper/supplementary/figures`. CSV summary checking and full saved-array replay are distinct verification levels. Full replay requires the original packed sensitivity inputs and cache; a fresh clone alone contains only the code and figure CSVs. See `README_SENSITIVITY.md`.

## 8. Failure handling and reproduction scope

* If a dependency pin fails, install the appropriate supplied requirements using the same Python interpreter.
* If a saved asset/checkpoint is missing, restore its original directory. An inspection ZIP may intentionally omit training weights.
* If a hash, preparation identity or exposure ledger differs, keep the original and report the discrepancy. Do not delete ledgers or edit checksums to make a different run appear identical.
* Retain failed/NA comparisons and all five seeds. A software pass is not a performance claim.
* Archive the console-printed run report with the tables it produced. A saved report and a fresh GPU retraining result have different reproduction scopes.

No citation file is required to run the project, and none is added by this preparation.

## 9. Restore frozen reporting inputs into a matching source clone

The local packaging workflow creates `dist\frozen_reporting_inputs.zip`. Obtain that package separately from the source repository; it is ignored by Git. Restore it without changing the source:

```powershell
python .\restore_saved_results.py --bundle "PATH\TO\frozen_reporting_inputs.zip" --project-root .
python .\main.py --step 8 --compact-results --device auto
python .\main.py --step 8 --primary-affinity --device auto
```

The restoration helper validates source/payload hashes and refuses to overwrite different existing artifacts. Restored reports use the exact saved predictions, not newly trained models. See `docs/RESULT_PACKAGES.md` for scope.

Windows validation on the project owner's Python 3.10.9 environment: 35 affinity-export self-checks and 11 transportation checks passed. The original primary-only export completed for all five seeds; the original compact export verified and reused the unchanged primary/external evidence. This does not establish fresh end-to-end training reproducibility.
