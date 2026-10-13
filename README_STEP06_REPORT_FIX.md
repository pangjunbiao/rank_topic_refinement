# Complete the saved Step 6 reports

This update is for the reviewed run `20261010T051100713971Z_6eab0566`.
Its model scoring is complete: 60 comparison cases and 60 control slots were saved.
The original execution stopped during held-out-coherence reporting because JSON
turns tuples into lists. The cached bytes were valid; the Python identity comparison
incorrectly rejected a repeated descriptor audit.

## Install in your existing PyCharm project

1. Extract the download into a temporary folder.
2. Copy its contents into `C:\Users\Sabir\PycharmProjects\rank_topic_refinement`.
   Replace matching files. `main.py` and `step06_finalize.py` belong directly in
   the project root. Merge the `src`, `configs`, `tests`, and `docs` folders.
3. Keep your existing `data`, `outputs`, `state`, and `logs`. This update does not
   supply replacement experiment data or replace any saved results.
4. Use the same Python interpreter and installed dependencies as the successful
   Step 6 scoring run. No package reinstall or CUDA replacement is needed.

Run these commands in PowerShell from the project root:

```powershell
python .\main.py --step 6 --self-check --device auto
python .\main.py --step 6 --finish-reports --device auto
```

The second command completes reports from the saved test arrays. It verifies
the original preparation artifacts, source indices, all prediction-case files,
frozen configurations, descriptors, development-selected rho, and test-exposure
ledger. It checks the saved NLL arithmetic and actual Proposed readout certificates.
It does not create an encoder, restore a model, optimize, train, infer new document
representations, or choose new parameters. Missing or genuinely changed saved
inputs cause a clear failure; it does not refit missing cases.

The original `src/rank_refinement` prediction code is byte-identical. The cache
repair lives in the new root reporting module. All descriptor slots, negative
values, and unavailable metrics/control slots are retained.

## Expected completion and files

The console should finish with `STEP06_FROZEN_PRIMARY_TEST_COMPLETE` and
`STEP 6 COMPLETE`, with `new_model_predictions=0` and `new_encoder_calls=0`.

Saved Step 6 reports:

```text
outputs/steps/step06_frozen_test/88b9b4de3e1f7d877709/
    SUMMARY.md
    run_result.json
    quality_gates.json
    primary_test_comparison_mean_sd.csv / .md / .tex
    primary_test_controls_mean_sd.csv / .md / .tex
    paired_prediction_intervals.json / .csv / .md / .tex
    report_completion_audit.json
    figures/
```

Main-paper and supplementary tables are also exported under
`outputs/paper/main/tables` and `outputs/paper/supplementary/tables`.
Each run writes a timestamped log/report and an inspection ZIP under
`logs`, `outputs/runs`, and `outputs/inspection`. The console prints its exact ZIP
path. Read `docs/step06_report_fix/REVIEW.md` for the independently inspected results.

The inspection ZIP is for reviewing your run. Keep the program files in the
project root/folders, rather than extracting inspection content into the project.

## Scientific scope

These are actual frozen primary-test predictions, including the Proposed model.
The selected Proposed model uses the NeuroMax base and realizes the projected
readout; its displayed label remains `Proposed`. Moment correction was already
disabled before test exposure. It remains an explicit control.

Fitting-reference C_v/NPMI are descriptive. Frozen test NLL measures document
completion on held-out documents. Full test-reference coherence is a separate
diagnostic; missing descriptor support remains NA. These reference scopes must
stay distinct in the manuscript.

This completion does not authorize tuning on the exposed primary test. Step 7
is external generalization with these same frozen models and a separately supplied
external dataset. Successful reporting does not establish universal metric wins.
