# Primary parameter sensitivity reproduction

Run from the project root using Python 3.10–3.12:

```powershell
python -m pip install -r requirements-sensitivity.txt
python .\main.py --step 8 --sensitivity-budget --self-check --device auto
python .\main.py --step 8 --sensitivity-budget --device auto
```

The self-check covers 37 tests. The normal command replays 65 support/rank-margin slots and 15 replacement-limit slots against the saved fitting/development arrays. Cached affinities are checked for ranks, nonnegativity and conserved totals; coherence and development losses are recalculated. No neural training, encoder calls, test rescoring or model reselection is required for the supplied cache.

The exact latest directory is recorded in `outputs/steps/step08_sensitivity_budget/LATEST.txt`.
Publication figures are `figures/SF1_parameter_sensitivity.pdf` and `figures/SF2_rank_margin_tradeoff.pdf`, with PNG copies. They use the same compact blue/red/gray style as the supplementary figures. Values and uncertainty are unchanged. Older R-prefixed diagnostic figures are also retained, but use the SF figures in the paper.

For figure-only regeneration:

```powershell
python .\reproduce_s5_figures.py --verify-only
python .\reproduce_s5_figures.py
```

This second path verifies the archived CSV hashes and recomputes all 192 summary checks; it does not replay predictions from model arrays. It saves figures to `outputs/paper/supplementary/figures`.

These are primary-corpus, post-exposure descriptive fitting/development sweeps. The frozen settings q=5 and R=4 are highlighted; this does not retrospectively establish selection or global optimality. Sample SD over five matched seeds is not a confidence interval.

`--sensitivity` retains the support/margin-only runner. `--sensitivity --sensitivity-recompute` explicitly recomputes alternative projections and may take substantially longer. Use the standard cached budget command above for saved-result reproduction.

Packed input records and caches are ignored by Git. A clean clone needs the separately supplied saved-input bundle for full array replay. The four small `data/supplementary_s5` CSVs support figure-only reproduction directly from the code repository.
