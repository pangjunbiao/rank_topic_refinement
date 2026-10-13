# Rank-Constrained Topic Refinement

Research code for selecting corpus-supported leading words and enforcing their ranks in a trained neural topic model's predictive affinity. The evaluated parent is NeuroMax. Neural parameters remain fixed during refinement.

Descriptor selection uses signed local word associations averaged across fixed fitting-data subsets. A primary linear program minimizes total-variation affinity change under rank and row/column conservation constraints. A secondary program limits decoder-logit disturbance within a declared additional allowance. The final affinity need not be the exact minimum-TV solution when that allowance is positive.

## Run and reproduce

**[RUN_COMMANDS.md](RUN_COMMANDS.md) is the command guide for all stages and paper exports.** It distinguishes model fitting, software self-checks, and reproduction of existing frozen results. Run commands from the folder containing `main.py`.

```powershell
python .\main.py --help
python .\main.py --step 8 --primary-affinity --self-check --device auto
```

The second command checks the reporting mathematics; it does not generate research results. Dependencies must already be installed, or installed using the guide.

## Project layout

| Location | Purpose |
|---|---|
| `main.py` | Staged command-line entrypoint |
| `src/rank_refinement/` | Preparation, acquisition, models, projection, evaluation and stages |
| `configs/` | Archived stage configurations and upstream source hashes |
| `step07_external/` | External-corpus preparation and fixed-recipe replication |
| `step08_assembly/`, `step08_compact/` | Verified saved-result tables, figures and divergence analysis |
| `primary_affinity_analysis/` | Primary-only English affinity figure and divergence export |
| `reproduce_transportation_analysis.py` | Matched descriptor edits and supporting-post audit |
| `tests/` | Software and mathematical checks |
| `data/README.md` | Input locations, encoder assets and saved-evidence requirements |
| `docs/REPRODUCTION_STATUS.md` | What this source snapshot can reproduce and what still needs packaging |

## Data and archived results

Primary inputs are the three Chinese transit Weibo CSV files. The external corpus is Transit Tweet Data. Local raw data, model weights, caches, saved outputs and exposure ledgers are excluded from Git. See [data/README.md](data/README.md) for the required paths.

Steps 6–8 refer to particular reviewed artifacts and hashes. Source code alone does not recreate those saved reports: either the matching archived inputs or an independently validated training workflow is needed. Do not remove integrity checks or change identifiers to make unrelated runs pass. No fresh end-to-end replication is claimed for this repository preparation.

## Interpretation

Coherence is measured against the fitting corpus; document-completion prediction uses observed-only inference. Five seeds share a fixed split. Affinity divergence measures changes, and source-grounded cases illustrate service themes; neither establishes labelled passenger-requirement accuracy. Unavailable scores and unsuccessful cases remain part of the record.

## Dependencies and upstream code

Use Python 3.10 for the archived Windows environment. Stage requirements specify direct dependency versions. `requirements-lock.txt` covers Step 1 only; it is not a complete environment lock. CUDA PyTorch must be installed separately in a new environment.

Neural competitor modules are retrieved from pinned upstream commits with byte/hash verification. Cached copies are not included in Git. The vendored GSDMM implementation retains its license in `src/rank_refinement/vendor/GSDMM_LICENSE.txt`. See [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md).

This preparation does not assign a new project license or dataset redistribution permission. Keep the repository private during review; determine the applicable release terms before a public release. No manuscript citation file is included.
