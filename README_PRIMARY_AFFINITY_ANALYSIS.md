# Primary affinity analysis

This read-only export compares the same five saved native NeuroMax models with their refined affinities. It produces an English before/after affinity figure, actual selected-word inclusion, divergence statistics and the matched document-completion result. It fits no model and selects no parameters.

## Run

From the project root, after restoring the matching saved inputs:

```powershell
python -m pip install -r requirements-primary-affinity.txt
python .\main.py --step 8 --primary-affinity --self-check --device auto
python .\main.py --step 8 --primary-affinity --device auto
```

The self-check uses mathematical fixtures. The full command independently verifies the saved input hashes and all five matched comparisons. Results go to `outputs/steps/step08_primary_affinity_analysis/<run-id>/`; the latest directory is recorded in `LATEST.txt`.

## Outputs

| File | Purpose |
|---|---|
| `main/figures/F1_primary_affinity_before_after_English.pdf` | English-labelled primary affinity figure; SVG/PNG accompany it |
| `main/tables/T6_primary_affinity_analysis.md` | Two-panel inclusion, prediction and divergence summary; LaTeX/CSV accompany it |
| `supplementary/divergence_per_seed.csv` | All five seed-level comparisons |
| `supplementary/divergence_per_topic.csv` | All fifty within-seed matched topic comparisons |
| `supplementary/divergence_per_document.csv` | All 2,050 seed/document comparisons |
| `source_provenance.json` | Verified input hashes and calculation scope |
| `primary_affinity_reproducibility.zip` | Code and exact used inputs for reproducing this primary-only analysis |

Jensen--Shannon divergence is reported in nats, not as its square-root distance. Affinity shares, unit-topic decoder distributions and final document predictions are different objects and are compared separately. English glosses affect presentation only. Numerical changes are evidence of adjustment, not evidence of predictive superiority or labelled requirement-detection accuracy.

The reproduction ZIP can be extracted into a separate empty folder and run using the same two analysis commands. Full pipeline training is a different reproduction task.
