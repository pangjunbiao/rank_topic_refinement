# Reproducible transportation case analysis

Replace the existing **Transportation Service Themes and Supporting Posts** subsection and its table with `transportation_subsection.tex`. Do not add a second repeated subsection. The original LaTeX subsection and table labels are preserved.

This package computes exact native-versus-refined descriptor edits and ties them to the same six existing primary fitting posts. It includes all six examples, all ten topics, and all five saved primary seeds. It does not establish passenger-requirement accuracy, human interpretability, or operational improvement. Some word changes improve association while removing service-specific words; those changes are retained.

## Extract into the project

Extract the ZIP contents into:

`C:\Users\Sabir\PycharmProjects\rank_topic_refinement`

The new script goes alongside `main.py`. Its evidence goes into `data\transportation_audit_v1`. No existing model, configuration, cache, or manuscript is replaced automatically.

## Run in PyCharm Terminal / PowerShell

First check the supplied historical fixture:

```powershell
python .\reproduce_transportation_analysis.py --self-check
```

Then read your project's original saved compact arrays and case selection directly:

```powershell
python .\reproduce_transportation_analysis.py --compact-dir ".\outputs\steps\step08_compact\171540c39f00cafba837" --device auto
```

If your saved compact folder has another ID, use that exact folder instead. To list the folders:

```powershell
Get-ChildItem .\outputs\steps\step08_compact -Directory | Select-Object FullName
```

The live folder must contain `numerical_evidence\application_selection.json` and the five `numerical_evidence\primary\seed809` through `seed813` folders, each with `affinity_before_after.npz` and `vocabulary.json`.

If you do not have those original arrays on that computer, reproduce the supplied historical primary evidence instead:

```powershell
python .\reproduce_transportation_analysis.py --device auto
```

This fallback is clearly marked as a saved snapshot. It cannot verify later revised model results or external/ablation benchmark values.

## Where results are saved

Each execution creates a new directory:

`outputs\steps\step08_transportation\<run-id>\`

The console prints the full paths. Important files:

| File | Purpose |
|---|---|
| `transportation_subsection.tex` | Complete replacement text and two-column table |
| `transportation_subsection_latex.txt` | Identical plain-text LaTeX for copying |
| `transportation_case_audit.csv` | Original Chinese posts, English summaries, all words, group IDs, and edit counts |
| `all_primary_descriptor_edits.csv` | All 50 matched topic cases across five seeds |
| `descriptor_edits_by_seed.csv` | Membership-change counts and diversity, calculated separately per seed |
| `matched_case_edit_counts.csv` | Exact edit counts for the six original seed-809 examples |
| `provenance.json` | Input paths, SHA-256 hashes, source mode, and evaluation scope |
| `run_result.json` | Completion status and elapsed time |
| `output_manifest.json` | Hashes of the saved report files |
| `SUMMARY.md` | Short explanation and all observed changes |

The ZIP also contains `reference_results` from an execution against the original saved arrays. Numerical tables and generated manuscript text were verified to match bundled-inventory mode byte for byte.

## Manuscript use

Copy the complete contents of `transportation_subsection.tex` over the existing transportation subsection and its table. Keep the main paper's `booktabs`, `tabularx`, and `array` packages. The table spans the two IEEE columns at normal `\footnotesize`; it does not shrink text using `\resizebox`.

In the supplementary **Lexical Evidence for Transportation Examples** section, append the short text in `supplementary_transportation_note.tex`. Keep the existing six-row lexical-overlap table unchanged: the same groups and one-based topic identifiers are retained. Overlap counts are archived values; this script does not replay tokenization or recalculate them.

Optional introduction sentence, after the first paragraph:

```latex
The intended use is service-feedback exploration: analysts inspect
topic descriptors together with supporting posts to identify
concerns for further investigation.
```

Do not add an abstract claim that this audit demonstrates better requirement detection or better transport operations. It provides a transparent qualitative application analysis.

## Validation and changed experiments

The bundled mode verifies input hashes and needs Python 3.10+ only. The live-array mode uses NumPy already installed in your project. There are no neural fits, encoder calls, GPU requirements, network downloads, or new dependency pins. `--device auto` is accepted for command consistency; file analysis uses CPU.

Live mode checks finite nonnegative arrays, fitting-vocabulary consistency, selected-word membership, conserved affinity totals, the frozen replacement limit, distinct promotions, descriptor diversity, case identity, and presentation-translation identity.

English summaries, post-type descriptions, and potential analyst follow-ups are presentation notes, not automatically predicted service labels. Original Chinese text and exact token IDs are preserved. Independent groups are used as provenance; topic identifiers are compared within a seed and are not assumed to correspond semantically across seeds.

If a revised compact run contains different posts or unfamiliar added/removed tokens, the program stops rather than attaching old translations or interpretations. A separately reviewed notes JSON can be supplied with `--notes PATH`. Do not edit the protected bundled input files to make a revised run look identical.

The primary arrays used for the shipped reference report come from the archived compact Step 8 run. These files do not resolve discrepancies in later external or ablation tables.
