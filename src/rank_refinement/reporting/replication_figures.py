"""Seed-paired tradeoffs complement the five-seed aggregate tables."""
from pathlib import Path
import numpy as np
from ..runtime.artifacts import write_csv, write_json


def plot_seed_tradeoffs(output, ablation_rows, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    points = []
    for seed in seeds:
        selected = [r for r in ablation_rows if r["seed"] == seed and r.get("status") == "scored"]
        native = next((r for r in selected if r["role"] == "native"), None)
        if native is None:
            continue
        for row in selected:
            if row["role"] not in {"proposed", "pair_acquisition", "without_subsample_stability", "positive_only_association"}:
                continue
            points.append({"seed": seed, "role": row["role"],
                "npmi_gain_over_native": row["fitting"]["c_npmi"] - native["fitting"]["c_npmi"],
                "nll_gain_over_native": native["prediction"]["macro_source_nll"] - row["prediction"]["macro_source_nll"]})
    output = Path(output)/"figures"; output.mkdir(parents=True, exist_ok=True)
    if not points:
        write_json(output/"seed_tradeoffs.json", {"available": False, "reason": "no_scored_seed_pairs"})
        return
    labels = {"proposed": "Proposed", "pair_acquisition": "Previous pair acquisition", "without_subsample_stability": "Without subsample stability", "positive_only_association": "Positive-only associations"}
    colors = dict(zip(labels, ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]))
    markers = dict(zip(labels, ["o", "s", "^", "D"]))
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for role, label in labels.items():
        part = [r for r in points if r["role"] == role]
        if not part: continue
        ax.scatter([r["npmi_gain_over_native"] for r in part], [r["nll_gain_over_native"] for r in part],
                   s=45, color=colors[role], marker=markers[role], label=label)
        for r in part: ax.annotate(str(r["seed"]), (r["npmi_gain_over_native"], r["nll_gain_over_native"]), xytext=(4,4), textcoords="offset points", fontsize=7)
    ax.axhline(0, color="#777777", linewidth=.8); ax.axvline(0, color="#777777", linewidth=.8)
    ax.set_xlabel("Fitting NPMI gain over matched native parent")
    ax.set_ylabel("Development NLL gain over matched native parent\nPositive values favor refinement")
    ax.set_title("Acquisition and prediction tradeoffs in each fixed seed", fontsize=11)
    ax.legend(frameon=False, fontsize=8); ax.spines[["top","right"]].set_visible(False)
    fig.text(.02,.01,"Every scored seed is shown; failed/NA cases remain in tables. Development replication, not final test evidence.",fontsize=7)
    fig.tight_layout(rect=(0,.05,1,1))
    fig.savefig(output/"seed_tradeoffs.png",dpi=220);fig.savefig(output/"seed_tradeoffs.pdf");plt.close(fig)
    write_csv(output/"seed_tradeoffs.csv",points,list(points[0]))
    write_json(output/"seed_tradeoffs.json",{"all_scored_seeds_shown":True,"axis_values_are_seed_paired_differences":True,"final_paper_figure":False})
