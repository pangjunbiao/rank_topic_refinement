"""Exploratory group-level prediction differences, distinct from mean tables."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..runtime.artifacts import write_csv, write_json


def plot_prediction_groups(output, paired_rows, means):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output) / "figures"
    chosen = {}
    for method in sorted({r["method"] for r in paired_rows}):
        candidates = [r for r in means if r["method"] == method and r["complete_seed_grid"]
                      and r["development_macro_nll_mean"] is not None]
        if candidates:
            chosen[method] = min(candidates, key=lambda r: (r["development_macro_nll_mean"], r["profile"]))["profile"]
    points = []
    for method, profile in chosen.items():
        subset = [r for r in paired_rows if (r["method"], r["profile"]) == (method, profile)]
        for group in sorted({r["group_id"] for r in subset}):
            rows = [r for r in subset if r["group_id"] == group]
            if len(rows) != 2 or len({r["seed"] for r in rows}) != 2:
                raise ValueError("Figure requires two training-seed values for every independent group.")
            points.append({"method": method, "profile": profile, "group_id": group, "source": rows[0]["source"],
                           "mean_competitor_minus_proposed_nll_per_token": float(np.mean([r["gain_per_token"] for r in rows]))})
    if not points:
        return
    methods = sorted(chosen)
    sources = sorted({r["source"] for r in points})
    colors = dict(zip(sources, ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]))
    rng = np.random.default_rng(4809)
    fig, ax = plt.subplots(figsize=(8, max(4., .65 * len(methods) + 1.6)))
    for i, method in enumerate(methods):
        for source in sources:
            values = [r["mean_competitor_minus_proposed_nll_per_token"] for r in points if r["method"] == method and r["source"] == source]
            ax.scatter(values, i + rng.uniform(-.18, .18, len(values)), s=10, alpha=.5,
                       color=colors[source], edgecolors="none", label=source if i == 0 else None)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(range(len(methods)), methods)
    ax.invert_yaxis()
    ax.set_xlabel("Competitor − Proposed document NLL (nats per target token)\nPositive values favor Proposed")
    ax.set_title("Development prediction differences across independent groups", fontsize=11)
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.02, .01, "Two-seed group averages; smallest mean development NLL profile per method. Exploratory, after selection.", fontsize=7)
    fig.tight_layout(rect=(0, .055, 1, 1))
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "paired_group_prediction_differences.png", dpi=220)
    fig.savefig(output / "paired_group_prediction_differences.pdf")
    plt.close(fig)
    write_csv(output / "paired_group_prediction_differences.csv", points, list(points[0]))
    write_json(output / "paired_group_prediction_differences.json", {"selected_predictive_profiles": chosen,
        "selection_rule": "smallest_complete_mean_development_macro_nll_then_profile_name",
        "display_unit": "independent_group_mean_over_two_training_seeds", "no_group_outlier_trimming": True,
        "scope": "exploratory_development_after_selection", "final_paper_figure": False})
