"""Bounded R1 settings, acquisition certificates, and frozen selection rule."""
from __future__ import annotations
from itertools import product
import numpy as np
from .graph import build_graph, UnsupportedAcquisition
from .partition import spectral_partition
from .cores import select_cores
from .global_cores import select_global_cores
from ..runtime.seeds import keyed_seed


def settings(policy):
    return [{"index": i + 1, "name": f"w{w}_p{p}_g{g:g}", "word_min": w, "pair_min": p, "gamma": g}
            for i, (w, p, g) in enumerate(product(policy["word_group_support"], policy["pair_group_support"], policy["gamma"]))]


def acquire(bundle, setting, seed, policy, stability=True, original=False, strategy=None, evidence_destination=None):
    graph, signed, support, active, diagnostics = build_graph(bundle, setting["word_min"], setting["pair_min"], setting["gamma"], stability)
    result = {"setting": setting, "seed": seed, "support": diagnostics, "structural_passed": False}
    try:
        if len(active) < policy["minimum_supported_nodes"]:
            raise UnsupportedAcquisition(f"positive_degree_words={len(active)} < required={policy['minimum_supported_nodes']}")
        selected_strategy = strategy or policy.get("core_search", "spectral_minimum_size_assignment")
        if original or selected_strategy == "spectral_minimum_size_assignment":
            communities, partition = spectral_partition(graph, active, keyed_seed(seed, "partition") % (2 ** 32 - 1),
                                                        restarts=policy.get("partition_restarts", 5),
                                                        iterations=policy.get("partition_max_iterations", 100))
            cores, reports, pairs = select_cores(communities, graph, signed, support, bundle["words"],
                                                eta=0.5 if original else policy["eta"], swaps=policy["max_accepted_swaps_per_core"], original=original)
            result.update(partition=partition, communities=communities)
        elif selected_strategy == "global_signed_candidate_packing":
            cores, reports, pairs, certificate = select_global_cores(graph, signed, support, active, bundle["words"],
                keyed_seed(seed, "global_core_search"), policy, destination=evidence_destination)
            result["core_search"] = certificate
            if not cores:
                raise UnsupportedAcquisition("Bounded candidate packing did not construct ten disjoint complete cores; no padding is permitted.")
        else:
            raise ValueError("Unknown declared core search: " + str(selected_strategy))
        result.update({"cores": cores, "core_reports": reports,
                       "all_core_pairs": pairs, "topics": [[bundle["words"][i] for i in c] for c in cores]})
        if not original and any(r["signed_cohesion"] <= 0 for r in reports):
            raise UnsupportedAcquisition("At least one complete ten-word core has nonpositive signed cohesion.")
        result["structural_passed"] = True
    except UnsupportedAcquisition as exc:
        result["structural_failure"] = str(exc)
    return result


def choose_setting(rows, anchors, grid, folds=(1, 2, 3), seeds=(809, 810)):
    """Missing/failed observations cannot disappear from endpoint means."""
    def complete(method_rows):
        expected = {(f, s) for f in folds for s in seeds}
        keys = [(r["fold"], r["seed"]) for r in method_rows]
        return len(keys) == len(set(keys)) and set(keys) == expected and all(r.get("scored") for r in method_rows)
    anchor_names = sorted({r["method"] for r in anchors})
    unavailable = [m for m in anchor_names if not complete([r for r in anchors if r["method"] == m])]
    if len(anchor_names) != 4 or unavailable:
        return None, [], {"status": "ANCHOR_PANEL_INCOMPLETE", "unavailable": unavailable, "anchor_names": anchor_names}
    anchor_fold = {m: {f: {metric: float(np.mean([r[metric] for r in anchors if r["method"] == m and r["fold"] == f]))
                           for metric in ["c_v", "c_npmi"]} for f in folds} for m in anchor_names}
    anchor_overall = {m: {metric: float(np.mean([anchor_fold[m][f][metric] for f in folds]))
                         for metric in ["c_v", "c_npmi"]} for m in anchor_names}
    strongest = {metric: max(anchor_overall[m][metric] for m in anchor_names) for metric in ["c_v", "c_npmi"]}
    diagnostics = []
    for setting in grid:
        selected = [r for r in rows if r["setting_name"] == setting["name"]]
        d = {**setting, "complete_six_runs": complete(selected), "qualified": False}
        if d["complete_six_runs"]:
            fold_values = {f: {metric: float(np.mean([r[metric] for r in selected if r["fold"] == f]))
                               for metric in ["c_v", "c_npmi"]} for f in folds}
            overall = {metric: float(np.mean([fold_values[f][metric] for f in folds])) for metric in ["c_v", "c_npmi"]}
            winning_folds = sum(all(fold_values[f][metric] > max(anchor_fold[m][f][metric] for m in anchor_names)
                                   for metric in ["c_v", "c_npmi"]) for f in folds)
            d.update(overall)
            d["folds_leading_both_endpoints"] = winning_folds
            d["coverage"] = float(np.mean([r["core_reference_token_fraction"] for r in selected]))
            d["qualified"] = all(r["structural_passed"] for r in selected) and overall["c_npmi"] > 0 and winning_folds >= 2 and all(overall[m] > strongest[m] for m in strongest)
        diagnostics.append(d)
    qualified = [d for d in diagnostics if d["qualified"]]
    winner = min(qualified, key=lambda d: (-d["c_npmi"], -d["c_v"], -d["coverage"], d["index"])) if qualified else None
    return winner, diagnostics, {"status": "COMPLETE", "strongest_endpoint_means": strongest,
                                 "anchor_means": anchor_overall, "fold_means": anchor_fold}
