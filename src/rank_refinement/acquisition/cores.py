"""Disjoint greedy cores and bounded deterministic improvement."""
from __future__ import annotations
from itertools import combinations
import numpy as np
from .graph import UnsupportedAcquisition


def core_objective(core, signed, degree, maximum, eta):
    m = len(core)
    cohesion = float(np.mean([signed[i, j] for i, j in combinations(core, 2)])) if m > 1 else 0.0
    return cohesion + eta * float(np.mean(degree[core] / maximum))


def select_cores(communities, adjacency, signed, support, words, eta=0.25, depth=10, swaps=10, original=False):
    cores, reports, pair_rows = [], [], []
    for topic, community in enumerate(communities):
        if len(community) < depth:
            raise UnsupportedAcquisition("An undersized community cannot supply a complete core.")
        degree = np.zeros(len(words))
        degree[community] = adjacency[np.ix_(community, community)].sum(axis=1)
        maximum = float(degree.max())
        if maximum <= 0:
            raise UnsupportedAcquisition("Community has no internal positive evidence.")
        first = min(community, key=lambda i: (-degree[i], i))
        core = [first]
        relation = adjacency if original else signed
        for _ in range(depth - 1):
            candidates = [i for i in community if i not in core]
            chosen = min(candidates, key=lambda i: (-float(relation[i, core].mean() + eta * degree[i] / maximum), i))
            core.append(chosen)
        accepted, history = 0, []
        if not original:
            for _ in range(swaps):
                current = core_objective(core, signed, degree, maximum, eta)
                best = None
                for removed in sorted(core):
                    for added in sorted(set(community) - set(core)):
                        candidate = sorted((set(core) - {removed}) | {added})
                        value = core_objective(candidate, signed, degree, maximum, eta)
                        if value > current + 1e-12 and (best is None or (-value, added, removed) < (-best[0], best[1], best[2])):
                            best = (value, added, removed, candidate)
                if best is None:
                    break
                value, added, removed, core = best
                accepted += 1
                history.append({"removed": words[removed], "added": words[added], "before": current, "after": value})
        core = sorted(core)
        values = [float(signed[i, j]) for i, j in combinations(core, 2)]
        cores.append(core)
        reports.append({"topic": topic, "community_size": len(community), "core_words": [words[i] for i in core],
                        "signed_cohesion": float(np.mean(values)), "minimum_signed_pair": min(values),
                        "zero_group_support_pairs": int(sum(support[i, j] == 0 for i, j in combinations(core, 2))),
                        "unsupported_or_nonpositive_pairs": sum(v <= 0 for v in values),
                        "objective": core_objective(core, signed, degree, maximum, eta),
                        "accepted_swaps": accepted, "swap_history": history})
        for i, j in combinations(core, 2):
            pair_rows.append({"topic": topic, "word_i": words[i], "word_j": words[j], "signed_score": float(signed[i, j]),
                              "independent_group_support": int(support[i, j]), "positive_graph_edge": float(adjacency[i, j])})
    if len(set(i for c in cores for i in c)) != len(cores) * depth:
        raise AssertionError("Cores are not disjoint.")
    return cores, reports, pair_rows
