"""Fitting-only signed core search; verified integer packing of finite candidates.

The finite candidate pool is a heuristic. An optimal packing is optimal only
inside that pool, never a certificate about all possible vocabulary subsets.
No validation texts, scores, labels or embeddings enter this module.
"""
from __future__ import annotations
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, vstack
from .graph import UnsupportedAcquisition


def pair_means(core, signed, adjacency):
    pairs = list(combinations(core, 2))
    if not pairs:
        raise ValueError("A core needs at least two distinct words.")
    return (float(np.mean([signed[i, j] for i, j in pairs])),
            float(np.mean([adjacency[i, j] for i, j in pairs])))


def verify_packing(cores, vocabulary_size, maximum_topics, depth):
    if len(cores) > maximum_topics:
        raise ValueError("Packing exceeds the common topic count.")
    flat = []
    for core in cores:
        if len(core) != depth or len(set(core)) != depth:
            raise ValueError("Packing contains a padded or repeated word slot.")
        if any(i < 0 or i >= vocabulary_size for i in core):
            raise ValueError("Packing leaves the fitting vocabulary.")
        flat.extend(core)
    if len(set(flat)) != len(flat):
        raise ValueError("Selected cores overlap.")
    return True


def pack_candidates(candidates, quality, vocabulary_size, topics=10, depth=10,
                    time_limit=10.0, node_limit=10000):
    """Maximize cardinality first, then summed quality, with at most K cores."""
    candidates = [tuple(c) for c in candidates]
    quality = np.asarray(quality, dtype=float)
    if quality.shape != (len(candidates),) or not np.isfinite(quality).all():
        raise ValueError("Candidate quality must be finite and aligned.")
    if time_limit <= 0 or node_limit < 1 or topics < 1:
        raise ValueError("Invalid bounded packing budget.")
    for core in candidates:
        verify_packing([core], vocabulary_size, topics, depth)
    if len(set(candidates)) != len(candidates):
        raise ValueError("Duplicate candidate cores are not separate evidence.")
    if not candidates:
        return [], {"status": "empty_pool", "selected_count": 0,
                    "optimal_within_candidate_pool": True}
    row, col = [], []
    for j, core in enumerate(candidates):
        for i in core:
            row.append(i); col.append(j)
    incidence = csr_matrix((np.ones(len(row)), (row, col)),
                           shape=(vocabulary_size, len(candidates)))
    matrix = vstack([incidence, csr_matrix(np.ones((1, len(candidates))))]).tocsr()
    # This dominates the maximum possible total quality change between two
    # packings, including negative diagnostic candidates. No tiny tie penalty.
    cardinality_weight = 2 * topics * max(1.0, float(np.max(np.abs(quality)))) + 1
    result = milp(-(cardinality_weight + quality), integrality=np.ones(len(candidates)),
                  bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, np.zeros(vocabulary_size + 1),
                                               np.r_[np.ones(vocabulary_size), topics]),
                  options={"time_limit": float(time_limit), "node_limit": int(node_limit),
                           "mip_rel_gap": 0.0, "presolve": True})
    selected = []
    if result.x is not None:
        x = np.asarray(result.x, dtype=float)
        if not np.isfinite(x).all() or np.max(np.abs(x - np.rint(x))) > 1e-6:
            raise ValueError("Solver did not provide a finite integer incumbent.")
        if (x < -1e-6).any() or (x > 1 + 1e-6).any():
            raise ValueError("Solver incumbent violates variable bounds.")
        selected = np.flatnonzero(x > .5).tolist()
        verify_packing([candidates[j] for j in selected], vocabulary_size, topics, depth)
    def finite_field(key):
        value = getattr(result, key, None)
        return float(value) if value is not None and np.isfinite(value) else None
    report = {"status": int(result.status), "message": str(result.message),
              "selected_count": len(selected), "verified_integer_incumbent": result.x is not None,
              "optimal_within_candidate_pool": result.status == 0,
              "mip_gap": finite_field("mip_gap"), "mip_node_count": finite_field("mip_node_count"),
              "objective": finite_field("fun"), "cardinality_weight": cardinality_weight,
              "time_limit_seconds": float(time_limit), "node_limit": int(node_limit)}
    return selected, report


def generate_candidates(available, relation, signed, adjacency, depth, swaps, pool):
    """Every available word initializes one greedy core plus bounded exchanges."""
    available = np.asarray(sorted(set(int(i) for i in available)), dtype=np.int64)
    if len(available) < depth:
        return 0
    local = relation[np.ix_(available, available)]
    generated = 0
    for anchor in range(len(available)):
        core = [anchor]
        running = local[:, anchor].copy()
        for _ in range(depth - 1):
            scores = running.copy(); scores[core] = -np.inf
            chosen = int(np.argmax(scores))
            core.append(chosen); running += local[:, chosen]
        for _ in range(swaps):
            sums = local[:, core].sum(axis=1)
            old = local[np.ix_(core, core)].sum(axis=1)
            gains = sums[:, None] - local[:, core] - old[None, :]
            gains[core, :] = -np.inf
            added, position = np.unravel_index(np.argmax(gains), gains.shape)
            if gains[added, position] <= 1e-12:
                break
            core[position] = int(added)
        ids = tuple(sorted(int(available[j]) for j in core))
        if ids not in pool:
            pool[ids] = pair_means(ids, signed, adjacency)
        generated += 1
    return generated


def select_global_cores(adjacency, signed, support, active, words, seed, policy,
                        topics=10, depth=10, destination=None):
    n = len(words); active = np.asarray(active, dtype=np.int64)
    matrices = [np.asarray(x) for x in (adjacency, signed, support)]
    if n != len(set(words)) or any(x.shape != (n, n) for x in matrices):
        raise ValueError("Global core matrices and unique fitting words must align.")
    if any(not np.isfinite(x).all() or not np.allclose(x, x.T) for x in matrices):
        raise ValueError("Core evidence must be finite and symmetric.")
    if (adjacency < 0).any() or (adjacency > 1 + 1e-12).any() or (support < 0).any():
        raise ValueError("Invalid positive graph or group support.")
    if (np.abs(signed) > 1 + 1e-12).any() or not np.allclose(np.diag(signed), 0):
        raise ValueError("Signed scores must lie in [-1,1] with zero diagonal.")
    if active.tolist() != sorted(set(active.tolist())) or any(i < 0 or i >= n for i in active):
        raise ValueError("Active fitting word IDs must be sorted and unique.")
    if len(active) < topics * depth:
        raise UnsupportedAcquisition("Insufficient distinct graph-supported words for all core slots.")
    search = policy["global_core_search"]; eta = float(policy["eta"])
    if eta < 0:
        raise ValueError("Negative graph-strength weight.")
    rng = np.random.default_rng(int(seed))
    relation = np.asarray(signed + eta * adjacency, dtype=float)
    np.fill_diagonal(relation, 0)
    fractions = search["word_subset_fractions"]
    if not fractions or any(not 0 < f <= 1 for f in fractions):
        raise ValueError("Invalid fitting-only candidate diversity fractions.")
    pool, masks = {}, [active]
    for draw in range(search["diversity_draws"]):
        size = min(len(active), max(2 * depth, int(len(active) * fractions[draw % len(fractions)])))
        masks.append(np.sort(rng.choice(active, size=size, replace=False)))
    emitted = 0
    for mask in masks:
        emitted += generate_candidates(mask, relation, signed, adjacency, depth,
                                       policy["max_accepted_swaps_per_core"], pool)
    history, chosen = [], []
    def positive_pack():
        candidates = sorted(c for c, (q, _) in pool.items() if q > 1e-12)
        ids, report = pack_candidates(candidates, [pool[c][0] + eta * pool[c][1] for c in candidates],
                                     n, topics, depth, search["solver_time_limit_seconds"],
                                     search["solver_node_limit"])
        report["positive_candidate_count"] = len(candidates)
        return [candidates[j] for j in ids], report
    chosen, report = positive_pack(); history.append({"phase": "initial", **report})
    for drops in search["repair_drop_counts"]:
        if len(chosen) == topics or not chosen:
            break
        before = len(pool); used = {i for c in chosen for i in c}
        views = [set(active) - used]
        for dropped in combinations(chosen, drops):
            views.append((set(active) - used) | {i for c in dropped for i in c})
        for available in views:
            emitted += generate_candidates(available, relation, signed, adjacency, depth,
                                           policy["max_accepted_swaps_per_core"], pool)
        updated, report = positive_pack()
        # Keep a verified earlier cardinality if a timed solver returns less.
        old_value = (len(chosen), sum(pool[c][0] + eta * pool[c][1] for c in chosen))
        new_value = (len(updated), sum(pool[c][0] + eta * pool[c][1] for c in updated))
        if new_value >= old_value:
            chosen = updated
        history.append({"phase": f"drop_{drops}_repair", "new_candidates": len(pool) - before,
                        "retained_core_count": len(chosen), **report})
    positive_count = len(chosen); diagnostic = False
    if positive_count < topics:
        candidates = sorted(pool)
        ids, report = pack_candidates(candidates, [pool[c][0] + eta * pool[c][1] for c in candidates],
                                     n, topics, depth, search["diagnostic_time_limit_seconds"],
                                     search["solver_node_limit"])
        history.append({"phase": "complete_failed_core_diagnostic", **report})
        complete = [candidates[j] for j in ids]
        if len(complete) == topics:
            chosen = complete; diagnostic = True
    candidates = sorted(pool)
    certificate = {"strategy": "global_signed_candidate_packing", "candidate_count": len(pool),
                   "positive_candidate_count": sum(q > 1e-12 for q, _ in pool.values()),
                   "generated_starts": emitted, "diversity_draws": search["diversity_draws"],
                   "mask_seed": int(seed), "word_subset_fractions": fractions,
                   "repair_drop_counts": search["repair_drop_counts"], "positive_packing_count": positive_count,
                   "returned_core_count": len(chosen), "failed_complete_diagnostic": diagnostic,
                   "search_history": history, "selected_core_ids": [list(c) for c in sorted(chosen)],
                   "heuristic_failure_does_not_prove_global_infeasibility": True,
                   "objective": "mean_all_signed_pairs + eta * mean_internal_positive_graph_edges",
                   "validation_used_in_search": False}
    if destination is not None:
        p = Path(destination); p.mkdir(parents=True, exist_ok=True)
        mask_matrix = np.zeros((len(masks), n), dtype=bool)
        for j, mask in enumerate(masks):
            mask_matrix[j, mask] = True
        np.savez_compressed(p / "candidate_pool.npz", cores=np.asarray(candidates, dtype=np.int64).reshape(-1, depth),
                            signed_cohesion=np.array([pool[c][0] for c in candidates]),
                            graph_strength=np.array([pool[c][1] for c in candidates]),
                            active_ids=active, fitting_word_masks=mask_matrix)
    if len(chosen) != topics:
        return [], [], [], certificate
    cores = [list(c) for c in sorted(chosen)]
    verify_packing(cores, n, topics, depth)
    reports, pair_rows = [], []
    for topic, core in enumerate(cores):
        pairs = list(combinations(core, 2)); values = [float(signed[i, j]) for i, j in pairs]
        q, graph_mean = pair_means(core, signed, adjacency)
        reports.append({"topic": topic, "core_words": [words[i] for i in core],
                        "signed_cohesion": q, "minimum_signed_pair": min(values),
                        "internal_graph_strength": graph_mean, "objective": q + eta * graph_mean,
                        "zero_group_support_pairs": int(sum(support[i, j] == 0 for i, j in pairs)),
                        "unsupported_or_nonpositive_pairs": sum(v <= 0 for v in values)})
        pair_rows.extend({"topic": topic, "word_i": words[i], "word_j": words[j],
                          "signed_score": float(signed[i, j]), "independent_group_support": int(support[i, j]),
                          "positive_graph_edge": float(adjacency[i, j])} for i, j in pairs)
    return cores, reports, pair_rows, certificate
