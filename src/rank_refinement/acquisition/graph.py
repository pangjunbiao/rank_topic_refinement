"""Positive evidence partitions; signed eligible evidence scores every core pair."""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class UnsupportedAcquisition(ValueError):
    pass


def build_graph(bundle, word_min, pair_min, gamma, stability=True):
    n = len(bundle["words"])
    signed = np.full((n, n), -1.0)
    np.fill_diagonal(signed, 0)
    pairs = bundle["pairs"]
    eligible = bundle["word_support"] >= word_min
    keep = (bundle["pair_support"] >= pair_min) & eligible[pairs[:, 0]] & eligible[pairs[:, 1]]
    scores = bundle["stable"] if stability else bundle["full"]
    # Support thresholds screen partition edges, not the signed all-pair loss.
    # Weak and zero-support pairs retain their measured signed association.
    signed[pairs[:, 0], pairs[:, 1]] = scores
    signed[pairs[:, 1], pairs[:, 0]] = scores
    p = pairs[keep]
    adjacency = np.zeros((n, n))
    weights = np.maximum(scores[keep], 0) * (bundle["pair_support"][keep] / bundle["groups"]) ** gamma
    adjacency[p[:, 0], p[:, 1]] = weights
    adjacency[p[:, 1], p[:, 0]] = weights
    maximum = float(adjacency.max())
    if maximum > 0:
        adjacency /= maximum
    active = np.flatnonzero(adjacency.sum(axis=1) > 0)
    components = connected_components(csr_matrix(adjacency[np.ix_(active, active)]), directed=False)[0] if len(active) else 0
    support = np.zeros((n, n), dtype=np.int64)
    support[pairs[:, 0], pairs[:, 1]] = bundle["pair_support"]
    support[pairs[:, 1], pairs[:, 0]] = bundle["pair_support"]
    diagnostics = {"eligible_word_count": int(eligible.sum()), "positive_degree_words": len(active),
                   "eligible_pairs": int(keep.sum()), "positive_edges": int((weights > 0).sum()),
                   "components": int(components), "maximum_unnormalized_edge": maximum,
                   "signed_policy": "all_pairs_gensim_smoothed_npmi; absent_marginal=-1; ubiquitous_joint=0"}
    return adjacency, signed, support, active, diagnostics
