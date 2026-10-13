"""Normalized spectral embedding and exact minimum-size assignment per iterate."""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from .graph import UnsupportedAcquisition


def minimum_size_assignment(cost, minimum):
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or not np.isfinite(cost).all():
        raise ValueError("Assignment cost must be a finite matrix.")
    n, k = cost.shape
    if minimum < 1 or n < k * minimum:
        raise UnsupportedAcquisition("Insufficient words for minimum community sizes.")
    labels = cost.argmin(axis=1)
    baseline = cost[np.arange(n), labels]
    slots = np.repeat(np.arange(k), minimum)
    mandatory = (cost[:, slots] - baseline[:, None]).T
    rows, points = linear_sum_assignment(mandatory)
    labels[points] = slots[rows]
    if (np.bincount(labels, minlength=k) < minimum).any():
        raise AssertionError("Minimum-size assignment violated its quota.")
    return labels


def spectral_partition(adjacency, active, seed, topics=10, minimum=10, restarts=5, iterations=100):
    from sklearn.cluster import kmeans_plusplus
    active = np.asarray(active, dtype=np.int64)
    if len(active) < topics * minimum:
        raise UnsupportedAcquisition(f"Only {len(active)} positive-degree words; need {topics * minimum}.")
    a = adjacency[np.ix_(active, active)]
    degree = a.sum(axis=1)
    norm = a / np.sqrt(degree[:, None] * degree[None, :])
    eigenvalues, vectors = eigh(norm, subset_by_index=[len(a) - topics, len(a) - 1], driver="evr")
    embedding = vectors[:, ::-1]
    for j in range(topics):
        pivot = np.argmax(np.abs(embedding[:, j]))
        if embedding[pivot, j] < 0:
            embedding[:, j] *= -1
    lengths = np.linalg.norm(embedding, axis=1)
    embedding /= np.maximum(lengths[:, None], np.finfo(float).eps)
    best = None
    for restart in range(restarts):
        centers, _ = kmeans_plusplus(embedding, n_clusters=topics, random_state=(int(seed) + restart) % (2 ** 32 - 1))
        previous, converged = None, False
        for iteration in range(iterations):
            costs = cdist(embedding, centers, "sqeuclidean")
            labels = minimum_size_assignment(costs, minimum)
            centers = np.vstack([embedding[labels == j].mean(axis=0) for j in range(topics)])
            if previous is not None and np.array_equal(labels, previous):
                converged = True
                break
            previous = labels.copy()
        objective = float(np.square(embedding - centers[labels]).sum())
        canonical = sorted([sorted(active[labels == j].tolist()) for j in range(topics)], key=lambda ids: ids[0])
        candidate = (objective, canonical, converged, iteration + 1)
        if best is None or (objective, canonical) < (best[0], best[1]):
            best = candidate
    return best[1], {"spectral_eigenvalues": eigenvalues[::-1].tolist(), "assignment_objective": best[0],
                     "assignment_converged": best[2], "iterations": best[3], "restarts": restarts,
                     "community_sizes": [len(c) for c in best[1]]}
