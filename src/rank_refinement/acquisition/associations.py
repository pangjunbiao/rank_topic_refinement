"""Signed window association, with explicit uninformative-event conventions."""
from __future__ import annotations
import numpy as np
from .subsampling import group_subsamples


def signed_npmi(word_counts, pair_counts, pairs, windows, epsilon=1e-12):
    counts = np.asarray(word_counts, dtype=np.float64)
    joint = np.asarray(pair_counts, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if windows <= 0 or np.any(counts < 0) or np.any(joint < 0):
        raise ValueError("Invalid association counts.")
    values = np.full(len(pairs), -1.0, dtype=np.float64)
    if len(pairs) == 0:
        return values
    a, b = counts[pairs[:, 0]] / windows, counts[pairs[:, 1]] / windows
    p = joint / windows
    ordinary = (a > 0) & (b > 0) & (p < 1)
    values[ordinary] = np.log((p[ordinary] + epsilon) / (a[ordinary] * b[ordinary])) / -np.log(p[ordinary] + epsilon)
    values[p >= 1] = 0.0  # ubiquitous joint event: no discriminating association
    return np.clip(values, -1, 1)


def association_bundle(evidence, group_ids, seed, policy):
    if list(group_ids) != sorted(set(group_ids)) or len(group_ids) != len(evidence["windows_per_group"]):
        raise ValueError("Evidence rows must match sorted unique fitting-group IDs.")
    indices, lists = group_subsamples(group_ids, seed, policy["subsamples"], policy["group_fraction"])
    pairs = evidence["pairs"]
    def measure(rows):
        words = evidence["word_windows"][rows].sum(axis=0)
        co = np.asarray(evidence["pair_windows"][rows].sum(axis=0)).ravel()
        total = int(evidence["windows_per_group"][rows].sum())
        return signed_npmi(words, co, pairs, total)
    samples = np.asarray([measure(rows) for rows in indices])
    stable = np.quantile(samples, policy["empirical_quantile"], axis=0, method="linear")
    full = measure(list(range(len(group_ids))))
    word_support = np.asarray(evidence["word_groups"].sum(axis=0)).ravel()
    pair_support = np.asarray(evidence["pair_groups"].sum(axis=0)).ravel()
    return {"words": evidence["words"], "pairs": pairs, "stable": stable, "full": full,
            "word_support": word_support, "pair_support": pair_support,
            "groups": len(group_ids), "subsample_group_ids": lists,
            "subsample_scores": samples, "sample_indices": indices}
