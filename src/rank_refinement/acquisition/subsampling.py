"""Finite, fitting-group subsamples; shared across settings and interventions."""
from __future__ import annotations
import math
import numpy as np
from ..runtime.seeds import keyed_seed


def group_subsamples(group_ids, seed, count=20, fraction=0.8):
    ids = sorted(group_ids)
    if len(ids) != len(set(ids)) or not ids or count < 1 or not 0 < fraction < 1:
        raise ValueError("Invalid independent-group subsampling request.")
    size = max(1, math.floor(fraction * len(ids)))
    rng = np.random.default_rng(keyed_seed(seed, "acquisition_group_subsamples_v1"))
    indices = [np.sort(rng.choice(len(ids), size=size, replace=False)).tolist() for _ in range(count)]
    return indices, [[ids[i] for i in row] for row in indices]
