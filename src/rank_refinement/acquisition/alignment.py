"""Exact affinity-based core-to-topic matching; no decoder modification."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def align_cores(cores, affinity):
    b = np.asarray(affinity, dtype=np.float64)
    if b.ndim != 2 or len(cores) != b.shape[0] or not np.isfinite(b).all() or (b < 0).any() or (b.sum(axis=1) <= 0).any():
        raise ValueError("Invalid nonnegative native affinity/core interface.")
    phi = b / b.sum(axis=1, keepdims=True)
    if any(not c or min(c) < 0 or max(c) >= b.shape[1] for c in cores):
        raise ValueError("A core index lies outside the native vocabulary.")
    scores = np.array([[phi[k, c].mean() for c in cores] for k in range(len(cores))])
    rows, columns = linear_sum_assignment(-scores)
    return [cores[int(columns[np.flatnonzero(rows == k)[0]])] for k in range(len(cores))], scores
