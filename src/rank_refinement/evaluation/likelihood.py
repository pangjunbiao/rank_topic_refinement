"""Nats/token aggregation, without invented predictors or log-probability clipping."""
import math
from collections import defaultdict


def document_loss(probabilities, target_ids):
    import numpy as np
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any(p < 0) or abs(float(p.sum()) - 1) > 1e-8:
        raise ValueError("Predictor must supply a finite normalized probability vector.")
    target = np.asarray(target_ids)
    if target.ndim != 1 or target.size == 0 or target.dtype.kind not in "iu" or int(target.min()) < 0 or int(target.max()) >= len(p):
        raise ValueError("Targets must be nonempty vocabulary IDs.")
    selected = p[target]
    if np.any(selected <= 0):
        return {"negative_log_probability": None, "target_tokens": len(target_ids), "status": "infinite_nll_zero_target_probability"}
    return {"negative_log_probability": float(-np.log(selected).sum()), "target_tokens": len(target_ids), "status": "finite"}


def aggregate_nll(records, expected_sources):
    if not expected_sources or len(set(expected_sources)) != len(expected_sources):
        raise ValueError("Collection sources must be nonempty and distinct.")
    sums, counts = defaultdict(float), defaultdict(int)
    for row in records:
        value, count = row["negative_log_probability"], int(row["target_tokens"])
        if value is None or not math.isfinite(value) or value < 0 or count <= 0:
            raise ValueError("Non-finite or invalid document loss; do not hide failures through clipping.")
        if row["source"] not in expected_sources:
            raise ValueError("Unexpected collection source.")
        sums[row["source"]] += value
        counts[row["source"]] += count
    if any(counts[s] == 0 for s in expected_sources):
        raise ValueError("Every declared collection source needs target tokens for macro NLL.")
    by_source = {s: sums[s] / counts[s] for s in expected_sources}
    return {"macro_source_nll": sum(by_source.values()) / len(expected_sources),
            "micro_nll": sum(sums.values()) / sum(counts.values()), "by_source": by_source,
            "target_tokens_by_source": dict(counts), "units": "nats_per_target_token"}
