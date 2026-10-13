"""Independent standard-metric requests; no slot removal or NA substitution."""
from __future__ import annotations
import warnings
import numpy as np


def audit_reference(topics, records, vocabulary):
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    texts = [r["tokens"] for r in records]
    if len(topics) != 10 or any(len(t) != 10 or len(set(t)) != 10 for t in topics):
        raise ValueError("Reference audit must retain all ten topics and all ten words.")
    dictionary = Dictionary([list(vocabulary)])
    reference_words = {w for text in texts for w in text}
    present = [[w in reference_words for w in t] for t in topics]
    result = {"reference_groups": len(records), "reference_absent_words": sorted({w for t in topics for w in t} - reference_words),
        "supported_slots": sum(sum(t) for t in present), "total_slots": 100,
        "topic_supported_slots": [sum(t) for t in present], "all_descriptor_slots_retained": True,
        "metrics": {}}
    for name, width in [("c_v", 110), ("c_npmi", 10)]:
        with warnings.catch_warnings(record=True) as caught, np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("always", RuntimeWarning)
            model = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence=name,
                                   window_size=width, topn=10, processes=1)
            raw = np.asarray(model.get_coherence_per_topic(), dtype=np.float64)
        finite = np.isfinite(raw)
        complete = bool(finite.all())
        result["metrics"][name] = {"window": width, "complete": complete,
            "mean": float(raw.mean()) if complete else None,
            "per_topic": [float(v) if ok else None for v, ok in zip(raw, finite)],
            "finite_topic_count": int(finite.sum()),
            "nonfinite_topic_indices": np.flatnonzero(~finite).tolist(),
            "failure": None if complete else "standard_metric_nonfinite_no_slot_removal",
            "warnings": sorted({str(w.message) for w in caught})}
    return result
