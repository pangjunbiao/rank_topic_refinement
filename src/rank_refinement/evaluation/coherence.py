"""Pinned standard Gensim coherence, with transparent whole-request caching."""
from __future__ import annotations
import importlib.metadata
import math
from pathlib import Path
from ..runtime.artifacts import digest, read_json, write_json


class UnscorableCoherence(ValueError):
    pass


def score_topics(topics, texts, vocabulary, topn=10, cv_window=110, npmi_window=10, cache_dir=None):
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    if not topics or any(len(t) != topn or len(set(t)) != topn for t in topics):
        raise ValueError("All descriptor slots must be present and distinct within a topic.")
    words = list(vocabulary)
    if len(set(words)) != len(words) or not words:
        raise ValueError("Invalid fitting vocabulary.")
    allowed = set(words)
    if any(word not in allowed for topic in topics for word in topic):
        raise ValueError("A descriptor is outside the common fitting vocabulary.")
    if not texts or not any(texts):
        raise UnscorableCoherence("Evaluation reference has no lexical tokens.")
    dictionary = Dictionary([words])
    key = digest({"topics": topics, "texts": texts, "vocabulary": words, "topn": topn,
                  "cv_window": cv_window, "npmi_window": npmi_window,
                  "gensim": importlib.metadata.version("gensim"), "policy": "standard_keep_all_slots_v1"})
    cache = Path(cache_dir) / (key + ".json") if cache_dir is not None else None
    if cache is not None and cache.is_file():
        result = read_json(cache)
        if result["cache_key"] != key:
            raise ValueError("Coherence cache identity mismatch.")
        return result
    result = {"cache_key": key, "topn": topn, "reference_documents": len(texts),
              "reference_absent_words": sorted({w for t in topics for w in t} - {w for text in texts for w in text}),
              "windows": {"c_v": cv_window, "c_npmi": npmi_window}, "gensim": importlib.metadata.version("gensim")}
    for name, window in [("c_v", cv_window), ("c_npmi", npmi_window)]:
        model = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence=name,
                               topn=topn, window_size=window, processes=1)
        values = [float(x) for x in model.get_coherence_per_topic()]
        if any(not math.isfinite(v) for v in values):
            raise UnscorableCoherence(f"Standard {name} returned a non-finite topic score; keep all slots and inspect support.")
        result[name] = {"mean": sum(values) / len(values), "per_topic": values}
    if cache is not None:
        write_json(cache, result)
    return result


def run_metric_oracles(folder, tolerance=1e-10):
    """Synthetic checks only: known events plus separate direct-library calls."""
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    from .diversity import topic_diversity
    from .likelihood import aggregate_nll, document_loss
    alpha = ["alpha" + str(i) for i in range(10)]
    beta = ["beta" + str(i) for i in range(10)]
    topics, words = [alpha, beta], alpha + beta
    texts = [alpha * 2, beta * 2, alpha + beta, list(reversed(alpha)), list(reversed(beta)), alpha[:5] + beta[:5]]
    wrapped = score_topics(topics, texts, words, cache_dir=Path(folder) / "oracle_cache")
    warm = score_topics(topics, texts, words, cache_dir=Path(folder) / "oracle_cache")
    checks = []
    for metric, width in [("c_v", 110), ("c_npmi", 10)]:
        direct = CoherenceModel(topics=topics, texts=texts, dictionary=Dictionary([words]),
                                coherence=metric, window_size=width, topn=10, processes=1)
        actual = [float(x) for x in direct.get_coherence_per_topic()]
        error = max(abs(a - b) for a, b in zip(actual, wrapped[metric]["per_topic"]))
        if error > tolerance or wrapped != warm:
            raise AssertionError("Pinned coherence oracle/cache mismatch.")
        checks.append({"name": metric + "_direct_and_cached", "passed": True, "max_absolute_error": error})
    pairs = score_topics([["a", "b"]], [["a", "b"], ["a", "b"], ["c", "d"], ["c", "d"]], ["a", "b", "c", "d"], topn=2)
    error = abs(pairs["c_npmi"]["mean"] - 1)
    if error > tolerance:
        raise AssertionError("Known perfect association NPMI differs from one beyond smoothing tolerance.")
    checks.append({"name": "npmi_known_probability_event", "passed": True, "max_absolute_error": error})
    absent = alpha[:-1] + ["absent"]
    reference = texts + [["alpha9", "beta9"]]
    import warnings
    import numpy as np
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        direct_values = {}
        for metric, width in [("c_v", 110), ("c_npmi", 10)]:
            direct = CoherenceModel(topics=[absent], texts=reference, dictionary=Dictionary([words + ["absent"]]),
                                    coherence=metric, topn=10, window_size=width, processes=1)
            direct_values[metric] = float(direct.get_coherence())
        try:
            absent_score = score_topics([absent], reference, words + ["absent"])
        except UnscorableCoherence:
            if all(math.isfinite(x) for x in direct_values.values()):
                raise AssertionError("Finite standard absent-word score was incorrectly rejected.")
            handling = "explicit_unscorable_matching_standard_nonfinite"
        else:
            if any(abs(absent_score[k]["mean"] - direct_values[k]) > tolerance for k in direct_values):
                raise AssertionError("Absent reference word was incorrectly zeroed/truncated.")
            handling = "finite_direct_scores_without_slot_removal"
    checks.append({"name": "reference_absent_word_keeps_all_slots", "passed": True, "handling": handling})
    if topic_diversity(topics) != 1 or topic_diversity([alpha, alpha]) != 0.5:
        raise AssertionError("TD known-set oracle failed.")
    checks.append({"name": "topic_diversity_known_sets", "passed": True})
    loss = document_loss([0.5, 0.5], [0, 1, 1])
    if abs(loss["negative_log_probability"] / 3 - math.log(2)) > tolerance:
        raise AssertionError("Natural-log target-token NLL oracle failed.")
    aggregate = aggregate_nll([{**document_loss([0.8, 0.2], [0] * 100), "source": "s1"},
                               {**document_loss([0.8, 0.2], [1]), "source": "s2"}], ["s1", "s2"])
    expected_macro = (-math.log(0.8) - math.log(0.2)) / 2
    expected_micro = (-100 * math.log(0.8) - math.log(0.2)) / 101
    if abs(aggregate["macro_source_nll"] - expected_macro) > tolerance or abs(aggregate["micro_nll"] - expected_micro) > tolerance:
        raise AssertionError("Macro/micro NLL weighting oracle failed.")
    checks.append({"name": "completion_nll_units_and_aggregation", "passed": True})
    return {"scope": "synthetic_evaluator_checks_not_model_results", "passed": True,
            "checks": checks, "windows": {"c_v": 110, "c_npmi": 10},
            "zero_pair_handling": "Pinned Gensim epsilon-smoothed standard measure; no extra full-topic zero rule."}
