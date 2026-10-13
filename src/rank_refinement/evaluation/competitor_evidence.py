"""Complete native-topic references and paired development diagnostics."""
from __future__ import annotations
import importlib.metadata
import statistics
import warnings
from pathlib import Path
import numpy as np
from ..runtime.artifacts import artifact_manifest, digest, read_json, write_json
from ..stages.step02_completion import check_indexed_tree
from .likelihood import aggregate_nll


def audit_native_reference(topics, records, vocabulary):
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    if not topics or any(len(t) != 10 or len(set(t)) != 10 for t in topics):
        raise ValueError("Retain every native topic and all ten genuine descriptor words.")
    texts = [r["tokens"] for r in records]
    reference_words = {w for row in texts for w in row}
    dictionary = Dictionary([list(vocabulary)])
    metrics = {}
    for name, width in [("c_v", 110), ("c_npmi", 10)]:
        with warnings.catch_warnings(record=True) as caught, np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("always", RuntimeWarning)
            model = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence=name,
                                   window_size=width, topn=10, processes=1)
            values = np.asarray(model.get_coherence_per_topic(), dtype=np.float64)
        finite = np.isfinite(values)
        metrics[name] = {"window": width, "complete": bool(finite.all()),
            "mean": float(values.mean()) if finite.all() else None,
            "per_topic": [float(v) if good else None for v, good in zip(values, finite)],
            "nonfinite_topic_indices": np.flatnonzero(~finite).tolist(),
            "warnings": sorted({str(w.message) for w in caught})}
    return {"reference_groups": len(records), "native_topics": len(topics), "total_slots": len(topics) * 10,
        "reference_absent_words": sorted({w for t in topics for w in t} - reference_words),
        "supported_slots": sum(w in reference_words for t in topics for w in t),
        "all_native_topic_slots_retained": True, "metrics": metrics}


def reference_scores(topics, data, cache):
    identity = {"topics": topics, "vocabulary": data["vocabulary"], "gensim": importlib.metadata.version("gensim"),
        "fit": [(r["group_id"], r["tokens"]) for r in data["fit_records"]],
        "development": [(r["group_id"], r["tokens"]) for r in data["dev_coherence_records"]],
        "windows": [110, 10], "retain_all_native_slots": True}
    key = digest(identity)
    folder = Path(cache) / key
    if (folder / "artifact_manifest.json").is_file():
        check_indexed_tree(folder)
        audit = read_json(folder / "reference_audit.json")
        if audit["identity_sha256"] != key:
            raise ValueError("Competitor reference cache identity changed.")
    else:
        audit = {"identity_sha256": key,
                 "fitting": audit_native_reference(topics, data["fit_records"], data["vocabulary"]),
                 "development": audit_native_reference(topics, data["dev_coherence_records"], data["vocabulary"])}
        write_json(folder / "reference_audit.json", audit)
        write_json(folder / "artifact_manifest.json", artifact_manifest(folder))
    def metric(name):
        entry = audit[name]
        complete = all(entry["metrics"][m]["complete"] for m in ["c_v", "c_npmi"])
        return {"c_v": entry["metrics"]["c_v"]["mean"], "c_npmi": entry["metrics"]["c_npmi"]["mean"],
            "scored": complete, "all_native_topic_slots_retained": True,
            "native_topics": entry["native_topics"], "reference_groups": entry["reference_groups"],
            "failure": None if complete else "standard_metric_nonfinite_no_slot_removal"}
    return metric("fitting"), metric("development"), audit


def summarize(rows, seeds):
    fields = {"fitting_c_v": ("fitting", "c_v"), "fitting_c_npmi": ("fitting", "c_npmi"),
        "development_c_v": ("development", "c_v"), "development_c_npmi": ("development", "c_npmi"),
        "development_macro_nll": ("prediction", "macro_source_nll"), "development_micro_nll": ("prediction", "micro_nll"),
        "native_fitting_c_v": ("native_fitting", "c_v"), "native_fitting_c_npmi": ("native_fitting", "c_npmi")}
    output = []
    for method, profile in sorted({(r["method"], r["profile"]) for r in rows}):
        subset = [r for r in rows if (r["method"], r["profile"]) == (method, profile)]
        complete = len(subset) == len(seeds) and {r["seed"] for r in subset} == set(seeds)
        entry = {"method": method, "profile": profile, "n_seeds": len(subset), "complete_seed_grid": complete,
            "all_cases_executed": bool(complete and all(r["status"] != "failed" for r in subset)),
            "fixed_k10_all_seeds": bool(complete and all(r.get("fixed_k10", True) for r in subset)),
            "native_topics_by_seed": [r.get("native_topic_count", 10) for r in sorted(subset, key=lambda r: r["seed"])],
            "prediction_interface": subset[0].get("prediction_interface"), "final_model_result": False}
        for name, (parent, child) in fields.items():
            values = [r.get(parent, {}).get(child) for r in subset]
            good = complete and all(v is not None and np.isfinite(v) for v in values)
            entry[name + "_mean"] = statistics.mean(values) if good else None
            entry[name + "_sd"] = statistics.stdev(values) if good and len(values) > 1 else None
        for name in ["topic_diversity", "native_topic_diversity"]:
            values = [r.get(name) for r in subset]
            good = complete and all(v is not None and np.isfinite(v) for v in values)
            entry[name + "_mean"] = statistics.mean(values) if good else None
            entry[name + "_sd"] = statistics.stdev(values) if good and len(values) > 1 else None
        output.append(entry)
    return output


def paired_groups(proposed, competitors, seeds):
    if {r["seed"] for r in proposed} != set(seeds) or {r["seed"] for r in competitors} != set(seeds):
        raise ValueError("Paired diagnostic requires the complete seed grid.")
    rows = []
    for seed in seeds:
        p = next(r for r in proposed if r["seed"] == seed)
        c = next(r for r in competitors if r["seed"] == seed)
        pm = {r["group_id"]: r for r in p["document_losses"]}
        cm = {r["group_id"]: r for r in c["document_losses"]}
        if len(pm) != len(p["document_losses"]) or len(cm) != len(c["document_losses"]) or set(pm) != set(cm):
            raise ValueError("Paired completion groups differ, are duplicated, or are missing.")
        for group in sorted(pm):
            a, b = pm[group], cm[group]
            if a["source"] != b["source"] or a["target_tokens"] != b["target_tokens"]:
                raise ValueError("Paired target/source contract differs.")
            rows.append({"seed": seed, "group_id": group, "source": a["source"], "target_tokens": a["target_tokens"],
                "proposed_negative_log_probability": a["negative_log_probability"],
                "competitor_negative_log_probability": b["negative_log_probability"],
                "gain_per_token": (b["negative_log_probability"] - a["negative_log_probability"]) / a["target_tokens"]})
    return rows


def bootstrap_gain(paired, seeds, resamples=10000, seed=4809):
    rng = np.random.default_rng(seed)
    sources = sorted({r["source"] for r in paired})
    draws = np.zeros(resamples)
    points = []
    for source in sources:
        sample = sorted((r for r in paired if r["seed"] == seeds[0] and r["source"] == source), key=lambda r: r["group_id"])
        if not sample:
            raise ValueError("Every source requires paired groups.")
        groups = [r["group_id"] for r in sample]
        weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=resamples)
        tokens = np.array([r["target_tokens"] for r in sample], dtype=np.float64)
        denominator = weights @ tokens
        for training_seed in seeds:
            indexed = {r["group_id"]: r for r in paired if r["seed"] == training_seed and r["source"] == source}
            if set(indexed) != set(groups):
                raise ValueError("Training seeds have different paired completion groups.")
            gain = np.array([indexed[g]["competitor_negative_log_probability"] - indexed[g]["proposed_negative_log_probability"] for g in groups])
            points.append(float(gain.sum() / tokens.sum()))
            draws += (weights @ gain) / denominator / (len(sources) * len(seeds))
    lower, upper = np.quantile(draws, [.025, .975])
    return {"mean_competitor_minus_proposed_macro_nll": statistics.mean(points),
        "percentile_95_low": float(lower), "percentile_95_high": float(upper),
        "resamples": resamples, "seed": seed, "unit": "independent_duplicate_group",
        "source_stratified": True, "shared_resampled_groups_across_training_seeds": True,
        "scope": "exploratory_post_selection_not_confirmatory", "training_seed_uncertainty_included": False}


def availability_gates(means, methods):
    descriptor = {m: any(r["method"] == m and r["fixed_k10_all_seeds"]
        and r["fitting_c_v_mean"] is not None and r["fitting_c_npmi_mean"] is not None for r in means) for m in methods}
    predictive = {m: any(r["method"] == m and r["development_macro_nll_mean"] is not None for r in means)
                  for m in methods if m != "BERTopic"}
    return {"fixed_k_descriptor_availability": descriptor, "compatible_prediction_availability": predictive,
        "bertopic_native_word_likelihood": "not_defined_not_a_win_or_loss",
        "fixed_k_descriptor_panel_complete": all(descriptor.values()),
        "compatible_prediction_panel_complete": all(predictive.values())}
