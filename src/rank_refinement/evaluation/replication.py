"""Complete-grid replication, paired seed/group uncertainty and readout audits."""
from __future__ import annotations
import copy
import statistics
import numpy as np
from .competitor_evidence import paired_groups
from ..models.adapters.pilot_anchors import topics_from_weights


def ablated_evidence(evidence, variant):
    result = copy.deepcopy(evidence)
    if variant == "without_subsample_stability":
        result["stable"] = result["full"].copy()
    elif variant == "positive_only_association":
        result["stable"] = np.maximum(result["stable"], 0.)
        result["full"] = np.maximum(result["full"], 0.)
    else:
        raise ValueError("Unknown predeclared acquisition ablation.")
    return result


def assert_grid(rows, profiles, seeds, allow_partial=False):
    if len(set(seeds)) != len(seeds) or not seeds or len(set(profiles)) != len(profiles):
        raise ValueError("Replication seeds and profiles must be unique and nonempty.")
    keys = [(r["method"], r["profile"], r["seed"]) for r in rows]
    expected = {(m, p, s) for m, p in profiles for s in seeds}
    if len(set(keys)) != len(keys) or not set(keys).issubset(expected):
        raise ValueError("Duplicate or foreign replication case; no seed substitution.")
    if not allow_partial and set(keys) != expected:
        raise ValueError("Incomplete replication grid; no partial-seed mean.")


def assert_ablation_grid(rows, roles, seeds, allow_partial=False):
    keys = [(r["role"], r["seed"]) for r in rows]
    expected = {(role, seed) for role in roles for seed in seeds}
    if len(set(keys)) != len(keys) or not set(keys).issubset(expected):
        raise ValueError("Duplicate or foreign ablation case.")
    if not allow_partial and set(keys) != expected:
        raise ValueError("Incomplete ablation grid; unavailable cases must remain explicit.")


def hierarchical_gain(proposed, competitors, seeds, resamples=10000, seed=5809):
    if not seeds or len(set(seeds)) != len(seeds) or resamples < 2:
        raise ValueError("Require distinct training seeds and at least two resamples.")
    if len(proposed) != len(seeds) or len(competitors) != len(seeds):
        raise ValueError("Require exactly one prediction row per training seed.")
    paired = paired_groups(proposed, competitors, seeds)
    if not paired or any(r["target_tokens"] <= 0 or not np.isfinite(r["proposed_negative_log_probability"])
                         or not np.isfinite(r["competitor_negative_log_probability"]) for r in paired):
        raise ValueError("Require finite losses and positive target counts.")
    rng = np.random.default_rng(seed)
    seed_weights = rng.multinomial(len(seeds), np.full(len(seeds), 1 / len(seeds)), size=resamples)
    sources = sorted({r["source"] for r in paired})
    draws = np.zeros(resamples); point = []
    for source in sources:
        base = sorted((r for r in paired if r["source"] == source and r["seed"] == seeds[0]), key=lambda r: r["group_id"])
        groups = [r["group_id"] for r in base]
        tokens = np.array([r["target_tokens"] for r in base], dtype=np.float64)
        weights = rng.multinomial(len(base), np.full(len(base), 1 / len(base)), size=resamples)
        denominator = weights @ tokens
        for column, training_seed in enumerate(seeds):
            indexed = {r["group_id"]: r for r in paired if r["source"] == source and r["seed"] == training_seed}
            if set(indexed) != set(groups) or any(indexed[g]["target_tokens"] != tokens[i] for i, g in enumerate(groups)):
                raise ValueError("All seeds must share the same independent completion groups.")
            loss_gain = np.array([indexed[g]["competitor_negative_log_probability"] - indexed[g]["proposed_negative_log_probability"] for g in groups])
            point.append(float(loss_gain.sum() / tokens.sum()))
            draws += (weights @ loss_gain) / denominator * seed_weights[:, column] / len(seeds) / len(sources)
    lower, upper = np.quantile(draws, [.025, .975])
    return {"mean_competitor_minus_proposed_macro_nll": statistics.mean(point),
        "percentile_95_low": float(lower), "percentile_95_high": float(upper),
        "resamples": int(resamples), "seed": int(seed), "training_seed_uncertainty_included": True,
        "independent_groups_resampled_within_source": True, "group_draws_shared_across_training_seeds": True,
        "scope": "exploratory_development_after_protocol_selection_not_confirmatory"}


def readout_mechanism(native, affinity, targets, vocabulary, role, seed):
    native = np.asarray(native, dtype=np.float64); affinity = np.asarray(affinity, dtype=np.float64)
    if (native.ndim != 2 or native.shape != affinity.shape or native.shape[1] != len(vocabulary)
            or not np.isfinite(native).all() or (native < 0).any() or native.sum() <= 0
            or not np.isfinite(affinity).all() or (affinity < 0).any()
            or len(targets) != native.shape[0] or any(len(t) != 10 or len(set(t)) != 10
                or any(not isinstance(i,(int,np.integer)) or i < 0 or i >= len(vocabulary) for i in t) for t in targets)):
        raise ValueError("Invalid actual readout in mechanism audit.")
    actual = topics_from_weights(affinity, vocabulary)
    membership = np.mean([len(set(words) & {vocabulary[i] for i in target}) / len(target)
                          for words, target in zip(actual, targets)])
    normalizer = float(native.mean())
    return {"role": role, "seed": seed, "target_top10_membership_recall": float(membership),
        "relative_l1_affinity_movement": float(np.abs(affinity-native).sum()/native.sum()),
        "maximum_topic_total_error_scaled": float(np.max(np.abs(affinity.sum(1)-native.sum(1)))/normalizer),
        "maximum_word_total_error_scaled": float(np.max(np.abs(affinity.sum(0)-native.sum(0)))/normalizer),
        "actual_readout_diversity": len({w for t in actual for w in t}) / (len(actual)*10)}
