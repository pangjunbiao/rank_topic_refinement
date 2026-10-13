"""Fitting-only, native-aware TOP10 acquisition with explicit abstention.

All 45 pairs contribute to each ten-word set's expected score. Missing
associations are retained with their signed score. The binary allocator chooses
one native/edited set per topic, uses disjoint supported seed pairs, and keeps
at least the native number of unique TOP10 words. The readout LP additionally
realizes the native TOP10 of every unedited row, so this diversity accounting
applies to the actual projected readout.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp
from .windows import count_group_evidence
from .associations import signed_npmi


def native_indices(native, topn=10):
    native = np.asarray(native, dtype=np.float64)
    if native.ndim != 2 or native.shape[1] <= topn or not np.isfinite(native).all() or (native < 0).any():
        raise ValueError("Invalid native readout or descriptor depth.")
    indices = []
    for row in native:
        order = np.lexsort((np.arange(len(row)), -row))
        if (row[order[:topn]] <= 0).any():
            raise ValueError("A native topic has insufficient positive descriptor words.")
        indices.append([int(x) for x in order[:topn]])
    return indices


def calibrate(fitting, words, seed, policy):
    """Half-size samples use only independent fitting groups, within source."""
    ids = [r["group_id"] for r in fitting]
    if not ids or len(set(ids)) != len(ids) or any(r["split"] != "fit" for r in fitting):
        raise ValueError("Association calibration requires unique fitting groups only.")
    evidence = count_group_evidence([r["tokens"] for r in fitting], words, policy["window"])
    rng = np.random.default_rng(int(seed))
    samples, selected_ids = [], []
    sources = sorted({r["source"] for r in fitting})
    for _ in range(policy["subsamples"]):
        chosen = []
        for source in sources:
            available = np.asarray([j for j, r in enumerate(fitting) if r["source"] == source], dtype=int)
            size = int(len(available) * policy["source_stratified_fraction"])
            if size < 1:
                raise ValueError("A source has insufficient groups for the declared half-sample.")
            chosen.extend(rng.choice(available, size=size, replace=False).tolist())
        chosen.sort()
        selected_ids.append([ids[j] for j in chosen])
        samples.append(signed_npmi(
            evidence["word_windows"][chosen].sum(axis=0),
            np.asarray(evidence["pair_windows"][chosen].sum(axis=0)).ravel(),
            evidence["pairs"], int(evidence["windows_per_group"][chosen].sum())))
    full = signed_npmi(evidence["word_windows"].sum(axis=0),
                       np.asarray(evidence["pair_windows"].sum(axis=0)).ravel(),
                       evidence["pairs"], int(evidence["windows_per_group"].sum()))
    return {"words": list(words), "pairs": evidence["pairs"],
            "stable": np.asarray(samples).mean(axis=0), "full": full,
            "pair_support": np.asarray(evidence["pair_groups"].sum(axis=0)).ravel(),
            "word_support": np.asarray(evidence["word_groups"].sum(axis=0)).ravel(),
            "sample_group_ids": selected_ids, "fit_groups": len(fitting)}


def allocate(native_sets, candidates, vocabulary_size, seconds=5.0, node_limit=10000,
             *, core_sizes=(2,), required_unique=None):
    """Choose complete descriptors, including an explicit native option.

Word-union binary variables have both upper and lower linking constraints;
unused words cannot falsely satisfy the diversity constraint. The default
pair policy has a feasible all-native allocation. A stricter union requirement
can make the expanded pool infeasible; native abstention then reports that the
requested diversity was not achieved. A time-limited incumbent is independently
checked; absence of an incumbent causes abstention, never an invented edit.
"""
    k = len(native_sets)
    if k < 1 or len({len(x) for x in native_sets}) != 1:
        raise ValueError("Invalid native descriptor sets.")
    for words in native_sets:
        if len(set(words)) != len(words) or any(w < 0 or w >= vocabulary_size for w in words):
            raise ValueError("Native descriptor word index is invalid.")
    if not core_sizes or any(not isinstance(n, int) or n < 1 for n in core_sizes):
        raise ValueError("Invalid supported-core sizes.")
    baseline_unique = len({w for words in native_sets for w in words})
    minimum_unique = baseline_unique if required_unique is None else required_unique
    if (not isinstance(minimum_unique, int) or not 0 <= minimum_unique <= min(vocabulary_size, sum(map(len, native_sets)))):
        raise ValueError("Invalid required descriptor union size.")
    options = [{"topic": t, "target": list(words), "core": [], "expected_gain": 0.0,
                "native_option": True} for t, words in enumerate(native_sets)]
    for item in candidates:
        t, target, core = item["topic"], item["target"], item["core"]
        if (not 0 <= t < k or len(target) != len(native_sets[t]) or len(set(target)) != len(target)
                or len(core) not in core_sizes or len(set(core)) != len(core) or not set(core).issubset(target)
                or any(w < 0 or w >= vocabulary_size for w in target)
                or not np.isfinite(item["expected_gain"]) or item["expected_gain"] <= 0):
            raise ValueError("Invalid acquisition candidate.")
        options.append({**item, "native_option": False})
    if not candidates:
        return [list(x) for x in native_sets], {"status": "no_supported_candidate", "selected": [],
            "candidate_count": 0, "native_unique_words": baseline_unique,
            "target_unique_words": baseline_unique, "allocation_verified": True,
            "optimal_within_pool": True, "required_unique_words": minimum_unique,
            "required_diversity_achieved": baseline_unique >= minimum_unique}
    union = sorted({w for option in options for w in option["target"]})
    union_index = {w: j for j, w in enumerate(union)}
    option_count = len(options)
    rr, cc, vv, lower, upper = [], [], [], [], []
    def add(entries, lo, hi):
        r = len(lower)
        for column, value in entries:
            rr.append(r); cc.append(column); vv.append(value)
        lower.append(lo); upper.append(hi)
    by_topic = [[j for j, option in enumerate(options) if option["topic"] == t] for t in range(k)]
    for indices in by_topic:
        add([(j, 1.0) for j in indices], 1.0, 1.0)
    for word in sorted({w for option in options for w in option["core"]}):
        add([(j, 1.0) for j, option in enumerate(options) if word in option["core"]], 0.0, 1.0)
    for word in union:
        u = option_count + union_index[word]
        included = [j for j, option in enumerate(options) if word in option["target"]]
        # u <= sum of all selected options containing word.
        add([(u, 1.0)] + [(j, -1.0) for j in included], -np.inf, 0.0)
        # For each topic, presence of word in that topic's chosen set implies u=1.
        for indices in by_topic:
            present = [j for j in indices if word in options[j]["target"]]
            if present:
                add([(u, -1.0)] + [(j, 1.0) for j in present], -np.inf, 0.0)
    add([(option_count + j, 1.0) for j in range(len(union))], minimum_unique, np.inf)
    matrix = sparse.coo_matrix((vv, (rr, cc)), shape=(len(lower), option_count + len(union))).tocsc()
    objective = np.r_[-np.asarray([x["expected_gain"] for x in options]), np.zeros(len(union))]
    solved = milp(objective, integrality=np.ones(len(objective)), bounds=Bounds(0.0, 1.0),
                  constraints=LinearConstraint(matrix, lower, upper),
                  options={"time_limit": seconds, "node_limit": node_limit, "mip_rel_gap": 0.0})
    selected = []
    targets = [list(x) for x in native_sets]
    if solved.x is not None:
        values = solved.x
        if np.max(np.abs(values - np.rint(values))) > 1e-6:
            raise ValueError("Allocator returned a noninteger incumbent.")
        for t, indices in enumerate(by_topic):
            chosen = [j for j in indices if values[j] > 0.5]
            if len(chosen) != 1:
                raise ValueError("Allocator did not choose one complete set per topic.")
            item = options[chosen[0]]
            targets[t] = list(item["target"])
            if not item["native_option"]:
                selected.append(item)
        core_words = [w for item in selected for w in item["core"]]
        if len(set(core_words)) != len(core_words):
            raise ValueError("Allocator reused a supported seed word.")
        actual_unique = len({w for words in targets for w in words})
        if actual_unique < minimum_unique:
            raise ValueError("Allocator falsely satisfied the diversity constraint.")
        chosen_union = {word for word in union if values[option_count + union_index[word]] > 0.5}
        if chosen_union != {w for words in targets for w in words}:
            raise ValueError("Allocator word-union certificate failed.")
    return targets, {"status": "allocated" if selected else "abstained", "selected": selected,
        "candidate_count": len(candidates), "solver_status": int(solved.status),
        "solver_message": str(solved.message), "optimal_within_pool": bool(solved.status == 0),
        "incumbent_available": solved.x is not None, "allocation_verified": True,
        "native_unique_words": baseline_unique,
        "target_unique_words": len({w for words in targets for w in words}),
        "required_unique_words": minimum_unique,
        "required_diversity_achieved": len({w for words in targets for w in words}) >= minimum_unique}


def choose_targets(native, fitting, vocabulary, metadata, seed, policy):
    native = np.asarray(native, dtype=np.float64)
    if native.shape != (10, len(vocabulary)):
        raise ValueError("The pilot requires ten native rows on the common fitting vocabulary.")
    if metadata["tokens"] != list(vocabulary):
        raise ValueError("Vocabulary metadata order differs from native columns.")
    top = native_indices(native)
    frequency = dict(zip(vocabulary, metadata["document_frequency"]))
    neighborhood = []
    for row in native:
        order = np.lexsort((np.arange(len(row)), -row))[:policy["neighborhood"]]
        neighborhood.append([int(j) for j in order if frequency[vocabulary[j]] >= policy["word_group_support"]])
    needed = sorted({vocabulary[j] for words in top + neighborhood for j in words})
    bundle = calibrate(fitting, needed, seed, policy)
    index = {word: i for i, word in enumerate(needed)}
    scores = np.zeros((len(needed), len(needed)), dtype=np.float64)
    supports = np.zeros_like(scores, dtype=np.int64)
    pairs = bundle["pairs"]
    scores[pairs[:, 0], pairs[:, 1]] = bundle["stable"]
    supports[pairs[:, 0], pairs[:, 1]] = bundle["pair_support"]
    scores += scores.T; supports += supports.T
    def quality(word_ids):
        ids = [index[vocabulary[j]] for j in word_ids]
        return float(np.mean([scores[i, j] for i, j in combinations(ids, 2)]))
    candidates, per_topic = [], []
    for topic in range(10):
        before = quality(top[topic])
        unique = {}
        for core in combinations(neighborhood[topic], 2):
            i, j = [index[vocabulary[w]] for w in core]
            if supports[i, j] < policy["pair_group_support"] or scores[i, j] < policy["minimum_mean_pair_npmi"]:
                continue
            order = np.lexsort((np.arange(len(vocabulary)), -native[topic]))
            target = list(core) + [int(w) for w in order if w not in core][:8]
            after = quality(target)
            gain = after - before
            if gain <= policy["minimum_expected_top10_gain"]:
                continue
            item = {"topic": topic, "core": list(core), "target": target,
                    "expected_gain": gain, "expected_before": before, "expected_after": after,
                    "pair_mean_npmi": float(scores[i, j]), "pair_group_support": int(supports[i, j]),
                    "word_group_support": [int(bundle["word_support"][x]) for x in [i, j]]}
            key = tuple(sorted(target))
            # Equivalent complete sets count once; the strongest supported pair
            # supplies a deterministic provenance choice.
            previous = unique.get(key)
            if previous is None or (item["pair_mean_npmi"], item["pair_group_support"], tuple(-w for w in core)) > (
                    previous["pair_mean_npmi"], previous["pair_group_support"], tuple(-w for w in previous["core"])):
                unique[key] = item
        candidates.extend(unique[key] for key in sorted(unique))
        per_topic.append({"topic": topic, "eligible_native_neighbors": len(neighborhood[topic]),
                          "native_expected_npmi": before, "distinct_candidates": len(unique)})
    targets, report = allocate(top, candidates, len(vocabulary), policy["allocation_seconds"], policy["allocation_nodes"])
    report.update(per_topic=per_topic, candidates=candidates, native_sets=top,
                  complete_target_sets=targets, only_fitting_groups_used=True)
    return targets, report, bundle
