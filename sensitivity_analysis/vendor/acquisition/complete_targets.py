"""R3 full TOP10 search, using verified fitting-only signed evidence.

Beam expansion replaces at most four native words. Every one of the 45
relationships contributes to the objective. New words need repeated independent
group support, two positive connections when introduced and at least one
positive supported connection in the final set. Exclusion
alternatives give the global allocator a way to avoid cross-topic collisions.
No development observations, scores, labels or targets enter this module.
"""
from __future__ import annotations
import numpy as np
from .native_targets import allocate, native_indices


def evidence_matrices(bundle, vocabulary, metadata, native, policy):
    words = bundle["words"]
    if (metadata["tokens"] != list(vocabulary) or len(words) != len(set(words))
            or not set(words).issubset(vocabulary)):
        raise ValueError("Association vocabulary/order differs from the readout.")
    pairs = np.asarray(bundle["pairs"])
    size = len(words)
    expected = np.column_stack(np.triu_indices(size, 1))
    if pairs.dtype.kind not in "iu" or not np.array_equal(pairs, expected):
        raise ValueError("Evidence must contain every distinct unordered pair exactly once.")
    matrices = []
    for name in ["stable", "full", "pair_support"]:
        values = np.asarray(bundle[name])
        if values.shape != (len(pairs),) or not np.isfinite(values).all():
            raise ValueError("Invalid signed association evidence: " + name)
        if name != "pair_support" and (np.abs(values) > 1 + 1e-12).any():
            raise ValueError("Signed NPMI is outside its declared range.")
        if name == "pair_support" and (values.dtype.kind not in "iu" or (values < 0).any()):
            raise ValueError("Independent pair support must be nonnegative integer counts.")
        matrix = np.zeros((size, size), dtype=float if name != "pair_support" else np.int64)
        matrix[pairs[:, 0], pairs[:, 1]] = values
        matrices.append(matrix + matrix.T)
    word_support = np.asarray(bundle["word_support"])
    lookup = {word: j for j, word in enumerate(vocabulary)}
    frequency = np.asarray(metadata["document_frequency"])
    if (word_support.shape != (size,) or word_support.dtype.kind not in "iu"
            or not np.array_equal(word_support, frequency[[lookup[w] for w in words]])):
        raise ValueError("Independent word support differs from accepted fitting frequencies.")
    if (matrices[2] > np.minimum(word_support[:, None], word_support[None, :])).any():
        raise ValueError("Pair evidence exceeds its independent word support.")
    index = {word: j for j, word in enumerate(words)}
    mapped = {j: index[w] for j, w in enumerate(vocabulary) if w in index}
    top = native_indices(native)
    neighborhoods = []
    for row in native:
        order = np.lexsort((np.arange(len(vocabulary)), -row))[:policy["neighborhood"]]
        neighborhoods.append([int(w) for w in order if frequency[w] >= policy["word_group_support"]])
    if any(w not in mapped for row in top + neighborhoods for w in row):
        raise ValueError("Saved evidence omits a required native neighborhood word.")
    return top, neighborhoods, mapped, matrices


def complete_quality(word_ids, mapped, matrix):
    indices = [mapped[w] for w in word_ids]
    if len(indices) != 10 or len(set(indices)) != 10:
        raise ValueError("Full-set scoring requires ten distinct words.")
    # The symmetric matrix counts each of the 45 pairs twice; diagonal is zero.
    return float(matrix[np.ix_(indices, indices)].sum() / 90.0)


def choose_complete_targets(native, vocabulary, metadata, bundle, policy):
    native = np.asarray(native, dtype=np.float64)
    if native.shape != (10, len(vocabulary)):
        raise ValueError("R3 requires ten topics on the accepted fitting vocabulary.")
    top, neighbors, mapped, (stable, full, support) = evidence_matrices(bundle, vocabulary, metadata, native, policy)
    native_union = sorted({w for row in top for w in row})
    candidates, details = [], []
    for topic in range(10):
        original = set(top[topic])
        before = complete_quality(top[topic], mapped, stable)
        before_full = complete_quality(top[topic], mapped, full)
        beam = [tuple(sorted(original))]
        visited = set(beam)
        depths = []
        for depth in range(1, policy["maximum_replacements"] + 1):
            pool = {}
            for current in beam:
                current_set = set(current)
                for new in neighbors[topic]:
                    if new in current_set:
                        continue
                    new_index = mapped[new]
                    connected = sum(support[new_index, mapped[w]] >= policy["pair_group_support"]
                                    and stable[new_index, mapped[w]] >= policy["positive_pair_threshold"] for w in current)
                    if connected < policy["minimum_positive_connections"]:
                        continue
                    for removed in current:
                        if removed not in original:
                            continue
                        target = tuple(sorted((current_set - {removed}) | {new}))
                        if target in visited:
                            continue
                        promoted = sorted(set(target) - original)
                        if any(sum(support[mapped[a], mapped[b]] >= policy["pair_group_support"]
                                   and stable[mapped[a], mapped[b]] >= policy["positive_pair_threshold"]
                                   for b in target if b != a) < policy["minimum_final_positive_connections"]
                               for a in promoted):
                            continue
                        gain = complete_quality(target, mapped, stable) - before
                        full_gain = complete_quality(target, mapped, full) - before_full
                        if (gain <= policy["minimum_stable_complete_set_gain"]
                                or full_gain <= policy["minimum_full_complete_set_gain"]):
                            continue
                        pool[target] = {"topic": topic, "core": promoted, "target": list(target),
                            "expected_gain": gain, "expected_before": before, "expected_after": before + gain,
                            "full_gain": full_gain, "replacements": depth}
            ordered = sorted(pool, key=lambda t: (-pool[t]["expected_gain"], t))[:policy["beam_width"]]
            keep = set(ordered)
            for word in native_union:
                alternatives = [t for t in pool if word not in t]
                if alternatives:
                    keep.add(min(alternatives, key=lambda t: (-pool[t]["expected_gain"], t)))
            candidates.extend(pool[t] for t in sorted(keep))
            depths.append({"depth": depth, "eligible_sets": len(pool), "retained_sets": len(keep)})
            visited.update(pool)
            beam = ordered
            if not beam:
                break
        details.append({"topic": topic, "native_stable_npmi": before, "native_full_npmi": before_full,
                        "eligible_neighbors": len(neighbors[topic]), "depths": depths})
    targets, report = allocate(top, candidates, len(vocabulary), policy["allocation_seconds"], policy["allocation_nodes"],
        core_sizes=tuple(range(1, policy["maximum_replacements"] + 1)), required_unique=policy["required_unique_words"])
    for selected in report["selected"]:
        if len(set(selected["target"]) & set(top[selected["topic"]])) < policy["retained_native_words_minimum"]:
            raise AssertionError("Complete-set search exceeded the native-retention budget.")
    report.update(candidates=candidates, per_topic=details, native_sets=top, complete_target_sets=targets,
                  only_fitting_groups_used=True, development_inputs_used=False,
                  fully_disjoint_targets=len({w for row in targets for w in row}) == 100,
                  search_is_bounded_not_global_optimum=True)
    return targets, report
