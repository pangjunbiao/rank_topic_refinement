"""Exact and conservative near-duplicate groups are built before any split."""
from collections import Counter, defaultdict
from itertools import combinations
from ..runtime.artifacts import digest


def group_records(records, near_policy):
    records = sorted(records, key=lambda r: r["record_id"])
    n = len(records)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    exact, edges = {}, []
    for i, record in enumerate(records):
        body = record["comparison_text"]
        if not body:
            continue
        if body in exact:
            j = exact[body]
            union(i, j)
            edges.append({"left": records[j]["record_id"], "right": record["record_id"], "kind": "exact", "jaccard": 1.0})
        else:
            exact[body] = i
    if near_policy["enabled"]:
        width, minimum = int(near_policy["shingle_size"]), int(near_policy["minimum_chars"])
        unique = sorted(exact.values())
        lengths, sizes, posting = {}, {}, defaultdict(list)
        for i in unique:
            text = records[i]["comparison_text"]
            if len(text) < minimum:
                continue
            grams = {text[t:t + width] for t in range(len(text) - width + 1)}
            lengths[i], sizes[i] = len(text), len(grams)
            for gram in sorted(grams):
                posting[gram].append(i)
        intersections = Counter()
        for indices in posting.values():
            for i, j in combinations(indices, 2):
                if min(lengths[i], lengths[j]) / max(lengths[i], lengths[j]) >= near_policy["minimum_length_ratio"]:
                    intersections[i, j] += 1
        for (i, j), common in sorted(intersections.items()):
            score = common / (sizes[i] + sizes[j] - common)
            if score >= near_policy["threshold"]:
                union(i, j)
                edges.append({"left": records[i]["record_id"], "right": records[j]["record_id"], "kind": "near", "jaccard": score})
    components = defaultdict(list)
    for i, record in enumerate(records):
        components[find(i)].append(record)
    groups = []
    for members in components.values():
        representative = min(members, key=lambda r: (-len(r["normalized_body"]), r["source_file"], r["source_id"]))
        member_ids = sorted(r["record_id"] for r in members)
        group_id = "g_" + digest(member_ids)[:24]
        groups.append({"group_id": group_id, "representative": representative,
                       "members": member_ids, "member_sources": sorted({r["source"] for r in members})})
    return sorted(groups, key=lambda g: g["group_id"]), edges
