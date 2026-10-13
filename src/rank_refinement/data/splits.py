"""Deterministic source-balanced assignment of already independent groups."""
from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path
from ..runtime.seeds import keyed_seed


def allocate_counts(n, ratios):
    names = ["fit", "development", "test"]
    quotas = [n * float(ratios[name]) for name in names]
    counts = [math.floor(q) for q in quotas]
    for i in sorted(range(3), key=lambda i: (-(quotas[i] - counts[i]), i))[:n - sum(counts)]:
        counts[i] += 1
    if n >= 3:
        for i in range(3):
            if counts[i] == 0:
                donor = max(range(3), key=lambda j: counts[j])
                if counts[donor] <= 1:
                    raise ValueError("Cannot represent each source in all splits.")
                counts[donor] -= 1
                counts[i] += 1
    return dict(zip(names, counts))


def recover_split(groups, path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"record_id", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("previous_split.csv must have record_id and split columns, e.g. beijing:1,fit.")
        previous = {}
        aliases = {"train": "fit", "dev": "development", "validation": "development"}
        for row in reader:
            record_id = row["record_id"].strip()
            split = aliases.get(row["split"].strip(), row["split"].strip())
            if split not in {"fit", "development", "test"} or record_id in previous:
                raise ValueError("Invalid or duplicate historical split record: " + record_id)
            previous[record_id] = split
    assignment = {}
    for group in groups:
        available = {previous[r] for r in group["members"] if r in previous}
        if len(available) != 1:
            raise ValueError("Historical split missing/conflicting for group " + group["group_id"] + "; do not silently reassign it.")
        assignment[group["group_id"]] = available.pop()
    return assignment


def assign_splits(groups, policy, previous_path=None):
    if previous_path is not None and Path(previous_path).is_file():
        return recover_split(groups, previous_path), "recovered_previous_grouped_split"
    by_source = defaultdict(list)
    for g in groups:
        by_source[g["representative"]["source"]].append(g)
    assignment = {}
    for source, items in sorted(by_source.items()):
        items.sort(key=lambda g: (keyed_seed(policy["split_seed"], source, g["group_id"]), g["group_id"]))
        counts = allocate_counts(len(items), policy["split_ratios"])
        start = 0
        for name in ["fit", "development", "test"]:
            for group in items[start:start + counts[name]]:
                assignment[group["group_id"]] = name
            start += counts[name]
    return assignment, "new_fixed_grouped_split_historical_exposure_retained"


def make_internal_folds(groups, assignment, number, seed):
    by_source = defaultdict(list)
    for group in groups:
        if assignment[group["group_id"]] == "fit":
            by_source[group["representative"]["source"]].append(group)
    folds = [set() for _ in range(number)]
    for source, items in sorted(by_source.items()):
        items.sort(key=lambda g: (keyed_seed(seed, source, g["group_id"]), g["group_id"]))
        for i, group in enumerate(items):
            folds[i % number].add(group["group_id"])
    fit_ids = {g["group_id"] for g in groups if assignment[g["group_id"]] == "fit"}
    return [{"fold": i + 1, "fit_group_ids": sorted(fit_ids - heldout), "validation_group_ids": sorted(heldout)}
            for i, heldout in enumerate(folds)]
