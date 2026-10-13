"""Step 1: real primary preparation, fold isolation, numerical evaluator oracles."""
from __future__ import annotations
import shutil
from collections import Counter
from pathlib import Path
from ..runtime.artifacts import artifact_manifest, digest, file_hash, read_json, write_csv, write_json, write_jsonl
from ..runtime.config import preparation_signature
from ..runtime.state import accepted_preparation, save_acceptance
from ..data.io import locate_primary, read_source
from ..data.duplicate_groups import group_records
from ..data.tokenization import ChineseTokenizer
from ..data.splits import assign_splits, make_internal_folds
from ..data.vocabulary import fit_features
from ..data.completion_masks import make_completion
from ..data.audit import raw_audit, feature_summary
from ..evaluation.coherence import run_metric_oracles
from ..reporting.tables import dataset_table, export_table


def code_identity(root):
    root = Path(root)
    dependencies = [root / "src/rank_refinement/stages/step01_prepare.py",
                    root / "src/rank_refinement/runtime/config.py",
                    root / "src/rank_refinement/runtime/seeds.py"]
    dependencies += list((root / "src/rank_refinement/data").glob("*.py"))
    dependencies += [root / "src/rank_refinement/evaluation" / name for name in ["coherence.py", "diversity.py", "likelihood.py"]]
    return digest({p.relative_to(root).as_posix(): file_hash(p) for p in sorted(dependencies) if p.stat().st_size})


def publish_preparation_tables(root, destination):
    import json
    root, destination = Path(root), Path(destination)
    def jsonl(name):
        return [json.loads(line) for line in (destination / name).read_text(encoding="utf-8").splitlines()]
    documents = {r["group_id"]: r for r in jsonl("base_records.jsonl")}
    features = {r["group_id"]: r for r in jsonl("feature_records.jsonl")}
    masks = {r["group_id"]: r for r in jsonl("completion_masks.jsonl")}
    metadata = read_json(destination / "vocabulary.json")
    raw = read_json(destination / "raw_audit.json")
    rows = dataset_table(raw, documents, {g: r["split"] for g, r in features.items()},
                         {g: r["tokens"] for g, r in features.items()}, masks, metadata["word_to_id"])
    # The external stage later owns the combined T1; preserve that accepted combined table.
    table_folder = root / "outputs/paper/main/tables"
    if not (root / "state/step07_accepted.json").is_file():
        export_table(table_folder, "T1_data_protocol", rows, list(rows[0]),
                     "Primary rows only. Fit-only vocabulary and one representative per duplicate group. External rows added in Step 7. No model-quality result is implied.")
    summary = read_json(destination / "coverage.json")
    export_table(root / "outputs/paper/supplementary/tables", "S3_step01_coverage", summary, list(summary[0]))
    export_table(root / "outputs/paper/supplementary/tables", "S3_step01_data_audit", raw["sources"], list(raw["sources"][0]))


def persist_features(folder, documents, transformed, metadata, assignments, completion_policy):
    import numpy as np
    from scipy.sparse import csr_matrix, save_npz
    folder = Path(folder)
    vocabulary = metadata["word_to_id"]
    masks = {g: make_completion(g, transformed[g], vocabulary, completion_policy) for g in sorted(documents)}
    ids = sorted(documents)
    feature_rows, observer_rows = [], []
    for g in ids:
        feature_rows.append({"group_id": g, "record_id": documents[g]["record_id"], "source": documents[g]["source"],
                             "split": assignments[g], "clean_text": documents[g]["clean_text"],
                             "tokens": transformed[g], "in_vocabulary_ids": [vocabulary[t] for t in transformed[g] if t in vocabulary],
                             "model_nonempty": any(t in vocabulary for t in transformed[g])})
        observer_rows.append({"group_id": g, "split": assignments[g], "eligible": masks[g]["eligible"],
                              "observed_text": masks[g]["observed_text"]})
    write_jsonl(folder / "feature_records.jsonl", feature_rows)
    write_jsonl(folder / "completion_masks.jsonl", [masks[g] for g in ids])
    write_jsonl(folder / "observed_encoder_view.jsonl", observer_rows)
    write_jsonl(folder / "fit_full_encoder_view.jsonl", [{"group_id": g, "text": documents[g]["clean_text"]} for g in ids if assignments[g] == "fit"])
    write_json(folder / "matrix_rows.json", ids)
    write_json(folder / "model_fit_group_ids.json", [g for g in ids if assignments[g] == "fit" and any(t in vocabulary for t in transformed[g])])
    write_json(folder / "completion_group_ids_by_split.json", {split: [g for g in ids if assignments[g] == split and masks[g]["eligible"]] for split in sorted(set(assignments.values()))})
    for name in ["full", "observed", "target"]:
        values, columns, pointers = [], [], [0]
        for g in ids:
            token_ids = [vocabulary[t] for t in transformed[g] if t in vocabulary] if name == "full" else masks[g][name + "_ids"]
            counts = Counter(token_ids)
            for col in sorted(counts):
                columns.append(col)
                values.append(counts[col])
            pointers.append(len(columns))
        matrix = csr_matrix((np.array(values, dtype=np.int32), np.array(columns, dtype=np.int32), np.array(pointers, dtype=np.int32)),
                            shape=(len(ids), len(vocabulary)))
        save_npz(folder / (name + "_counts.npz"), matrix, compressed=True)
    summary = feature_summary(documents, transformed, metadata, assignments, masks)
    write_json(folder / "coverage.json", summary)
    write_csv(folder / "coverage.csv", summary, list(summary[0]))
    return masks, summary


def verify_saved_matrices(folder):
    import json
    from scipy.sparse import load_npz
    folder = Path(folder)
    rows = read_json(folder / "matrix_rows.json")
    metadata = read_json(folder / "vocabulary.json")
    vocabulary = metadata["word_to_id"]
    records = {r["group_id"]: r for r in (json.loads(line) for line in (folder / "feature_records.jsonl").read_text(encoding="utf-8").splitlines())}
    masks = {r["group_id"]: r for r in (json.loads(line) for line in (folder / "completion_masks.jsonl").read_text(encoding="utf-8").splitlines())}
    matrices = {name: load_npz(folder / (name + "_counts.npz")) for name in ["full", "observed", "target"]}
    for name, matrix in matrices.items():
        if matrix.shape != (len(rows), len(vocabulary)) or (matrix.data < 0).any():
            raise AssertionError("Saved count-matrix shape or sign error: " + name)
    for i, group_id in enumerate(rows):
        expected = Counter(vocabulary[t] for t in records[group_id]["tokens"] if t in vocabulary)
        row = matrices["full"].getrow(i)
        actual = dict(zip((int(x) for x in row.indices), (int(x) for x in row.data)))
        if actual != dict(expected):
            raise AssertionError("Saved full-count matrix differs from token occurrences.")
        for name in ["observed", "target"]:
            row = matrices[name].getrow(i)
            actual = dict(zip((int(x) for x in row.indices), (int(x) for x in row.data)))
            if actual != dict(Counter(masks[group_id][name + "_ids"])):
                raise AssertionError("Saved completion matrix differs from its occurrence mask.")
        if masks[group_id]["eligible"] and (matrices["full"].getrow(i) - matrices["observed"].getrow(i) - matrices["target"].getrow(i)).nnz:
            raise AssertionError("Saved observed/target counts do not sum to full counts.")
    return {"name": "saved_sparse_matrices_match_occurrences", "passed": True, "rows": len(rows)}


def verify_preparation(documents, groups, assignments, folds, transformed, metadata, masks, fold_summaries):
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError("Preparation invariant failed: " + name)

    check("every_group_has_one_split", set(documents) == set(assignments))
    members = [member for g in groups for member in g["members"]]
    check("raw_members_never_repeat_across_groups", len(members) == len(set(members)))
    fit = {g for g in documents if assignments[g] == "fit"}
    check("fitting_vocabulary_provenance", metadata["fit_group_ids_sha256"] == digest(sorted(fit)))
    frequency = Counter()
    for g in fit:
        frequency.update(set(transformed[g]))
    check("vocabulary_df_recomputed_from_fit_only", all(frequency[w] == count for w, count in zip(metadata["tokens"], metadata["document_frequency"])))
    validation = [g for fold in folds for g in fold["validation_group_ids"]]
    check("internal_validation_folds_partition_outer_fit", set(validation) == fit and len(validation) == len(fit))
    for fold, summary in zip(folds, fold_summaries):
        own_fit, own_validation = set(fold["fit_group_ids"]), set(fold["validation_group_ids"])
        check(f"fold{fold['fold']}_no_outer_heldout_groups", own_fit | own_validation == fit and not own_fit & own_validation)
        check(f"fold{fold['fold']}_vocabulary_fit_provenance", summary["fit_ids_sha256"] == digest(sorted(own_fit)))
    for g, mask in masks.items():
        if mask["eligible"]:
            observed, target = set(mask["observed_positions"]), set(mask["target_positions"])
            expected = {i for i, token in enumerate(transformed[g]) if token in metadata["word_to_id"]}
            if observed & target or observed | target != expected or not observed or not target:
                raise AssertionError("Completion occurrence partition failed: " + g)
            encoded = " ".join(transformed[g][i].replace("_", "") for i in mask["observed_positions"])
            if encoded != mask["observed_text"]:
                raise AssertionError("Observed encoder view contains a non-observed token.")
    checks.append({"name": "all_completion_partitions_and_encoder_views", "passed": True})
    return checks


def quality_checks(summary, metadata, fold_summaries, policy):
    limits = policy["quality_gates"]
    checks = [{"name": "at_least_100_fitting_vocabulary_words", "passed": len(metadata["tokens"]) >= limits["minimum_vocabulary"],
               "actual": len(metadata["tokens"]), "required": limits["minimum_vocabulary"]}]
    for row in summary:
        floor = limits["minimum_pooled_coverage"] if row["source"] == "ALL" else limits["minimum_source_coverage"]
        checks.append({"name": row["split"] + "_" + row["source"] + "_coverage", "passed": row["coverage"] >= floor,
                       "actual": row["coverage"], "required": floor})
        if row["split"] in {"development", "test"}:
            floor = limits["minimum_heldout_completion_fraction"]
            checks.append({"name": row["split"] + "_" + row["source"] + "_completion_eligibility", "passed": row["completion_eligible_fraction"] >= floor,
                           "actual": row["completion_eligible_fraction"], "required": floor})
    for fold in fold_summaries:
        checks.append({"name": f"fold{fold['fold']}_vocabulary_size", "passed": fold["vocabulary"] >= limits["minimum_vocabulary"],
                       "actual": fold["vocabulary"], "required": limits["minimum_vocabulary"]})
    return checks


def run(root, configs, versions, hardware, log):
    root = Path(root)
    policy = configs["data"]["primary"]
    log.info("PHASE 1/7 | strict CSV decoding and input provenance")
    paths = locate_primary(root, policy)
    records, inputs = [], []
    for source, path in sorted(paths.items()):
        rows, metadata = read_source(path, source, policy)
        records.extend(rows)
        inputs.append(metadata)
        log.info("INPUT | %s | raw=%d | encoding=%s | sha256=%s", source, len(rows), metadata["encoding"], metadata["sha256"][:12])
    previous_path = root / policy["previous_split_path"]
    previous_hash = file_hash(previous_path) if previous_path.is_file() else None
    tokenizer = ChineseTokenizer(root, policy["tokenizer"])
    preparation_code_hash = code_identity(root)
    signature = preparation_signature(configs, {"csvs": inputs, "previous_split_sha256": previous_hash}, tokenizer.asset_manifest, versions, preparation_code_hash)
    code_manifest = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted((root / "src").rglob("*.py")) if p.stat().st_size}
    code_manifest["main.py"] = file_hash(root / "main.py")
    write_json(log.folder / "code_manifest.json", {"files": code_manifest, "execution_code_sha256": digest(code_manifest), "preparation_code_sha256": preparation_code_hash})
    write_json(log.folder / "configuration_snapshot.json", configs)
    write_json(log.folder / "input_manifest.json", inputs)
    write_json(log.folder / "hardware.json", hardware)
    write_json(log.folder / "versions.json", versions)
    write_json(log.folder / "lexical_assets.json", tokenizer.asset_manifest)
    log.history("input_identity", {"preparation_id": signature, "configuration_hash": digest(configs), "inputs_hash": digest(inputs), "device": hardware["future_model_device"]})
    log.info("PHASE 2/7 | independent standard-metric oracles (synthetic, not model results)")
    oracles = run_metric_oracles(log.folder, configs["evaluation"]["coherence"]["oracle_tolerance"])
    write_json(log.folder / "metric_oracles.json", oracles)
    accepted = accepted_preparation(root, signature)
    if accepted is not None:
        report = read_json(root / accepted["processed_relative_path"] / "preparation_report.json")
        report = {**report, "reused_verified_artifacts": True, "current_run_id": log.run_id}
        write_json(log.folder / "preparation_report.json", report)
        write_json(root / "outputs/steps/step01_prepare/latest.json", {"run_id": log.run_id, "status": report["status"], "preparation_id": signature,
                   "report_relative_path": (log.folder / "preparation_report.json").relative_to(root).as_posix()})
        publish_preparation_tables(root, root / accepted["processed_relative_path"])
        log.info("CACHE | accepted preparation reverified by artifact hashes; no data/model refit")
        return report, root / accepted["processed_relative_path"]
    log.info("PHASE 3/7 | grouping exact/near duplicates and fixed Chinese tokenization")
    groups, edges = group_records(records, policy["near_duplicates"])
    raw = raw_audit(records, groups, edges)
    documents, eligible_groups, exclusions = {}, [], []
    for group in groups:
        representative = group["representative"]
        tokens = tokenizer.tokenize(representative["clean_text"])
        if len(tokens) < policy["tokenizer"]["minimum_lexical_tokens"]:
            exclusions.append({"group_id": group["group_id"], "record_id": representative["record_id"],
                               "reason": "empty_after_common_cleaning_or_lexical_rules", "members": group["members"]})
            continue
        eligible_groups.append(group)
        documents[group["group_id"]] = {"group_id": group["group_id"], "record_id": representative["record_id"],
                                         "source": representative["source"], "source_id": representative["source_id"],
                                         "clean_text": representative["clean_text"], "base_tokens": tokens}
    assignments, split_origin = assign_splits(eligible_groups, policy, previous_path)
    folds = make_internal_folds(eligible_groups, assignments, policy["internal_folds"], policy["fold_seed"])
    destination = root / "data" / "processed" / "primary" / signature[:20]
    if destination.exists():
        # Immutable run-scoped attempt directories preserve failed/interrupted attempts too.
        destination = destination.with_name(signature[:20] + "_" + log.run_id)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(log.folder / "preparation_working_path.json", {"relative_path": destination.relative_to(root).as_posix()})
    tokenizer.save_assets(destination)
    write_json(destination / "raw_audit.json", raw)
    write_json(destination / "input_manifest.json", inputs)
    write_json(destination / "lexical_assets.json", tokenizer.asset_manifest)
    write_json(destination / "configuration_snapshot.json", configs)
    write_json(destination / "excluded_records.json", exclusions)
    write_json(destination / "duplicate_edges.json", edges)
    sample_ids = set()
    from ..runtime.seeds import keyed_seed
    for source in sorted(paths):
        subset = [r for r in records if r["source"] == source]
        sample_ids.update(r["record_id"] for r in sorted(subset, key=lambda r: keyed_seed(policy["split_seed"], "cleaning_audit", r["record_id"]))[:10])
    sample_ids.update(item["record_id"] for item in exclusions)
    write_jsonl(destination / "cleaning_examples.jsonl", [{"record_id": r["record_id"], "raw_body": r["raw_body"], "clean_text": r["clean_text"],
                   "base_tokens": tokenizer.tokenize(r["clean_text"])} for r in records if r["record_id"] in sample_ids])
    write_jsonl(destination / "base_records.jsonl", [documents[g] for g in sorted(documents)])
    write_jsonl(destination / "duplicate_groups.jsonl", [{"group_id": g["group_id"], "representative_id": g["representative"]["record_id"],
                    "members": g["members"], "member_sources": g["member_sources"], "split": assignments.get(g["group_id"]),
                    "eligible": g["group_id"] in documents} for g in groups])
    write_csv(destination / "split_records.csv", [{"record_id": member, "group_id": g["group_id"], "split": assignments[g["group_id"]]} for g in eligible_groups for member in g["members"]],
              ["record_id", "group_id", "split"])
    write_json(destination / "internal_folds.json", folds)
    log.info("GROUPS | raw=%d | independent_before_filter=%d | eligible=%d | duplicate_groups=%d | near_edges=%d | excluded_groups=%d", len(records), len(groups), len(documents), raw["duplicate_groups"], raw["near_edges"], len(exclusions))
    log.info("SPLITS | %s | fit=%d | development=%d | test=%d", split_origin, *(sum(v == name for v in assignments.values()) for name in ["fit", "development", "test"]))
    log.info("PHASE 4/7 | fitting-only phrases, vocabulary and completion views")
    fit_ids = [g for g in sorted(documents) if assignments[g] == "fit"]
    transformed, metadata = fit_features(documents, fit_ids, policy["phrases"], policy["vocabulary"], destination)
    masks, summary = persist_features(destination, documents, transformed, metadata, assignments, policy["completion"])
    log.info("VOCABULARY | V=%d | fitting_groups=%d | learned_phrases=%d", len(metadata["tokens"]), len(fit_ids), metadata["phrase_count"])
    for row in summary:
        if row["source"] == "ALL":
            log.info("COVERAGE | %s | %.2f%% | completion=%d/%d | nonempty_model=%d", row["split"], 100 * row["coverage"], row["completion_eligible_documents"], row["representatives"], row["model_nonempty_documents"])
    log.info("PHASE 5/7 | three independently fitting-derived fold feature bundles")
    fold_documents = {g: documents[g] for g in sorted(fit_ids)}
    fold_summaries = []
    for fold in folds:
        folder = destination / "folds" / ("fold" + str(fold["fold"]).zfill(2))
        roles = {g: "fit" if g in set(fold["fit_group_ids"]) else "validation" for g in fold_documents}
        views, features = fit_features(fold_documents, fold["fit_group_ids"], policy["phrases"], policy["vocabulary"], folder)
        _, coverage = persist_features(folder, fold_documents, views, features, roles, policy["completion"])
        supports = {str(threshold): sum(count >= threshold for count in features["document_frequency"]) for threshold in [10, 20]}
        record = {"fold": fold["fold"], "fit_groups": len(fold["fit_group_ids"]), "validation_groups": len(fold["validation_group_ids"]),
                  "vocabulary": len(features["tokens"]), "phrase_count": features["phrase_count"], "word_group_support_counts": supports,
                  "fit_ids_sha256": features["fit_group_ids_sha256"], "coverage": coverage}
        fold_summaries.append(record)
        log.info("FOLD %d | fit=%d | validation=%d | V=%d | eligible_words_df10=%d | df20=%d", fold["fold"], record["fit_groups"], record["validation_groups"], record["vocabulary"], supports["10"], supports["20"])
    write_json(destination / "fold_summaries.json", fold_summaries)
    log.info("PHASE 6/7 | data firewall, provenance and predeclared coverage gates")
    invariants = verify_preparation(documents, eligible_groups, assignments, folds, transformed, metadata, masks, fold_summaries)
    invariants.append(verify_saved_matrices(destination))
    for fold in folds:
        check = verify_saved_matrices(destination / "folds" / ("fold" + str(fold["fold"]).zfill(2)))
        check["name"] = "fold" + str(fold["fold"]) + "_" + check["name"]
        invariants.append(check)
    gates = quality_checks(summary, metadata, fold_summaries, policy)
    passed = all(c["passed"] for c in gates)
    report = {"preparation_id": signature, "preparation_code_sha256": preparation_code_hash, "run_id": log.run_id, "current_run_id": log.run_id,
              "status": "PREPARATION_PASSED" if passed else "PREPARATION_REVIEW_REQUIRED",
              "engineering_passed": True, "preparation_gate_passed": passed, "model_quality_established": False,
              "raw_records": len(records), "eligible_representatives": len(documents), "vocabulary": len(metadata["tokens"]),
              "split_counts": dict(Counter(assignments.values())), "split_origin": split_origin, "previous_split_sha256": previous_hash,
              "historically_exposed_primary": configs["default"]["project"]["primary_historically_exposed"],
              "coverage": summary, "fold_summaries": fold_summaries, "invariant_checks": invariants, "quality_gates": gates,
              "metric_oracles": oracles, "reused_verified_artifacts": False,
              "processed_relative_path": destination.relative_to(root).as_posix(),
              "next_stage": "Step 2 asset preflight and bounded acquisition pilot, after inspection",
              "remaining_unknowns": ["graph pair support and coherent disjoint cores", "held-out model coherence leadership", "compatible native decoder and full-pipeline NLL", "large model asset revision pins before scored pilots"]}
    write_json(destination / "preparation_report.json", report)
    write_json(log.folder / "preparation_report.json", report)
    write_json(log.folder / "fold_summaries.json", fold_summaries)
    supports = {str(threshold): sum(count >= threshold for count in metadata["document_frequency"]) for threshold in [10, 20]}
    write_json(destination / "lexical_diagnostics.json", {"fitting_word_group_support_counts": supports,
               "single_character_vocabulary_words": sum(len(w) == 1 for w in metadata["tokens"]),
               "protected_tokens_present_in_vocabulary": sorted(set(tokenizer.asset_manifest["protected_tokens"]) & set(metadata["tokens"])),
               "candidate_support_not_graph_feasibility": True})
    write_json(destination / "artifact_manifest.json", artifact_manifest(destination))
    log.info("PHASE 7/7 | table exports, accepted-state manifest and inspection inputs")
    publish_preparation_tables(root, destination)
    for item in gates:
        if not item["passed"]:
            log.info("REVIEW REQUIRED | %s | actual=%s | required=%s", item["name"], item["actual"], item["required"])
    if passed:
        save_acceptance(root, {"preparation_id": signature, "run_id": log.run_id, "engineering_passed": True,
                             "preparation_gate_passed": True, "model_quality_established": False,
                             "previous_split_sha256": previous_hash,
                             "processed_relative_path": destination.relative_to(root).as_posix(), "artifacts": read_json(destination / "artifact_manifest.json")})
        write_json(root / "data/processed/primary/CURRENT.json", {"preparation_id": signature, "relative_path": destination.relative_to(root).as_posix()})
    write_json(root / "outputs/steps/step01_prepare/latest.json", {"run_id": log.run_id, "status": report["status"], "preparation_id": signature,
               "report_relative_path": (log.folder / "preparation_report.json").relative_to(root).as_posix()})
    return report, destination
