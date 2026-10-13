"""Step 2: fitting-only support first, then the bounded held-out acquisition pilot.

An early scientific failure is a completed diagnostic, not a software error.
It never creates an accepted Step 2 state or starts neural training.
"""
from __future__ import annotations
import json
import time
import warnings
from pathlib import Path
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from ..runtime.artifacts import artifact_manifest, digest, file_hash, read_json, verify_artifacts, write_csv, write_json, write_jsonl
from ..runtime.state import require_preparation
from ..runtime.step02_contract import CORE_PINS, ANCHOR_PINS, dependency_versions, validate_contract
from ..acquisition.windows import count_group_evidence
from ..acquisition.associations import association_bundle, signed_npmi
from ..acquisition.pilot import acquire, settings, choose_setting
from ..evaluation.coherence import score_topics, UnscorableCoherence
from ..evaluation.diversity import topic_diversity
from ..reporting.tables import export_table


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fit_and_validation(folder, outer_fit_ids=None):
    rows = load_jsonl(Path(folder) / "feature_records.jsonl")
    if len({r["group_id"] for r in rows}) != len(rows):
        raise AssertionError("A preparation feature view repeats duplicate-group representatives.")
    if outer_fit_ids is not None and ({r["group_id"] for r in rows} != set(outer_fit_ids) or any(r["split"] not in ["fit", "validation"] for r in rows)):
        raise AssertionError("An internal pilot fold contains an outer held-out group or loses a fitting group.")
    fit = sorted([r for r in rows if r["split"] == "fit"], key=lambda r: r["group_id"])
    validation = sorted([r for r in rows if r["split"] == "validation"], key=lambda r: r["group_id"])
    metadata = read_json(Path(folder) / "vocabulary.json")
    if metadata["fit_group_ids_sha256"] != digest([r["group_id"] for r in fit]):
        raise AssertionError("Pilot vocabulary provenance is not this fold's fitting set.")
    if validation and set(r["group_id"] for r in fit) & set(r["group_id"] for r in validation):
        raise AssertionError("Pilot fitting and validation groups overlap.")
    return fit, validation, metadata


def cached_result(destination, producer):
    destination = Path(destination)
    manifest = destination / "artifact_manifest.json"
    result = destination / "result.json"
    if manifest.is_file() and result.is_file():
        verify_artifacts(destination, read_json(manifest))
        return read_json(result), True
    destination.mkdir(parents=True, exist_ok=True)
    produced = producer(destination)
    write_json(result, produced)
    write_json(manifest, artifact_manifest(destination))
    return produced, False


def summarize(acquisition, fold, seed, method, setting_name, validation, vocabulary, cache):
    result = {"fold": fold, "seed": seed, "method": method, "setting_name": setting_name,
              "structural_passed": acquisition["structural_passed"], "scored": False,
              "positive_degree_words": acquisition["support"]["positive_degree_words"],
              "eligible_word_count": acquisition["support"]["eligible_word_count"],
              "graph_components": acquisition["support"]["components"],
              "failure": acquisition.get("structural_failure", ""), "c_v": None, "c_npmi": None,
              "td": None, "core_reference_token_fraction": None, "reference_absent_word_count": None,
              "minimum_core_signed_cohesion": min((r["signed_cohesion"] for r in acquisition.get("core_reports", [])), default=None)}
    if "topics" not in acquisition:
        return result
    topics = acquisition["topics"]
    texts = [r["tokens"] for r in validation]
    selected = set(w for t in topics for w in t)
    reference = set(w for text in texts for w in text)
    result["td"] = topic_diversity(topics)
    result["core_reference_token_fraction"] = sum(w in selected for text in texts for w in text) / max(1, sum(map(len, texts)))
    result["reference_absent_word_count"] = len(selected - reference)
    result["reference_absent_words"] = sorted(selected - reference)
    # Even failed complete cores are scored for diagnosis. They remain ineligible.
    with warnings.catch_warnings(record=True) as caught, np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("always", RuntimeWarning)
        try:
            scores = score_topics(topics, texts, vocabulary, cache_dir=cache)
            result.update(scored=True, c_v=scores["c_v"]["mean"], c_npmi=scores["c_npmi"]["mean"])
            result["per_topic"] = {k: scores[k]["per_topic"] for k in ["c_v", "c_npmi"]}
        except UnscorableCoherence as exc:
            result["coherence_failure"] = str(exc)
    result["coherence_warnings"] = [str(w.message) for w in caught]
    return result


def evidence_for(fit, metadata, destination, policy):
    words = [w for w, df in zip(metadata["tokens"], metadata["document_frequency"]) if df >= min(policy["word_group_support"])]
    if len(words) < 100:
        raise ValueError("The accepted fold lacks the minimum candidate words; inspect the data gate.")
    evidence = count_group_evidence([r["tokens"] for r in fit], words, policy["window"])
    write_json(destination / "evidence_rows.json", {"group_ids": [r["group_id"] for r in fit], "words": words,
                                                   "window": policy["window"], "counting": "gensim_4.4.0_WordOccurrenceAccumulator",
                                                   "independent_pair_support": "one_true_local_window_event_per_duplicate_group",
                                                   "oov_positions_retained": True})
    np.savez_compressed(destination / "word_window_counts.npz", values=evidence["word_windows"],
                        windows_per_group=evidence["windows_per_group"], pairs=evidence["pairs"])
    for key in ["word_groups", "pair_groups", "pair_windows"]:
        sparse.save_npz(destination / (key + ".npz"), evidence[key])
    return evidence


def original_bundle(evidence, group_ids, bundle):
    binary = evidence["word_groups"].astype(np.int64)
    co = (binary.T @ binary).toarray()
    pairs = evidence["pairs"]
    pair_support = co[pairs[:, 0], pairs[:, 1]]
    frequency = np.asarray(binary.sum(axis=0)).ravel()
    # Original manuscript equation has unsmoothed document NPMI and zero edges
    # for absent/ubiquitous pairs. Keep that documented distinction explicit.
    values = np.full(len(pairs), -1.0)
    p = pair_support / len(group_ids)
    valid = (p > 0) & (p < 1)
    values[valid] = np.log(p[valid] / ((frequency[pairs[valid, 0]] / len(group_ids)) * (frequency[pairs[valid, 1]] / len(group_ids)))) / -np.log(p[valid])
    values[p >= 1] = 0
    return {**bundle, "word_support": frequency, "pair_support": pair_support, "stable": values, "full": values}


def export_results(root, folder, rows, summaries, anchor_rows, diagnostics):
    columns = ["method", "setting_name", "fold", "seed", "structural_passed", "scored", "eligible_word_count", "positive_degree_words",
               "graph_components", "minimum_core_signed_cohesion", "c_v", "c_npmi", "td", "core_reference_token_fraction", "reference_absent_word_count", "failure"]
    simple = [{k: r.get(k) for k in columns} for r in rows]
    export_table(folder, "pilot_runs", simple, columns, "Internal validation only; failed or unscorable observations remain explicit. These are pilot diagnostics, not final primary results.")
    export_table(root / "outputs/paper/supplementary/tables", "S3_step02_acquisition_runs", simple, columns)
    write_json(folder / "selection_diagnostics.json", diagnostics)
    write_json(folder / "anchor_runs.json", anchor_rows)
    write_json(folder / "setting_summary.json", summaries)
    summary_rows = []
    for setting in settings(read_json(folder / "protocol.json")["acquisition"]["pilot"]):
        selected = [r for r in rows if r["method"] == "proposed_stable" and r["setting_name"] == setting["name"]]
        scored = [r for r in selected if r["scored"]]
        summary_rows.append({**setting, "structural_passes": sum(r["structural_passed"] for r in selected),
                             "required_runs": 6, "scored_runs": len(scored),
                             "diagnostic_cv_mean": float(np.mean([r["c_v"] for r in scored])) if len(scored) == 6 else None,
                             "diagnostic_npmi_mean": float(np.mean([r["c_npmi"] for r in scored])) if len(scored) == 6 else None,
                             "minimum_positive_degree_words": min(r["positive_degree_words"] for r in selected),
                             "minimum_core_cohesion": min((r["minimum_core_signed_cohesion"] for r in selected if r["minimum_core_signed_cohesion"] is not None), default=None),
                             "eligible_for_selection": all(r["structural_passed"] for r in selected) and len(scored) == 6})
    export_table(folder, "acquisition_grid", summary_rows, list(summary_rows[0]))
    export_table(root / "outputs/paper/supplementary/tables", "S3_step02_acquisition_grid", summary_rows, list(summary_rows[0]))
    # Source data for a later, distinct sensitivity figure, not a duplicate T2.
    write_csv(root / "outputs/paper/main/figures/source_data/F_R1_acquisition_grid.csv", summary_rows, list(summary_rows[0]))
    write_json(folder / "all_run_records.json", rows)


def anchor_panel(root, configs, folder, prepared, fold_data, hardware, log):
    versions = dependency_versions(ANCHOR_PINS, "requirements-step02-anchors.txt")
    from ..models.encoder_assets import fitting_embeddings
    from ..models.adapters.pilot_anchors import fit_anchor, UnusableAnchor
    a = configs["acquisition_anchors"]
    outer_rows = [r for r in load_jsonl(prepared / "feature_records.jsonl") if r["split"] == "fit"]
    embeddings, embed_index, embedding_manifest = fitting_embeddings(root, outer_rows, a["encoder"], hardware["future_model_device"], log, folder)
    write_json(folder / "anchor_dependency_versions.json", versions)
    write_json(folder / "assets_resolved_before_scoring.json", {"resolved": True, "encoder": a["encoder"], "embedding_key": embedding_manifest["key"]})
    all_rows, panel = [], []
    for method in a["panel"]:
        method_rows = []
        for profile_index, profile in enumerate(a["profiles"][method]):
            candidate = []
            for fold, (fold_folder, fit, validation, metadata) in fold_data.items():
                all_ids = read_json(fold_folder / "matrix_rows.json")
                index = {g: i for i, g in enumerate(all_ids)}
                fit = [r for r in fit if r["model_nonempty"]]
                x = sparse.load_npz(fold_folder / "full_counts.npz")[[index[r["group_id"]] for r in fit]]
                vectors = embeddings[[embed_index[r["group_id"]] for r in fit]]
                for seed in configs["default"]["project"]["pilot_seeds"]:
                    dest = folder / "anchors" / method / profile["name"] / f"fold{fold:02d}_seed{seed}"
                    def produce(destination):
                        started = time.perf_counter()
                        try:
                            topics, weights, details = fit_anchor(method, x, metadata["tokens"], seed, profile, vectors if method == "bertopic" else None)
                        except UnusableAnchor as exc:
                            return {"method": method, "profile": profile["name"], "profile_index": profile_index,
                                    "fold": fold, "seed": seed, "scored": False, "failure": str(exc)}
                        np.savez_compressed(destination / "native_descriptor_weights.npz", weights=weights)
                        write_json(destination / "fit_details.json", details)
                        acq = {"structural_passed": True, "support": {"positive_degree_words": len(metadata["tokens"]),
                                                                      "eligible_word_count": len(metadata["tokens"]), "components": None}, "topics": topics}
                        result = summarize(acq, fold, seed, method, profile["name"], validation, metadata["tokens"], folder / "coherence_cache")
                        result.update(profile=profile["name"], profile_index=profile_index, topics=topics, wall_seconds=time.perf_counter() - started)
                        return result
                    row, reused = cached_result(dest, produce)
                    log.info("ANCHOR | %s/%s | fold=%d seed=%d | scored=%s | C_v=%s NPMI=%s | reused=%s", method, profile["name"], fold, seed, row["scored"], row.get("c_v"), row.get("c_npmi"), reused)
                    candidate.append(row); all_rows.append(row)
            valid = len(candidate) == 6 and all(r["scored"] for r in candidate)
            method_rows.append({"profile": profile["name"], "index": profile_index, "valid": valid, "rows": candidate,
                                "c_v": float(np.mean([r["c_v"] for r in candidate])) if valid else None,
                                "c_npmi": float(np.mean([r["c_npmi"] for r in candidate])) if valid else None})
        usable = [r for r in method_rows if r["valid"]]
        if usable:
            selected = min(usable, key=lambda r: (-r["c_npmi"], -r["c_v"], r["index"]))
            panel.extend(selected["rows"])
        else:
            panel.extend({"method": method, "fold": fold, "seed": seed, "scored": False,
                          "failure": "No declared profile yields six complete finite scores."} for fold in fold_data for seed in [809, 810])
    write_json(folder / "anchor_candidate_runs.json", all_rows)
    return panel


def run(root, configs, versions, hardware, log):
    root = Path(root)
    validate_contract(configs)
    extras = dependency_versions(CORE_PINS, "requirements-step02.txt")
    accepted = require_preparation(root)
    prepared = root / accepted["processed_relative_path"]
    # Do not reprepare or reclean the accepted corpus. Its file hashes are verified.
    files = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted((root / "src").rglob("*.py")) if p.stat().st_size}
    files["main.py"] = file_hash(root / "main.py")
    for name in ["gsdmm_provenance.json", "GSDMM_LICENSE.txt"]:
        path = root / "src/rank_refinement/vendor" / name
        files[path.relative_to(root).as_posix()] = file_hash(path)
    protocol = {"version": "R2_STEP02_GLOBAL_SIGNED_PACKING_20261009", "preparation_id": accepted["preparation_id"],
                "acquisition": configs["acquisition"], "anchors": configs["acquisition_anchors"],
                "evaluation": configs["evaluation"], "seeds": [809, 810], "versions": {**versions, **extras}, "code": files,
                "early_stop": "No structurally complete six-run setting: skip all anchor fits/assets; score complete failed cores only as diagnostics."}
    protocol_id = digest(protocol)
    frozen = root / "state/step02_accepted.json"
    if frozen.is_file():
        previous = read_json(frozen)
        if previous["protocol_id"] != protocol_id:
            raise RuntimeError("Accepted Step 2 is frozen. Preserve it and use a separate project for a changed protocol.")
        verify_artifacts(root / previous["relative_path"], previous["artifacts"])
    folder = root / "outputs/steps/step02_acquisition_pilot" / protocol_id[:20]
    folder.mkdir(parents=True, exist_ok=True)
    write_json(log.folder / "preparation_working_path.json", {"relative_path": folder.relative_to(root).as_posix()})
    write_json(folder / "protocol.json", protocol)
    write_json(log.folder / "protocol.json", protocol)
    write_json(log.folder / "code_manifest.json", {"files": files, "execution_code_sha256": digest(files)})
    write_json(log.folder / "configuration_snapshot.json", configs)
    log.history("input_identity", {"protocol_id": protocol_id, "preparation_id": accepted["preparation_id"]})
    outer_fit = {r["group_id"] for r in load_jsonl(prepared / "feature_records.jsonl") if r["split"] == "fit"}
    policy, grid = configs["acquisition"]["pilot"], settings(configs["acquisition"]["pilot"])
    fold_data, acquisitions, bundles = {}, [], {}
    log.info("PHASE 1/5 | verified Step 1 preparation=%s | support-first pilot; no primary development/test scoring", accepted["preparation_id"][:20])
    with threadpool_limits(limits=1):
        for fold in [1, 2, 3]:
            fold_folder = prepared / "folds" / f"fold{fold:02d}"
            fit, validation, metadata = fit_and_validation(fold_folder, outer_fit)
            fold_data[fold] = fold_folder, fit, validation, metadata
            evidence_dir = folder / "evidence" / f"fold{fold:02d}"
            evidence_dir.mkdir(parents=True, exist_ok=True)
            evidence = evidence_for(fit, metadata, evidence_dir, policy)
            log.info("EVIDENCE | fold=%d | fitting groups=%d | candidate words=%d | virtual windows=%d", fold, len(fit), len(evidence["words"]), int(evidence["windows_per_group"].sum()))
            for seed in [809, 810]:
                bundle = association_bundle(evidence, [r["group_id"] for r in fit], seed, policy)
                bundles[fold, seed] = bundle
                write_json(evidence_dir / f"seed{seed}_subsamples.json", bundle["subsample_group_ids"])
                np.savez_compressed(evidence_dir / f"seed{seed}_signed_associations.npz", stable=bundle["stable"], full=bundle["full"],
                                    pair_support=bundle["pair_support"], word_support=bundle["word_support"], samples=bundle["subsample_scores"])
                for setting in grid:
                    dest = folder / "acquisition" / setting["name"] / f"fold{fold:02d}_seed{seed}"
                    result, reused = cached_result(dest, lambda d: acquire(bundle, setting, seed, policy, evidence_destination=d))
                    result.update(fold=fold, method="proposed_stable", setting_name=setting["name"])
                    acquisitions.append(result)
                    log.info("SUPPORT | fold=%d seed=%d %s | positive-degree words=%d | core gate=%s | min cohesion=%s | reused=%s", fold, seed, setting["name"], result["support"]["positive_degree_words"], result["structural_passed"], min((q["signed_cohesion"] for q in result.get("core_reports", [])), default=None), reused)
                for control_name, control in configs["acquisition"]["controls"].items():
                    original = control_name == "original_document_positive"
                    b = original_bundle(evidence, [r["group_id"] for r in fit], bundle) if original else bundle
                    control_setting = {"index": 0, "name": control_name, "word_min": control["word_min"], "pair_min": control["pair_min"], "gamma": control["gamma"]}
                    result, _ = cached_result(folder / "controls" / control_name / f"fold{fold:02d}_seed{seed}",
                                               lambda d: acquire(b, control_setting, seed, policy,
                                                   stability=control_name == "r1_partition_diagnostic", original=original,
                                                   strategy="spectral_minimum_size_assignment" if control_name == "r1_partition_diagnostic" else None,
                                                   evidence_destination=d))
                    result.update(fold=fold, method=control_name, setting_name=control_name)
                    acquisitions.append(result)
        structural_candidates = [s for s in grid if all(r["structural_passed"] for r in acquisitions if r["method"] == "proposed_stable" and r["setting_name"] == s["name"])]
        log.info("PHASE 2/5 | structural candidates passing all six runs=%d/%d", len(structural_candidates), len(grid))
        panel = []
        if structural_candidates:
            log.info("ASSET PREFLIGHT | all four anchors retained; pinned Chinese embeddings use %s", hardware["future_model_device"])
            panel = anchor_panel(root, configs, folder, prepared, fold_data, hardware, log)
        else:
            log.info("EARLY SCIENTIFIC STOP | no qualifying support setting; skip encoder download and all model fits")
            write_json(folder / "assets_resolved_before_scoring.json", {"resolved": False, "reason": "No setting survives fitting-only structural gate.", "comparative_selection_permitted": False, "diagnostic_failed_core_scoring_only": True})
        log.info("PHASE 3/5 | standard held-out coherence for complete cores; unsupported/absent slots remain visible")
        rows = []
        for acquisition in acquisitions:
            fold, seed = acquisition["fold"], acquisition["seed"]
            _, _, validation, metadata = fold_data[fold]
            dest = folder / "scores" / acquisition["setting_name"] / f"fold{fold:02d}_seed{seed}"
            row, reused = cached_result(dest, lambda d: summarize(acquisition, fold, seed, acquisition["method"], acquisition["setting_name"], validation, metadata["tokens"], folder / "coherence_cache"))
            rows.append(row)
            log.info("COHERENCE | fold=%d seed=%d %s | C_v=%s NPMI=%s TD=%s | support_and_score_pass=%s", fold, seed, acquisition["setting_name"], row["c_v"], row["c_npmi"], row["td"], row["structural_passed"] and row["scored"])
        log.info("PHASE 4/5 | exact six-run selection rule; no anchor omission or failed-run averaging")
        winner, summaries, diagnostics = choose_setting([r for r in rows if r["method"] == "proposed_stable"], panel, grid)
        if not structural_candidates:
            diagnostics = {"status": "NO_STRUCTURALLY_QUALIFYING_SETTING", "anchor_panel": "not_run_after_fitting_only_failure", "comparison_claims": "none", "reason": "All settings fail at least one of six structural checks."}
            summaries = [{**s, "qualified": False,
                          "complete_six_runs": len([r for r in rows if r["method"] == "proposed_stable" and r["setting_name"] == s["name"] and r["scored"]]) == 6,
                          "reason": "Fitting-only structural gate failed."} for s in grid]
        deployed = []
        if winner is not None:
            # The selected recipe must also be realizable on the outer fitting pool.
            fit, _, metadata = fit_and_validation(prepared)
            destination = folder / "selected_outer_fit"
            destination.mkdir(parents=True, exist_ok=True)
            evidence = evidence_for(fit, metadata, destination, policy)
            for seed in [809, 810]:
                bundle = association_bundle(evidence, [r["group_id"] for r in fit], seed, policy)
                chosen = acquire(bundle, winner, seed, policy, evidence_destination=destination / f"seed{seed}_search")
                chosen["common_vocabulary_core_ids"] = [[metadata["word_to_id"][w] for w in t] for t in chosen.get("topics", [])]
                write_json(destination / f"seed{seed}_cores.json", chosen)
                write_json(destination / f"seed{seed}_subsamples.json", bundle["subsample_group_ids"])
                deployed.append(chosen)
        deployment_ok = bool(deployed) and all(r["structural_passed"] for r in deployed)
        passed = winner is not None and deployment_ok
        status = "ACQUISITION_PILOT_PASSED" if passed else "STOP_ACQUISITION_SUPPORT" if not structural_candidates else "STOP_DEPLOYMENT_SUPPORT" if winner is not None else "STOP_ACQUISITION_PERFORMANCE_OR_PANEL"
        result = {"status": status, "engineering_passed": True, "preparation_gate_passed": True,
                  "acquisition_gate_passed": passed, "model_quality_established": False,
                  "preparation_id": accepted["preparation_id"], "protocol_id": protocol_id,
                  "run_id": log.run_id, "relative_path": folder.relative_to(root).as_posix(), "grid_settings": len(grid),
                  "required_fold_seed_runs_per_setting": 6, "proposed_structural_passes": sum(r["structural_passed"] for r in acquisitions if r["method"] == "proposed_stable"),
                  "proposed_attempts": 72, "structurally_qualifying_settings": len(structural_candidates),
                  "complete_core_sets_constructed": sum("topics" in r for r in acquisitions if r["method"] == "proposed_stable"),
                  "anchor_panel_run": bool(structural_candidates), "primary_test_metrics_computed": False,
                  "primary_outer_development_metrics_computed": False, "selected_setting": winner,
                  "selection_diagnostics": diagnostics,
                  "next_stage": "Review acquisition pilot before Step 3" if passed else "Stop before neural training; review structural failure and design a separate bounded fitting/development revision.",
                  "failed_heuristic_does_not_prove_no_feasible_word_sets_exist": True}
        log.info("PHASE 5/5 | incremental source data, failure evidence and inspection exports")
        export_results(root, folder, rows, summaries, panel, diagnostics)
        write_json(folder / "run_result.json", result)
        write_json(log.folder / "run_result.json", result)
        write_json(root / "state/step02_last_result.json", result)
        manifest = artifact_manifest(folder)
        write_json(folder / "artifact_manifest.json", manifest)
        if passed:
            write_json(root / "state/step02_accepted.json", {**result, "artifacts": manifest})
        write_json(root / "outputs/steps/step02_acquisition_pilot/latest.json", result)
        return result, folder
