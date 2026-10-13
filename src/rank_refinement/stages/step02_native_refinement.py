"""Runnable Step 2R3: a bounded, exploratory native-readout mechanism pilot.

Step 1 is required and verified; its data, vocabulary and masks are not edited.
Unigrams are a separate fitting-only candidate view, not a new canonical task.
This stage measures real projected TOP10 descriptors on LDA/NMF readouts. It
does not train EnCOT/NeuroMax, evaluate NLL, or create final paper benchmark rows.
"""
from __future__ import annotations
import io
import json
import platform
import statistics
import warnings
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from ..acquisition.native_targets import choose_targets
from ..projection.minimum_tv import minimum_tv, certify
from ..models.adapters.pilot_anchors import fit_anchor, topics_from_weights
from ..data.vocabulary import fit_features
from ..evaluation.coherence import score_topics, UnscorableCoherence, run_metric_oracles
from ..evaluation.diversity import topic_diversity
from ..runtime.artifacts import (artifact_manifest, atomic_bytes, digest, file_hash, read_json,
    verify_artifacts, write_csv, write_json, write_jsonl, write_text)
from ..runtime.state import require_preparation
from ..runtime.step02_contract import CORE_PINS, dependency_versions
from .step02_acquisition_pilot import fit_and_validation, load_jsonl


def validate_protocol(configs):
    p = configs.get("step02_refinement", {})
    required = {"protocol": "R3_NATIVE_TOP10_DIVERSITY_20261009",
                "scope": "exploratory_readout_mechanism_pilot",
                "representations": ["accepted_phrases", "unigrams"], "bases": ["lda", "nmf"],
                "folds": [1, 2, 3], "seeds": [809, 810], "topics": 10, "topn": 10,
                "prediction_evaluated": False, "external_data_used": False}
    for key, value in required.items():
        if p.get(key) != value:
            raise ValueError("Step 2R3 protocol mismatch: " + key + ". Declare a new protocol before changing the pilot.")
    expected = {
        "profiles": {"lda": {"alpha": "auto", "eta": "auto", "passes": 20, "iterations": 100},
            "nmf": {"input": "tfidf", "init": "nndsvda", "solver": "cd", "beta_loss": "frobenius", "max_iter": 400, "tol": 0.0001}},
        "acquisition": {"window": 10, "neighborhood": 50, "word_group_support": 20, "pair_group_support": 5,
            "subsamples": 20, "source_stratified_fraction": 0.5, "minimum_mean_pair_npmi": 0.1,
            "minimum_expected_top10_gain": 0.02, "pair_words_disjoint": True,
            "preserve_global_top10_diversity": True, "allocation_seconds": 5.0, "allocation_nodes": 10000},
        "projection": {"rank_margin_scaled": 1e-6, "seconds": 30.0, "certificate_tolerance_scaled": 1e-7,
            "primary_objective_tolerance_scaled": 1e-5, "max_relative_l1_movement": 0.05,
            "numerical_retry": "same_lp_ipm_only_for_tiny_negative_affinity"},
        "acceptance": {"fitting_cv_gain": 0.0, "fitting_npmi_gain": 0.0, "allow_diversity_loss": False,
            "validation_used_for_case_acceptance": False, "complete_reference_required_for_benchmark": True,
            "all_cases_retained": True, "minimum_matched_finite_cases_for_promising_profile": 4}}
    for key, value in expected.items():
        if p.get(key) != value:
            raise ValueError("The frozen R3 " + key + " budget/policy changed; do not retune it after viewing validation.")
    return p


def metric_score(topics, records, vocabulary, cache):
    texts = [r["tokens"] for r in records]
    absent = sorted({w for topic in topics for w in topic} - {w for text in texts for w in text})
    with warnings.catch_warnings(record=True) as caught, np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("always", RuntimeWarning)
        try:
            values = score_topics(topics, texts, vocabulary, topn=10, cv_window=110,
                                  npmi_window=10, cache_dir=cache)
            result = {"scored": True, "c_v": values["c_v"]["mean"], "c_npmi": values["c_npmi"]["mean"],
                "per_topic": {m: values[m]["per_topic"] for m in ["c_v", "c_npmi"]}, "failure": None}
        except UnscorableCoherence as exc:
            result = {"scored": False, "c_v": None, "c_npmi": None, "per_topic": None, "failure": str(exc)}
    result.update(absent_words=absent, warnings=[str(w.message) for w in caught],
                  reference_groups=len(records), all_ten_topics_and_words_retained=True)
    return result


def fitting_acceptance(native_score, candidate_score, native_td, candidate_td, certificate, protocol):
    reasons = []
    if not certificate.get("success") or not certificate.get("certified"):
        reasons.append("projection_not_numerically_certified")
    movement = certificate.get("relative_l1_movement")
    if movement is not None and movement > protocol["projection"]["max_relative_l1_movement"]:
        reasons.append("movement_exceeds_five_percent")
    if candidate_score is None:
        reasons.append("no_scored_projection_candidate")
    elif not native_score["scored"] or not candidate_score["scored"]:
        reasons.append("fitting_coherence_unscorable")
    else:
        for metric, threshold in [("c_v", "fitting_cv_gain"), ("c_npmi", "fitting_npmi_gain")]:
            if candidate_score[metric] - native_score[metric] <= protocol["acceptance"][threshold]:
                reasons.append("fitting_" + metric + "_does_not_improve")
    if candidate_td is not None and candidate_td + 1e-12 < native_td:
        reasons.append("actual_top10_diversity_decreases")
    return not reasons, reasons


def features(preparation, outer_fit_ids, base_records, variant, fold, destination, vocabulary_policy):
    fit, validation, metadata = fit_and_validation(preparation / f"folds/fold{fold:02d}", outer_fit_ids)
    rows = fit + validation
    if variant == "unigrams":
        # Only outer-fitting groups are presented to this candidate feature view.
        fitting_ids = [r["group_id"] for r in fit]
        transformed, metadata = fit_features(base_records, fitting_ids, {"enabled": False},
                                             vocabulary_policy, destination)
        rows = [{**r, "tokens": transformed[r["group_id"]]} for r in rows]
        fit = [r for r in rows if r["split"] == "fit"]
        validation = [r for r in rows if r["split"] == "validation"]
    elif variant == "accepted_phrases":
        write_json(destination / "vocabulary.json", metadata)
    else:
        raise ValueError("Undeclared representation.")
    if metadata["fit_group_ids_sha256"] != digest(sorted(r["group_id"] for r in fit)):
        raise AssertionError("Candidate vocabulary was not fitted on this fitting fold.")
    wmap = metadata["word_to_id"]
    rr, cc, vv = [], [], []
    for i, row in enumerate(fit):
        for j, count in Counter(wmap[w] for w in row["tokens"] if w in wmap).items():
            rr.append(i); cc.append(j); vv.append(count)
    counts = sparse.csr_matrix((vv, (rr, cc)), shape=(len(fit), len(wmap)), dtype=np.float64)
    nonempty = counts.sum(axis=1).A1 > 0
    diagnostic = {"representation": variant, "fold": fold, "fit_groups": len(fit),
                  "validation_groups": len(validation), "vocabulary": len(wmap),
                  "fit_empty_count": int((~nonempty).sum()),
                  "anchor_fitting_group_ids": [r["group_id"] for r, keep in zip(fit, nonempty) if keep],
                  "fit_group_ids_sha256": metadata["fit_group_ids_sha256"],
                  "outer_development_test_excluded": True, "canonical_step01_edited": False}
    for name, records in [("fitting", fit), ("validation", validation)]:
        total = sum(len(r["tokens"]) for r in records)
        diagnostic[name + "_token_coverage"] = sum(w in wmap for r in records for w in r["tokens"]) / max(1, total)
    write_json(destination / "feature_diagnostic.json", diagnostic)
    write_jsonl(destination / "feature_records.jsonl", fit + validation)
    return fit, validation, metadata, counts[nonempty], diagnostic


def save_arrays(path, **arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic_bytes(path, stream.getvalue())


def run_case(fit, validation, metadata, counts, variant, fold, seed, base, protocol, folder, cache, log):
    manifest = folder / "artifact_manifest.json"
    if manifest.is_file():
        records = read_json(manifest)
        if {x["path"] for x in records} != {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"}:
            raise ValueError("Cached case contains unindexed or missing artifacts: " + str(folder))
        verify_artifacts(folder, records)
        result = read_json(folder / "result.json")
        with np.load(folder / "readout.npz", allow_pickle=False) as saved:
            native_saved, final_saved = saved["native"], saved["final"]
        if result["fitting_accepted"]:
            independent = certify(native_saved, final_saved, result["targets"],
                protocol["projection"]["rank_margin_scaled"], protocol["projection"]["certificate_tolerance_scaled"],
                result["certificate"]["solver_tv_scaled"], protocol["projection"]["primary_objective_tolerance_scaled"])
            if not independent["certified"]:
                raise ValueError("Cached projection no longer verifies.")
        elif not np.array_equal(native_saved, final_saved):
            raise ValueError("A rejected case changed its native readout.")
        log.info("CACHE | %s %s fold=%d seed=%d | verified complete case reused", variant, base, fold, seed)
        return result
    folder.mkdir(parents=True, exist_ok=True)
    vocabulary = metadata["tokens"]
    native_topics, native, details = fit_anchor(base, counts, vocabulary, seed, protocol["profiles"][base])
    native = np.asarray(native, dtype=np.float64)
    native_fit = metric_score(native_topics, fit, vocabulary, cache)
    native_td = topic_diversity(native_topics)
    targets, selection, bundle = choose_targets(native, fit, vocabulary, metadata, seed, protocol["acquisition"])
    write_json(folder / "selection.json", selection)
    write_json(folder / "sampling_group_ids.json", bundle.pop("sample_group_ids"))
    save_arrays(folder / "association_evidence.npz", **{key: bundle[key] for key in
        ["pairs", "stable", "full", "pair_support", "word_support"]})
    write_json(folder / "association_words.json", {"words": bundle["words"], "fit_groups": bundle["fit_groups"]})
    write_json(folder / "native_fit_details.json", details)
    candidate, candidate_topics, candidate_fit, candidate_td = None, None, None, None
    certificate = {"success": True, "certified": False, "status": "no_selected_candidate", "prediction_evaluated": False}
    if selection["selected"]:
        p = protocol["projection"]
        candidate, certificate = minimum_tv(native, targets, p["rank_margin_scaled"], p["seconds"],
                                             p["certificate_tolerance_scaled"], p["primary_objective_tolerance_scaled"])
        if candidate is not None:
            candidate_topics = topics_from_weights(candidate, vocabulary)
            # All rows have complete rank targets, including abstaining rows.
            if any(set(topic) != {vocabulary[j] for j in target} for topic, target in zip(candidate_topics, targets)):
                raise AssertionError("The scored readout is not the allocated complete TOP10 target.")
            candidate_fit = metric_score(candidate_topics, fit, vocabulary, cache)
            candidate_td = topic_diversity(candidate_topics)
    accepted, reasons = fitting_acceptance(native_fit, candidate_fit, native_td, candidate_td, certificate, protocol)
    final = candidate if accepted else native
    final_topics = candidate_topics if accepted else native_topics
    final_td = topic_diversity(final_topics)
    # Validation is accessed for metrics only after the complete fitting decision.
    native_validation = metric_score(native_topics, validation, vocabulary, cache)
    candidate_validation = metric_score(candidate_topics, validation, vocabulary, cache) if candidate_topics else None
    final_validation = candidate_validation if accepted else native_validation
    changed_topics = sum(set(a) != set(b) for a, b in zip(native_topics, final_topics))
    result = {"representation": variant, "base": base, "fold": fold, "seed": seed,
              "scope": protocol["scope"], "native_fitting": native_fit, "candidate_fitting": candidate_fit,
              "native_validation": native_validation, "candidate_validation": candidate_validation,
              "final_validation": final_validation, "native_topics": native_topics, "candidate_topics": candidate_topics,
              "final_topics": final_topics, "targets": targets, "certificate": certificate,
              "allocation_verified": selection["allocation_verified"], "selected_topics": len(selection["selected"]),
              "changed_topics": changed_topics, "fitting_accepted": accepted, "rejection_reasons": reasons,
              "native_td": native_td, "candidate_td": candidate_td, "final_td": final_td,
              "prediction_evaluated": False, "validation_used_for_acceptance": False,
              "display_only_same_final_words_has_identical_word_set_metrics": True,
              "relative_l1_final": float(np.abs(final - native).sum() / native.sum())}
    arrays = {"native": native, "final": final}
    if candidate is not None:
        arrays["candidate"] = candidate
    save_arrays(folder / "readout.npz", **arrays)
    write_json(folder / "result.json", result)
    write_json(manifest, artifact_manifest(folder))
    return result


def summarize(results, protocol):
    profiles = []
    for variant in protocol["representations"]:
        for base in protocol["bases"]:
            cases = [r for r in results if r["representation"] == variant and r["base"] == base]
            paired = [r for r in cases if r["native_validation"]["scored"] and r["final_validation"]["scored"]]
            cv_gains = [r["final_validation"]["c_v"] - r["native_validation"]["c_v"] for r in paired]
            npmi_gains = [r["final_validation"]["c_npmi"] - r["native_validation"]["c_npmi"] for r in paired]
            avg = lambda values: statistics.mean(values) if values else None
            promising = bool(len(paired) >= protocol["acceptance"]["minimum_matched_finite_cases_for_promising_profile"]
                             and avg(cv_gains) > 0 and avg(npmi_gains) > 0
                             and all(r["final_td"] + 1e-12 >= r["native_td"] for r in cases))
            profiles.append({"representation": variant, "base": base, "expected_cases": 6,
                "completed_cases": len(cases), "matched_finite_cases": len(paired), "unscorable_cases": len(cases) - len(paired),
                "fitting_accepted": sum(r["fitting_accepted"] for r in cases),
                "both_coherence_wins": sum(a > 0 and b > 0 for a, b in zip(cv_gains, npmi_gains)),
                "native_cv_finite_diagnostic": avg([r["native_validation"]["c_v"] for r in paired]),
                "refined_cv_finite_diagnostic": avg([r["final_validation"]["c_v"] for r in paired]),
                "native_npmi_finite_diagnostic": avg([r["native_validation"]["c_npmi"] for r in paired]),
                "refined_npmi_finite_diagnostic": avg([r["final_validation"]["c_npmi"] for r in paired]),
                "mean_cv_change_finite_diagnostic": avg(cv_gains), "mean_npmi_change_finite_diagnostic": avg(npmi_gains),
                "minimum_td_change_all_cases": min(r["final_td"] - r["native_td"] for r in cases),
                "maximum_relative_l1_final": max(r["relative_l1_final"] for r in cases),
                "mechanism_promising": promising,
                "full_six_case_coherence_summary_available": len(paired) == 6})
    all_complete = all(p["completed_cases"] == p["expected_cases"] for p in profiles)
    finite = sum(p["matched_finite_cases"] for p in profiles)
    promising = any(p["mechanism_promising"] for p in profiles)
    complete_reference = finite == len(results)
    return {"status": "STEP02_COMPLETED_PROMISING" if promising else "STEP02_COMPLETED_REVIEW_NEEDED",
            "engineering_passed": all_complete, "preparation_gate_passed": True,
            "acquisition_gate_passed": bool(promising and complete_reference),
            "mechanism_promising": promising, "complete_coherence_reference_gate_passed": complete_reference,
            "full_experiment_gate_passed": False, "model_quality_established": False,
            "expected_cases": 24, "completed_cases": len(results), "matched_finite_cases": finite,
            "unscorable_cases_retained": len(results) - finite,
            "fitting_accepted_cases": sum(r["fitting_accepted"] for r in results),
            "diversity_losses_in_deployed_readouts": sum(r["final_td"] + 1e-12 < r["native_td"] for r in results),
            "prediction_evaluated": False, "primary_development_test_metrics_evaluated": False,
            "external_data_used": False, "canonical_step01_edited": False,
            "profiles": profiles, "selected_final_base": None,
            "next_action": "review_step02_then_bounded_encot_neuromax_predictive_pilot" if promising else "review_saved_rejections_before_further_training",
            "summary_scope": "Finite matched diagnostic cases only; all NA cases remain in paired_results.csv. Folds and seeds are dependent, not 24 independent experiments."}


def export_results(folder, results, summary):
    paired, changes, certificates = [], [], []
    for r in results:
        n, f = r["native_validation"], r["final_validation"]
        finite = n["scored"] and f["scored"]
        key = {name: r[name] for name in ["representation", "base", "fold", "seed"]}
        paired.append({**key, "fitting_accepted": r["fitting_accepted"], "selected_topics": r["selected_topics"],
            "changed_topics": r["changed_topics"], "native_scored": n["scored"], "refined_scored": f["scored"],
            "native_cv": n["c_v"], "refined_cv": f["c_v"], "native_npmi": n["c_npmi"], "refined_npmi": f["c_npmi"],
            "cv_change": f["c_v"] - n["c_v"] if finite else None,
            "npmi_change": f["c_npmi"] - n["c_npmi"] if finite else None,
            "native_td": r["native_td"], "refined_td": r["final_td"], "td_change": r["final_td"] - r["native_td"],
            "relative_l1_final": r["relative_l1_final"], "native_absent_words": " | ".join(n["absent_words"]),
            "refined_absent_words": " | ".join(f["absent_words"]), "rejection_reasons": " | ".join(r["rejection_reasons"]),
            "native_coherence_failure": n["failure"], "refined_coherence_failure": f["failure"]})
        c = r["certificate"]
        certificates.append({**key, "fitting_accepted": r["fitting_accepted"], "candidate_certified": c.get("certified", False),
            "row_sum_error_scaled": c.get("row_sum_max_abs_error_scaled"),
            "column_sum_error_scaled": c.get("column_sum_max_abs_error_scaled"),
            "minimum_rank_gap_scaled": c.get("rank_gap_min_scaled"), "tv_scaled": c.get("tv_scaled"),
            "objective_error_scaled": c.get("objective_error_scaled"), "lp_seconds": c.get("seconds"),
            "actual_top10_targets_realized": c.get("actual_top_sets_realized", False),
            "relative_l1_final": r["relative_l1_final"], "post_solve_clipping": c.get("post_solve_clipping", False)})
        for topic, (native, final) in enumerate(zip(r["native_topics"], r["final_topics"])):
            changes.append({**key, "topic": topic, "changed": set(native) != set(final),
                "native_top10": " | ".join(native), "refined_top10": " | ".join(final),
                "added_words": " | ".join(sorted(set(final) - set(native))),
                "removed_words": " | ".join(sorted(set(native) - set(final)))})
    for name, records in [("paired_results", paired), ("topic_changes", changes), ("projection_certificates", certificates),
                           ("comparison", summary["profiles"])]:
        write_csv(folder / (name + ".csv"), records, list(records[0]))
    def fmt(x):
        return "NA" if x is None else f"{x:.6f}"
    text = ["# Step 2R3 result", "", "This is an exploratory LDA/NMF readout-mechanism pilot. It is not the final neural model or a completed competitor benchmark.", "",
        f"Completed: {summary['completed_cases']}/24 cases. Matched finite coherence: {summary['matched_finite_cases']}/24. Fitting-accepted edits: {summary['fitting_accepted_cases']}/24.",
        f"Deployed diversity losses: {summary['diversity_losses_in_deployed_readouts']}. Mechanism promising: {summary['mechanism_promising']}.", "",
        "| Representation | Base | Finite / planned | Native NPMI | Refined NPMI | Change C_v | Change NPMI |", "|---|---|---:|---:|---:|---:|---:|"]
    for p in summary["profiles"]:
        text.append(f"| {p['representation']} | {p['base']} | {p['matched_finite_cases']}/6 | {fmt(p['native_npmi_finite_diagnostic'])} | {fmt(p['refined_npmi_finite_diagnostic'])} | {fmt(p['mean_cv_change_finite_diagnostic'])} | {fmt(p['mean_npmi_change_finite_diagnostic'])} |")
    text += ["", "The displayed means use the same finite cases for native and refined. They are diagnostic summaries, not complete six-run benchmark means. All cases and missing-reference failures remain saved.", "",
        "Negative NPMI remains negative. No topic or word was removed to change its sign. The published Gensim TOP10 definitions and windows are unchanged.", "",
        "Acceptance uses fitting metrics, numerical conservation, actual target realization, and no TD loss. Validation never decides whether a case deploys the edit. Rejected edits retain the exact native matrix.", "",
        "Projection and same-word display-only must tie descriptor coherence. Only projection realizes the rank sets in the affinity. This stage has no NLL claim.", "",
        "The unigram view does not overwrite the accepted preparation. EnCOT/NeuroMax interfaces, full prediction/correction/backoff, NLL, competitor selection, ablations, sensitivity and external evaluation require later stages.", "",
        "Next action: " + summary["next_action"], "", "Send the automatically printed inspection ZIP for review before a full experiment."]
    write_text(folder / "SUMMARY.md", "\n".join(text) + "\n")


def run(root, configs, versions, hardware, log):
    root = Path(root).resolve()
    protocol = validate_protocol(configs)
    versions.update(dependency_versions(CORE_PINS, "requirements-step02.txt"))
    state = require_preparation(root)
    preparation = (root / state["processed_relative_path"]).resolve()
    if not preparation.is_relative_to(root):
        raise ValueError("Accepted preparation path escapes the project.")
    outer = load_jsonl(preparation / "feature_records.jsonl")
    outer_fit_ids = {r["group_id"] for r in outer if r["split"] == "fit"}
    base_records = {r["group_id"]: r for r in load_jsonl(preparation / "base_records.jsonl") if r["group_id"] in outer_fit_ids}
    if set(base_records) != outer_fit_ids:
        raise ValueError("Accepted fitting representatives lack lexical base records.")
    step01_before = file_hash(root / "state/step01_accepted.json")
    source_dir = Path(__file__).resolve().parents[1]
    source_names = ["stages/step02_native_refinement.py", "acquisition/native_targets.py", "projection/minimum_tv.py",
        "models/adapters/pilot_anchors.py", "evaluation/coherence.py", "acquisition/windows.py", "acquisition/associations.py",
        "data/vocabulary.py", "stages/step02_acquisition_pilot.py", "runtime/artifacts.py"]
    fingerprint = digest({"preparation": state["preparation_id"], "protocol": protocol, "versions": versions,
                          "source_hashes": {name: file_hash(source_dir / name) for name in source_names},
                          "platform": platform.system(), "python": platform.python_version()})
    folder = root / "outputs/steps/step02_native_refinement" / fingerprint[:20]
    folder.mkdir(parents=True, exist_ok=True)
    write_json(log.folder / "preparation_working_path.json", {"relative_path": folder.relative_to(root).as_posix()})
    write_json(folder / "protocol_snapshot.json", protocol)
    write_json(folder / "input_provenance.json", {"preparation_id": state["preparation_id"], "signature": fingerprint,
        "preparation_artifacts_verified": len(state["artifacts"]), "outer_fitting_groups": len(outer_fit_ids),
        "outer_fit_ids_sha256": digest(sorted(outer_fit_ids)), "accepted_step01_state_sha256": step01_before,
        "primary_inputs": read_json(preparation / "input_manifest.json"),
        "canonical_step01_edited": False, "primary_development_test_scored": False,
        "external_data_used": False, "device_for_this_step": "cpu", "future_neural_device": hardware["future_model_device"]})
    log.info("STEP 2R3 | native-aware complete TOP10 | 2 representations x 2 bases x 3 folds x 2 seeds = 24 cases")
    log.info("PREPARATION | %d artifacts verified | outer fitting groups=%d | fitting-only internal folds", len(state["artifacts"]), len(outer_fit_ids))
    log.info("CPU WORK | Gensim LDA, sklearn NMF, sparse HiGHS LP/MILP | neural GPU work belongs to the predictive pilot")
    with threadpool_limits(limits=1):
        write_json(folder / "metric_oracles.json", run_metric_oracles(folder / "metric_oracles"))
        results, feature_reports = [], []
        for variant in protocol["representations"]:
            for fold in protocol["folds"]:
                dest = folder / "features" / variant / f"fold{fold:02d}"
                fit, val, metadata, counts, diagnostic = features(preparation, outer_fit_ids, base_records,
                    variant, fold, dest, configs["data"]["primary"]["vocabulary"])
                feature_reports.append(diagnostic)
                log.info("FEATURES | %s fold=%d | fit=%d validation=%d V=%d fitting_empty=%d", variant, fold, len(fit), len(val), len(metadata["tokens"]), diagnostic["fit_empty_count"])
                for seed in protocol["seeds"]:
                    for base in protocol["bases"]:
                        case = folder / "cases" / variant / base / f"fold{fold:02d}_seed{seed}"
                        log.info("CASE %d/24 | %s %s fold=%d seed=%d | fit, acquire, allocate, project, verify, then validate", len(results) + 1, variant, base, fold, seed)
                        result = run_case(fit, val, metadata, counts, variant, fold, seed, base, protocol,
                                          case, folder / "coherence_cache", log)
                        results.append(result)
                        n, f = result["native_validation"], result["final_validation"]
                        display = lambda value: "NA" if value is None else f"{value:.6f}"
                        log.info("RESULT | accepted=%s targets=%d changed=%d | validation Cv=%s -> %s NPMI=%s -> %s TD=%.3f -> %.3f | movement=%.3f%%",
                            result["fitting_accepted"], result["selected_topics"], result["changed_topics"], display(n["c_v"]), display(f["c_v"]),
                            display(n["c_npmi"]), display(f["c_npmi"]), result["native_td"], result["final_td"], 100 * result["relative_l1_final"])
                        if result["rejection_reasons"]:
                            log.info("RETAIN NATIVE | %s", "; ".join(result["rejection_reasons"]))
                        write_json(folder / "case_results.json", results)
                        interim = summarize(results, protocol) if len(results) == 24 else {"status": "RUNNING", "completed_cases": len(results), "expected_cases": 24}
                        write_json(folder / "progress.json", interim)
    verify_artifacts(preparation, state["artifacts"])
    if file_hash(root / "state/step01_accepted.json") != step01_before:
        raise AssertionError("The accepted Step 1 state changed during Step 2.")
    summary = summarize(results, protocol)
    summary.update(preparation_id=state["preparation_id"], step02_id=fingerprint,
                   processed_relative_path=folder.relative_to(root).as_posix(),
                   preparation_artifacts_verified_before_and_after=len(state["artifacts"]))
    write_json(folder / "feature_reports.json", feature_reports)
    export_results(folder, results, summary)
    write_json(folder / "summary.json", summary)
    write_json(folder / "artifact_manifest.json", artifact_manifest(folder))
    write_json(root / "state/step02_r3_last_result.json", summary)
    write_json(root / "outputs/steps/step02_native_refinement/LATEST.json", {
        "relative_path": folder.relative_to(root).as_posix(), "summary_relative_path": (folder / "summary.json").relative_to(root).as_posix(),
        "run_id": log.run_id, "step02_id": fingerprint})
    for profile in summary["profiles"]:
        log.info("PROFILE | %s %s | finite=%d/6 accepted=%d/6 both coherence wins=%d | promising=%s", profile["representation"], profile["base"],
            profile["matched_finite_cases"], profile["fitting_accepted"], profile["both_coherence_wins"], profile["mechanism_promising"])
    log.info("STEP 2 CHECK | mechanism_promising=%s | complete_reference=%s | deployed_TD_losses=%d | NLL=NOT_MEASURED | full_experiment_ready=False",
        summary["mechanism_promising"], summary["complete_coherence_reference_gate_passed"], summary["diversity_losses_in_deployed_readouts"])
    log.info("SAVED RESULTS | %s", folder)
    log.info("READ FIRST | %s", folder / "SUMMARY.md")
    return summary, folder
