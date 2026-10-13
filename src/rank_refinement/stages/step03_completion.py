"""R2: complete the bounded saved neural pilot, not the final benchmark.

The methodology revision is explicit: no moment correction, a certified
primary minimum-TV readout, and common fitting-reference descriptor scores.
Unavailable development descriptor scores remain unavailable. Neither test
nor external data participate in model selection.
"""
from __future__ import annotations
import copy
import importlib.metadata
import statistics
from pathlib import Path
import numpy as np
from ..runtime.artifacts import artifact_manifest, digest, file_hash, read_json, write_csv, write_json, write_text
from ..runtime.state import require_preparation
from ..models.neural_training import load_canonical
from ..models.adapters.pilot_anchors import topics_from_weights
from ..models.classical_completion import fit_and_predict, select_probability_backoff
from ..models.neural_prediction import score_log_predictions
from ..evaluation.reference_audit import audit_reference
from ..evaluation.diversity import topic_diversity
from ..projection.minimum_tv import certify
from .step02_completion import check_indexed_tree
from .step02_native_refinement import fitting_acceptance, save_arrays
from .step03_neural_pilot import prediction_row
from .step03_saved_replay import validate_repair, source_folder, verify_saved_source, case_folder, native_replay_check, REPAIR_POLICY_SHA256

PROTOCOL = "STEP03_COMPLETION_R2_20261009"
POLICY_SHA256 = "03bcb1984a5cc74c956512ae937f14c2e9b6ed42bcc19b941e33138730341b45"
SCOPE = "bounded_development_pilot_r2"


def validate_completion(configs):
    native, repair = validate_repair(configs)
    p = configs.get("step03_completion", {})
    if p.get("protocol") != PROTOCOL or digest(p) != POLICY_SHA256:
        raise ValueError("Step 3 completion policy changed; declare a named revision before scoring.")
    if p["source_protocol"] != repair["protocol"] or p["moment_correction"] or p["canonical_projection"] != "primary_minimum_tv":
        raise ValueError("Incorrect canonical Proposed revision.")
    if p["prediction"]["shared_backoff_grid"] != native["projection_pilot"]["backoff_grid"]:
        raise ValueError("All predictive paths must use the same declared backoff grid.")
    if p["prediction"]["unigram_pseudocount"] != native["projection_pilot"]["unigram_pseudocount"]:
        raise ValueError("Shared fitting prior changed.")
    return native, p


def repaired_source_folder(root):
    root = Path(root).resolve()
    pointer = root / "outputs/steps/step03_saved_replay/LATEST.json"
    state = root / "state/step03_repair_result.json"
    if pointer.is_file():
        rel = read_json(pointer)["relative_path"]
    elif state.is_file():
        rel = read_json(state)["processed_relative_path"]
    else:
        raise RuntimeError("Keep the completed Step 3 --repair outputs and state. No neural retraining is needed.")
    if not isinstance(rel, str) or "\\" in rel:
        raise ValueError("Invalid repaired-source path.")
    parts = Path(rel)
    if parts.is_absolute() or ".." in parts.parts or len(parts.parts) != 4 or parts.parts[:3] != ("outputs", "steps", "step03_saved_replay"):
        raise ValueError("Unsafe repaired-source path.")
    folder = (root / parts).resolve()
    if not folder.is_relative_to(root) or not folder.is_dir():
        raise ValueError("Repaired source is missing or unsafe.")
    return folder


def reference_scores(topics, data, cache):
    identity = {"topics": topics, "vocabulary": data["vocabulary"], "gensim": importlib.metadata.version("gensim"),
        "fitting": [(r["group_id"], r["tokens"]) for r in data["fit_records"]],
        "development": [(r["group_id"], r["tokens"]) for r in data["dev_coherence_records"]],
        "windows": {"c_v": 110, "c_npmi": 10}, "all_slots": True}
    key = digest(identity)
    destination = Path(cache) / key
    if (destination / "artifact_manifest.json").is_file():
        check_indexed_tree(destination)
        value = read_json(destination / "reference_audit.json")
        if value["identity_sha256"] != key:
            raise ValueError("Coherence cache identity changed.")
    else:
        value = {"identity_sha256": key, "fitting": audit_reference(topics, data["fit_records"], data["vocabulary"]),
                 "development": audit_reference(topics, data["dev_coherence_records"], data["vocabulary"])}
        write_json(destination / "reference_audit.json", value)
        write_json(destination / "artifact_manifest.json", artifact_manifest(destination))
    def metric(name):
        audit = value[name]
        return {"scored": all(audit["metrics"][m]["complete"] for m in ["c_v", "c_npmi"]),
            "c_v": audit["metrics"]["c_v"]["mean"], "c_npmi": audit["metrics"]["c_npmi"]["mean"],
            "per_topic": {m: audit["metrics"][m]["per_topic"] for m in ["c_v", "c_npmi"]},
            "reference_groups": audit["reference_groups"], "absent_words": audit["reference_absent_words"],
            "all_ten_topics_and_words_retained": True,
            "failure": None if all(audit["metrics"][m]["complete"] for m in ["c_v", "c_npmi"]) else "standard_metric_nonfinite_no_slot_removal",
            "scope": "in_sample_descriptive" if name == "fitting" else "heldout_reference_diagnostic"}
    return metric("fitting"), metric("development"), value


def verify_sources(original, repaired, data, native_policy):
    originals = verify_saved_source(original, data, native_policy)
    check_indexed_tree(repaired)
    saved_run = read_json(repaired / "run_result.json")
    if not saved_run["engineering_passed"] or saved_run["protocol"] != "STEP03_SAVED_REPLAY_R1_20261009":
        raise ValueError("Require the completed certified numerical repair.")
    reference = read_json(repaired / "source_neural_pilot_reference.json")
    if reference["artifact_manifest_sha256"] != file_hash(original / "artifact_manifest.json") or reference["relative_path"] != original.relative_to(original.parents[3]).as_posix():
        raise ValueError("Repair does not refer to the verified original native pilot.")
    snapshot = read_json(repaired / "replay_protocol.json")
    if digest(snapshot["repair_policy"]) != REPAIR_POLICY_SHA256:
        raise ValueError("Wrong repair snapshot.")
    cases = read_json(repaired / "case_results.json")
    expected = {(c["base"], c["profile"], c["seed"]): c for c in originals}
    keys = [(c["base"], c["profile"], c["seed"]) for c in cases]
    if len(keys) != 8 or len(set(keys)) != 8 or set(keys) != set(expected):
        raise ValueError("Require all eight unique repaired cases, with no dropped failures.")
    for case in cases:
        key = (case["base"], case["profile"], case["seed"])
        folder = case_folder(repaired, case)
        check_indexed_tree(folder)
        if read_json(folder / "case_result.json") != case:
            raise ValueError("Repaired case does not match its summary.")
        original_case = case_folder(original, expected[key])
        if case["source_case_result_sha256"] != file_hash(original_case / "case_result.json"):
            raise ValueError("Repaired case refers to different native source.")
        with np.load(folder / "readouts_and_latents.npz", allow_pickle=False) as z:
            arrays = {n: z[n] for n in z.files}
        with np.load(original_case / "readouts_and_latents.npz", allow_pickle=False) as z:
            for name in ["native", "theta_fit", "theta_development", "decoder_gain", "decoder_offset", "prior"]:
                if not np.array_equal(arrays[name], z[name]):
                    raise ValueError("Repair changed saved native arrays: " + name)
        native_replay_check(arrays, expected[key], data, native_policy)
        for role in ["primary", "index", "range", "l2"]:
            certificate = case["certificates"].get(role, {})
            if not certificate.get("success") or not certificate.get("certified") or "candidate_" + role not in arrays:
                raise ValueError("A repaired candidate lacks its certificate: " + role)
            if not certify(arrays["native"], arrays["candidate_" + role], case["targets"], margin=1e-6)["certified"]:
                raise ValueError("Independent conservation/ranking certification failed.")
    return cases


def revised_neural_case(root, repaired, case, data, native_policy, configs, destination, log):
    source = case_folder(repaired, case)
    identity = {"source_manifest": file_hash(source / "artifact_manifest.json"), "protocol": POLICY_SHA256}
    if (destination / "artifact_manifest.json").is_file():
        check_indexed_tree(destination)
        cached = read_json(destination / "case_result.json")
        if cached["identity"] != identity:
            raise ValueError("Revised case cache identity differs.")
        log.info("REVISED CACHE | %s %s seed=%d | verified saved results", case["base"], case["profile"], case["seed"])
        return cached
    with np.load(source / "readouts_and_latents.npz", allow_pickle=False) as z:
        arrays = {n: z[n] for n in z.files}
    common = dict(theta_fit=arrays["theta_fit"], theta_dev=arrays["theta_development"], native=arrays["native"],
        gain=arrays["decoder_gain"], offset=arrays["decoder_offset"], prior=arrays["prior"], data=data,
        policy=native_policy["projection_pilot"], base=case["base"], profile=case["profile"], seed=case["seed"])
    rows, audits = [], {}
    variants = [("native", case["base"], "native", False, True),
                ("proposed", "Proposed", "primary", False, True),
                ("moment_control", "Proposed + moment correction (removed component)", "primary", True, True),
                ("no_backoff", "Proposed without backoff", "primary", False, False),
                ("index_control", "Index secondary without moments", "index", False, True),
                ("range_control", "Range secondary without moments", "range", False, True),
                ("l2_control", "L2 projection without moments", "l2", False, True)]
    accepted, reasons = False, []
    for role, label, candidate, moment, use_backoff in variants:
        affinity = arrays["native"] if candidate == "native" else arrays["candidate_" + candidate]
        topics = topics_from_weights(affinity, data["vocabulary"])
        fit, dev, audit = reference_scores(topics, data, root / "data/cache/step03_completion_coherence")
        if candidate in ["native", "primary"]:
            audits[candidate] = audit
        row = prediction_row(role, label, affinity, topics, fit, dev, topic_diversity(topics),
            correction=moment, use_backoff=use_backoff, deployed=candidate != "native", **common)
        row.update(scope=SCOPE, canonical_projection=candidate,
            fitting_coherence_scope="in_sample_descriptive", final_model_result=False,
            prediction_interface="frozen_native_affine_decoder", development_tuned=True)
        if role == "proposed":
            accepted, reasons = fitting_acceptance(rows[0]["fitting"], fit, rows[0]["topic_diversity"],
                row["topic_diversity"], case["certificates"]["primary"], configs["step02_refinement"])
            row.update(fitting_accepted=accepted, deployment_status="deployed_research_candidate" if accepted else "candidate_rejected_not_native_relabelled")
        rows.append(row)
    display = copy.deepcopy(rows[0])
    display.update(method="Same-descriptor display-only", role="display_only", topics=rows[1]["topics"],
        fitting=rows[1]["fitting"], development=rows[1]["development"], topic_diversity=rows[1]["topic_diversity"],
        descriptor_sets_from_proposed=True, projection_deployed=False, canonical_projection="native")
    if display["prediction"] != rows[0]["prediction"] or display["document_losses"] != rows[0]["document_losses"]:
        raise AssertionError("Display-only changed predictions.")
    rows.append(display)
    result = {"base": case["base"], "profile": case["profile"], "seed": case["seed"], "identity": identity,
        "fitting_accepted": accepted, "acceptance_reasons": reasons, "rows": rows,
        "primary_certificate": case["certificates"]["primary"], "source_relative_path": source.relative_to(root).as_posix()}
    write_json(destination / "case_result.json", result)
    write_json(destination / "reference_audit.json", audits)
    write_json(destination / "projection_certificates.json", case["certificates"])
    save_arrays(destination / "selected_readout.npz", proposed=arrays["candidate_primary"], native=arrays["native"])
    write_json(destination / "artifact_manifest.json", artifact_manifest(destination))
    proposed = rows[1]
    log.info("PROPOSED | base=%s profile=%s seed=%d | fitting Cv=%.6f NPMI=%.6f TD=%.2f | dev NLL=%.9f native=%.9f | moments=False | accepted=%s",
        case["base"], case["profile"], case["seed"], proposed["fitting"]["c_v"], proposed["fitting"]["c_npmi"],
        proposed["topic_diversity"], proposed["prediction"]["macro_source_nll"], rows[0]["prediction"]["macro_source_nll"], accepted)
    return result


def classical_case(root, method, seed, profile, data, policy, prior, destination, log):
    identity = {"method": method, "seed": seed, "profile": profile, "protocol": POLICY_SHA256}
    if (destination / "artifact_manifest.json").is_file():
        check_indexed_tree(destination)
        saved = read_json(destination / "case_result.json")
        if saved["identity"] != identity:
            raise ValueError("Classical cache identity differs.")
        log.info("CLASSICAL CACHE | %s %s seed=%d", method, profile["name"], seed)
        return saved
    log.info("CLASSICAL FIT | %s %s seed=%d | full fitting counts only; development observed tokens only", method, profile["name"], seed)
    log.new_classical_fits = getattr(log, "new_classical_fits", 0) + 1
    topics, weights, probabilities, details = fit_and_predict(method, data["fit"], data["dev_observed"], data["vocabulary"], seed, profile)
    fit, dev, audit = reference_scores(topics, data, root / "data/cache/step03_completion_coherence")
    chosen, curve = select_probability_backoff(probabilities, prior, policy["prediction"]["shared_backoff_grid"],
        data["dev_records"], data["masks"], sorted({r["source"] for r in data["fit_records"]}))
    row = {"method": method, "role": "classical", "underlying_base": method, "profile": profile["name"], "seed": seed,
        "status": "scored", "topics": topics, "fitting": fit, "development": dev, "topic_diversity": topic_diversity(topics),
        "prediction": chosen["scores"], "rho": chosen["rho"], "backoff_curve": curve, "document_losses": chosen["rows"],
        "prediction_interface": details["prediction_interface"], "scope": SCOPE, "fitting_coherence_scope": "in_sample_descriptive",
        "projection_deployed": False, "moment_correction": False, "development_tuned": True,
        "final_model_result": False, "test_or_external_used": False}
    result = {"identity": identity, "row": row, "training_details": details}
    write_json(destination / "case_result.json", result)
    write_json(destination / "reference_audit.json", audit)
    save_arrays(destination / "native_readout_and_predictions.npz", weights=weights, probabilities=probabilities)
    write_json(destination / "artifact_manifest.json", artifact_manifest(destination))
    log.info("CLASSICAL SAVED | %s %s seed=%d | fitting Cv=%.6f NPMI=%.6f | dev NLL=%.9f | rho=%.2f",
        method, profile["name"], seed, fit["c_v"], fit["c_npmi"], chosen["scores"]["macro_source_nll"], chosen["rho"])
    return result


def flat_row(row):
    return {"method": row["method"], "role": row["role"], "base": row["underlying_base"],
        "profile": row["profile"], "seed": row["seed"], "fitting_c_v": row["fitting"]["c_v"],
        "fitting_c_npmi": row["fitting"]["c_npmi"], "development_c_v": row["development"]["c_v"],
        "development_c_npmi": row["development"]["c_npmi"], "topic_diversity": row["topic_diversity"],
        "development_macro_nll": row["prediction"]["macro_source_nll"], "development_micro_nll": row["prediction"]["micro_nll"],
        "rho": row["rho"], "moment_correction": row["moment_correction"], "projection_deployed": row["projection_deployed"],
        "coherence_scope": "fitting_in_sample_descriptive", "prediction_scope": "development_selection",
        "final_model_result": False}


def export_progress(output, neural, classical):
    rows = [flat_row(row) for case in neural for row in case["rows"]]
    rows += [flat_row(case["row"]) for case in classical]
    if rows:
        write_csv(output / "all_case_metrics.csv", rows, list(rows[0]))
    write_json(output / "progress.json", {"neural_cases_saved": len(neural), "classical_cases_saved": len(classical),
        "expected_neural_cases": 8, "expected_classical_cases": 8, "partial_results_are_not_final": True})


def profile_decisions(cases, native_policy, policy):
    decisions = []
    for base in native_policy["bases"]:
        for profile in native_policy["profiles"]:
            subset = sorted((c for c in cases if c["base"] == base and c["profile"] == profile), key=lambda c: c["seed"])
            complete = len(subset) == 2 and {c["seed"] for c in subset} == set(native_policy["seeds"])
            rows = [{r["role"]: r for r in c["rows"]} for c in subset]
            nll = [r["proposed"]["prediction"]["macro_source_nll"] for r in rows]
            gains = [r["native"]["prediction"]["macro_source_nll"] - r["proposed"]["prediction"]["macro_source_nll"] for r in rows]
            mechanism = bool(complete and all(c["fitting_accepted"] for c in subset))
            predictive = bool(complete and all(g > 0 for g in gains) and statistics.mean(gains) >= policy["selection"]["mean_matched_native_nll_gain_minimum"])
            decisions.append({"base": base, "profile": profile, "seed_order": [c["seed"] for c in subset],
                "complete_two_seeds": complete, "mechanism_gate": mechanism, "prediction_gate": predictive,
                "eligible_for_followup": bool(mechanism and predictive), "development_nll_by_seed": nll,
                "native_minus_proposed_nll_by_seed": gains, "mean_development_nll": statistics.mean(nll) if complete else None,
                "mean_native_minus_proposed_nll": statistics.mean(gains) if complete else None})
    eligible = [d for d in decisions if d["eligible_for_followup"]]
    chosen = min(eligible, key=lambda d: (d["mean_development_nll"], d["base"], d["profile"])) if eligible else None
    return decisions, chosen


def aggregate_rows(rows, expected_seeds):
    keys = {(r["method"], r["role"], r["underlying_base"], r["profile"]) for r in rows}
    result = []
    fields = ["fitting_c_v", "fitting_c_npmi", "development_c_v", "development_c_npmi", "topic_diversity", "development_macro_nll", "development_micro_nll"]
    for method, role, base, profile in sorted(keys):
        subset = [flat_row(r) for r in rows if (r["method"], r["role"], r["underlying_base"], r["profile"]) == (method, role, base, profile)]
        complete = len(subset) == len(expected_seeds) and {r["seed"] for r in subset} == set(expected_seeds)
        entry = {"method": method, "role": role, "base": base, "profile": profile, "n_seeds": len(subset),
            "complete_seed_grid": complete, "coherence_scope": "fitting_in_sample_descriptive", "prediction_scope": "development_selection", "final_model_result": False}
        for field in fields:
            values = [r[field] for r in subset]
            finite = complete and all(v is not None and np.isfinite(v) for v in values)
            entry[field + "_mean"] = statistics.mean(values) if finite else None
            entry[field + "_sd"] = statistics.stdev(values) if finite and len(values) > 1 else None
        result.append(entry)
    return result


def complete_all(root, original, repaired, prepared, output, configs, log):
    native_policy, policy = validate_completion(configs)
    data = load_canonical(prepared)
    cases = verify_sources(original, repaired, data, native_policy)
    write_json(output / "data_firewall.json", {"fitting_group_ids": data["fit_ids"],
        "fitting_observed_ids": data["fit_observed_ids"], "development_completion_ids": data["dev_ids"],
        "development_coherence_group_ids": [r["group_id"] for r in data["dev_coherence_records"]],
        "coherence_common_reference": "accepted_fitting_groups_in_sample",
        "development_targets_used_only_for_prediction_selection": True, "test_or_external_scored_or_trained": False,
        "new_encoder_calls": 0, "new_neural_fits": 0})
    neural, classical = [], []
    for case in cases:
        destination = output / "neural_cases" / case["base"] / case["profile"] / ("seed" + str(case["seed"]))
        neural.append(revised_neural_case(root, repaired, case, data, native_policy, configs, destination, log))
        export_progress(output, neural, classical)
        log.info("PROGRESS | neural=%d/8 | classical=0/8 | saved=%s", len(neural), output / "all_case_metrics.csv")
    prior = data["fit"].astype(np.float64).sum(axis=0) + policy["prediction"]["unigram_pseudocount"]
    prior /= prior.sum()
    for method in policy["classical_panel"]["methods"]:
        for profile in policy["classical_panel"]["profiles"][method]:
            for seed in policy["classical_panel"]["seeds"]:
                destination = output / "classical_cases" / method / profile["name"] / ("seed" + str(seed))
                classical.append(classical_case(root, method, seed, profile, data, policy, prior, destination, log))
                export_progress(output, neural, classical)
                log.info("PROGRESS | neural=8/8 | classical=%d/8 | saved=%s", len(classical), output / "all_case_metrics.csv")
    rows = [r for c in neural for r in c["rows"]] + [c["row"] for c in classical]
    means = aggregate_rows(rows, native_policy["seeds"])
    write_csv(output / "profile_metrics_mean_sd.csv", means, list(means[0]))
    decisions, chosen = profile_decisions(neural, native_policy, policy)
    # Each competitor receives its own best predictive profile on the same
    # development population. No weaker profile is chosen to favor Proposed.
    competitors = []
    for method in ["encot", "neuromax", "lda", "nmf"]:
        candidates = [m for m in means if m["method"] == method and m["role"] in ["native", "classical"] and m["complete_seed_grid"]]
        if len(candidates) != 2 or any(m["development_macro_nll_mean"] is None for m in candidates):
            raise ValueError("Competitor predictive profile grid is incomplete.")
        competitors.append(min(candidates, key=lambda m: (m["development_macro_nll_mean"], m["profile"])))
    unigram_scores, unigram_docs = score_log_predictions(np.tile(np.log(prior), (len(data["dev_ids"]), 1)),
        data["dev_records"], data["masks"], sorted({r["source"] for r in data["fit_records"]}))
    write_json(output / "fitting_unigram_sanity_predictor.json", {"scores": unigram_scores, "document_losses": unigram_docs,
        "no_topics": True, "test_or_external_used": False})
    proposed_mean = next((m for m in means if chosen and m["role"] == "proposed" and m["base"] == chosen["base"] and m["profile"] == chosen["profile"]), None)
    selected_ablations = [m for m in means if chosen and m["base"] == chosen["base"] and m["profile"] == chosen["profile"]]
    write_csv(output / "selected_proposed_and_ablation_mean_sd.csv", selected_ablations, list(means[0]))
    write_json(output / "selected_ablation_diagnostics.json", {"rows": selected_ablations,
        "same_descriptors_can_tie_by_construction": True,
        "primary_projection_is_selected_for_minimum_tv_not_best_secondary_nll": True,
        "proposed_dominates_every_ablation_nll": bool(proposed_mean and all(
            proposed_mean["development_macro_nll_mean"] <= m["development_macro_nll_mean"]
            for m in selected_ablations if m["role"] != "proposed")),
        "development_selection_results_are_not_final_inference": True})
    benchmark = competitors + ([proposed_mean] if proposed_mean else [])
    write_csv(output / "bounded_competitor_comparison.csv", benchmark, list(means[0]))
    comparisons = []
    if proposed_mean:
        for c in competitors:
            comparisons.append({"competitor": c["method"], "profile": c["profile"],
                "proposed_minus_competitor_fitting_cv": proposed_mean["fitting_c_v_mean"] - c["fitting_c_v_mean"] if c["fitting_c_v_mean"] is not None else None,
                "proposed_minus_competitor_fitting_npmi": proposed_mean["fitting_c_npmi_mean"] - c["fitting_c_npmi_mean"] if c["fitting_c_npmi_mean"] is not None else None,
                "proposed_minus_competitor_diversity": proposed_mean["topic_diversity_mean"] - c["topic_diversity_mean"],
                "competitor_minus_proposed_development_nll": c["development_macro_nll_mean"] - proposed_mean["development_macro_nll_mean"]})
    descriptor_winners = {}
    for metric in ["fitting_c_v_mean", "fitting_c_npmi_mean", "topic_diversity_mean"]:
        finite = [m for m in means if m["role"] in ["native", "classical"] and m[metric] is not None]
        descriptor_winners[metric] = max(finite, key=lambda m: (m[metric], m["method"], m["profile"])) if finite else None
    all_metric_wins = bool(comparisons and all(c["proposed_minus_competitor_fitting_cv"] is not None and
        c["proposed_minus_competitor_fitting_cv"] > 0 and c["proposed_minus_competitor_fitting_npmi"] is not None and
        c["proposed_minus_competitor_fitting_npmi"] > 0 and c["proposed_minus_competitor_diversity"] >= -1e-12 and
        c["competitor_minus_proposed_development_nll"] > 0 for c in comparisons))
    strongest_nll = min([m["development_macro_nll_mean"] for m in competitors] + [unigram_scores["macro_source_nll"]])
    beats_panel_nll = bool(proposed_mean and proposed_mean["development_macro_nll_mean"] < strongest_nll)
    ready = bool(chosen and beats_panel_nll)
    summary = {"protocol": PROTOCOL, "method_revision": {"moment_correction": False, "canonical_projection": "primary_minimum_tv",
        "index_range_l2_retained_as_controls": True, "no_neural_retraining": True},
        "evaluation_revision": {"common_coherence_reference": "fitting_in_sample_descriptive", "development_coherence_na_preserved": True,
            "original_development_coherence_gate_passed": False, "heldout_coherence_superiority_established": False},
        "completed_neural_cases": len(neural), "completed_classical_cases": len(classical), "profiles": decisions,
        "selected_candidate": chosen, "competitor_profile_comparisons": comparisons,
        "descriptor_only_best_profiles_not_hidden": descriptor_winners,
        "bounded_panel_nll_win": beats_panel_nll, "all_metrics_winning_in_predictively_selected_panel": all_metric_wins,
        "ready_for_step4_bounded_benchmark_implementation": ready,
        "ready_for_full_final_experiment": False, "full_final_competitor_benchmark_completed": False,
        "final_model_result": False, "test_or_external_used": False,
        "remaining_final_panel": ["gsdmm", "bertopic", "combinedtm", "fastopic", "glocom"],
        "validation_after_development_driven_revision_required": True}
    write_json(output / "case_results.json", {"neural": neural, "classical": classical})
    write_json(output / "summary.json", summary)
    write_json(output / "selected_candidate_for_step04.json", {"protocol": PROTOCOL,
        "candidate": {"base": chosen["base"], "profile": chosen["profile"], "canonical_projection": "primary_minimum_tv", "moment_correction": False} if chosen else None,
        "authorized_next_scope": "bounded_full_panel_implementation_and_development_checks" if ready else "diagnose_pilot_before_expansion",
        "selection_is_development_only": True, "test_or_external_used": False,
        "full_final_experiment_allowed": False, "ready_for_step4": ready})
    table = "| Method | Profile | Fitting C_v | Fitting NPMI | Diversity | Development NLL |\n|---|---|---:|---:|---:|---:|\n"
    def fmt(value):
        return "NA" if value is None else f"{value:.6f}"
    for row in benchmark:
        table += "| " + " | ".join([row["method"], row["profile"], fmt(row["fitting_c_v_mean"]), fmt(row["fitting_c_npmi_mean"]), fmt(row["topic_diversity_mean"]), fmt(row["development_macro_nll_mean"])]) + " |\n"
    write_text(output / "SUMMARY.md", "# Step 3 R2 bounded pilot\n\n"
        "Proposed uses the certified primary minimum-TV affinity in the actual frozen native decoder, without moment correction. "
        "The separable index secondary is not a canonical contribution; index, range, L2, moment restoration, display-only and no-backoff remain controls. "
        "Original outputs are preserved. Two seeds are pilot replications, not the final five-seed benchmark.\n\n"
        "C_v (window 110) and NPMI (window 10) use the same accepted fitting reference for all methods, retain every topic/word, "
        "and are in-sample descriptive results. Development descriptor scores are retained as NA wherever standard scoring is nonfinite. "
        "The original heldout-coherence gate has not passed. The revised prediction-and-descriptor pilot gate is a different, explicit scope. "
        "Development NLL is source-balanced, observed-mask-only, and used for selection, not final unbiased evaluation. "
        "The positive shared-prior backoff grid is the same for every predictive model. NMF reports reconstruction cross-entropy, not native generative likelihood.\n\n"
        + table + "\n"
        + f"Selected candidate: {chosen['base'] + '/' + chosen['profile'] if chosen else 'NONE'}. "
        + f"Bounded-panel NLL win: {beats_panel_nll}. All metrics in predictively selected panel win/tie: {all_metric_wins}. "
        + f"Step 4 bounded implementation allowed: {ready}. Full final experiment allowed: False.\n\n"
        "Every baseline profile appears in profile_metrics_mean_sd.csv. Descriptor-best profiles are separately identified in summary.json; "
        "a prediction-selected comparator table does not prove descriptor dominance over every baseline configuration. "
        "The unigram sanity predictor is in fitting_unigram_sanity_predictor.json. "
        "selected_proposed_and_ablation_mean_sd.csv retains every matched control. Secondary objectives may return a slightly better NLL "
        "on some seeds; primary minimum-TV is not selected by chasing the best development NLL among readouts. "
        "Same-descriptor display-only and identical-topic controls necessarily tie on descriptor metrics.\n\n"
        "The bounded panel contains LDA, NMF, EnCOT and NeuroMax, not the full planned nine-family competitor benchmark. "
        "GSDMM, BERTopic, CombinedTM, FASTopic and GloCOM, final five-seed validation and external generalization remain for the benchmark stage. "
        "No final paper table or winning claim is approved here.\n")
    return summary


def run(root, configs, versions, hardware, log):
    root = Path(root).resolve()
    native, policy = validate_completion(configs)
    preparation = require_preparation(root)
    original, repaired = source_folder(root), repaired_source_folder(root)
    for source in [original, repaired]:
        check_indexed_tree(source)
        if read_json(source / "run_result.json")["preparation_id"] != preparation["preparation_id"]:
            raise ValueError("Source pilot and accepted preparation differ.")
    source_hashes = {p.relative_to(root).as_posix(): file_hash(p / "artifact_manifest.json") for p in [original, repaired]}
    implementation = {p.relative_to(root).as_posix(): file_hash(p) for p in (root / "src/rank_refinement").rglob("*.py")}
    identity = {"protocol": policy, "source_indexes": source_hashes, "preparation_id": preparation["preparation_id"],
        "implementation": implementation, "versions": {n: importlib.metadata.version(n) for n in ["numpy", "scipy", "gensim", "scikit-learn", "osqp"]}}
    key = digest(identity)
    output = root / "outputs/steps/step03_completion" / key[:20]
    output.mkdir(parents=True, exist_ok=True)
    write_json(log.folder / "preparation_working_path.json", {"relative_path": output.relative_to(root).as_posix()})
    write_json(output / "completion_protocol_resolved_before_scoring.json", identity)
    write_json(output / "source_neural_pilot_reference.json", {"relative_path": original.relative_to(root).as_posix(),
        "artifact_manifest_sha256": source_hashes[original.relative_to(root).as_posix()]})
    write_json(output / "repaired_saved_source_reference.json", {"relative_path": repaired.relative_to(root).as_posix(),
        "artifact_manifest_sha256": source_hashes[repaired.relative_to(root).as_posix()]})
    write_json(root / "outputs/steps/step03_completion/LATEST.json", {"relative_path": output.relative_to(root).as_posix()})
    log.info("STEP 3 COMPLETION | Proposed moments=False | primary minimum-TV | new neural fits=0 | encoder calls=0 | at most 8 small classical fits")
    summary = complete_all(root, original, repaired, root / preparation["processed_relative_path"], output, configs, log)
    for source in [original, repaired]:
        if file_hash(source / "artifact_manifest.json") != source_hashes[source.relative_to(root).as_posix()]:
            raise AssertionError("Completion modified a historical source index.")
        check_indexed_tree(source)
    ready = summary["ready_for_step4_bounded_benchmark_implementation"]
    result = {"status": "STEP03_COMPLETED_BOUNDED_FOLLOWUP_ALLOWED" if ready else "STEP03_COMPLETED_QUALITY_GATE_BLOCKED",
        "engineering_passed": True, "preparation_gate_passed": True, "model_quality_established": False,
        "protocol": PROTOCOL, "step03_completion_id": key, "preparation_id": preparation["preparation_id"],
        "processed_relative_path": output.relative_to(root).as_posix(), "completed_neural_cases": 8, "completed_classical_cases": 8,
        "new_neural_fits": 0, "new_encoder_calls": 0, "moment_correction": False,
        "new_classical_fits": getattr(log, "new_classical_fits", 0),
        "selected_candidate": summary["selected_candidate"], "bounded_panel_nll_win": summary["bounded_panel_nll_win"],
        "ready_for_step4": ready, "full_experiment_gate_passed": False,
        "original_development_coherence_gate_passed": False, "final_model_result": False, "test_or_external_used": False}
    write_json(output / "run_result.json", result)
    write_json(output / "artifact_manifest.json", artifact_manifest(output))
    write_json(root / "state/step03_completion_result.json", {**result, "artifacts": read_json(output / "artifact_manifest.json")})
    log.info("SAVED RESULTS | %s", output)
    log.info("READ FIRST | %s", output / "SUMMARY.md")
    log.info("STEP 3 COMPLETE | bounded followup=%s | full final experiment=False | heldout coherence remains diagnostic", ready)
    return result, output
