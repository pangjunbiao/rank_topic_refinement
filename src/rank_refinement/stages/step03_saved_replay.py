"""Repair saved Step 3 readouts, with zero fitting or encoder calls."""
from __future__ import annotations
import copy
import importlib.metadata
import statistics
from pathlib import Path
import numpy as np
from ..runtime.artifacts import (artifact_manifest, digest, file_hash, read_json, verify_artifacts,
    write_csv, write_json, write_text)
from ..runtime.state import require_preparation
from .step02_completion import check_indexed_tree
from .step02_native_refinement import fitting_acceptance, metric_score, save_arrays
from .step03_neural_pilot import prediction_row, validate_protocol as validate_native_protocol
from ..models.neural_training import load_canonical
from ..models.neural_prediction import log_predictions, select_backoff
from ..models.adapters.pilot_anchors import topics_from_weights
from ..evaluation.diversity import topic_diversity
from ..evaluation.reference_audit import audit_reference
from ..projection.minimum_tv import certify
from ..projection.predictive_controls import l2_projection
from ..projection.replay_solver import primary_tv, index_secondary, range_secondary

PROTOCOL = "STEP03_SAVED_REPLAY_R1_20261009"


def validate_repair(configs):
    native = validate_native_protocol(configs)
    repair = configs.get("step03_repair", {})
    if repair.get("protocol") != PROTOCOL or digest(repair) != REPAIR_POLICY_SHA256:
        raise ValueError("The declared numerical-replay protocol changed.")
    if repair["source_protocol"] != native["protocol"] or repair["new_model_fits"] != 0:
        raise ValueError("This repair may only replay the completed native pilot.")
    if importlib.metadata.version("osqp") != "1.0.4":
        raise RuntimeError("Use the already installed requirements-step03.txt dependencies; OSQP must be 1.0.4.")
    return native, repair


def source_folder(root):
    pointer = Path(root) / "outputs/steps/step03_neural_pilot/LATEST.json"
    if pointer.is_file():
        relative = Path(read_json(pointer)["relative_path"])
    else:
        saved_state = Path(root) / "state/step03_pilot_result.json"
        if not saved_state.is_file():
            raise RuntimeError("Keep your original completed Step 3 outputs and state folders for no-training repair.")
        relative = Path(read_json(saved_state)["processed_relative_path"])
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 4 or relative.parts[:3] != ("outputs", "steps", "step03_neural_pilot"):
        raise ValueError("Unsafe saved neural-pilot path.")
    folder = (Path(root) / relative).resolve()
    if not folder.is_relative_to(Path(root).resolve()) or not folder.is_dir():
        raise ValueError("Missing saved Step 3 source folder.")
    return folder


def verify_saved_source(source, data, policy):
    check_indexed_tree(source)
    cases = read_json(source / "case_results.json")
    expected = {(b, p, s) for b in policy["bases"] for p in policy["profiles"] for s in policy["seeds"]}
    keys = [(c["base"], c["profile"], c["seed"]) for c in cases]
    if len(keys) != 8 or set(keys) != expected:
        raise ValueError("Replay requires all eight unique completed native cases.")
    snapshot = read_json(source / "protocol_resolved_before_scoring.json")
    if snapshot["protocol"] != policy:
        raise ValueError("Saved native pilot differs from the frozen protocol.")
    firewall = read_json(source / "data_firewall.json")
    for name, values in [("fitting_group_ids", data["fit_ids"]),
                         ("fitting_observed_ids", data["fit_observed_ids"]),
                         ("development_completion_ids", data["dev_ids"])]:
        if firewall[name] != values:
            raise ValueError("Saved latent rows differ from canonical data: " + name)
    return cases


def case_folder(source, case):
    return source / "cases" / case["base"] / case["profile"] / ("seed" + str(case["seed"]))


def native_replay_check(arrays, case, data, policy):
    expected_shape = (10, len(data["vocabulary"]))
    if arrays["native"].shape != expected_shape or arrays["theta_fit"].shape != (len(data["fit_observed_ids"]), 10) or arrays["theta_development"].shape != (len(data["dev_ids"]), 10):
        raise ValueError("Saved affinity/latent dimensions do not match canonical data.")
    if not case["interface"]["verified"] or any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Native interface or saved array finite check failed.")
    if (arrays["native"] < 0).any() or (arrays["theta_fit"] < 0).any() or (arrays["theta_development"] < 0).any():
        raise ValueError("Saved native affinity/latents are negative.")
    topics = topics_from_weights(arrays["native"], data["vocabulary"])
    saved = next(r for r in case["rows"] if r["role"] == "native")
    if topics != saved["topics"]:
        raise ValueError("Saved native descriptors differ from the exported affinity.")
    probability = log_predictions(arrays["theta_development"], arrays["native"], arrays["decoder_gain"], arrays["decoder_offset"])
    chosen, curve = select_backoff(probability, arrays["prior"], policy["projection_pilot"]["backoff_grid"],
        data["dev_records"], data["masks"], sorted({r["source"] for r in data["fit_records"]}))
    error = max(abs(chosen["scores"][name] - saved["prediction"][name]) for name in ["macro_source_nll", "micro_nll"])
    if error > 1e-10 or chosen["rho"] != saved["rho"]:
        raise ValueError("Native predictor replay does not reproduce its saved development score.")
    old_docs = {r["group_id"]: r for r in saved["document_losses"]}
    if len(old_docs) != len(chosen["rows"]):
        raise ValueError("Native document-completion population changed.")
    maximum_doc_error = max(abs(r["negative_log_probability"] - old_docs[r["group_id"]]["negative_log_probability"]) for r in chosen["rows"])
    if maximum_doc_error > 1e-8:
        raise ValueError("Per-document native replay failed.")
    # Also independently derive the shared prior from canonical fitting counts.
    prior = data["fit"].astype(np.float64).sum(0) + policy["projection_pilot"]["unigram_pseudocount"]
    prior /= prior.sum()
    if not np.allclose(prior, arrays["prior"], rtol=0, atol=1e-14):
        raise ValueError("Native backoff prior was not derived from the accepted fitting counts.")
    return {"verified": True, "macro_micro_nll_max_error": error,
            "per_document_loss_max_error": maximum_doc_error, "same_observed_masks": True,
            "native_affinity_sha256": __import__("hashlib").sha256(arrays["native"].tobytes()).hexdigest()}


def replay_case(root, source, saved, data, policy, repair, r3, output, log):
    original = case_folder(source, saved)
    check_indexed_tree(original)
    if read_json(original / "case_result.json") != saved:
        raise ValueError("Saved case and source summary differ.")
    if (output / "artifact_manifest.json").is_file():
        check_indexed_tree(output)
        cached = read_json(output / "case_result.json")
        if cached["source_case_result_sha256"] != file_hash(original / "case_result.json"):
            raise ValueError("Repaired cache differs from the original saved case.")
        log.info("REPLAY CACHE | %s %s seed=%d | verified complete repaired case reused", saved["base"], saved["profile"], saved["seed"])
        return cached
    output.mkdir(parents=True, exist_ok=True)
    with np.load(original / "readouts_and_latents.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    check = native_replay_check(arrays, saved, data, policy)
    write_json(output / "native_replay_certificate.json", check)
    native, targets = arrays["native"], saved["targets"]
    p3, pp = policy["projection_pilot"], r3["projection"]
    context = read_json(original / "range_contexts.json")
    index = {g: i for i, g in enumerate(data["fit_observed_ids"])}
    ids = context["group_ids"]
    if len(ids) != 64 or len(set(ids)) != 64 or not set(ids).issubset(index):
        raise ValueError("Range contexts are not the same 64 unique fitting contexts.")
    context_latents = arrays["theta_fit"][[index[g] for g in ids]]
    log.info("SAVED REPLAY | %s %s seed=%d | new neural fits=0 | native NLL verified", saved["base"], saved["profile"], saved["seed"])
    primary, pcert = primary_tv(native, targets, pp["rank_margin_scaled"], repair["primary_seconds"])
    certificates = {"primary": pcert}
    candidates = {}
    if primary is not None:
        candidates["primary"] = primary
        indexed, icert = index_secondary(native, targets, pcert["solver_tv_scaled"], pp["rank_margin_scaled"],
            p3["tv_cap_slack_scaled"], repair["index_seconds"])
        certificates["index"] = icert
        if indexed is not None:
            candidates["index"] = indexed
        ranged, rcert = range_secondary(native, targets, pcert["solver_tv_scaled"], context_latents,
            arrays["decoder_gain"], primary, pp["rank_margin_scaled"], p3["tv_cap_slack_scaled"],
            repair["range_seconds"], repair["range_maximum_iterations"], repair["range_full_constraint_tolerance"], log)
        certificates["range"] = rcert
        if ranged is not None:
            candidates["range"] = ranged
        old_l2 = saved["certificates"].get("l2", {})
        if old_l2.get("success") and "candidate_l2" in arrays:
            l2 = arrays["candidate_l2"]
            lcert = certify(native, l2, targets, margin=pp["rank_margin_scaled"])
            lcert.update(success=lcert["certified"], verified_original_l2_reused=True,
                original_feasible_set=True, tv_cap_applied=False, source_certificate=old_l2)
        else:
            l2, lcert = l2_projection(native, targets, pp["rank_margin_scaled"], p3["qp_seconds"], p3["qp_eps"])
        certificates["l2"] = lcert
        if l2 is not None and lcert["success"]:
            candidates["l2"] = l2
    common = dict(theta_fit=arrays["theta_fit"], theta_dev=arrays["theta_development"], native=native,
        gain=arrays["decoder_gain"], offset=arrays["decoder_offset"], prior=arrays["prior"], data=data,
        policy=p3, base=saved["base"], profile=saved["profile"], seed=saved["seed"])
    native_row = copy.deepcopy(next(r for r in saved["rows"] if r["role"] == "native"))
    rows, acceptances = [native_row], {}
    cache = Path(root) / "data/cache/step03_replay_coherence"
    for role, label in [("primary", "Minimum-TV primary control"), ("index", "Index-secondary candidate"),
                         ("range", "Range-secondary candidate"), ("l2", "L2 projection control")]:
        if role not in candidates:
            rows.append({"role": role, "method": label, "status": "failed", "failure": "projection_not_certified",
                         "underlying_base": saved["base"], "profile": saved["profile"], "seed": saved["seed"]})
            acceptances[role] = {"accepted": False, "reasons": ["projection_not_certified"]}
            continue
        affinity = candidates[role]
        topics = topics_from_weights(affinity, data["vocabulary"])
        fit_metric = metric_score(topics, data["fit_records"], data["vocabulary"], cache)
        dev_metric = metric_score(topics, data["dev_coherence_records"], data["vocabulary"], cache)
        td = topic_diversity(topics)
        accepted, reasons = fitting_acceptance(native_row["fitting"], fit_metric,
            native_row["topic_diversity"], td, certificates[role], r3)
        acceptances[role] = {"accepted": accepted, "reasons": reasons}
        row = prediction_row(role, label, affinity, topics, fit_metric, dev_metric, td, **common)
        row.update(fitting_accepted=accepted, scope="saved_readout_repair_development_pilot",
                   new_model_fits=0, fitting_coherence_scope="in_sample_descriptive")
        rows.append(row)
    # Preserve separate native and candidate support audits. Fitting/reference
    # scopes cannot be swapped merely to turn the original scientific gate green.
    references = {"native": audit_reference(native_row["topics"], data["dev_coherence_records"], data["vocabulary"])}
    for role in ["index", "range"]:
        row = next(r for r in rows if r["role"] == role)
        if row["status"] == "scored":
            references[role] = audit_reference(row["topics"], data["dev_coherence_records"], data["vocabulary"])
    write_json(output / "reference_audit.json", references)
    write_json(output / "projection_certificates.json", certificates)
    write_json(output / "range_contexts.json", context)
    save_arrays(output / "readouts_and_latents.npz", **{n: arrays[n] for n in
        ["native", "theta_fit", "theta_development", "decoder_gain", "decoder_offset", "prior"]},
        **{("candidate_" + role): value for role, value in candidates.items()})
    result = {"base": saved["base"], "profile": saved["profile"], "seed": saved["seed"],
        "source_case_result_sha256": file_hash(original / "case_result.json"), "source_interface": saved["interface"],
        "native_replay": check, "targets": targets, "certificates": certificates, "acceptances": acceptances,
        "rows": rows, "new_model_fits": 0, "new_encoder_calls": 0, "test_or_external_used": False,
        "full_experiment_gate_passed": False}
    write_json(output / "case_result.json", result)
    write_json(output / "artifact_manifest.json", artifact_manifest(output))
    range_row = next(r for r in rows if r["role"] == "range")
    log.info("REPLAY SAVED | %s %s seed=%d | primary=%s range=%s | fitting accepted=%s | %s",
        saved["base"], saved["profile"], saved["seed"], pcert["success"], certificates.get("range", {}).get("success", False),
        acceptances.get("range", {}).get("accepted", False), output)
    return result


def summarize_replay(cases, policy):
    pp, profiles = policy["projection_pilot"], []
    chosen_by_profile = {}
    for base in policy["bases"]:
        for profile in policy["profiles"]:
            subset = [c for c in cases if c["base"] == base and c["profile"] == profile]
            if not subset:
                continue
            complete = len(subset) == 2 and {c["seed"] for c in subset} == set(policy["seeds"])
            diffs = []
            for case in subset:
                rows = {r["role"]: r for r in case["rows"]}
                if rows["range"]["status"] == "scored" and rows["index"]["status"] == "scored":
                    diffs.append(rows["index"]["prediction"]["macro_source_nll"] - rows["range"]["prediction"]["macro_source_nll"])
            adopt_range = bool(complete and len(diffs) == 2 and all(d > 0 for d in diffs)
                and statistics.mean(diffs) >= pp["secondary_minimum_mean_nll_improvement"])
            criterion = "range" if adopt_range else "index"
            chosen_by_profile[(base, profile)] = criterion
            paired = [(c, {r["role"]: r for r in c["rows"]}) for c in subset]
            scored = complete and all(rows[criterion]["status"] == "scored" for _, rows in paired)
            accepted = scored and all(c["acceptances"][criterion]["accepted"] for c, _ in paired)
            nlls, deltas, cv_gains, npmi_gains = [], [], [], []
            reference_complete = scored and all(rows["native"]["development"]["scored"] and rows[criterion]["development"]["scored"] for _, rows in paired)
            if scored:
                for c, rows in paired:
                    native, proposed = rows["native"], rows[criterion]
                    nlls.append(proposed["prediction"]["macro_source_nll"])
                    deltas.append(nlls[-1] - native["prediction"]["macro_source_nll"])
                    cv_gains.append(proposed["fitting"]["c_v"] - native["fitting"]["c_v"])
                    npmi_gains.append(proposed["fitting"]["c_npmi"] - native["fitting"]["c_npmi"])
            within_budget = scored and all(d <= pp["maximum_development_nll_damage"] for d in deltas)
            profiles.append({"underlying_base": base, "profile": profile, "complete_two_seed_rows": bool(scored),
                "chosen_secondary": criterion, "range_revision_adopted": adopt_range,
                "index_minus_range_nll_by_seed": diffs, "fitting_acceptance_both_seeds": bool(accepted),
                "fitting_cv_gains": cv_gains, "fitting_npmi_gains": npmi_gains,
                "development_nll": nlls, "development_nll_damage": deltas,
                "mean_development_nll": statistics.mean(nlls) if nlls else None,
                "mean_native_development_nll": statistics.mean(rows["native"]["prediction"]["macro_source_nll"] for _, rows in paired) if scored else None,
                "mean_development_nll_damage": statistics.mean(deltas) if deltas else None,
                "proposed_beats_matched_native_nll_both_seeds": bool(scored and all(d < 0 for d in deltas)),
                "development_prediction_within_budget": bool(within_budget),
                "complete_development_coherence_reference": bool(reference_complete),
                "descriptive_mechanism_and_predictive_budget_supported": bool(accepted and within_budget),
                "original_step03_gate_passed": False,
                "fit_coherence_does_not_establish_heldout_superiority": True})
    eligible = [p for p in profiles if p["descriptive_mechanism_and_predictive_budget_supported"]]
    best = min(eligible, key=lambda p: (p["mean_development_nll"], p["underlying_base"], p["profile"])) if eligible else None
    strongest_native = min((p["mean_native_development_nll"] for p in profiles if p["complete_two_seed_rows"]), default=None)
    numerical = len(cases) == 8 and all(all(c["certificates"].get(role, {}).get("success", False) for role in ["primary", "index", "range", "l2"]) for c in cases)
    return {"completed_cases": len(cases), "new_model_fits": 0, "new_encoder_calls": 0,
        "numerical_replay_gate_passed": numerical, "profiles": profiles,
        "candidate_base_for_research": {k: best[k] for k in ["underlying_base", "profile", "chosen_secondary"]} if best else None,
        "candidate_is_not_a_final_selected_model": True,
        "strongest_native_mean_development_nll": strongest_native,
        "candidate_beats_strongest_native_mean_nll": bool(best and strongest_native is not None and best["mean_development_nll"] < strongest_native),
        "all_metrics_winning_established": False,
        "original_development_coherence_gate_passed": bool(len(cases) == 8 and profiles and all(p["complete_development_coherence_reference"] for p in profiles)),
        "original_step03_gate_passed": False, "ready_for_step4": False,
        "full_experiment_gate_passed": False, "final_model_result": False,
        "test_or_external_used": False}, chosen_by_profile


def export_replay(folder, cases, policy):
    summary, decisions = summarize_replay(cases, policy)
    records = []
    for case in cases:
        for row in case["rows"]:
            scored = row["status"] == "scored"
            records.append({"method": row["method"], "role": row["role"], "base": case["base"],
                "profile": case["profile"], "seed": case["seed"], "status": row["status"],
                "fitting_c_v": row["fitting"]["c_v"] if scored else None,
                "fitting_c_npmi": row["fitting"]["c_npmi"] if scored else None,
                "development_c_v": row["development"]["c_v"] if scored else None,
                "development_c_npmi": row["development"]["c_npmi"] if scored else None,
                "topic_diversity": row["topic_diversity"] if scored else None,
                "macro_source_nll": row["prediction"]["macro_source_nll"] if scored else None,
                "micro_nll": row["prediction"]["micro_nll"] if scored else None,
                "projection_deployed": row.get("projection_deployed", False) if scored else False,
                "scope": "saved_readout_repair_development_pilot", "final_model_result": False,
                "rho": row["rho"] if scored else None})
        criterion = decisions[(case["base"], case["profile"])]
        selected = next(r for r in case["rows"] if r["role"] == criterion)
        if selected["status"] == "scored":
            # The selected candidate is reported as Proposed only when the
            # original fitting-only deployment checks accept it.
            accepted = case["acceptances"][criterion]["accepted"]
            if accepted:
                derived = dict(records[-len(case["rows"]) + next(i for i, r in enumerate(case["rows"]) if r["role"] == criterion)])
                derived.update(method="Proposed", role="proposed", status="deployed_research_candidate")
                records.append(derived)
    write_json(folder / "case_results.json", cases)
    write_json(folder / "summary.json", summary)
    columns = ["method", "role", "base", "profile", "seed", "status", "fitting_c_v", "fitting_c_npmi",
               "development_c_v", "development_c_npmi", "topic_diversity", "macro_source_nll", "micro_nll",
               "projection_deployed", "scope", "final_model_result", "rho"]
    write_csv(folder / "repaired_case_metrics.csv", records, columns)
    write_text(folder / "SUMMARY.md", "# Step 3 no-training numerical repair\n\n"
        f"Completed cases: {len(cases)}/8. New model fits: 0. New encoder calls: 0.\n\n"
        f"Numerical repair gate: {summary['numerical_replay_gate_passed']}. Step 4 ready: False.\n\n"
        "The original development coherence gate is retained. Fitting-reference C_v/NPMI are descriptive, in-sample results; "
        "they do not replace missing development scores. Every topic and every descriptor slot is retained.\n\n"
        "Range secondary is adopted only if corrected/backoff NLL beats index in both seeds with mean gain >=0.001. "
        "Otherwise the original index secondary is retained. This optional revision does not define whether the projection method exists.\n\n"
        "The blocked scientific gate is an evaluation outcome, not a software exception. "
        "A successful repair reports STEP03_NUMERICS_REPAIRED_RESEARCH_GATE_BLOCKED.\n\n"
        "The original index-weighted secondary objective is retained as a control. Its linear weights decompose "
        "into a row term and a column term, so conservation makes that objective constant; it cannot select a unique readout. "
        "Equivalent solvers can consequently return different certified readouts. No solution is selected by trying solvers until scores improve.\n\n"
        "Inspect repaired_case_metrics.csv, proposed_and_ablation_metrics.csv, component_diagnostics.json, "
        "summary.json, cases/*/reference_audit.json, and numerical certificates. "
        "No final model, final paper table or full-benchmark approval is created.\n")
    return summary


def assess_selected_candidates(root, output, cases, data, policy, log):
    """Restore the actual Proposed and matched ablations after two-seed selection.

    This scores saved arrays only. Display-only uses the same descriptors but
    retains the native predictor; no synthetic predictor benefit is assigned.
    """
    summary, decisions = summarize_replay(cases, policy)
    reports, table = [], []
    p3 = policy["projection_pilot"]
    for case in cases:
        folder = output / "cases" / case["base"] / case["profile"] / ("seed" + str(case["seed"]))
        with np.load(folder / "readouts_and_latents.npz", allow_pickle=False) as archive:
            arrays = {n: archive[n] for n in archive.files}
        criterion = decisions[(case["base"], case["profile"])]
        by_role = {r["role"]: r for r in case["rows"]}
        accepted = case["acceptances"][criterion]["accepted"]
        selected = by_role[criterion] if accepted else by_role["native"]
        affinity = arrays["candidate_" + criterion] if accepted else arrays["native"]
        common = dict(theta_fit=arrays["theta_fit"], theta_dev=arrays["theta_development"],
            native=arrays["native"], gain=arrays["decoder_gain"], offset=arrays["decoder_offset"],
            prior=arrays["prior"], data=data, policy=p3, base=case["base"], profile=case["profile"], seed=case["seed"])
        rows = []
        for role, label, correction, backoff in [
            ("proposed", "Proposed", True, True),
            ("proposed_uncorrected", "Proposed without moment correction", False, True),
            ("proposed_no_backoff", "Proposed without backoff", True, False)]:
            row = prediction_row(role, label, affinity, selected["topics"], selected["fitting"],
                selected["development"], selected["topic_diversity"], correction=correction,
                use_backoff=backoff, deployed=accepted, **common)
            row.update(scope="saved_readout_repair_development_pilot", fitting_coherence_scope="in_sample_descriptive",
                       fitting_accepted=accepted, chosen_secondary=criterion,
                       deployment_status="deployed_research_candidate" if accepted else "abstained_native_fallback")
            rows.append(row)
        display = copy.deepcopy(by_role["native"])
        display.update(role="display_only", method="Same-descriptor display-only", topics=selected["topics"],
            fitting=selected["fitting"], development=selected["development"], topic_diversity=selected["topic_diversity"],
            projection_deployed=False, readout_is_native=True, descriptor_sets_from_proposed=True,
            scope="saved_readout_repair_development_pilot", fitting_coherence_scope="in_sample_descriptive")
        if display["document_losses"] != by_role["native"]["document_losses"] or display["prediction"] != by_role["native"]["prediction"]:
            raise AssertionError("Display-only altered the native predictor.")
        rows.append(display)
        if abs(rows[0]["prediction"]["macro_source_nll"] - selected["prediction"]["macro_source_nll"]) > 1e-10:
            raise AssertionError("Selected Proposed does not replay its candidate prediction.")
        report = {"base": case["base"], "profile": case["profile"], "seed": case["seed"],
            "chosen_secondary": criterion, "projection_deployed": accepted, "rows": rows,
            "numerical_certificate_path": folder.relative_to(output).as_posix() + "/projection_certificates.json",
            "final_model_result": False, "ready_for_step4": False, "test_or_external_used": False}
        reports.append(report)
        destination = output / "selected_candidate_assessments" / case["base"] / case["profile"] / ("seed" + str(case["seed"]))
        write_json(destination / "assessment.json", report)
        save_arrays(destination / "selected_readout.npz", proposed=affinity, native=arrays["native"])
        write_json(destination / "artifact_manifest.json", artifact_manifest(destination))
        for row in rows:
            table.append({"method": row["method"], "role": row["role"], "base": case["base"],
                "profile": case["profile"], "seed": case["seed"], "chosen_secondary": criterion,
                "projection_deployed": row["projection_deployed"],
                "fitting_c_v": row["fitting"]["c_v"], "fitting_c_npmi": row["fitting"]["c_npmi"],
                "development_c_v": row["development"]["c_v"], "development_c_npmi": row["development"]["c_npmi"],
                "topic_diversity": row["topic_diversity"], "macro_source_nll": row["prediction"]["macro_source_nll"],
                "micro_nll": row["prediction"]["micro_nll"], "rho": row["rho"], "final_model_result": False})
        log.info("PROPOSED SAVED | base=%s profile=%s seed=%d | fitting Cv=%.6f NPMI=%.6f | development NLL=%.9f | deployed=%s | final result=False",
            case["base"], case["profile"], case["seed"], selected["fitting"]["c_v"], selected["fitting"]["c_npmi"],
            rows[0]["prediction"]["macro_source_nll"], accepted)
    write_json(output / "selected_candidate_rows.json", reports)
    write_csv(output / "proposed_and_ablation_metrics.csv", table, ["method", "role", "base", "profile", "seed",
        "chosen_secondary", "projection_deployed", "fitting_c_v", "fitting_c_npmi", "development_c_v",
        "development_c_npmi", "topic_diversity", "macro_source_nll", "micro_nll", "rho", "final_model_result"])
    diagnostics = []
    for base in policy["bases"]:
        for profile in policy["profiles"]:
            subset = [c for c in reports if c["base"] == base and c["profile"] == profile]
            if not subset:
                continue
            pairs = [{r["role"]: r for r in c["rows"]} for c in subset]
            complete = len(subset) == 2 and {c["seed"] for c in subset} == set(policy["seeds"])
            values = {role: [rows[role]["prediction"]["macro_source_nll"] for rows in pairs]
                      for role in ["proposed", "proposed_uncorrected", "proposed_no_backoff", "display_only"]}
            diagnostics.append({"base": base, "profile": profile, "complete_two_seed_rows": complete,
                "macro_source_nll_by_role_and_seed": values,
                "mean_macro_source_nll_by_role": {k: statistics.mean(v) if complete else None for k, v in values.items()},
                "no_correction_beats_native_both_seeds": bool(complete and all(a < b for a, b in zip(values["proposed_uncorrected"], values["display_only"]))),
                "no_correction_beats_corrected_both_seeds": bool(complete and all(a < b for a, b in zip(values["proposed_uncorrected"], values["proposed"]))),
                "full_proposed_equals_no_backoff_both_seeds": bool(complete and all(rows["proposed"]["rho"] == 0 for rows in pairs)),
                "correction_policy_changed": False, "no_correction_control_relabelled_as_proposed": False,
                "display_only_has_identical_descriptor_metrics": True,
                "heldout_coherence_superiority_established": False, "ready_for_step4": False})
    write_json(output / "component_diagnostics.json", diagnostics)
    return reports


def replay_all(root, source, prepared, output, configs, log):
    policy, repair = validate_repair(configs)
    data = load_canonical(prepared)
    cases = verify_saved_source(source, data, policy)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for saved in cases:
        result = replay_case(root, source, saved, data, policy, repair, configs["step02_refinement"],
            output / "cases" / saved["base"] / saved["profile"] / ("seed" + str(saved["seed"])), log)
        results.append(result)
        export_replay(output, results, policy)
        log.info("REPLAY PROGRESS | saved %d/8 | %s", len(results), output / "repaired_case_metrics.csv")
    summary = export_replay(output, results, policy)
    assess_selected_candidates(root, output, results, data, policy, log)
    return summary


def run(root, configs, versions, hardware, log):
    root = Path(root).resolve()
    policy, repair = validate_repair(configs)
    preparation = require_preparation(root)
    source = source_folder(root)
    check_indexed_tree(source)
    source_before = file_hash(source / "artifact_manifest.json")
    if read_json(source / "run_result.json")["preparation_id"] != preparation["preparation_id"]:
        raise ValueError("Saved neural pilot used a different accepted preparation.")
    implementation = {p.name: file_hash(p) for p in [Path(__file__), Path(__file__).parents[1] / "projection/replay_solver.py",
                                                   Path(__file__).parents[1] / "evaluation/reference_audit.py"]}
    identity = {"repair_policy": repair, "source_manifest_sha256": source_before,
                "preparation_id": preparation["preparation_id"], "implementation": implementation,
                "numpy": importlib.metadata.version("numpy"), "scipy": importlib.metadata.version("scipy"),
                "gensim": importlib.metadata.version("gensim"), "osqp": importlib.metadata.version("osqp")}
    key = digest(identity)
    output = root / "outputs/steps/step03_saved_replay" / key[:20]
    output.mkdir(parents=True, exist_ok=True)
    write_json(log.folder / "preparation_working_path.json", {"relative_path": output.relative_to(root).as_posix()})
    write_json(output / "replay_protocol.json", identity)
    write_json(output / "source_neural_pilot_reference.json", {"relative_path": source.relative_to(root).as_posix(),
        "artifact_manifest_sha256": source_before, "new_model_fits": 0, "new_encoder_calls": 0})
    write_json(root / "outputs/steps/step03_saved_replay/LATEST.json", {"relative_path": output.relative_to(root).as_posix()})
    log.info("STEP 3 REPAIR | saved native arrays only | neural fits=0 | encoder calls=0 | solver=CPU")
    summary = replay_all(root, source, root / preparation["processed_relative_path"], output, configs, log)
    if file_hash(source / "artifact_manifest.json") != source_before:
        raise AssertionError("Repair changed the original neural-pilot index.")
    check_indexed_tree(source)
    result = {"status": "STEP03_NUMERICS_REPAIRED_RESEARCH_GATE_BLOCKED" if summary["numerical_replay_gate_passed"] else "STEP03_REPAIR_INCOMPLETE",
        "engineering_passed": summary["numerical_replay_gate_passed"], "preparation_gate_passed": True,
        "protocol": PROTOCOL, "preparation_id": preparation["preparation_id"], "step03_repair_id": key,
        "model_quality_established": False, "prediction_evaluated": True, "completed_cases": summary["completed_cases"],
        "new_model_fits": 0, "new_encoder_calls": 0, "candidate_base_for_research": summary["candidate_base_for_research"],
        "original_step03_gate_passed": False, "ready_for_step4": False, "full_experiment_gate_passed": False,
        "final_model_result": False, "test_or_external_used": False, "processed_relative_path": output.relative_to(root).as_posix()}
    write_json(output / "run_result.json", result)
    write_json(output / "artifact_manifest.json", artifact_manifest(output))
    write_json(root / "state/step03_repair_result.json", {**result, "artifacts": read_json(output / "artifact_manifest.json")})
    log.info("SAVED RESULTS | %s", output)
    log.info("READ FIRST | %s", output / "SUMMARY.md")
    log.info("STEP 3 REPAIR | numerics=%s | development coherence complete=%s | Step 4 allowed=False",
        summary["numerical_replay_gate_passed"], summary["original_development_coherence_gate_passed"])
    return result, output


REPAIR_POLICY_SHA256 = "c5d2d6872b460bd47424dfeea55e2530888155f92512d54a74fd8a54b2021888"
