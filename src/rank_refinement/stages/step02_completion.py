"""Finalize the completed exploratory pilot without fitting or modifying it.

The user's revised authorization permits a bounded neural follow-up despite
incomplete descriptor references in the classical pilot. This handoff gate is
separate from the original R3 acquisition/reference and final-benchmark gates.
"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import statistics
import numpy as np
from ..models.adapters.pilot_anchors import topics_from_weights
from ..projection.minimum_tv import certify
from ..reporting.result_labels import result_identity
from ..runtime.artifacts import (artifact_manifest, digest, file_hash, read_json,
    verify_artifacts, write_csv, write_json, write_text)
from ..runtime.state import require_preparation
from .step02_native_refinement import validate_protocol, summarize, fitting_acceptance

SCOPE = "exploratory_readout_mechanism_pilot"
PROTOCOL = "STEP02_R3_COMPLETION_R1_20261009"


def safe_source_path(root, relative):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("Invalid source pilot path.")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 4:
        raise ValueError("Unsafe source pilot path.")
    if rel.parts[:3] != ("outputs", "steps", "step02_native_refinement"):
        raise ValueError("Completion must use a saved Step 2 native-refinement pilot.")
    path = (Path(root) / rel).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_dir():
        raise ValueError("Saved Step 2 source folder is missing or unsafe.")
    return path


def check_indexed_tree(folder):
    """Require complete indexed payloads, including size checks and safe paths."""
    records = read_json(folder / "artifact_manifest.json")
    expected = {p.relative_to(folder).as_posix() for p in folder.rglob("*")
                if p.is_file() and p.name != "artifact_manifest.json"}
    indexed = [r["path"] for r in records]
    if len(set(indexed)) != len(indexed) or set(indexed) != expected:
        raise ValueError("Saved pilot has missing, extra or duplicate indexed payloads.")
    verify_artifacts(folder, records)
    for entry in records:
        if (folder / entry["path"]).stat().st_size != entry["bytes"]:
            raise ValueError("Saved artifact size differs: " + entry["path"])
    return records


def expected_case_keys(protocol):
    return {(rep, base, fold, seed) for rep in protocol["representations"]
            for base in protocol["bases"] for fold in protocol["folds"] for seed in protocol["seeds"]}


def check_case_grid(results, protocol):
    keys = [tuple(r[k] for k in ["representation", "base", "fold", "seed"]) for r in results]
    if len(keys) != 24 or len(set(keys)) != 24 or set(keys) != expected_case_keys(protocol):
        raise ValueError("Finalization requires the exact 24 unique planned cases.")


def compare_saved_summary(results, protocol, saved):
    derived = summarize(results, protocol)
    for key, value in derived.items():
        if saved.get(key) != value:
            raise ValueError("Saved summary differs from its individual cases: " + key)
    return derived


def check_feature_groups(records, expected_fit, expected_validation):
    groups = [r["group_id"] for r in records]
    if len(set(groups)) != len(groups) or any(r["split"] not in ["fit", "validation"] for r in records):
        raise ValueError("Saved feature view contains duplicate or undeclared groups.")
    if ({r["group_id"] for r in records if r["split"] == "fit"} != set(expected_fit)
            or {r["group_id"] for r in records if r["split"] == "validation"} != set(expected_validation)
            or set(expected_fit) & set(expected_validation)):
        raise ValueError("Saved feature view differs from the accepted fitting-only internal fold.")


def validate_completion_config(configs):
    policy = configs.get("step02_completion", {})
    required = {"protocol": PROTOCOL, "source_protocol": "R3_NATIVE_TOP10_DIVERSITY_20261009",
                "scope": "bounded_neural_pilot_handoff", "retrain_models": False,
                "allow_partial_reference_for_handoff": True,
                "allow_partial_reference_for_benchmark": False,
                "reported_proposed_name": "Proposed", "next_native_fit_cap": 8,
                "next_final_base_count": 1}
    if {k: policy.get(k) for k in required} != required:
        raise ValueError("The declared completion/handoff protocol changed.")
    names = {"stages/step02_native_refinement.py", "acquisition/native_targets.py", "projection/minimum_tv.py",
             "models/adapters/pilot_anchors.py", "evaluation/coherence.py", "acquisition/windows.py",
             "acquisition/associations.py", "data/vocabulary.py", "stages/step02_acquisition_pilot.py",
             "runtime/artifacts.py"}
    if set(policy.get("scientific_source_sha256", {})) != names:
        raise ValueError("Missing scientific-source provenance guard.")
    return policy


def inspect_case(case, source, protocol, records, metadata):
    rep, base, fold, seed = [case[k] for k in ["representation", "base", "fold", "seed"]]
    case_folder = source / "cases" / rep / base / f"fold{fold:02d}_seed{seed}"
    check_indexed_tree(case_folder)
    if read_json(case_folder / "result.json") != case:
        raise ValueError("Individual result differs from saved case_results.json.")
    if case.get("scope") != SCOPE or case.get("prediction_evaluated") is not False or case.get("validation_used_for_acceptance") is not False:
        raise ValueError("Case scope or fitting/validation firewall declaration changed.")
    fitting = {r["group_id"]: r for r in records if r["split"] == "fit"}
    validation = {r["group_id"] for r in records if r["split"] == "validation"}
    if not fitting or not validation or set(fitting) & validation:
        raise ValueError("Invalid internal fitting/validation groups.")
    if metadata["fit_group_ids_sha256"] != digest(sorted(fitting)):
        raise ValueError("Saved vocabulary is not fitted on this fitting fold.")
    samples = read_json(case_folder / "sampling_group_ids.json")
    if len(samples) != protocol["acquisition"]["subsamples"]:
        raise ValueError("Wrong number of fitting association subsamples.")
    expected_sources = {s: int(n * protocol["acquisition"]["source_stratified_fraction"])
                        for s, n in Counter(r["source"] for r in fitting.values()).items()}
    for sample in samples:
        if len(set(sample)) != len(sample) or not set(sample).issubset(fitting):
            raise ValueError("Association sample contains a duplicate or non-fitting group.")
        if dict(Counter(fitting[g]["source"] for g in sample)) != expected_sources:
            raise ValueError("Association sample violates source stratification.")
    with np.load(case_folder / "readout.npz", allow_pickle=False) as archive:
        native, final = archive["native"], archive["final"]
    shape = (10, len(metadata["tokens"]))
    if native.shape != shape or final.shape != shape or not np.isfinite(native).all() or not np.isfinite(final).all():
        raise ValueError("Invalid saved readout shape or nonfinite affinity.")
    if (native < 0).any() or (final < 0).any():
        raise ValueError("A saved affinity is negative.")
    if topics_from_weights(native, metadata["tokens"]) != case["native_topics"] or topics_from_weights(final, metadata["tokens"]) != case["final_topics"]:
        raise ValueError("Saved descriptors differ from the actual saved readout.")
    selection = read_json(case_folder / "selection.json")
    if selection["complete_target_sets"] != case["targets"] or not selection["allocation_verified"] or not selection["only_fitting_groups_used"]:
        raise ValueError("Target allocation provenance mismatch.")
    native_td = len(set(sum(case["native_topics"], []))) / 100
    final_td = len(set(sum(case["final_topics"], []))) / 100
    if native_td != case["native_td"] or final_td != case["final_td"] or final_td + 1e-12 < native_td:
        raise ValueError("Actual descriptor diversity differs or decreases.")
    if case["fitting_accepted"]:
        p = protocol["projection"]
        certificate = certify(native, final, case["targets"], p["rank_margin_scaled"],
            p["certificate_tolerance_scaled"], case["certificate"]["solver_tv_scaled"], p["primary_objective_tolerance_scaled"])
        if not certificate["certified"]:
            raise ValueError("Saved deployed projection fails independent certification.")
        if not fitting_acceptance(case["native_fitting"], case["candidate_fitting"],
            native_td, final_td, {**certificate, "success": True}, protocol)[0]:
            raise ValueError("Saved deployed edit fails the original fitting-only acceptance criteria.")
        if abs(certificate["relative_l1_movement"] - case["relative_l1_final"]) > 1e-10:
            raise ValueError("Saved movement differs from the actual projected readout.")
    else:
        if not np.array_equal(native, final) or case["relative_l1_final"] != 0:
            raise ValueError("A rejected edit changed the native readout.")
        certificate = {"certified": False, "rejected_edit_exact_native_retained": True}
    support = []
    for role, topics in [("native", case["native_topics"]), ("proposed", case["final_topics"])]:
        score = case["native_validation"] if role == "native" else case["final_validation"]
        absent = sorted(set(sum(topics, [])) - {w for r in records if r["split"] == "validation" for w in r["tokens"]})
        if absent != score["absent_words"]:
            raise ValueError("Saved reference-support report differs from validation texts.")
        for word in absent:
            support.append({"representation": rep, "base": base, "fold": fold, "seed": seed,
                **result_identity(role, base, SCOPE), "word": word,
                "fitting_group_df": sum(word in r["tokens"] for r in fitting.values()), "validation_group_df": 0})
    return certificate, support


def long_results(results):
    rows, topics = [], []
    for case in results:
        key = {k: case[k] for k in ["representation", "fold", "seed"]}
        for role in ["native", "proposed"]:
            score = case["native_validation"] if role == "native" else case["final_validation"]
            identity = result_identity(role, case["base"], SCOPE)
            td = case["native_td"] if role == "native" else case["final_td"]
            rows.append({**key, **identity, "scored": score["scored"], "c_v": score["c_v"],
                "npmi": score["c_npmi"], "td": td, "prediction_evaluated": False,
                "final_proposed_model_result": False,
                "relative_l1_movement": 0.0 if role == "native" else case["relative_l1_final"],
                "reference_absent_words": " | ".join(score["absent_words"]), "coherence_failure": score["failure"]})
            for number, words in enumerate(case["native_topics"] if role == "native" else case["final_topics"], 1):
                topics.append({**key, **identity, "topic": number, "top10": " | ".join(words)})
    return rows, topics


def comparison_rows(results, summary):
    rows = []
    for p in summary["profiles"]:
        cases = [r for r in results if r["representation"] == p["representation"] and r["base"] == p["base"]]
        for role, prefix, td_key in [("native", "native", "native_td"), ("proposed", "refined", "final_td")]:
            rows.append({"representation": p["representation"], **result_identity(role, p["base"], SCOPE),
                "planned_cases": 6, "matched_finite_cases": p["matched_finite_cases"],
                "unscorable_matched_cases": p["unscorable_cases"],
                "mean_cv_matched_diagnostic": p[prefix + "_cv_finite_diagnostic"],
                "mean_npmi_matched_diagnostic": p[prefix + "_npmi_finite_diagnostic"],
                "mean_td_all_cases": statistics.mean(r[td_key] for r in cases),
                "td_cases": len(cases), "final_paper_benchmark_row": False})
    return rows


def require_pilot_handoff(root):
    """Read-only guard for the later bounded stage; never authorizes a benchmark."""
    root = Path(root).resolve()
    path = root / "state/step02_pilot_accepted.json"
    if not path.is_file():
        raise RuntimeError("Finalize Step 2 before starting the bounded neural pilot.")
    state = read_json(path)
    if (state.get("completion_protocol") != PROTOCOL or state.get("step02_pilot_gate_passed") is not True
            or state.get("bounded_neural_pilot_allowed") is not True
            or state.get("handoff_scope") != "bounded_neural_predictive_pilot_only"
            or state.get("full_experiment_gate_passed") is not False
            or state.get("final_paper_benchmark_ready") is not False):
        raise ValueError("Invalid bounded-pilot acceptance scope.")
    preparation = require_preparation(root)
    if preparation["preparation_id"] != state["preparation_id"]:
        raise ValueError("Accepted preparation changed after the Step 2 handoff.")
    folder = (root / state["processed_relative_path"]).resolve()
    if not folder.is_relative_to(root / "outputs/steps/step02_completion"):
        raise ValueError("Unsafe accepted completion path.")
    verify_artifacts(folder, state["artifacts"])
    if read_json(folder / "completion_result.json") != {k: v for k, v in state.items() if k not in ["artifacts", "source_manifest_sha256"]}:
        raise ValueError("Accepted completion result differs from its state.")
    source = safe_source_path(root, state["source_processed_relative_path"])
    if file_hash(source / "artifact_manifest.json") != state["source_manifest_sha256"]:
        raise ValueError("The source pilot index changed after acceptance.")
    check_indexed_tree(source)
    return state


def run(root, configs, versions, hardware, log):
    root = Path(root).resolve()
    protocol, policy = validate_protocol(configs), validate_completion_config(configs)
    source_dir = Path(__file__).resolve().parents[1]
    for name, expected in policy["scientific_source_sha256"].items():
        if file_hash(source_dir / name) != expected:
            raise ValueError("Scientific pilot source changed; do not finalize a different method as the reviewed run: " + name)
    preparation = require_preparation(root)
    step01_before = file_hash(root / "state/step01_accepted.json")
    pointer = root / "outputs/steps/step02_native_refinement/LATEST.json"
    if not pointer.is_file():
        raise RuntimeError("No completed Step 2 LATEST.json. Keep your existing outputs folder. If no Step 2 run exists, run python main.py --step 2 --device auto first.")
    source = safe_source_path(root, read_json(pointer)["relative_path"])
    source_index = check_indexed_tree(source)
    snapshot, saved = read_json(source / "protocol_snapshot.json"), read_json(source / "summary.json")
    provenance = read_json(source / "input_provenance.json")
    if snapshot != protocol:
        raise ValueError("Saved pilot protocol differs from the frozen configuration.")
    if saved["preparation_id"] != preparation["preparation_id"] or provenance["preparation_id"] != preparation["preparation_id"]:
        raise ValueError("Saved Step 2 used a different accepted preparation.")
    if saved["processed_relative_path"] != source.relative_to(root).as_posix() or saved["step02_id"] != provenance["signature"]:
        raise ValueError("Saved source identity mismatch.")
    if provenance["accepted_step01_state_sha256"] != step01_before or provenance["primary_inputs"] != read_json(root / preparation["processed_relative_path"] / "input_manifest.json"):
        raise ValueError("Saved pilot input provenance differs from the accepted preparation.")
    if provenance["primary_development_test_scored"] is not False or provenance["external_data_used"] is not False or provenance["canonical_step01_edited"] is not False:
        raise ValueError("The source pilot used an undeclared scope.")
    results = read_json(source / "case_results.json")
    check_case_grid(results, protocol)
    derived = compare_saved_summary(results, protocol, saved)
    log.info("FINALIZE | existing 24-case Step 2 pilot | new model fits=0 | unchanged metrics and NA cases")
    feature_views = {}
    accepted_folds = {r["fold"]: r for r in read_json(root / preparation["processed_relative_path"] / "internal_folds.json")}
    for rep in protocol["representations"]:
        for fold in protocol["folds"]:
            folder = source / "features" / rep / f"fold{fold:02d}"
            rows = [json.loads(s) for s in (folder / "feature_records.jsonl").read_text(encoding="utf-8").splitlines()]
            check_feature_groups(rows, accepted_folds[fold]["fit_group_ids"], accepted_folds[fold]["validation_group_ids"])
            feature_views[rep, fold] = rows, read_json(folder / "vocabulary.json")
    verified, absent = [], []
    for number, case in enumerate(results, 1):
        certificate, missing = inspect_case(case, source, protocol, *feature_views[case["representation"], case["fold"]])
        verified.append({"representation": case["representation"], "base": case["base"], "fold": case["fold"],
            "seed": case["seed"], "fitting_accepted": case["fitting_accepted"], "verification": certificate})
        absent.extend(missing)
        log.info("VERIFIED %d/24 | Proposed | base=%s representation=%s fold=%d seed=%d | deployment=%s",
            number, case["base"], case["representation"], case["fold"], case["seed"], case["fitting_accepted"])
    verify_artifacts(root / preparation["processed_relative_path"], preparation["artifacts"])
    if file_hash(root / "state/step01_accepted.json") != step01_before:
        raise ValueError("Accepted preparation changed during finalization.")
    source_manifest_hash = file_hash(source / "artifact_manifest.json")
    source_code = file_hash(Path(__file__))
    identity = digest({"source_manifest": source_manifest_hash, "source_id": saved["step02_id"],
        "completion_protocol": policy, "completion_source_sha256": source_code,
        "labels_source_sha256": file_hash(source_dir / "reporting/result_labels.py")})
    destination = root / "outputs/steps/step02_completion" / identity[:20]
    destination.mkdir(parents=True, exist_ok=True)
    write_json(log.folder / "preparation_working_path.json", {"relative_path": destination.relative_to(root).as_posix()})
    permitted = bool(derived["engineering_passed"] and derived["mechanism_promising"] and derived["diversity_losses_in_deployed_readouts"] == 0)
    completed = {**saved, "status": "STEP02_PILOT_ACCEPTED" if permitted else "STEP02_COMPLETED_REVIEW_NEEDED",
        "completion_protocol": PROTOCOL, "completion_id": identity,
        "processed_relative_path": destination.relative_to(root).as_posix(),
        "source_processed_relative_path": source.relative_to(root).as_posix(),
        "step02_pilot_gate_passed": permitted, "bounded_neural_pilot_allowed": permitted,
        "full_experiment_gate_passed": False, "model_quality_established": False,
        "final_paper_benchmark_ready": False, "final_proposed_model_result": False,
        "new_model_fits": 0, "original_metrics_changed": False, "reported_proposed_name": "Proposed",
        "independently_reverified_cases": len(verified), "next_action": "bounded_encot_neuromax_predictive_pilot" if permitted else "review_saved_pilot",
        "handoff_scope": "bounded_neural_predictive_pilot_only" if permitted else "none",
        "original_reference_and_acquisition_flags_preserved": True}
    handoff = {"protocol": PROTOCOL, "allowed": permitted, "allowed_scope": completed["handoff_scope"],
        "reported_model": "Proposed", "native_candidates": ["EnCOT", "NeuroMax"], "select_final_base_count": 1,
        "maximum_new_native_fits": policy["next_native_fit_cap"], "pilot_training_seeds": [809, 810],
        "preferred_neural_device": hardware["future_model_device"], "gpu_used_by_this_completion": False,
        "native_interfaces_verified": False, "prediction_evaluated": False,
        "controls_required": ["matched native", "same-word display-only", "original secondary", "L2 projection"],
        "select_and_accept_using_actual_full_pipeline_development_nll": True,
        "primary_test_and_external_reserved": True, "negative_npmi_is_not_a_software_failure": True,
        "complete_reference_required_before_final_benchmark_claim": True,
        "full_experiment_authorized": False, "step03_implementation_included": False}
    rows, topic_rows = long_results(results)
    for name, data in [("model_results.csv", rows), ("comparison_diagnostic.csv", comparison_rows(results, derived)),
                       ("topic_words.csv", topic_rows)]:
        write_csv(destination / name, data, list(data[0]))
    missing_columns = ["representation", "base", "fold", "seed", "model", "method_role", "underlying_base", "base_key",
        "result_scope", "word", "fitting_group_df", "validation_group_df"]
    write_csv(destination / "missing_reference_words.csv", absent, missing_columns)
    write_json(destination / "verification.json", {"cases": verified, "source_payloads_verified": len(source_index),
        "training_performed": False, "metric_recomputation_performed": False,
        "saved_case_metrics_preserved": True, "reference_support_independently_rechecked": True})
    write_json(destination / "source_result_reference.json", {"relative_path": source.relative_to(root).as_posix(),
        "source_manifest_sha256": source_manifest_hash, "source_step02_id": saved["step02_id"]})
    write_json(destination / "source_summary.json", saved)
    write_json(destination / "completion_protocol.json", policy)
    write_json(destination / "step03_handoff.json", handoff)
    write_json(destination / "completion_result.json", completed)
    lines = ["# Step 2 completion", "", "Status: " + completed["status"], "",
        "The completed classical readout pilot is accepted for a bounded neural predictive follow-up. Full benchmarking is not approved." if permitted else "The completed pilot needs review before further fitting.", "",
        "No models were retrained. Original scores and all missing-reference cases were preserved.",
        "Proposed labels the acquisition-plus-projection readout. Underlying base and exploratory scope are recorded separately.",
        "These are not the final proposed neural-model results. EnCOT and NeuroMax have not been evaluated by this command.", "",
        f"Cases: 24/24; matched finite comparisons: {saved['matched_finite_cases']}/24; fitting-accepted edits: {saved['fitting_accepted_cases']}/24.",
        f"Both-coherence improvements: {sum(p['both_coherence_wins'] for p in saved['profiles'])}/{saved['matched_finite_cases']} matched finite cases; diversity losses: {saved['diversity_losses_in_deployed_readouts']}.",
        f"Complete reference: {saved['complete_coherence_reference_gate_passed']}; original acquisition gate: {saved['acquisition_gate_passed']}; NLL: not measured.", "",
        "The limited follow-up approval is a separate operational gate, authorized after reviewing the pilot. It does not turn a failed scientific/reference gate into a pass.", "",
        "| Representation | Underlying base | Proposed C_v | Proposed NPMI | Matched finite / planned |",
        "|---|---|---:|---:|---:|"]
    for p in saved["profiles"]:
        fmt = lambda x: "NA" if x is None else f"{x:.6f}"
        lines.append(f"| {p['representation']} | {p['base']} | {fmt(p['refined_cv_finite_diagnostic'])} | {fmt(p['refined_npmi_finite_diagnostic'])} | {p['matched_finite_cases']}/6 |")
        log.info("PROPOSED PILOT | underlying_base=%s representation=%s | matched=%d/6 Cv=%s NPMI=%s",
            p["base"], p["representation"], p["matched_finite_cases"], fmt(p["refined_cv_finite_diagnostic"]), fmt(p["refined_npmi_finite_diagnostic"]))
    lines += ["", "Means use matched finite cases only. They are diagnostic, not complete final-paper benchmark means.", "",
        "Next implementation: test both verified neural interfaces within the eight-fit budget; retain one selected final base; assess actual full correction/backoff prediction, matched native and the declared controls. Keep test and external results reserved.", "",
        "Files: model_results.csv (48 native/Proposed case rows), comparison_diagnostic.csv, topic_words.csv, missing_reference_words.csv, verification.json, step03_handoff.json, completion_result.json."]
    write_text(destination / "SUMMARY.md", "\n".join(lines) + "\n")
    write_json(destination / "artifact_manifest.json", artifact_manifest(destination))
    if file_hash(source / "artifact_manifest.json") != source_manifest_hash:
        raise ValueError("Source pilot changed during finalization.")
    # Accept only after every original payload and deployed readout was checked.
    if permitted:
        write_json(root / "state/step02_pilot_accepted.json", {**completed,
            "artifacts": read_json(destination / "artifact_manifest.json"),
            "source_manifest_sha256": source_manifest_hash})
    write_json(root / "outputs/steps/step02_completion/LATEST.json", {
        "relative_path": destination.relative_to(root).as_posix(), "completion_id": identity, "run_id": log.run_id})
    log.info("SAVED RESULTS | %s", destination)
    log.info("READ FIRST | %s", destination / "SUMMARY.md")
    return completed, destination
