"""Explicit exploratory full-set revision, reusing the completed neural pilot.

Every descriptor is scored from the actual certified affinity. Baselines are
reused only from the indexed R2 completion with identical data/metric contracts.
Selection uses fitting descriptors and development prediction; test/external
evaluation is never performed. A quality failure is saved, never suppressed.
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
from ..acquisition.complete_targets import choose_complete_targets
from ..projection.replay_solver import primary_tv, index_secondary, range_secondary
from ..projection.predictive_controls import l2_projection
from ..projection.minimum_tv import certify
from ..evaluation.diversity import topic_diversity
from .step02_completion import check_indexed_tree
from .step02_native_refinement import save_arrays
from .step03_neural_pilot import prediction_row
from .step03_saved_replay import source_folder, case_folder
from .step03_completion import (validate_completion, repaired_source_folder, verify_sources,
    reference_scores, flat_row, aggregate_rows)

PROTOCOL = "STEP03_FULLSET_R3_20261009"
POLICY_SHA256 = "08d895844e51da0944ff2c92623d5bd85a894f40a85a8d1be55cacfef3e6872f"


def validate_improvement(configs):
    native, completion = validate_completion(configs)
    policy = configs.get("step03_improvement", {})
    if policy.get("protocol") != PROTOCOL or digest(policy) != POLICY_SHA256:
        raise ValueError("Full-set policy changed; use a named, recorded research revision.")
    if (policy["source_protocol"] != completion["protocol"] or policy["moment_correction"]
            or policy["seeds"] != native["seeds"] or policy["test_or_external_used"]):
        raise ValueError("Incorrect full-set revision scope.")
    return native, policy


def completion_source_folder(root):
    root = Path(root).resolve()
    pointer = root / "outputs/steps/step03_completion/LATEST.json"
    state = root / "state/step03_completion_result.json"
    if pointer.is_file():
        relative = read_json(pointer)["relative_path"]
    elif state.is_file():
        relative = read_json(state)["processed_relative_path"]
    else:
        raise RuntimeError("Keep your completed Step 3 --complete outputs and state. If absent, run --step 3 --complete first.")
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("Unsafe completion source path.")
    parts = Path(relative)
    if (parts.is_absolute() or ".." in parts.parts or len(parts.parts) != 4
            or parts.parts[:3] != ("outputs", "steps", "step03_completion")):
        raise ValueError("Unsafe completion source path.")
    source = (root / parts).resolve()
    if not source.is_relative_to(root) or not source.is_dir():
        raise ValueError("Completed R2 pilot folder is missing or outside the project.")
    return source


def verify_completion_source(source, original, repaired, preparation_id):
    check_indexed_tree(source)
    result = read_json(source / "run_result.json")
    if (result.get("protocol") != "STEP03_COMPLETION_R2_20261009" or not result.get("engineering_passed")
            or result["preparation_id"] != preparation_id):
        raise ValueError("Require an engineering-passed R2 completion on the accepted data.")
    snapshot = read_json(source / "completion_protocol_resolved_before_scoring.json")
    from .step03_completion import POLICY_SHA256 as old_hash
    if digest(snapshot["protocol"]) != old_hash:
        raise ValueError("R2 completion policy does not match the frozen source contract.")
    for folder, name in [(original, "source_neural_pilot_reference.json"),
                         (repaired, "repaired_saved_source_reference.json")]:
        ref = read_json(source / name)
        if (ref["relative_path"] != folder.relative_to(folder.parents[3]).as_posix()
                or ref["artifact_manifest_sha256"] != file_hash(folder / "artifact_manifest.json")):
            raise ValueError("Completed R2 pilot references a different saved model source.")
    summary = read_json(source / "summary.json")
    chosen = summary["selected_candidate"]
    if not chosen or (chosen["base"], chosen["profile"]) != ("neuromax", "random_words"):
        raise ValueError("This bounded revision requires the already selected NeuroMax/random_words pilot. Other sources need a declared revision.")
    return read_json(source / "case_results.json"), summary


def load_evidence(folder, fitting):
    with np.load(folder / "association_evidence.npz", allow_pickle=False) as z:
        evidence = {n: z[n] for n in z.files}
    evidence.update(read_json(folder / "association_words.json"))
    fitting_ids = {r["group_id"] for r in fitting}
    if len(fitting_ids) != len(fitting) or any(r["split"] != "fit" for r in fitting):
        raise ValueError("Require unique fitting groups only.")
    samples = read_json(folder / "sampling_group_ids.json")
    if (evidence["fit_groups"] != len(fitting) or len(samples) != 20
            or any(len(ids) != len(set(ids)) or not set(ids).issubset(fitting_ids) for ids in samples)):
        raise ValueError("Saved calibration sample contains foreign or duplicate evidence units.")
    return evidence


def failed_row(role, method, seed, failure, policy):
    return {"role": role, "method": method, "underlying_base": policy["base"], "profile": policy["profile"],
        "seed": seed, "status": "failed", "failure": failure, "topics": None,
        "fitting": {"c_v": None, "c_npmi": None}, "development": {"c_v": None, "c_npmi": None},
        "topic_diversity": None, "prediction": {"macro_source_nll": None, "micro_nll": None},
        "rho": None, "moment_correction": False, "projection_deployed": False,
        "final_model_result": False, "test_or_external_used": False}


def run_case(root, original, repaired, old_case, data, native_policy, policy, destination, log, identity):
    if (destination / "artifact_manifest.json").is_file():
        check_indexed_tree(destination)
        cached = read_json(destination / "case_result.json")
        if cached["identity"] != identity:
            raise ValueError("Full-set case cache differs from its declared inputs.")
        log.info("FULLSET CACHE | seed=%d | all indexed outputs verified", old_case["seed"])
        return cached
    seed = old_case["seed"]
    destination.mkdir(parents=True, exist_ok=True)
    saved = case_folder(repaired, old_case)
    original_case = case_folder(original, old_case)
    with np.load(saved / "readouts_and_latents.npz", allow_pickle=False) as z:
        arrays = {n: z[n] for n in z.files}
    evidence = load_evidence(original_case, data["fit_records"])
    targets, selection = choose_complete_targets(arrays["native"], data["vocabulary"], data["metadata"], evidence, policy["acquisition"])
    write_json(destination / "selection.json", selection)
    log.info("FULLSET ACQUISITION | seed=%d | candidate sets=%d | edited topics=%d | unique words=%d/100 | fitting only",
        seed, selection["candidate_count"], len(selection["selected"]), selection["target_unique_words"])
    margin = policy["projection"]["rank_margin_scaled"]
    affinities, certificates = {}, {}
    if selection["required_diversity_achieved"] and selection["selected"]:
        affinity, cert = primary_tv(arrays["native"], targets, margin, policy["projection"]["primary_seconds"])
        certificates["primary"] = cert
        if affinity is not None:
            affinities["primary"] = affinity
    else:
        certificates["primary"] = {"success": False, "certified": False, "failure": "no_feasible_disjoint_supported_edit_in_bounded_pool"}
    if "primary" in affinities:
        controls = policy["controls"]
        cp = certificates["primary"]["solver_tv_scaled"]
        affinity, cert = index_secondary(arrays["native"], targets, cp, margin, controls["tv_cap_slack_scaled"], controls["secondary_seconds"])
        certificates["index"] = cert
        if affinity is not None:
            affinities["index"] = affinity
        contexts = read_json(original_case / "range_contexts.json")["group_ids"]
        positions = {g: i for i, g in enumerate(data["fit_observed_ids"])}
        if len(contexts) != 64 or len(set(contexts)) != 64 or not set(contexts).issubset(positions):
            raise ValueError("Range control requires the same 64 fitting-observed contexts.")
        affinity, cert = range_secondary(arrays["native"], targets, cp, arrays["theta_fit"][[positions[g] for g in contexts]],
            arrays["decoder_gain"], affinities["primary"], margin, controls["tv_cap_slack_scaled"],
            controls["secondary_seconds"], controls["range_maximum_iterations"], controls["range_tolerance"], log)
        certificates["range"] = cert
        if affinity is not None:
            affinities["range"] = affinity
        affinity, cert = l2_projection(arrays["native"], targets, margin, controls["l2_seconds"], controls["l2_epsilon"])
        certificates["l2"] = cert
        if affinity is not None:
            affinities["l2"] = affinity
    common = dict(theta_fit=arrays["theta_fit"], theta_dev=arrays["theta_development"], native=arrays["native"],
        gain=arrays["decoder_gain"], offset=arrays["decoder_offset"], prior=arrays["prior"], data=data,
        policy=native_policy["projection_pilot"], base=policy["base"], profile=policy["profile"], seed=seed)
    rows, audits = [], {}
    variants = [("native", "NeuroMax", "native", False, True),
        ("proposed", "Proposed", "primary", False, True),
        ("pair_acquisition", "Proposed with previous pair acquisition (R2)", "previous", False, True),
        ("moment_control", "Proposed + moment correction", "primary", True, True),
        ("no_backoff", "Proposed without backoff", "primary", False, False),
        ("index_control", "Index secondary without moments", "index", False, True),
        ("range_control", "Range secondary without moments", "range", False, True),
        ("l2_control", "L2 projection without moments", "l2", False, True)]
    for role, label, mode, correction, use_backoff in variants:
        affinity = arrays["native"] if mode == "native" else arrays["candidate_primary"] if mode == "previous" else affinities.get(mode)
        if affinity is None:
            rows.append(failed_row(role, label, seed, "readout_not_certified", policy))
            continue
        topics = topics_from_weights(affinity, data["vocabulary"])
        fit, dev, audit = reference_scores(topics, data, root / "data/cache/step03_fullset_coherence")
        row = prediction_row(role, label, affinity, topics, fit, dev, topic_diversity(topics),
                             correction=correction, use_backoff=use_backoff, deployed=mode != "native", **common)
        row.update(scope="exploratory_fullset_pilot", canonical_projection=mode, final_model_result=False)
        rows.append(row)
        audits[role] = audit
        log.info("RESULT | %s | seed=%d | Cv=%.6f NPMI=%.6f TD=%.2f | dev NLL=%.9f", label, seed,
            fit["c_v"], fit["c_npmi"], row["topic_diversity"], row["prediction"]["macro_source_nll"])
    by_role = {r["role"]: r for r in rows}
    if by_role["proposed"]["status"] == "scored":
        display = copy.deepcopy(by_role["native"])
        display.update(method="Same-descriptor display-only", role="display_only", topics=by_role["proposed"]["topics"],
            fitting=by_role["proposed"]["fitting"], development=by_role["proposed"]["development"],
            topic_diversity=by_role["proposed"]["topic_diversity"], descriptor_sets_from_proposed=True,
            projection_deployed=False, canonical_projection="native")
        rows.append(display)
    else:
        rows.append(failed_row("display_only", "Same-descriptor display-only", seed, "proposed_descriptor_unavailable", policy))
    result = {"identity": identity, "base": policy["base"], "profile": policy["profile"], "seed": seed,
        "rows": rows, "targets": targets, "selection": {k: v for k, v in selection.items() if k != "candidates"},
        "certificates": certificates, "new_neural_fits": 0, "new_classical_fits": 0, "new_encoder_calls": 0}
    write_json(destination / "case_result.json", result)
    write_json(destination / "reference_audit.json", audits)
    write_json(destination / "projection_certificates.json", certificates)
    save_arrays(destination / "readouts.npz", native=arrays["native"], previous_pair=arrays["candidate_primary"],
                **{"candidate_" + n: a for n, a in affinities.items()})
    write_json(destination / "artifact_manifest.json", artifact_manifest(destination))
    return result


def quality_gates(cases, means, baselines, old_cases, policy):
    complete = len(cases) == 2 and {c["seed"] for c in cases} == set(policy["seeds"])
    passed = complete
    per_seed = []
    for case in cases:
        rows = {r["role"]: r for r in case["rows"]}
        p, n, previous = rows["proposed"], rows["native"], rows["pair_acquisition"]
        cert = case["certificates"]["primary"]
        valid = p["status"] == "scored" and cert.get("success") and cert.get("certified")
        descriptor = bool(valid and all(p["fitting"][metric] > old["fitting"][metric]
            for metric in ["c_v", "c_npmi"] for old in [n, previous]) and p["topic_diversity"] == 1.0)
        gain = n["prediction"]["macro_source_nll"] - p["prediction"]["macro_source_nll"] if valid else None
        movement = bool(valid and cert["relative_l1_movement"] <= policy["projection"]["max_relative_l1_movement"])
        per_seed.append({"seed": case["seed"], "certified": bool(valid), "fitting_descriptor_gate": descriptor,
            "movement_gate": movement, "native_minus_proposed_nll": gain, "prediction_gate": gain is not None and gain > 0})
        passed = bool(passed and descriptor and movement and gain is not None and gain > 0)
    proposed = next(m for m in means if m["role"] == "proposed")
    previous = next(m for m in means if m["role"] == "pair_acquisition")
    if passed:
        passed = (statistics.mean(r["native_minus_proposed_nll"] for r in per_seed) >= policy["acceptance"]["mean_native_nll_gain_minimum"]
                  and proposed["development_macro_nll_mean"] <= previous["development_macro_nll_mean"])
    comparisons = []
    for b in baselines:
        fields = ["fitting_c_v_mean", "fitting_c_npmi_mean", "topic_diversity_mean", "development_macro_nll_mean"]
        available = all(proposed[f] is not None and b[f] is not None for f in fields)
        comparisons.append({"competitor": b["method"], "profile": b["profile"], "available": available,
            "cv_gain": proposed[fields[0]] - b[fields[0]] if available else None,
            "npmi_gain": proposed[fields[1]] - b[fields[1]] if available else None,
            "td_gain": proposed[fields[2]] - b[fields[2]] if available else None,
            "nll_gain": b[fields[3]] - proposed[fields[3]] if available else None})
    panel = bool(comparisons and all(c["available"] and c["cv_gain"] > 0 and c["npmi_gain"] > 0
                 and c["td_gain"] >= -1e-12 and c["nll_gain"] > 0 for c in comparisons))
    return {"complete_two_seed_grid": complete, "per_seed": per_seed, "revision_gate_passed": bool(passed),
        "all_tested_baseline_profiles_win_or_tie": panel, "baseline_comparisons": comparisons,
        "bounded_followup_allowed": bool(passed and panel), "full_final_experiment_allowed": False,
        "heldout_coherence_superiority_established": False, "final_model_result": False, "test_or_external_used": False}


def execute(root, original, repaired, completed, prepared, preparation_id, output, configs, log):
    native_policy, policy = validate_improvement(configs)
    data = load_canonical(prepared)
    old_cases, old_summary = verify_completion_source(completed, original, repaired, preparation_id)
    verified = verify_sources(original, repaired, data, native_policy)
    selected = sorted([c for c in verified if (c["base"], c["profile"]) == (policy["base"], policy["profile"])], key=lambda c: c["seed"])
    if len(selected) != 2 or {c["seed"] for c in selected} != set(policy["seeds"]):
        raise ValueError("Require both declared selected-base seeds.")
    old_rows = [r for c in old_cases["neural"] for r in c["rows"]] + [c["row"] for c in old_cases["classical"]]
    baselines = [m for m in aggregate_rows(old_rows, policy["seeds"]) if m["role"] in ["native", "classical"]]
    if len(baselines) != 8 or any(not b["complete_seed_grid"] for b in baselines):
        raise ValueError("Do not drop a baseline profile or missing seed.")
    write_json(output / "data_firewall.json", {"fitting_group_ids": data["fit_ids"], "fitting_observed_ids": data["fit_observed_ids"],
        "development_completion_ids": data["dev_ids"], "target_acquisition_development_inputs": False,
        "coherence_reference": "common_fitting_in_sample_descriptive", "prediction_reference": "development_selection",
        "test_or_external_used": False, "new_neural_fits": 0, "new_classical_fits": 0, "new_encoder_calls": 0})
    cases = []
    for c in selected:
        identity = {"policy_sha256": POLICY_SHA256, "implementation_sha256": digest({p.relative_to(root).as_posix(): file_hash(p)
            for p in (root / "src/rank_refinement").rglob("*.py")}), "source_case_sha256": file_hash(case_folder(repaired, c) / "artifact_manifest.json")}
        cases.append(run_case(root, original, repaired, c, data, native_policy, policy, output / "cases" / ("seed" + str(c["seed"])), log, identity))
        flat = [flat_row(r) for case in cases for r in case["rows"]]
        write_csv(output / "all_case_metrics.csv", flat, list(flat[0]))
        write_json(output / "progress.json", {"saved_seeds": [c["seed"] for c in cases], "expected_seeds": policy["seeds"], "partial_result": len(cases) < 2})
    means = aggregate_rows([r for c in cases for r in c["rows"]], policy["seeds"])
    gates = quality_gates(cases, means, baselines, old_cases, policy)
    proposed = next(m for m in means if m["role"] == "proposed")
    write_csv(output / "proposed_and_controls_mean_sd.csv", means, list(means[0]))
    write_csv(output / "all_baseline_profiles_mean_sd.csv", baselines, list(baselines[0]))
    write_csv(output / "all_profile_comparisons.csv", gates["baseline_comparisons"], list(gates["baseline_comparisons"][0]))
    benchmark = []
    for method in ["encot", "neuromax", "lda", "nmf"]:
        options = [m for m in baselines if m["method"] == method]
        if len(options) != 2:
            raise ValueError("Baseline family grid changed.")
        benchmark.append(min(options, key=lambda m: (m["development_macro_nll_mean"], m["profile"])))
    benchmark.append(proposed)
    write_csv(output / "bounded_competitor_comparison.csv", benchmark, list(means[0]))
    write_json(output / "case_results.json", cases)
    write_json(output / "quality_gates.json", gates)
    write_json(output / "selected_candidate.json", {"candidate": {"base": policy["base"], "profile": policy["profile"],
        "protocol": PROTOCOL, "moment_correction": False, "canonical_projection": "primary_minimum_tv"} if gates["bounded_followup_allowed"] else None,
        "bounded_followup_allowed": gates["bounded_followup_allowed"], "full_final_experiment_allowed": False})
    controls = [m for m in means if m["role"] not in ["proposed", "native", "pair_acquisition"]]
    descriptor_ties = {m["role"]: all(m[f] == proposed[f] for f in ["fitting_c_v_mean", "fitting_c_npmi_mean", "topic_diversity_mean"])
                       for m in controls}
    dominance = bool(proposed["development_macro_nll_mean"] is not None and all(m["development_macro_nll_mean"] is not None
        and proposed["development_macro_nll_mean"] <= m["development_macro_nll_mean"] for m in controls))
    write_json(output / "ablation_diagnostics.json", {"proposed_nll_no_worse_than_every_control": dominance,
        "descriptor_ties_by_construction": descriptor_ties,
        "index_secondary_is_separable_and_constant_under_fixed_margins": True,
        "canonical_primary_not_selected_by_best_control_nll": True,
        "all_control_failures_retained": True})
    def fmt(mean, sd):
        return "NA" if mean is None else f"{mean:.6f} ± {sd:.6f}" if sd is not None else f"{mean:.6f}"
    table = "| Method | Fitting C_v ↑ | Fitting NPMI ↑ | TD ↑ | Development NLL ↓ |\n|---|---:|---:|---:|---:|\n"
    for row in benchmark:
        table += "| " + " | ".join([row["method"]] + [fmt(row[n + "_mean"], row[n + "_sd"])
            for n in ["fitting_c_v", "fitting_c_npmi", "topic_diversity", "development_macro_nll"]]) + " |\n"
    write_text(output / "SUMMARY.md", "# Step 3 R3 full-set pilot\n\n"
        "This named research revision replaces the pair-plus-eight-tail acquisition with bounded complete TOP10 selection. "
        "At most four native words change; all 45 signed fitting relationships contribute; every promoted word needs two "
        "supported positive connections when introduced and at least one in the final target. The allocation seeks 100 distinct descriptor words. The actual affinity "
        "must realize those sets while preserving row/column totals with certified minimum TV. Moment correction remains removed.\n\n"
        + table + "\n"
        + f"Revision gate: {gates['revision_gate_passed']}. All eight tested baseline profiles win/tie: {gates['all_tested_baseline_profiles_win_or_tie']}. "
        + f"Bounded followup allowed: {gates['bounded_followup_allowed']}. All matched controls NLL no worse: {dominance}.\n\n"
        "These are genuine Proposed readout pilot results, not final publication results. Coherence uses the common fitting corpus and "
        "is in-sample. Development NLL is used for research selection. Two seeds cannot establish final superiority. Heldout coherence "
        "remains NA when any standard metric is nonfinite; no words/topics are dropped. NMF reports reconstruction cross-entropy. "
        "GSDMM, BERTopic, CombinedTM, FASTopic and GloCOM remain outside this bounded panel. Neither test nor external data were scored.\n\n"
        "All baseline profiles, every new control and per-seed results are saved. Display-only shares Proposed descriptors and native "
        "predictions, so descriptor ties are required. Projection controls sharing targets also tie on descriptors. Index weighting is "
        "constant under the fixed margins; it is a numerical control, not a new objective contribution. Any better control NLL remains "
        "visible. This protocol was developed after earlier pilot inspection: fresh confirmation after freezing is required.\n")
    log.info("PROPOSED MEAN | Cv=%s | NPMI=%s | TD=%s | development NLL=%s", proposed["fitting_c_v_mean"],
        proposed["fitting_c_npmi_mean"], proposed["topic_diversity_mean"], proposed["development_macro_nll_mean"])
    return gates


def run(root, configs, versions, hardware, log):
    root = Path(root).resolve()
    _, policy = validate_improvement(configs)
    preparation = require_preparation(root)
    original, repaired, completed = source_folder(root), repaired_source_folder(root), completion_source_folder(root)
    sources = [original, repaired, completed]
    before = {p.relative_to(root).as_posix(): file_hash(p / "artifact_manifest.json") for p in sources}
    identity = {"policy": policy, "source_indexes": before, "preparation_id": preparation["preparation_id"],
        "implementation": {p.relative_to(root).as_posix(): file_hash(p) for p in (root / "src/rank_refinement").rglob("*.py")},
        "versions": {n: importlib.metadata.version(n) for n in ["numpy", "scipy", "gensim", "scikit-learn", "osqp"]}}
    key = digest(identity)
    output = root / "outputs/steps/step03_coherent_refinement" / key[:20]
    output.mkdir(parents=True, exist_ok=True)
    write_json(log.folder / "preparation_working_path.json", {"relative_path": output.relative_to(root).as_posix()})
    write_json(output / "protocol_resolved_before_scoring.json", identity)
    for source, name in [(original, "source_neural_pilot_reference.json"), (repaired, "repaired_saved_source_reference.json"),
                         (completed, "completed_pilot_reference.json")]:
        write_json(output / name, {"relative_path": source.relative_to(root).as_posix(),
            "artifact_manifest_sha256": before[source.relative_to(root).as_posix()]})
    log.info("STEP 3 FULLSET R3 | fitting-only complete descriptors | saved neural models | new fits=0 | CPU readout/graph")
    gates = execute(root, original, repaired, completed, root / preparation["processed_relative_path"],
                    preparation["preparation_id"], output, configs, log)
    for source in sources:
        if file_hash(source / "artifact_manifest.json") != before[source.relative_to(root).as_posix()]:
            raise AssertionError("Full-set revision modified a historical source.")
        check_indexed_tree(source)
    result = {"status": "STEP03_R3_BOUNDED_FOLLOWUP_ALLOWED" if gates["bounded_followup_allowed"] else "STEP03_R3_QUALITY_GATE_BLOCKED",
        "engineering_passed": True, "preparation_gate_passed": True, "model_quality_established": False, "protocol": PROTOCOL,
        "preparation_id": preparation["preparation_id"], "step03_improvement_id": key,
        "processed_relative_path": output.relative_to(root).as_posix(), "completed_proposed_cases": 2,
        "new_model_fits": 0, "new_neural_fits": 0, "new_classical_fits": 0, "new_encoder_calls": 0,
        "revision_gate_passed": gates["revision_gate_passed"], "bounded_panel_all_profile_win_or_tie": gates["all_tested_baseline_profiles_win_or_tie"],
        "ready_for_bounded_followup": gates["bounded_followup_allowed"], "full_experiment_gate_passed": False,
        "final_model_result": False, "test_or_external_used": False}
    write_json(output / "run_result.json", result)
    write_json(output / "artifact_manifest.json", artifact_manifest(output))
    write_json(root / "outputs/steps/step03_coherent_refinement/LATEST.json", {"relative_path": output.relative_to(root).as_posix()})
    write_json(root / "state/step03_improvement_result.json", result)
    log.info("SAVED RESULTS | %s", output)
    log.info("READ FIRST | %s", output / "SUMMARY.md")
    log.info("QUALITY | revision=%s | all tested baseline profiles win/tie=%s | bounded followup=%s | final experiment=False",
        gates["revision_gate_passed"], gates["all_tested_baseline_profiles_win_or_tie"], gates["bounded_followup_allowed"])
    return result, output
