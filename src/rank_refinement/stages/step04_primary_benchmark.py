"""Bounded remaining-competitor pilot with accepted Proposed frozen.

Twenty prespecified fits, incremental exports, native prediction interfaces,
complete seed accounting and untouched test/external partitions.
"""
from __future__ import annotations
import copy
import traceback
from pathlib import Path
import numpy as np
from ..runtime.artifacts import artifact_manifest, digest, file_hash, read_json, write_csv, write_json, write_text
from ..runtime.state import require_preparation
from ..runtime.step02_contract import CORE_PINS, ANCHOR_PINS, dependency_versions
from ..models.neural_training import load_canonical, encode_views
from ..models.competitor_sources import resolve_sources
from ..models.competitor_neural import cluster_contexts, train_model, decode_model
from ..models.competitor_classical import fit_gsdmm, fit_bertopic, native_topics
from ..models.classical_completion import select_probability_backoff
from ..evaluation.diversity import topic_diversity
from ..evaluation.competitor_evidence import reference_scores, summarize, paired_groups, bootstrap_gain, availability_gates
from .step02_completion import check_indexed_tree
from .step02_native_refinement import save_arrays
from .step03_completion import flat_row
from .step03_improvement import validate_improvement

PROTOCOL = "STEP04_FROZEN_PROPOSED_COMPETITOR_PILOT_R1_20261010"
POLICY_SHA256 = "a9a29d19a33ab4100b4ecee7e5e4b0f98dca9f96b480ca929b689f35f1f31e62"
SOURCE_SHA256 = "7362a2a67d96ce831d8aa6dc7e21924853f7f2142fb2a1ec6ae94933a6e70bb2"
LABELS = {"gsdmm": "GSDMM", "bertopic": "BERTopic", "combinedtm": "CombinedTM", "fastopic": "FASTopic", "glocom": "GloCOM"}


def dependencies():
    versions = dependency_versions({**CORE_PINS, **ANCHOR_PINS, "geomloss": "0.2.6", "torch-kmeans": "0.2.0",
        "osqp": "1.0.4", "pynndescent": "0.5.13", "matplotlib": "3.9.2"}, "requirements-step04.txt")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Keep or install a working PyTorch 2.4–2.7 runtime; retain the CUDA build for your RTX 3060.") from exc
    version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if not (2, 4) <= version <= (2, 7):
        raise RuntimeError("This stage is validated with PyTorch 2.4–2.7. Keep your working 2.7 CUDA build.")
    versions["torch"] = torch.__version__
    return versions


def validate_protocol(configs, root=None):
    validate_improvement(configs)
    policy = configs.get("step04_competitors", {})
    if policy.get("protocol") != PROTOCOL or digest(policy) != POLICY_SHA256:
        raise ValueError("Step 4 policy changed; use a named research revision instead of silently expanding the pilot.")
    if (len(policy["seeds"]) * sum(len(policy["profiles"][m]) for m in policy["methods"]) != 20
            or not policy["freeze_proposed"] or policy["retrain_proposed"]
            or policy["test_evaluation"] or policy["external_evaluation"]):
        raise ValueError("Incorrect bounded Step 4 scope.")
    if root is not None and digest(read_json(Path(root) / "configs/upstream_competitor_sources.json")) != SOURCE_SHA256:
        raise ValueError("Pinned competitor source identity changed.")
    return policy


def safe_step_folder(root, relative, stage):
    root = Path(root).resolve()
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("Unsafe saved stage reference.")
    path = Path(relative)
    if (path.is_absolute() or ".." in path.parts or len(path.parts) != 4
            or path.parts[:3] != ("outputs", "steps", stage)):
        raise ValueError("Unsafe saved stage reference.")
    folder = (root / path).resolve()
    if not folder.is_relative_to(root) or not folder.is_dir():
        raise RuntimeError("Saved stage folder is missing: " + relative + ". Keep your existing data/state/outputs.")
    return folder


def proposed_source(root, preparation_id):
    root = Path(root).resolve()
    pointer, state = root / "outputs/steps/step03_coherent_refinement/LATEST.json", root / "state/step03_improvement_result.json"
    if pointer.is_file():
        relative = read_json(pointer)["relative_path"]
    elif state.is_file():
        relative = read_json(state)["processed_relative_path"]
    else:
        raise RuntimeError("Step 4 requires your successful Step 3 --improve outputs and state. It does not rerun Proposed.")
    source = safe_step_folder(root, relative, "step03_coherent_refinement")
    check_indexed_tree(source)
    result = read_json(source / "run_result.json")
    if (result.get("protocol") != "STEP03_FULLSET_R3_20261009" or result["preparation_id"] != preparation_id
            or not result.get("engineering_passed") or not result.get("ready_for_bounded_followup")):
        raise ValueError("Require the accepted two-seed R3 Proposed pilot on the current preparation.")
    cases = read_json(source / "case_results.json")
    proposed = [r for c in cases for r in c["rows"] if r["role"] == "proposed"]
    if len(proposed) != 2 or {r["seed"] for r in proposed} != {809, 810} or any(not r["projection_deployed"] or r["status"] != "scored" for r in proposed):
        raise ValueError("Frozen Proposed does not have two scored deployed readouts.")
    ref = read_json(source / "completed_pilot_reference.json")
    completed = safe_step_folder(root, ref["relative_path"], "step03_completion")
    if file_hash(completed / "artifact_manifest.json") != ref["artifact_manifest_sha256"]:
        raise ValueError("Frozen baseline source differs from the accepted Proposed reference.")
    check_indexed_tree(completed)
    historical = read_json(completed / "case_results.json")
    baseline = [r for c in historical["neural"] for r in c["rows"] if r["role"] == "native"]
    baseline += [c["row"] for c in historical["classical"]]
    if len(baseline) != 16:
        raise ValueError("All eight original baseline profiles and their two seeds must remain visible.")
    baseline = copy.deepcopy(baseline)
    display_names = {"encot": "EnCOT", "neuromax": "NeuroMax", "lda": "LDA", "nmf": "NMF"}
    for row in baseline:
        row["source_method_name"] = row["method"]
        row["method"] = display_names.get(row["method"], row["method"])
    return source, completed, copy.deepcopy(proposed), baseline, cases


def unavailable_metric(reason):
    return {"c_v": None, "c_npmi": None, "scored": False, "failure": reason}


def failed_case(method, profile, seed, identity, error):
    row = {"method": LABELS[method], "role": "competitor", "underlying_base": method,
        "profile": profile["name"], "seed": seed, "status": "failed", "failure": error,
        "topics": None, "native_topics": None, "native_topic_count": None, "fixed_k10": False,
        "fitting": unavailable_metric("execution_failed"), "development": unavailable_metric("execution_failed"),
        "native_fitting": unavailable_metric("execution_failed"), "native_development": unavailable_metric("execution_failed"),
        "topic_diversity": None, "native_topic_diversity": None,
        "prediction": {"macro_source_nll": None, "micro_nll": None}, "rho": None, "document_losses": [],
        "prediction_interface": None, "projection_deployed": False, "moment_correction": False,
        "test_or_external_used": False, "final_model_result": False}
    return {"identity": identity, "row": row, "training_details": {"failure": error}}


def competitor_case(root, method, profile, seed, data, vectors, source_folders, source_specs, policy,
                    device, prior, destination, log, identity):
    if (destination / "artifact_manifest.json").is_file():
        check_indexed_tree(destination)
        value = read_json(destination / "case_result.json")
        if value["identity"] != identity:
            raise ValueError("Competitor case cache identity changed.")
        log.info("COMPETITOR CACHE | %s %s seed=%d | status=%s", LABELS[method], profile["name"], seed, value["row"]["status"])
        return value
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "case_inputs.json", identity)
    log.info("COMPETITOR CASE | %s %s seed=%d | fitting only | observed-only development inference", LABELS[method], profile["name"], seed)
    try:
        extra_arrays = {}
        if method in ["gsdmm", "bertopic"]:
            log.new_classical_fits = getattr(log, "new_classical_fits", 0) + 1
            if method == "gsdmm":
                topics, weights, probabilities, theta, details = fit_gsdmm(data["fit"], data["dev_observed"], data["vocabulary"], seed, profile, log)
            else:
                topics, weights, probabilities, theta, details = fit_bertopic(data["fit"], data["dev_observed"], data["vocabulary"], seed, profile, vectors, log)
        else:
            contexts = cluster_contexts(data["fit"], vectors["fit_full"], data["dev_observed"], vectors["development_observed"], profile, seed) if method == "glocom" else None
            model, details = train_model(root, method, profile, seed, source_folders[method], data["fit"], vectors,
                data["fit_ids"], source_specs, device, policy["checkpoint_every_epochs"], log, contexts)
            weights, probabilities, theta, decoding = decode_model(method, model, data["dev_observed"], vectors,
                contexts, seed, policy["prediction"]["combinedtm_posterior_samples"])
            details.update(decoding)
            topics, topic_info = native_topics(weights, data["vocabulary"])
            details.update(topic_info)
            if contexts:
                extra_arrays = {"global_bank": contexts["bank"], "global_centers": contexts["centers"],
                                "fit_context_assignments": contexts["fit_assignments"], "observed_context_assignments": contexts["observed_assignments"]}
            del model
        if topics is None:
            native_fit = native_dev = unavailable_metric("native_topics_without_ten_supported_words")
            audit = {"failure": "native_descriptor_slots_unavailable_no_padding", "details": details}
            native_td = None
        else:
            native_fit, native_dev, audit = reference_scores(topics, data, root / "data/cache/step04_coherence")
            native_td = topic_diversity(topics)
        fixed = bool(details["fixed_k10"] and topics is not None)
        fit = native_fit if fixed else unavailable_metric("native_topic_count_not_fixed_k10_no_padding")
        dev = native_dev if fixed else unavailable_metric("native_topic_count_not_fixed_k10_no_padding")
        if probabilities is None:
            prediction, document_losses, rho, curve = {"macro_source_nll": None, "micro_nll": None}, [], None, []
        else:
            chosen, curve = select_probability_backoff(probabilities, prior, policy["prediction"]["shared_backoff_grid"],
                data["dev_records"], data["masks"], sorted({r["source"] for r in data["fit_records"]}))
            prediction, document_losses, rho = chosen["scores"], chosen["rows"], chosen["rho"]
        row = {"method": LABELS[method], "role": "competitor", "underlying_base": method,
            "profile": profile["name"], "seed": seed, "status": "scored" if fixed else "native_variable_k_or_unavailable_descriptors",
            "topics": topics if fixed else None, "native_topics": topics, "native_topic_count": details["native_topic_count"], "fixed_k10": fixed,
            "fitting": fit, "development": dev, "native_fitting": native_fit, "native_development": native_dev,
            "topic_diversity": native_td if fixed else None, "native_topic_diversity": native_td,
            "prediction": prediction, "rho": rho, "backoff_curve": curve, "document_losses": document_losses,
            "prediction_interface": details["prediction_interface"], "projection_deployed": False, "moment_correction": False,
            "test_or_external_used": False, "final_model_result": False, "fitting_coherence_scope": "in_sample_descriptive", "development_tuned": True,
            "prediction_unavailable_reason": details.get("prediction_unavailable_reason")}
        details.update(profile=profile, seed=seed, fit_documents=len(data["fit_ids"]),
                       observed_development_documents=len(data["dev_ids"]), shared_vocabulary_size=len(data["vocabulary"]))
        value = {"identity": identity, "row": row, "training_details": details}
        arrays = {"native_descriptor_weights": weights, **extra_arrays}
        if probabilities is not None:
            arrays.update(native_decoded_probabilities=probabilities, native_observed_latents=theta)
        save_arrays(destination / "native_readout_and_predictions.npz", **arrays)
        write_json(destination / "reference_audit.json", audit)
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        write_text(destination / "failure_traceback.txt", traceback.format_exc())
        log.logger.error("COMPETITOR FAILED | %s %s seed=%d | %s | retained in the panel", LABELS[method], profile["name"], seed, error)
        value = failed_case(method, profile, seed, identity, error)
    write_json(destination / "case_result.json", value)
    write_json(destination / "artifact_manifest.json", artifact_manifest(destination))
    r = value["row"]
    def display(v):
        return "NA" if v is None else f"{v:.6f}"
    log.info("COMPETITOR SAVED | %s %s seed=%d | K=%s | Cv=%s NPMI=%s TD=%s | dev NLL=%s | status=%s",
        r["method"], r["profile"], seed, r["native_topic_count"], display(r["fitting"]["c_v"]), display(r["fitting"]["c_npmi"]),
        display(r["topic_diversity"]), display(r["prediction"]["macro_source_nll"]), r["status"])
    try:
        import torch
        if device == "cuda":
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return value


def enriched(row):
    r = copy.deepcopy(row)
    r.setdefault("native_topic_count", 10)
    r.setdefault("fixed_k10", True)
    r.setdefault("native_fitting", r["fitting"])
    r.setdefault("native_development", r["development"])
    r.setdefault("native_topic_diversity", r["topic_diversity"])
    return r


def export_progress(output, frozen, cases, expected=20):
    rows = frozen + [c["row"] for c in cases]
    flat = []
    for r in rows:
        entry = flat_row(r)
        entry.update(status=r["status"], native_topic_count=r.get("native_topic_count", 10), fixed_k10=r.get("fixed_k10", True),
                     prediction_interface=r.get("prediction_interface"), failure=r.get("failure"))
        flat.append(entry)
    write_csv(output / "all_case_metrics.csv", flat, list(flat[0]))
    write_json(output / "competitor_case_results.json", cases)
    write_json(output / "progress.json", {"completed_cases": len(cases), "expected_competitor_cases": expected,
        "failed_cases": sum(c["row"]["status"] == "failed" for c in cases), "partial_results_are_not_final": True})


def compare_profiles(means):
    proposed = next(r for r in means if r["method"] == "Proposed")
    result = []
    for row in means:
        if row["method"] == "Proposed":
            continue
        item = {"method": row["method"], "profile": row["profile"], "complete_seed_grid": row["complete_seed_grid"],
                "fixed_k10_all_seeds": row["fixed_k10_all_seeds"]}
        for field, direction in [("fitting_c_v", 1), ("fitting_c_npmi", 1), ("topic_diversity", 1), ("development_macro_nll", -1)]:
            a, b = proposed[field + "_mean"], row[field + "_mean"]
            delta = direction * (a - b) if a is not None and b is not None else None
            item[field + "_proposed_gain"] = delta
            item[field + "_win_or_tie"] = delta >= -1e-12 if delta is not None else None
        result.append(item)
    return result


def selected_profiles(means):
    decisions = []
    for method in sorted({r["method"] for r in means} - {"Proposed"}):
        rows = [r for r in means if r["method"] == method and r["all_cases_executed"]
                and r["fixed_k10_all_seeds"] and r["fitting_c_v_mean"] is not None and r["fitting_c_npmi_mean"] is not None]
        predictive = [r for r in rows if r["development_macro_nll_mean"] is not None]
        if predictive:
            choice = min(predictive, key=lambda r: (r["development_macro_nll_mean"], -r["fitting_c_npmi_mean"], r["profile"]))
            rule = "mean_development_nll_then_fitting_npmi_then_profile_name"
        elif rows:
            choice = min(rows, key=lambda r: (-r["fitting_c_npmi_mean"], -r["fitting_c_v_mean"], r["profile"]))
            rule = "mean_fitting_npmi_then_cv_then_profile_name"
        else:
            choice, rule = None, "no_complete_fixed_k10_profile"
        decisions.append({"method": method, "selected_profile": choice["profile"] if choice else None, "rule": rule})
    return decisions


def summarize_panel(output, frozen, cases, policy, log):
    rows = [enriched(r) for r in frozen + [c["row"] for c in cases]]
    means = summarize(rows, policy["seeds"])
    for r in means:
        r["native_topics_by_seed"] = "/".join("NA" if n is None else str(n) for n in r["native_topics_by_seed"])
    comparisons = compare_profiles(means)
    selection = selected_profiles(means)
    new_methods = [LABELS[m] for m in policy["methods"]]
    availability = availability_gates(means, new_methods)
    metrics = ["fitting_c_v", "fitting_c_npmi", "topic_diversity", "development_macro_nll"]
    observed = [r[m + "_win_or_tie"] for r in comparisons for m in metrics if r[m + "_win_or_tie"] is not None]
    all_executed = len(cases) == 20 and all(c["row"]["status"] != "failed" for c in cases)
    panel_complete = all_executed and availability["fixed_k_descriptor_panel_complete"] and availability["compatible_prediction_panel_complete"]
    gates = {**availability, "engineering_passed": all_executed, "twenty_prespecified_cases_recorded": len(cases) == 20,
        "failed_cases": sum(c["row"]["status"] == "failed" for c in cases), "full_bounded_panel_complete": panel_complete,
        "all_available_profile_metrics_win_or_tie": bool(observed and all(observed)),
        "complete_bounded_panel_win_or_tie": bool(panel_complete and observed and all(observed)),
        "proposed_retrained_or_retuned": False, "test_or_external_used": False,
        "final_paper_results": False, "full_experiment_authorized_by_pilot": False,
        "next_action": "review_full_panel_before_freezing_five_seed_experiment" if panel_complete else "review_failures_and_metric_availability_before_expanding"}
    from ..reporting.tables import export_table
    export_table(output, "all_profiles_mean_sd", means, list(means[0]))
    export_table(output, "all_profile_comparisons", comparisons, list(comparisons[0]))
    write_json(output / "profile_selection.json", selection)
    write_json(output / "quality_gates.json", gates)
    write_json(output / "all_rows.json", rows)
    proposed = [r for r in rows if r["method"] == "Proposed"]
    uncertainty, paired_export = [], []
    keys = {(r["method"], r["profile"]) for r in rows} - {("Proposed", proposed[0]["profile"])}
    for method, profile in sorted(keys):
        subset = [r for r in rows if (r["method"], r["profile"]) == (method, profile)]
        if len(subset) != 2 or any(r["prediction"]["macro_source_nll"] is None for r in subset):
            continue
        paired = paired_groups(proposed, subset, policy["seeds"])
        diagnostic = bootstrap_gain(paired, policy["seeds"], policy["selection"]["bootstrap_resamples"], policy["selection"]["bootstrap_seed"])
        diagnostic.update(method=method, profile=profile)
        uncertainty.append(diagnostic)
        paired_export += [{"method": method, "profile": profile, **r} for r in paired]
    write_json(output / "paired_prediction_bootstrap.json", uncertainty)
    if paired_export:
        write_csv(output / "paired_document_prediction_gains.csv", paired_export, list(paired_export[0]))
        from ..reporting.competitor_figures import plot_prediction_groups
        plot_prediction_groups(output, paired_export, means)
    lines = ["# Step 4: frozen Proposed and remaining competitors", "", "Bounded two-seed development pilot; not final manuscript evidence.", "",
        "Fitting C_v (110), fitting NPMI (10), and TD use every ten-word descriptor slot and the accepted common reference.",
        "Prediction uses the same 203 observed-only completion documents and shared fitting-prior backoff grid.",
        "BERTopic NLL is NA because its native c-TF-IDF and memberships do not define word probabilities.",
        "Variable native topic counts remain in the supplement; fixed-K10 cells are NA without padding.", "",
        "| Model | Profile | C_v | NPMI | TD | Dev macro NLL | Native K |", "|---|---|---:|---:|---:|---:|---|"]
    def fmt(row, name):
        value, sd = row[name + "_mean"], row[name + "_sd"]
        return "NA" if value is None else f"{value:.6f} ± {sd:.6f}" if sd is not None else f"{value:.6f}"
    for r in sorted(means, key=lambda r: (r["method"] != "Proposed", r["method"], r["profile"])):
        lines.append("| " + " | ".join([r["method"], r["profile"], fmt(r, "fitting_c_v"), fmt(r, "fitting_c_npmi"),
            fmt(r, "topic_diversity"), fmt(r, "development_macro_nll"), r["native_topics_by_seed"]]) + " |")
    lines += ["", f"Engineering complete: **{all_executed}**. Fixed-K descriptor panel complete: **{availability['fixed_k_descriptor_panel_complete']}**.",
        f"Compatible prediction panel complete: **{availability['compatible_prediction_panel_complete']}**.",
        f"Complete bounded panel win/tie gate: **{gates['complete_bounded_panel_win_or_tie']}**.", "",
        "A negative Proposed gain in all_profile_comparisons.csv is a real competitor advantage and is retained.",
        "Held-out coherence stays NA when the standard metric is nonfinite; no topic/word is removed to repair it.",
        "Bootstrap intervals are exploratory group uncertainty after development selection, not confirmatory significance or training-seed uncertainty.",
        "The original Proposed, EnCOT, NeuroMax, LDA, NMF and controls are reused from indexed sources with unchanged metrics.",
        "Next: inspect this complete panel, freeze a fair final protocol, then evaluate five seeds and the untouched test/external sets in later stages.",
        "Do not retune Proposed using the final test or external dataset.", ""]
    write_text(output / "SUMMARY.md", "\n".join(lines))
    log.info("PANEL | cases=%d/20 | engineering=%s | fixed-K descriptors=%s | compatible prediction=%s | complete win/tie=%s",
        len(cases), all_executed, availability["fixed_k_descriptor_panel_complete"], availability["compatible_prediction_panel_complete"], gates["complete_bounded_panel_win_or_tie"])
    return gates


def run(root, configs, versions, hardware, log):
    root = Path(root).resolve()
    policy = validate_protocol(configs, root)
    prepared = require_preparation(root)
    source, historical, proposed, baselines, source_cases = proposed_source(root, prepared["preparation_id"])
    data = load_canonical(root / prepared["processed_relative_path"])
    if len(data["dev_ids"]) != 203 or len(data["fit_ids"]) != 1436 or len(data["vocabulary"]) != 1472:
        raise ValueError("This bounded continuation expects the accepted 1436-fit/203-development/V1472 corpus.")
    source_folders, source_specs = resolve_sources(root, log)
    source_hashes = {p.relative_to(root).as_posix(): file_hash(p / "artifact_manifest.json") for p in [source, historical]}
    identity = {"protocol": policy, "source_indexes": source_hashes, "preparation_id": prepared["preparation_id"],
        "competitor_sources": source_specs, "versions": versions, "device": hardware["future_model_device"],
        "implementation": {p.relative_to(root).as_posix(): file_hash(p) for p in sorted((root / "src/rank_refinement").rglob("*.py"))}}
    key = digest(identity)
    output = root / "outputs/steps/step04_competitor_pilot" / key[:20]
    output.mkdir(parents=True, exist_ok=True)
    write_json(log.folder / "preparation_working_path.json", {"relative_path": output.relative_to(root).as_posix()})
    write_json(output / "protocol_resolved_before_scoring.json", identity)
    write_json(output / "frozen_proposed_reference.json", {"relative_path": source.relative_to(root).as_posix(), "artifact_manifest_sha256": file_hash(source / "artifact_manifest.json")})
    for name in ["source_neural_pilot_reference.json", "repaired_saved_source_reference.json", "completed_pilot_reference.json"]:
        write_json(output / name, read_json(source / name))
    write_json(output / "frozen_proposed_and_controls.json", source_cases)
    write_json(output / "data_firewall.json", {"fitting_group_ids": data["fit_ids"], "development_completion_ids": data["dev_ids"],
        "development_coherence_ids": [r["group_id"] for r in data["dev_coherence_records"]],
        "development_full_text_encoded": False, "test_or_external_used": False, "fit_encoder": False,
        "proposed_frozen": True, "shared_fitting_vocabulary": data["vocabulary"],
        "shared_backoff_grid": policy["prediction"]["shared_backoff_grid"], "coherence_scope": "common_fitting_reference_in_sample_descriptive"})
    frozen, cases = proposed + baselines, []
    export_progress(output, frozen, cases)
    prior = data["fit"].astype(np.float64).sum(0) + policy["prediction"]["unigram_pseudocount"]
    prior /= prior.sum()
    vectors = embedding_manifest = None
    log.info("STEP 4 | frozen Proposed | 5 methods x 2 profiles x 2 seeds = 20 fits maximum | test/external untouched")
    for method in policy["methods"]:
        if method != "gsdmm" and vectors is None:
            vectors, embedding_manifest = encode_views(root, data, configs["acquisition_anchors"]["encoder"], hardware["future_model_device"], log)
            write_json(output / "shared_encoder_manifest.json", embedding_manifest)
        for profile in policy["profiles"][method]:
            for seed in policy["seeds"]:
                case_identity = {"benchmark_identity_sha256": key, "method": method, "profile": profile, "seed": seed,
                    "fit_group_ids_sha256": digest(data["fit_ids"]), "development_group_ids_sha256": digest(data["dev_ids"]),
                    "encoder_vectors_sha256": embedding_manifest["vectors_sha256"] if embedding_manifest and method != "gsdmm" else None}
                destination = output / "cases" / method / profile["name"] / ("seed" + str(seed))
                cases.append(competitor_case(root, method, profile, seed, data, vectors, source_folders, source_specs, policy,
                    hardware["future_model_device"], prior, destination, log, case_identity))
                export_progress(output, frozen, cases)
                log.info("PROGRESS | competitors=%d/20 | incremental results=%s", len(cases), output / "all_case_metrics.csv")
    gates = summarize_panel(output, frozen, cases, policy, log)
    for folder in [source, historical]:
        if file_hash(folder / "artifact_manifest.json") != source_hashes[folder.relative_to(root).as_posix()]:
            raise AssertionError("Competitor benchmark modified frozen Proposed or a historical baseline.")
        check_indexed_tree(folder)
    result = {"status": "STEP04_BOUNDED_PANEL_COMPLETE" if gates["full_bounded_panel_complete"] else "STEP04_PANEL_REVIEW_REQUIRED" if gates["engineering_passed"] else "STEP04_EXECUTION_INCOMPLETE",
        "engineering_passed": gates["engineering_passed"], "preparation_gate_passed": True, "model_quality_established": False,
        "protocol": PROTOCOL, "preparation_id": prepared["preparation_id"], "step04_id": key,
        "processed_relative_path": output.relative_to(root).as_posix(), "competitor_cases_recorded": len(cases), "failed_cases": gates["failed_cases"],
        "bounded_panel_complete": gates["full_bounded_panel_complete"], "complete_bounded_panel_win_or_tie": gates["complete_bounded_panel_win_or_tie"],
        "new_neural_fits": getattr(log, "new_neural_fits", 0), "new_classical_fits": getattr(log, "new_classical_fits", 0),
        "new_proposed_fits": 0, "full_experiment_gate_passed": False, "final_model_result": False, "test_or_external_used": False}
    write_json(output / "run_result.json", result)
    write_json(output / "artifact_manifest.json", artifact_manifest(output))
    write_json(root / "outputs/steps/step04_competitor_pilot/LATEST.json", {"relative_path": output.relative_to(root).as_posix()})
    write_json(root / "state/step04_competitor_result.json", result)
    log.info("SAVED RESULTS | %s", output)
    log.info("READ FIRST | %s", output / "SUMMARY.md")
    return result, output
