"""Bounded neural development pilot; no test, external or final-paper rows."""
from __future__ import annotations
import importlib.metadata
import statistics
from pathlib import Path
import numpy as np
from ..runtime.artifacts import (artifact_manifest, digest, file_hash, read_json,
    verify_artifacts, write_csv, write_json, write_text)
from ..runtime.state import require_preparation
from .step02_completion import require_pilot_handoff, check_indexed_tree
from .step02_native_refinement import metric_score, fitting_acceptance, save_arrays
from ..models.adapters.pilot_anchors import topics_from_weights
from ..models.neural_sources import resolve_sources
from ..models.neural_training import (load_canonical, encode_views, global_context,
    train_native, export_latents)
from ..models.neural_prediction import (affine_parameters, moment_parameters,
    log_predictions, select_backoff, score_log_predictions, verify_native_interface)
from ..acquisition.native_targets import choose_targets
from ..evaluation.diversity import topic_diversity
from ..projection.minimum_tv import minimum_tv, certify
from ..projection.predictive_controls import secondary_lp, l2_projection

PROTOCOL = "STEP03_NEURAL_PREDICTIVE_R1_20261009"
PINS = {"geomloss": "0.2.6", "torch-kmeans": "0.2.0", "osqp": "1.0.4",
        "sentence-transformers": "3.4.1", "transformers": "4.51.3", "tokenizers": "0.21.1",
        "huggingface-hub": "0.30.2", "safetensors": "0.5.3"}


def dependencies():
    import torch
    found, problems = {"torch": torch.__version__}, []
    for name, expected in PINS.items():
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
        if found[name] != expected:
            problems.append(f"{name}: installed={found[name]}, required={expected}")
    if torch.__version__.split("+")[0] != "2.7.0":
        problems.append("The verified interface requires your existing PyTorch 2.7.0 (CUDA suffix is allowed).")
    if problems:
        raise RuntimeError("Step 3 dependencies are incomplete. Run python -m pip install -r requirements-step03.txt. " + "; ".join(problems))
    return found


def validate_protocol(configs):
    policy = configs.get("step03_neural", {})
    # The delivered full policy hash is inserted at packaging time and prevents
    # silent changes to the search budget, masks, or metric conventions.
    if policy.get("protocol") != PROTOCOL or digest(policy) != POLICY_SHA256:
        raise ValueError("Step 3's declared protocol changed. A revision must be named before development scoring.")
    if len(policy["bases"]) * len(policy["profiles"]) * len(policy["seeds"]) != 8:
        raise ValueError("Step 3 native-fit cap must remain eight.")
    return policy


def fitting_context_ids(records, count, seed):
    sources = sorted({r["source"] for r in records})
    rng = np.random.default_rng(seed)
    chosen = []
    slots = min(count, len(records))
    ordered = {s: list(rng.permutation([i for i, r in enumerate(records) if r["source"] == s])) for s in sources}
    while len(chosen) < slots:
        for source in sources:
            if ordered[source] and len(chosen) < slots:
                chosen.append(ordered[source].pop())
    return np.asarray(sorted(chosen), dtype=int)


def prediction_row(role, method, affinity, topics, fit_metric, dev_metric, td, theta_fit, theta_dev,
                   native, gain, offset, prior, data, policy, base, profile, seed,
                   correction=True, use_backoff=True, deployed=True):
    moment = moment_parameters(theta_fit, native, affinity, policy["moment_std_floor"]) if correction else None
    logp = log_predictions(theta_dev, affinity, gain, offset, moment)
    sources = sorted({r["source"] for r in data["fit_records"]})
    grid = policy["backoff_grid"] if use_backoff else [0.]
    chosen, curve = select_backoff(logp, prior, grid, data["dev_records"], data["masks"], sources)
    no_backoff, _ = score_log_predictions(logp, data["dev_records"], data["masks"], sources)
    return {"role": role, "method": method, "underlying_base": base, "profile": profile, "seed": seed,
        "scope": "bounded_development_pilot", "status": "scored", "topics": topics,
        "fitting": fit_metric, "development": dev_metric, "topic_diversity": td,
        "prediction": chosen["scores"], "rho": chosen["rho"], "backoff_curve": curve,
        "prediction_without_backoff": no_backoff, "document_losses": chosen["rows"],
        "moment_correction": correction, "projection_deployed": deployed,
        "readout_is_native": bool(np.array_equal(affinity, native)), "final_paper_result": False,
        "development_tuned": True, "test_or_external_used": False}


def run_case(root, base, profile, seed, data, vectors, contexts, folders, source_specs,
             policy, r3, device, log, folder, coherence_cache):
    if (folder / "artifact_manifest.json").is_file():
        check_indexed_tree(folder)
        result = read_json(folder / "case_result.json")
        if (result["base"], result["profile"], result["seed"]) != (base, profile, seed):
            raise ValueError("Cached case identity differs from the requested base/profile/seed.")
        with np.load(folder / "readouts_and_latents.npz", allow_pickle=False) as archive:
            native, proposed = archive["native"], archive["proposed"]
        if result["proposed_fitting_accepted"]:
            check = certify(native, proposed, result["targets"], margin=r3["projection"]["rank_margin_scaled"])
            if not check["certified"]:
                raise ValueError("Cached Proposed readout failed independent certification.")
        elif not np.array_equal(native, proposed):
            raise ValueError("An abstaining cached Proposed row changed the predictor.")
        log.info("CASE CACHE | %s %s seed=%d | verified finished case reused", base, profile, seed)
        return result
    folder.mkdir(parents=True, exist_ok=True)
    model, training = train_native(root, base, profile, seed, policy, folders, source_specs,
                                  data, vectors, contexts, device, log)
    write_json(folder / "native_training.json", training)
    inputs_fit = contexts["fit_observed"] if base == "encot" else data["fit_observed"]
    inputs_dev = contexts["development_observed"] if base == "encot" else data["dev_observed"]
    import torch
    interface = verify_native_interface(model, torch.from_numpy(inputs_dev[:16]).to(device))
    write_json(folder / "native_interface_certificate.json", interface)
    theta_fit, theta_dev = export_latents(model, inputs_fit, device), export_latents(model, inputs_dev, device)
    with torch.no_grad():
        native = model.get_beta().cpu().double().numpy()
    gain, offset = affine_parameters(model)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    native_topics = topics_from_weights(native, data["vocabulary"])
    native_fit = metric_score(native_topics, data["fit_records"], data["vocabulary"], coherence_cache)
    native_dev = metric_score(native_topics, data["dev_coherence_records"], data["vocabulary"], coherence_cache)
    native_td = topic_diversity(native_topics)
    targets, selection, bundle = choose_targets(native, data["fit_records"], data["vocabulary"], data["metadata"], seed, r3["acquisition"])
    write_json(folder / "selection.json", selection)
    write_json(folder / "sampling_group_ids.json", bundle.pop("sample_group_ids"))
    save_arrays(folder / "association_evidence.npz", **{key: bundle[key] for key in ["pairs", "stable", "full", "pair_support", "word_support"]})
    write_json(folder / "association_words.json", {"words": bundle["words"], "fit_groups": bundle["fit_groups"]})
    pp, p3 = r3["projection"], policy["projection_pilot"]
    candidates, certificates = {}, {}
    accepted, reasons = False, ["acquisition_abstained"]
    final, final_topics, final_fit, final_dev, final_td = native, native_topics, native_fit, native_dev, native_td
    candidate_fit, candidate_dev, candidate_topics = None, None, None
    context_rows = fitting_context_ids(data["fit_observed_records"], p3["context_count"], p3["context_seed"])
    write_json(folder / "range_contexts.json", {"group_ids": [data["fit_observed_ids"][int(i)] for i in context_rows],
        "count": len(context_rows), "only_fitting_observed_contexts": True})
    if selection["selected"]:
        primary, report = minimum_tv(native, targets, pp["rank_margin_scaled"], pp["seconds"],
                                     pp["certificate_tolerance_scaled"], pp["primary_objective_tolerance_scaled"])
        certificates["primary"] = report
        if primary is not None:
            candidates["primary"] = primary
            for mode in ["index", "range"]:
                affinity, cert = secondary_lp(native, targets, report["tv_scaled"], pp["rank_margin_scaled"],
                    p3["tv_cap_slack_scaled"], p3["secondary_seconds"], mode=mode,
                    contexts=theta_fit[context_rows], decoder_gain=gain)
                certificates[mode] = cert
                if affinity is not None:
                    candidates[mode] = affinity
                log.info("PROJECTION | %s seed=%d | %s certified=%s | seconds=%.2f", base, seed, mode, cert["certified"], cert["seconds"])
            affinity, cert = l2_projection(native, targets, pp["rank_margin_scaled"], p3["qp_seconds"], p3["qp_eps"])
            certificates["l2"] = cert
            if affinity is not None:
                candidates["l2"] = affinity
            if "range" in candidates:
                candidate_topics = topics_from_weights(candidates["range"], data["vocabulary"])
                candidate_fit = metric_score(candidate_topics, data["fit_records"], data["vocabulary"], coherence_cache)
                candidate_dev = metric_score(candidate_topics, data["dev_coherence_records"], data["vocabulary"], coherence_cache)
                candidate_td = topic_diversity(candidate_topics)
                accepted, reasons = fitting_acceptance(native_fit, candidate_fit, native_td, candidate_td, certificates["range"], r3)
                if accepted:
                    final, final_topics, final_fit, final_dev, final_td = candidates["range"], candidate_topics, candidate_fit, candidate_dev, candidate_td
            else:
                reasons = ["range_secondary_failed"]
        else:
            reasons = ["minimum_tv_failed"]
    write_json(folder / "projection_certificates.json", certificates)
    prior = data["fit"].astype(np.float64).sum(axis=0) + p3["unigram_pseudocount"]
    prior /= prior.sum()
    common = dict(theta_fit=theta_fit, theta_dev=theta_dev, native=native, gain=gain, offset=offset,
        prior=prior, data=data, policy=p3, base=base, profile=profile, seed=seed)
    rows = [prediction_row("native", "EnCOT" if base == "encot" else "NeuroMax", native,
        native_topics, native_fit, native_dev, native_td, correction=False, deployed=False, **common)]
    for role, label in [("primary", "Minimum-TV primary control"), ("index", "Index-secondary control"), ("l2", "L2 projection control")]:
        affinity = candidates.get(role)
        if affinity is None and selection["selected"]:
            rows.append({"role": role, "method": label, "underlying_base": base, "profile": profile,
                "seed": seed, "status": "failed", "failure": "projection_not_certified", "final_paper_result": False})
        else:
            affinity = native if affinity is None else affinity
            topics = topics_from_weights(affinity, data["vocabulary"])
            fit_metric = metric_score(topics, data["fit_records"], data["vocabulary"], coherence_cache)
            dev_metric = metric_score(topics, data["dev_coherence_records"], data["vocabulary"], coherence_cache)
            rows.append(prediction_row(role, label, affinity, topics, fit_metric, dev_metric,
                                      topic_diversity(topics), deployed=bool(affinity is not native), **common))
    for role, label, correction, use_backoff in [
            ("proposed_uncorrected", "Proposed without moment correction", False, True),
            ("proposed_no_backoff", "Proposed without backoff", True, False),
            ("proposed", "Proposed", True, True)]:
        rows.append(prediction_row(role, label, final, final_topics, final_fit, final_dev, final_td,
                                  correction=correction, use_backoff=use_backoff, deployed=accepted, **common))
    display = dict(rows[0])
    display.update(role="display_only", method="Same-descriptor display-only", topics=final_topics,
        fitting=final_fit, development=final_dev, topic_diversity=final_td,
        projection_deployed=False, readout_is_native=True, descriptor_sets_from_proposed=True)
    rows.append(display)
    if display["prediction"] != rows[0]["prediction"] or display["document_losses"] != rows[0]["document_losses"]:
        raise AssertionError("Display-only changed the native predictor.")
    save_arrays(folder / "readouts_and_latents.npz", native=native, proposed=final, theta_fit=theta_fit,
        theta_development=theta_dev, decoder_gain=gain, decoder_offset=offset, prior=prior,
        **{("candidate_" + key): affinity for key, affinity in candidates.items()})
    proposed_row = next(r for r in rows if r["role"] == "proposed")
    result = {"base": base, "profile": profile, "seed": seed, "scope": "bounded_development_pilot",
        "interface": interface, "targets": targets, "proposed_fitting_accepted": accepted,
        "acceptance_reasons": reasons, "all_controls_scored": all(r["status"] == "scored" for r in rows),
        "rows": rows, "certificates": certificates, "prediction_evaluated": True,
        "test_or_external_used": False, "final_model_result": False}
    write_json(folder / "case_result.json", result)
    write_json(folder / "artifact_manifest.json", artifact_manifest(folder))
    log.info("CASE SAVED | Proposed | base=%s profile=%s seed=%d | accepted=%s | Cv=%s NPMI=%s NLL=%.6f | native NLL=%.6f",
        base, profile, seed, accepted, final_dev["c_v"], final_dev["c_npmi"],
        proposed_row["prediction"]["macro_source_nll"], rows[0]["prediction"]["macro_source_nll"])
    return result


def summarize(cases, policy):
    profiles, p3 = [], policy["projection_pilot"]
    roles = ["native", "primary", "index", "l2", "proposed_uncorrected", "proposed_no_backoff", "proposed", "display_only"]
    for base in policy["bases"]:
        for profile in policy["profiles"]:
            subset = [c for c in cases if c["base"] == base and c["profile"] == profile]
            if not subset:
                continue
            metrics = {}
            for role in roles:
                rows = [next(r for r in c["rows"] if r["role"] == role) for c in subset]
                finite = [r for r in rows if r["status"] == "scored" and r["development"]["scored"]]
                full = (len(rows) == len(policy["seeds"]) and len(finite) == len(rows)
                        and {c["seed"] for c in subset} == set(policy["seeds"]))
                item = {"cases": len(rows), "finite_descriptor_cases": len(finite), "complete": full}
                # Incomplete scopes retain individual rows and NA aggregate values.
                for key, getter in [("c_v", lambda r: r["development"]["c_v"]),
                                    ("c_npmi", lambda r: r["development"]["c_npmi"]),
                                    ("topic_diversity", lambda r: r["topic_diversity"]),
                                    ("macro_source_nll", lambda r: r["prediction"]["macro_source_nll"])]:
                    values = [getter(r) for r in finite] if full else []
                    item[key] = statistics.mean(values) if values else None
                    item[key + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
                metrics[role] = item
            complete = all(m["complete"] for m in metrics.values())
            both_accepted = len(subset) == 2 and all(c["proposed_fitting_accepted"] for c in subset)
            coherence_both, prediction_both = False, False
            deltas, secondary_deltas = [], []
            if complete:
                coherence_both = True
                prediction_both = True
                for case in subset:
                    by_role = {r["role"]: r for r in case["rows"]}
                    n, p, i = [by_role[r] for r in ["native", "proposed", "index"]]
                    coherence_both &= p["development"]["c_v"] > n["development"]["c_v"] and p["development"]["c_npmi"] > n["development"]["c_npmi"] and p["topic_diversity"] + 1e-12 >= n["topic_diversity"]
                    delta = p["prediction"]["macro_source_nll"] - n["prediction"]["macro_source_nll"]
                    prediction_both &= delta <= p3["maximum_development_nll_damage"]
                    deltas.append(delta)
                    secondary_deltas.append(i["prediction"]["macro_source_nll"] - p["prediction"]["macro_source_nll"])
            secondary = bool(complete and all(x > 0 for x in secondary_deltas)
                and statistics.mean(secondary_deltas) >= p3["secondary_minimum_mean_nll_improvement"])
            profiles.append({"underlying_base": base, "profile": profile, "metrics": metrics,
                "complete_two_seed_reference_and_controls": complete, "both_fitting_edits_accepted": both_accepted,
                "both_seeds_development_coherence_improve": bool(coherence_both),
                "both_seeds_prediction_cost_within_budget": bool(prediction_both),
                "development_nll_damage_by_seed": deltas, "index_minus_proposed_nll_by_seed": secondary_deltas,
                "range_secondary_performance_supported": secondary,
                "investment_gate_passed": bool(complete and both_accepted and coherence_both and prediction_both and secondary)})
    qualifying = [p for p in profiles if p["investment_gate_passed"]]
    ordered = sorted(qualifying, key=lambda p: (p["metrics"]["proposed"]["macro_source_nll"],
        -p["metrics"]["proposed"]["c_npmi"], -p["metrics"]["proposed"]["c_v"], p["underlying_base"], p["profile"]))
    selected = {k: ordered[0][k] for k in ["underlying_base", "profile"]} if ordered else None
    expected = {(b, p, s) for b in policy["bases"] for p in policy["profiles"] for s in policy["seeds"]}
    keys = [(c["base"], c["profile"], c["seed"]) for c in cases]
    completed = len(keys) == len(expected) and set(keys) == expected
    strongest = min((p["metrics"]["native"]["macro_source_nll"] for p in profiles if p["metrics"]["native"]["complete"]), default=None)
    competitive = bool(ordered and strongest is not None and ordered[0]["metrics"]["proposed"]["macro_source_nll"] <= strongest)
    return {"completed_cases": len(cases), "planned_cases": 8, "all_native_cases_completed": completed,
        "profiles": profiles, "selected_base": selected, "strongest_native_development_nll": strongest,
        "selected_proposed_beats_best_native_nll_in_this_pilot": competitive,
        "pilot_investment_gate_passed": bool(completed and selected is not None),
        "all_metrics_winning_established": False, "final_paper_benchmark_ready": False,
        "test_or_external_used": False, "scope": "bounded_development_pilot"}


def export_progress(folder, cases, policy):
    summary = summarize(cases, policy)
    write_json(folder / "case_results.json", cases)
    write_json(folder / "summary.json", summary)
    rows = []
    for case in cases:
        for row in case["rows"]:
            scored = row["status"] == "scored"
            rows.append({"method": row["method"], "underlying_base": case["base"], "profile": case["profile"],
                "seed": case["seed"], "status": row["status"], "scope": "development_pilot",
                "c_v": row["development"]["c_v"] if scored else None,
                "c_npmi": row["development"]["c_npmi"] if scored else None,
                "topic_diversity": row["topic_diversity"] if scored else None,
                "macro_source_nll": row["prediction"]["macro_source_nll"] if scored else None,
                "micro_nll": row["prediction"]["micro_nll"] if scored else None,
                "rho": row["rho"] if scored else None,
                "projection_deployed": row["projection_deployed"] if scored else False})
    columns = ["method", "underlying_base", "profile", "seed", "status", "scope", "c_v", "c_npmi",
        "topic_diversity", "macro_source_nll", "micro_nll", "rho", "projection_deployed"]
    write_csv(folder / "development_case_metrics.csv", rows, columns)
    state = "SUPPORTED" if summary["pilot_investment_gate_passed"] else "NOT YET SUPPORTED"
    text = ["# Step 3 bounded neural development pilot", "", f"Completed native cases: {len(cases)}/8.",
        "", "These are development pilot results, not final test or paper benchmark results.",
        "Proposed means the certified refined neural readout, fitting-only moment correction and development-selected backoff.",
        "The underlying base and all failed/abstaining cases remain visible.", "",
        f"Investment gate: **{state}**. Selected base: {summary['selected_base']}.",
        "", "The range-secondary objective must improve actual corrected/backoff NLL against the index control in both seeds.",
        "Negative NPMI is retained. Full-reference NA cases prevent an aggregate approval; no topic or word is silently dropped.",
        "The display-only row has exactly the native predictor and Proposed's displayed descriptors.",
        "An actual projection supplies readout realization; this pilot does not establish human requirement accuracy.", "",
        "Inspect development_case_metrics.csv, summary.json and each cases/ folder.",
        "Checkpoints live in data/cache/step03_native; rerunning reuses completed training and resumes saved epochs."]
    write_text(folder / "SUMMARY.md", "\n".join(text) + "\n")
    return summary


def run(root, configs, versions, hardware, log):
    root = Path(root).resolve()
    policy = validate_protocol(configs)
    handoff = require_pilot_handoff(root)
    preparation = require_preparation(root)
    before = file_hash(root / "state/step01_accepted.json")
    prepared = root / preparation["processed_relative_path"]
    folders, specs = resolve_sources(root, log)
    data = load_canonical(prepared)
    source_names = ["models/neural_sources.py", "models/neural_training.py", "models/neural_prediction.py",
                    "projection/predictive_controls.py", "stages/step03_neural_pilot.py"]
    source_root = Path(__file__).resolve().parents[1]
    identity = {"protocol": policy, "preparation_id": preparation["preparation_id"], "handoff": handoff["completion_id"],
        "upstream_sources": specs, "versions": versions, "device": hardware["future_model_device"],
        "implementation": {name: file_hash(source_root / name) for name in source_names}}
    key = digest(identity)
    folder = root / "outputs/steps/step03_neural_pilot" / key[:20]
    folder.mkdir(parents=True, exist_ok=True)
    write_json(log.folder / "preparation_working_path.json", {"relative_path": folder.relative_to(root).as_posix()})
    write_json(folder / "protocol_resolved_before_scoring.json", identity)
    write_json(folder / "data_firewall.json", {"preparation_id": preparation["preparation_id"],
        "fitting_group_ids": data["fit_ids"], "fitting_observed_ids": data["fit_observed_ids"],
        "development_completion_ids": data["dev_ids"],
        "development_coherence_ids": [r["group_id"] for r in data["dev_coherence_records"]],
        "canonical_vocabulary_or_masks_changed": False, "test_or_external_used": False,
        "development_encoded_text": "accepted observed completion view only"})
    write_json(folder / "source_handoff.json", handoff)
    write_json(root / "outputs/steps/step03_neural_pilot/LATEST.json", {"relative_path": folder.relative_to(root).as_posix()})
    device = hardware["future_model_device"]
    log.info("STEP 3 | 2 bases x 2 word-initialization profiles x 2 seeds | native-fit cap=8 | development only")
    log.info("DATA | fit=%d | fitting observed=%d | development completion=%d | V=%d | masks/vocabulary unchanged",
        len(data["fit_ids"]), len(data["fit_observed_ids"]), len(data["dev_ids"]), len(data["vocabulary"]))
    vectors, embedding_manifest = encode_views(root, data, configs["acquisition_anchors"]["encoder"], device, log)
    write_json(folder / "encoder_views_manifest.json", embedding_manifest)
    contexts = global_context(data, vectors, policy["global_context"])
    save_arrays(folder / "fitting_global_context.npz", bank=contexts["bank"], centers=contexts["centers"], assignments=contexts["fit_assignments"])
    cases = []
    for profile in policy["profiles"]:
        for base in policy["bases"]:
            for seed in policy["seeds"]:
                case_folder = folder / "cases" / base / profile / ("seed" + str(seed))
                result = run_case(root, base, profile, seed, data, vectors, contexts, folders, specs,
                    policy, configs["step02_refinement"], device, log, case_folder, root / "data/cache/coherence")
                cases.append(result)
                export_progress(folder, cases, policy)
                log.info("PROGRESS | saved %d/8 neural cases | %s", len(cases), folder / "development_case_metrics.csv")
    summary = export_progress(folder, cases, policy)
    if file_hash(root / "state/step01_accepted.json") != before:
        raise AssertionError("Step 3 changed accepted Step 1 state.")
    result = {"status": "STEP03_PILOT_SUPPORTED" if summary["pilot_investment_gate_passed"] else "STEP03_PILOT_NOT_SUPPORTED",
        "protocol": PROTOCOL, "step03_id": key, "preparation_id": preparation["preparation_id"],
        "engineering_passed": True, "preparation_gate_passed": True, "prediction_evaluated": True,
        "model_quality_established": False, "final_model_result": False, "full_experiment_gate_passed": False,
        "pilot_investment_gate_passed": summary["pilot_investment_gate_passed"],
        "selected_base": summary["selected_base"], "completed_cases": len(cases),
        "test_or_external_used": False, "processed_relative_path": folder.relative_to(root).as_posix()}
    write_json(folder / "run_result.json", result)
    write_json(folder / "artifact_manifest.json", artifact_manifest(folder))
    write_json(root / "state/step03_pilot_result.json", {**result, "artifacts": read_json(folder / "artifact_manifest.json")})
    log.info("SAVED RESULTS | %s", folder)
    log.info("READ FIRST | %s", folder / "SUMMARY.md")
    log.info("NEURAL PILOT | investment gate=%s | selected base=%s | final competitor benchmark ready=False",
             summary["pilot_investment_gate_passed"], summary["selected_base"])
    return result, folder


POLICY_SHA256 = "4745a4d4822b84d962a318b7bdbd7e894aa6c38465bc840f6e4bc1394d8aa953"
