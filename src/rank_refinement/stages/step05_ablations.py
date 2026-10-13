"""Frozen five-seed development replication and matched mechanism ablations.

The two inspected seeds are reused exactly. Three new seeds share the same
training, acquisition, projection, vocabulary and metric contracts. Test and
external scoring are deliberately separate, subsequent stages.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from ..runtime.artifacts import artifact_manifest, digest, file_hash, read_json, write_json, write_csv, write_text
from ..runtime.state import require_preparation
from ..models.neural_training import load_canonical, encode_views, global_context, train_native, export_latents
from ..models.neural_sources import resolve_sources as resolve_native_sources
from ..models.neural_prediction import affine_parameters, verify_native_interface
from ..models.competitor_sources import resolve_sources as resolve_competitor_sources
from ..models.classical_completion import fit_and_predict, select_probability_backoff
from ..models.adapters.pilot_anchors import topics_from_weights
from ..evaluation.competitor_evidence import reference_scores, summarize, availability_gates
from ..evaluation.replication import ablated_evidence, assert_grid, assert_ablation_grid, hierarchical_gain, readout_mechanism
from ..evaluation.diversity import topic_diversity
from ..acquisition.native_targets import choose_targets
from ..acquisition.complete_targets import choose_complete_targets
from ..projection.replay_solver import primary_tv
from ..reporting.replication_figures import plot_seed_tradeoffs
from ..reporting.tables import export_table
from .step02_completion import check_indexed_tree
from .step02_native_refinement import save_arrays
from .step03_neural_pilot import prediction_row, fitting_context_ids
from .step03_saved_replay import case_folder
from .step03_improvement import run_case as fullset_case, failed_row, load_evidence
from .step04_primary_benchmark import (validate_protocol as validate_panel, safe_step_folder,
    proposed_source, competitor_case, enriched, compare_profiles)

PROTOCOL = "STEP05_FROZEN_FIVE_SEED_REPLICATION_R1_20261010"
POLICY_SHA256 = "6f6bf0156360c455fea97eb7352d3cbabf15e2cdaa4c25545669e3fd8db2352d"
LABEL_TO_METHOD = {"GSDMM":"gsdmm", "BERTopic":"bertopic", "CombinedTM":"combinedtm", "FASTopic":"fastopic", "GloCOM":"glocom", "LDA":"lda", "NMF":"nmf", "EnCOT":"encot", "NeuroMax":"neuromax"}


def validate_protocol(configs, root=None):
    validate_panel(configs, root)
    p = configs.get("step05_replication", {})
    if p.get("protocol") != PROTOCOL or digest(p) != POLICY_SHA256:
        raise ValueError("Step 5 protocol changed; declare a named revision before scoring.")
    if p["test_evaluation"] or p["external_evaluation"] or p["moment_correction"] or p["fit_encoder"]:
        raise ValueError("Step 5 cannot evaluate held-out test/external data or change Proposed.")
    return p


def profiles_to_run(policy):
    result = list(policy["primary_profiles"].items())
    result += [(m, p) for m, ps in policy["descriptor_challengers"].items() for p in ps]
    if len(result) != 11 or len(set(result)) != 11:
        raise ValueError("Require nine locked families and both declared descriptor challengers.")
    return sorted(result)


def accepted_panel(root, preparation_id, policy):
    root = Path(root).resolve()
    pointer = root/"outputs/steps/step04_competitor_pilot/LATEST.json"
    state = root/"state/step04_competitor_result.json"
    if pointer.is_file(): relative = read_json(pointer)["relative_path"]
    elif state.is_file(): relative = read_json(state)["processed_relative_path"]
    else: raise RuntimeError("Keep your completed Step 4 outputs and state; run Step 4 first if absent.")
    panel = safe_step_folder(root, relative, "step04_competitor_pilot")
    check_indexed_tree(panel)
    r = read_json(panel/"run_result.json"); gates = read_json(panel/"quality_gates.json")
    if (r.get("protocol") != policy["source_protocol"] or r.get("preparation_id") != preparation_id
            or not r.get("engineering_passed") or not r.get("bounded_panel_complete")
            or not gates.get("complete_bounded_panel_win_or_tie") or r.get("test_or_external_used")):
        raise ValueError("Step 5 requires the complete reviewed Step 4 pilot on the accepted preparation.")
    selection = {r["method"]: r["selected_profile"] for r in read_json(panel/"profile_selection.json")}
    if selection != policy["primary_profiles"]:
        raise ValueError("The locked primary profiles differ from the inspected Step 4 selection.")
    rows = read_json(panel/"all_rows.json")
    expected = [("Proposed",policy["proposed_profile"])] + profiles_to_run(policy)
    reused = [copy.deepcopy(r) for r in rows if (r["method"],r["profile"]) in expected]
    assert_grid(reused, expected, policy["reuse_seeds"])
    return panel, reused


def profile_settings(configs, method, name):
    if method in {"encot","neuromax"}:
        if name not in configs["step03_neural"]["profiles"]: raise ValueError("Unknown native profile.")
        return {"name":name}
    source = configs["step03_completion"]["classical_panel"]["profiles"] if method in {"lda","nmf"} else configs["step04_competitors"]["profiles"]
    matches = [copy.deepcopy(p) for p in source[method] if p["name"] == name]
    if len(matches) != 1: raise ValueError("Unknown/ambiguous frozen baseline profile.")
    return matches[0]


def native_case(root, base, profile, seed, data, vectors, contexts, folders, specs, native_policy,
                device, prior, destination, log, identity):
    if (destination/"artifact_manifest.json").is_file():
        check_indexed_tree(destination); value=read_json(destination/"case_result.json")
        if value["identity"] != identity: raise ValueError("Native case identity changed.")
        return value
    destination.mkdir(parents=True,exist_ok=True)
    model, training = train_native(root, base, profile, seed, native_policy, folders, specs, data, vectors, contexts, device, log)
    log.new_neural_fits = getattr(log,"new_neural_fits",0) + int(training["new_training_epochs"] > 0)
    fit_inputs = contexts["fit_observed"] if base=="encot" else data["fit_observed"]
    dev_inputs = contexts["development_observed"] if base=="encot" else data["dev_observed"]
    import torch
    interface = verify_native_interface(model,torch.from_numpy(dev_inputs[:16]).to(device))
    theta_fit = export_latents(model,fit_inputs,device); theta_dev = export_latents(model,dev_inputs,device)
    with torch.no_grad(): native=model.get_beta().cpu().double().numpy()
    gain,offset=affine_parameters(model)
    del model
    if device=="cuda": torch.cuda.empty_cache()
    topics=topics_from_weights(native,data["vocabulary"])
    fit,dev,audit=reference_scores(topics,data,root/"data/cache/step05_coherence")
    row=prediction_row("native","EnCOT" if base=="encot" else "NeuroMax",native,topics,fit,dev,topic_diversity(topics),
        theta_fit,theta_dev,native,gain,offset,prior,data,native_policy["projection_pilot"],base,profile,seed,
        correction=False,deployed=False)
    row.update(prediction_interface="frozen_native_affine_decoder",scope="five_seed_development_replication",final_model_result=False)
    save_arrays(destination/"readouts_and_latents.npz",native=native,theta_fit=theta_fit,theta_development=theta_dev,
        decoder_gain=gain,decoder_offset=offset,prior=prior)
    write_json(destination/"native_training.json",training);write_json(destination/"native_interface_certificate.json",interface)
    write_json(destination/"reference_audit.json",audit)
    value={"identity":identity,"row":enriched(row),"training_details":training}
    write_json(destination/"case_result.json",value);write_json(destination/"artifact_manifest.json",artifact_manifest(destination))
    return value


def classical_case(root,method,profile,seed,data,prior,folder,log,identity,grid):
    if (folder/"artifact_manifest.json").is_file():
        check_indexed_tree(folder);value=read_json(folder/"case_result.json")
        if value["identity"]!=identity: raise ValueError("Classical case identity changed.")
        return value
    folder.mkdir(parents=True,exist_ok=True);log.new_classical_fits=getattr(log,"new_classical_fits",0)+1
    topics,weights,p,details=fit_and_predict(method,data["fit"],data["dev_observed"],data["vocabulary"],seed,profile)
    fit,dev,audit=reference_scores(topics,data,root/"data/cache/step05_coherence")
    chosen,curve=select_probability_backoff(p,prior,grid,data["dev_records"],data["masks"],sorted({r["source"] for r in data["fit_records"]}))
    row=enriched({"method": "LDA" if method=="lda" else "NMF", "role":"native", "underlying_base":method,
        "profile":profile["name"],"seed":seed,"status":"scored","topics":topics,"fitting":fit,"development":dev,
        "prediction":chosen["scores"],"rho":chosen["rho"],"backoff_curve":curve,"document_losses":chosen["rows"],
        "topic_diversity":topic_diversity(topics),"prediction_interface":details["prediction_interface"],
        "projection_deployed":False,"moment_correction":False,"final_model_result":False,"test_or_external_used":False})
    save_arrays(folder/"native_readout_and_predictions.npz",native_descriptor_weights=weights,native_decoded_probabilities=p)
    write_json(folder/"reference_audit.json",audit)
    value={"identity":identity,"row":row,"training_details":details}
    write_json(folder/"case_result.json",value);write_json(folder/"artifact_manifest.json",artifact_manifest(folder))
    return value


def build_refinement_source(root,output,seed,native_folder,data,configs,log):
    source=output/"refinement_sources";folder=case_folder(source,{"base":"neuromax","profile":"random_words","seed":seed})
    if (folder/"artifact_manifest.json").is_file():
        check_indexed_tree(folder)
        ref=read_json(folder/"native_source_reference.json")
        if ref["sha256"]!=file_hash(native_folder/"artifact_manifest.json"): raise ValueError("Refinement parent changed.")
        return source
    folder.mkdir(parents=True,exist_ok=True)
    with np.load(native_folder/"readouts_and_latents.npz",allow_pickle=False) as z: arrays={n:z[n] for n in z.files}
    targets,selection,bundle=choose_targets(arrays["native"],data["fit_records"],data["vocabulary"],data["metadata"],seed,configs["step02_refinement"]["acquisition"])
    previous=arrays["native"].copy();cert={"no_edit":True}
    if selection["selected"]:
        previous,cert=primary_tv(arrays["native"],targets,configs["step03_improvement"]["projection"]["rank_margin_scaled"],180.)
        if previous is None: raise RuntimeError("Previous-pair acquisition control could not be certified; no native substitution.")
    save_arrays(folder/"readouts_and_latents.npz",**arrays,candidate_primary=previous)
    write_json(folder/"sampling_group_ids.json",bundle.pop("sample_group_ids"))
    save_arrays(folder/"association_evidence.npz",**{n:bundle[n] for n in ["pairs","stable","full","pair_support","word_support"]})
    write_json(folder/"association_words.json",{"words":bundle["words"],"fit_groups":bundle["fit_groups"]})
    ids=fitting_context_ids(data["fit_observed_records"],64,configs["step03_neural"]["projection_pilot"]["context_seed"])
    write_json(folder/"range_contexts.json",{"group_ids":[data["fit_observed_ids"][int(i)] for i in ids]})
    write_json(folder/"previous_pair_selection.json",selection);write_json(folder/"previous_pair_certificate.json",cert)
    write_json(folder/"native_source_reference.json",{"relative_path":native_folder.relative_to(root).as_posix(),"sha256":file_hash(native_folder/"artifact_manifest.json")})
    write_json(folder/"artifact_manifest.json",artifact_manifest(folder));return source


def acquisition_ablation(root,seed,variant,source,readout_folder,data,configs,folder,log,identity):
    if (folder/"artifact_manifest.json").is_file():
        check_indexed_tree(folder);value=read_json(folder/"case_result.json")
        if value["identity"]!=identity: raise ValueError("Acquisition ablation identity changed.")
        return value
    folder.mkdir(parents=True,exist_ok=True)
    with np.load(readout_folder/"readouts_and_latents.npz",allow_pickle=False) as z: a={n:z[n] for n in z.files}
    evidence=load_evidence(source,data["fit_records"]);changed=ablated_evidence(evidence,variant)
    policy=configs["step03_improvement"]
    targets,selection=choose_complete_targets(a["native"],data["vocabulary"],data["metadata"],changed,policy["acquisition"])
    label={"without_subsample_stability":"Proposed without subsample stability","positive_only_association":"Proposed with positive-only associations"}[variant]
    affinity=None;cert={"certified":False,"failure":"no_feasible_supported_disjoint_edit"}
    if selection["required_diversity_achieved"] and selection["selected"]:
        affinity,cert=primary_tv(a["native"],targets,policy["projection"]["rank_margin_scaled"],policy["projection"]["primary_seconds"])
    if affinity is None:
        row=failed_row(variant,label,seed,cert.get("failure","projection_not_certified"),policy)
        row.update(status="abstained",failure_kind="readout_unavailable_no_native_substitution")
    else:
        topics=topics_from_weights(affinity,data["vocabulary"]);fit,dev,audit=reference_scores(topics,data,root/"data/cache/step05_coherence")
        row=prediction_row(variant,label,affinity,topics,fit,dev,topic_diversity(topics),a["theta_fit"],a["theta_development"],
            a["native"],a["decoder_gain"],a["decoder_offset"],a["prior"],data,configs["step03_neural"]["projection_pilot"],
            "neuromax","random_words",seed,correction=False)
        save_arrays(folder/"readout.npz",native=a["native"],refined=affinity)
        write_json(folder/"reference_audit.json",audit)
    value={"identity":identity,"row":row,"targets":targets,"certificate":cert,"selection":selection}
    write_json(folder/"case_result.json",value);write_json(folder/"artifact_manifest.json",artifact_manifest(folder));return value


def export_progress(output,rows,ablations,seeds):
    write_json(output/"all_rows.json",rows);write_json(output/"ablation_rows.json",ablations)
    fields=["method","profile","seed","role","status","c_v","npmi","td","dev_macro_nll","rho"]
    def flat(r):
        return {"method":r["method"],"profile":r["profile"],"seed":r["seed"],"role":r["role"],"status":r["status"],
            "c_v":r.get("fitting",{}).get("c_v"),"npmi":r.get("fitting",{}).get("c_npmi"),"td":r.get("topic_diversity"),
            "dev_macro_nll":r.get("prediction",{}).get("macro_source_nll"),"rho":r.get("rho")}
    write_csv(output/"all_case_metrics.csv",[flat(r) for r in rows],fields)
    write_csv(output/"ablation_case_metrics.csv",[flat(r) for r in ablations],fields)
    means=summarize([enriched(r) for r in rows],seeds)
    if means: export_table(output,"five_seed_comparison",means,list(means[0]))
    am=summarize([enriched(r) for r in ablations],seeds)
    if am: export_table(output,"five_seed_ablations",am,list(am[0]))
    return means,am


def final_reports(output,rows,ablations,policy,log,projection_checks=None):
    expected=[("Proposed",policy["proposed_profile"])]+profiles_to_run(policy)
    assert_grid(rows,expected,policy["seeds"])
    assert_ablation_grid(ablations,policy["retained_controls"]+policy["new_acquisition_ablations"],policy["seeds"])
    means,am=export_progress(output,rows,ablations,policy["seeds"])
    comparisons=compare_profiles(means)
    write_json(output/"all_profile_comparisons.json",comparisons)
    if comparisons: export_table(output,"all_profile_comparisons",comparisons,list(comparisons[0]))
    availability=availability_gates(means,list(policy["primary_profiles"]))
    proposed=[r for r in rows if r["method"]=="Proposed"]
    intervals=[]
    for method,profile in profiles_to_run(policy):
        part=[r for r in rows if (r["method"],r["profile"])==(method,profile)]
        if all(r.get("document_losses") for r in proposed+part):
            intervals.append({"method":method,"profile":profile,**hierarchical_gain(proposed,part,policy["seeds"],policy["bootstrap"]["resamples"],policy["bootstrap"]["seed"])})
    write_json(output/"hierarchical_prediction_bootstrap.json",intervals)
    plot_seed_tradeoffs(output,ablations,policy["seeds"])
    all_scored=all(r["status"]=="scored" for r in rows)
    complete=all(r["complete_seed_grid"] for r in means)
    descriptor=availability["fixed_k_descriptor_panel_complete"]; predictive=availability["compatible_prediction_panel_complete"]
    observed=[value for c in comparisons for key,value in c.items() if key.endswith("_win_or_tie") and value is not None]
    winner=bool(observed) and all(observed)
    gains=[]
    for seed in policy["seeds"]:
        p=next(r for r in proposed if r["seed"]==seed)
        n=next(r for r in rows if r["method"]=="NeuroMax" and r["seed"]==seed)
        gain=None if p["status"]!="scored" else n["prediction"]["macro_source_nll"]-p["prediction"]["macro_source_nll"]
        gains.append({"seed":seed,"native_minus_proposed_nll":gain})
    ablation_available=all(r["status"]=="scored" for r in ablations)
    parent_mean=next(r for r in means if r["method"]=="NeuroMax")
    proposed_mean=next(r for r in means if r["method"]=="Proposed")
    finite_gains=[r["native_minus_proposed_nll"] for r in gains if r["native_minus_proposed_nll"] is not None]
    mean_parent_gain=float(np.mean(finite_gains)) if len(finite_gains)==len(policy["seeds"]) else None
    parent_prediction=mean_parent_gain is not None and mean_parent_gain>=policy["acceptance"]["minimum_mean_native_nll_gain"]
    parent_descriptors=all(proposed_mean[k+"_mean"] is not None and parent_mean[k+"_mean"] is not None
        and proposed_mean[k+"_mean"]>parent_mean[k+"_mean"] for k in ["fitting_c_v","fitting_c_npmi"])
    projection_checks=projection_checks or []
    projection_passed=(len(projection_checks)==len(policy["seeds"]) and {r["seed"] for r in projection_checks}==set(policy["seeds"])
        and all(r["certified"] and r["within_declared_movement_budget"] for r in projection_checks))
    ablation_comparisons=[]
    for r in am:
        entry={"method":r["method"],"profile":r["profile"],"complete_seed_grid":r["complete_seed_grid"]}
        for k in ["fitting_c_v","fitting_c_npmi","topic_diversity","development_macro_nll"]:
            x,y=proposed_mean[k+"_mean"],r[k+"_mean"]
            entry[k+"_proposed_gain"]=None if x is None or y is None else (y-x if k=="development_macro_nll" else x-y)
        ablation_comparisons.append(entry)
    write_json(output/"ablation_comparisons.json",ablation_comparisons)
    export_table(output,"ablation_comparisons",ablation_comparisons,list(ablation_comparisons[0]))
    gates={"engineering_passed":bool(all_scored and complete),"complete_five_seed_grid":complete,**availability,
        "all_available_profile_means_win_or_tie":winner,"native_parent_gains_by_seed":gains,
        "mean_native_parent_nll_gain":mean_parent_gain,"minimum_mean_native_nll_gain_passed":parent_prediction,
        "mean_descriptor_gain_over_native_passed":parent_descriptors,"complete_ablation_grid":True,
        "primary_projection_checks":projection_checks,"primary_projection_gate_passed":projection_passed,
        "every_seed_improves_parent_nll":all(r["native_minus_proposed_nll"] is not None and r["native_minus_proposed_nll"]>0 for r in gains),
        "all_ablation_readouts_available":ablation_available,
        "unavailable_ablation_cases":[{"role":r["role"],"seed":r["seed"],"status":r["status"]} for r in ablations if r["status"]!="scored"],
        "ready_for_untouched_primary_test_review":bool(all_scored and complete and descriptor and predictive and ablation_available
            and winner and parent_prediction and parent_descriptors and projection_passed),
        "test_or_external_used":False,"final_model_result":False,"coherence_scope":"common_fitting_reference_in_sample_descriptive",
        "next_action":"review_five_seed_replication_before_untouched_test_and_sensitivity"}
    write_json(output/"quality_gates.json",gates)
    lines=["# Step 5: frozen five-seed replication and mechanism ablation","",
        "Seeds 809/810 are unchanged inspected pilot cases; seeds 811/812/813 are new fixed replications.",
        "Nine selected baseline families plus the two stronger coherence challengers; every historical two-seed profile remains separately retained.",
        "C_v/NPMI use the common fitting reference. Prediction is observed-only development completion, not final test.",
        "", "| Model | Profile | C_v | NPMI | TD | Dev macro NLL |", "|---|---|---:|---:|---:|---:|"]
    def f(r,key):
        value,sd=r[key+"_mean"],r[key+"_sd"]
        return "NA" if value is None else f"{value:.6f} ± {sd:.6f}"
    for r in sorted(means,key=lambda r:(r["method"]!="Proposed",r["method"],r["profile"])):
        lines.append("| "+" | ".join([r["method"],r["profile"],f(r,"fitting_c_v"),f(r,"fitting_c_npmi"),f(r,"topic_diversity"),f(r,"development_macro_nll")])+" |")
    abb_lines=["# Five-seed mechanism ablations","","C_v/NPMI are fitting-reference scores. NLL is development completion.","",
        "| Variant | C_v | NPMI | TD | Dev macro NLL |", "|---|---:|---:|---:|---:|"]
    for r in sorted(am,key=lambda r:(r["method"]!="Proposed",r["method"])):
        abb_lines.append("| "+" | ".join([r["method"],f(r,"fitting_c_v"),f(r,"fitting_c_npmi"),f(r,"topic_diversity"),f(r,"development_macro_nll")])+" |")
    abb_lines += ["","Identical descriptor sets legitimately tie. Display-only retains the native predictor; actual target membership is audited separately.",
        "Native, index/range/L2, previous-pair, moment, no-backoff, no-stability and positive-only controls retain every seed, loss and unavailable case.",""]
    write_text(output/"ABLATION_SUMMARY.md","\n".join(abb_lines))
    lines += ["",f"Engineering complete: {gates['engineering_passed']}. Available-profile mean win/tie: {winner}.",
        f"All ablation readouts available: {ablation_available}. Ready for untouched-test review: {gates['ready_for_untouched_primary_test_review']}.",
        "", "Read ABLATION_SUMMARY.md, readout_mechanism.csv, hierarchical_prediction_bootstrap.json and all_profile_comparisons.json.",
        "Same-target projection controls and display-only can legitimately have identical descriptor scores. No-backoff can tie Proposed.",
        "Zero SD is retained for deterministic outputs. Failed/NA cases are not replaced or omitted. Negative gains remain losses.",
        "The bootstrap resamples both training seeds and independent groups; it remains exploratory after development protocol selection.",
        "Do not describe these results as untouched-test performance. No automatic profile change occurs after replication.",""]
    write_text(output/"SUMMARY.md","\n".join(lines));return gates


def parent_replication_gate(rows,policy):
    subset=[r for r in rows if r["method"] in {"Proposed","NeuroMax"}]
    assert_grid(subset,[("Proposed",policy["proposed_profile"]),("NeuroMax",policy["proposed_profile"])],policy["seeds"])
    means=summarize([enriched(r) for r in subset],policy["seeds"])
    p=next(r for r in means if r["method"]=="Proposed");n=next(r for r in means if r["method"]=="NeuroMax")
    nl=p["development_macro_nll_mean"];nn=n["development_macro_nll_mean"]
    gain=None if nl is None or nn is None else nn-nl
    descriptors=all(p[k+"_mean"] is not None and n[k+"_mean"] is not None and p[k+"_mean"]>n[k+"_mean"] for k in ["fitting_c_v","fitting_c_npmi"])
    passed=bool(all(r["status"]=="scored" for r in subset) and descriptors and p["topic_diversity_mean"]==1.
        and gain is not None and gain>=policy["acceptance"]["minimum_mean_native_nll_gain"])
    return {"passed":passed,"mean_native_minus_proposed_nll":gain,"descriptor_gain_passed":descriptors,
        "complete_five_seed_parent_grid":True,"means":means,"test_or_external_used":False,
        "purpose":"resource_gate_before_thirty_additional_competitor_fits_not_a_final_model_claim"}


def replicate_new_seeds(root,output,key,rows,ablations,data,vectors,contexts,native_folders,native_specs,comp_folders,comp_specs,configs,policy,device,prior,embedding,log):
    # Replicate all three new parents first; only then spend the remaining baseline budget.
    ordered=sorted(profiles_to_run(policy),key=lambda p:(p[0]!="NeuroMax",p[0],p[1]))
    jobs=[(seed,label,name) for label,name in ordered for seed in policy["new_seeds"]]
    for seed,label,name in jobs:
        method=LABEL_TO_METHOD[label];profile=profile_settings(configs,method,name)
        folder=output/"baseline_cases"/method/name/("seed"+str(seed))
        case_identity={"stage05_identity":key,"method":method,"profile":profile,"seed":seed,"embedding_sha256":embedding["vectors_sha256"]}
        if method in {"encot","neuromax"}:
            case=native_case(root,method,name,seed,data,vectors,contexts,native_folders,native_specs,configs["step03_neural"],device,prior,folder,log,case_identity)
        elif method in {"lda","nmf"}:
            case=classical_case(root,method,profile,seed,data,prior,folder,log,case_identity,configs["step04_competitors"]["prediction"]["shared_backoff_grid"])
        else:
            case=competitor_case(root,method,profile,seed,data,vectors,comp_folders,comp_specs,configs["step04_competitors"],device,prior,folder,log,case_identity)
        rows.append(case["row"])
        if method=="neuromax":
            generated=build_refinement_source(root,output,seed,folder,data,configs,log)
            case_policy=copy.deepcopy(configs["step03_improvement"]);case_policy.update(scope=policy["scope"],seeds=policy["seeds"])
            projection=fullset_case(root,generated,generated,{"base":method,"profile":name,"seed":seed},data,
                configs["step03_neural"],case_policy,output/"proposed_cases"/("seed"+str(seed)),log,{"stage05_identity":key,"seed":seed,"parent_index":file_hash(folder/"artifact_manifest.json")})
            ablations.extend(projection["rows"]);rows.append(next(r for r in projection["rows"] if r["role"]=="proposed"))
        assert_grid(rows,[("Proposed",policy["proposed_profile"])]+profiles_to_run(policy),policy["seeds"],allow_partial=True)
        export_progress(output,rows,ablations,policy["seeds"])
        r=case["row"]
        log.info("REPLICATION SAVED | %s %s seed=%d | status=%s | Cv=%s NPMI=%s dev NLL=%s | rows=%d/60",label,name,seed,r["status"],r["fitting"].get("c_v"),r["fitting"].get("c_npmi"),r["prediction"].get("macro_source_nll"),len(rows))
        if method=="neuromax" and seed==policy["new_seeds"][-1]:
            gate=parent_replication_gate(rows,policy)
            certificates=[read_json(output/"proposed_cases"/("seed"+str(s))/"case_result.json")["certificates"]["primary"] for s in policy["new_seeds"]]
            gate["new_primary_projection_gate_passed"]=all(c.get("success") and c.get("certified")
                and c.get("relative_l1_movement",float("inf"))<=configs["step03_improvement"]["projection"]["max_relative_l1_movement"] for c in certificates)
            gate["passed"] &= gate["new_primary_projection_gate_passed"]
            write_json(output/"parent_replication_gate.json",gate)
            log.info("FIVE-SEED PARENT GATE | passed=%s | mean NLL gain=%s | saved before 30 other baseline fits",gate["passed"],gate["mean_native_minus_proposed_nll"])
            if not gate["passed"] and policy["stop_before_other_baselines_if_parent_gate_fails"]:
                return rows,ablations
    return rows,ablations


def run(root,configs,versions,hardware,log):
    root=Path(root).resolve();policy=validate_protocol(configs,root);prepared=require_preparation(root)
    panel,rows=accepted_panel(root,prepared["preparation_id"],policy)
    snapshot=read_json(panel/"protocol_resolved_before_scoring.json")
    device=hardware["future_model_device"]
    if snapshot["device"]!=device: raise ValueError("Keep the same CPU/CUDA device as the accepted pilot for the mixed reused/new-seed replication.")
    for name,value in snapshot.get("versions",{}).items():
        if name in versions and value != versions[name]:
            raise ValueError("Keep the inspected Step 4 dependency versions for replication: "+name+" was "+str(value)+", now "+str(versions[name]))
    data=load_canonical(root/prepared["processed_relative_path"])
    source,completed,_,_,source_cases=proposed_source(root,prepared["preparation_id"])
    identity={"policy":policy,"preparation_id":prepared["preparation_id"],"step04_index_sha256":file_hash(panel/"artifact_manifest.json"),
        "versions":versions,"device":device,"implementation":{p.relative_to(root).as_posix():file_hash(p) for p in sorted((root/"src/rank_refinement").rglob("*.py"))}}
    key=digest(identity);output=root/"outputs/steps/step05_replication"/key[:20];output.mkdir(parents=True,exist_ok=True)
    write_json(log.folder/"preparation_working_path.json",{"relative_path":output.relative_to(root).as_posix()})
    if (output/"artifact_manifest.json").is_file():
        check_indexed_tree(output)
        prior_result=read_json(output/"run_result.json")
        if prior_result.get("step05_id")!=key or read_json(output/"protocol_resolved_before_scoring.json")!=identity:
            raise ValueError("Completed Step 5 cache identity changed.")
        replay=copy.deepcopy(prior_result)
        replay.update(cache_replay=True,new_neural_fits=0,new_classical_fits=0)
        log.info("STEP 5 CACHE | completed indexed result reused unchanged | new model fits=0")
        log.info("READ FIRST | %s",output/"SUMMARY.md")
        return replay,output
    write_json(output/"protocol_resolved_before_scoring.json",identity)
    write_json(output/"completed_step04_reference.json",{"relative_path":panel.relative_to(root).as_posix(),"artifact_manifest_sha256":file_hash(panel/"artifact_manifest.json")})
    for name in ["frozen_proposed_reference.json","source_neural_pilot_reference.json","repaired_saved_source_reference.json","completed_pilot_reference.json"]:
        write_json(output/name,read_json(panel/name))
    write_json(output/"historical_two_seed_profiles.json",read_json(panel/"all_rows.json"))
    write_json(output/"locked_final_profiles.json",policy["primary_profiles"])
    write_json(output/"data_firewall.json",{"fitting_ids":data["fit_ids"],"development_observed_ids":data["dev_ids"],
        "test_or_external_used":False,"encoder_fitted":False,"full_development_text_encoded":False,
        "acquisition_sampling_seed_varies_with_pipeline_seed":True,"human_labels_or_assessments":False})
    ablations=[copy.deepcopy(r) for c in source_cases for r in c["rows"]]
    export_progress(output,rows,ablations,policy["seeds"])
    native_folders,native_specs=resolve_native_sources(root,log);comp_folders,comp_specs=resolve_competitor_sources(root,log)
    vectors,embedding=encode_views(root,data,configs["acquisition_anchors"]["encoder"],device,log)
    write_json(output/"shared_encoder_manifest.json",embedding)
    contexts=global_context(data,vectors,configs["step03_neural"]["global_context"])
    prior=data["fit"].astype(np.float64).sum(0)+.5;prior/=prior.sum()
    log.info("STEP 5 | seeds=%s | reuse 24 comparison rows | maximum 33 new baseline fits | Proposed shares NeuroMax parent | test/external untouched",policy["seeds"])
    rows,ablations=replicate_new_seeds(root,output,key,rows,ablations,data,vectors,contexts,native_folders,native_specs,comp_folders,comp_specs,configs,policy,device,prior,embedding,log)
    parent_gate=read_json(output/"parent_replication_gate.json")
    if not parent_gate["passed"]:
        result={"status":"STEP05_PARENT_REVIEW_REQUIRED","engineering_passed":True,"preparation_gate_passed":True,
            "model_quality_established":False,"protocol":PROTOCOL,"preparation_id":prepared["preparation_id"],"step05_id":key,
            "processed_relative_path":output.relative_to(root).as_posix(),"comparison_rows":len(rows),"ablation_rows":len(ablations),
            "training_seeds":policy["seeds"],"complete_five_seed_grid":False,"ready_for_untouched_primary_test_review":False,
            "new_neural_fits":getattr(log,"new_neural_fits",0),"new_classical_fits":getattr(log,"new_classical_fits",0),
            "final_model_result":False,"full_experiment_gate_passed":False,"test_or_external_used":False}
        write_json(output/"quality_gates.json",{**result,"parent_replication_gate":parent_gate})
        write_text(output/"SUMMARY.md","# Step 5 paused after the five-seed parent check\n\nThe three new NeuroMax parent fits and their Proposed/retained controls are saved. The declared descriptor, prediction or certificate gate failed. Thirty additional baseline fits were avoided.\n\nRead parent_replication_gate.json, all_case_metrics.csv and ablation_case_metrics.csv. The full comparison panel and new acquisition-control panel are incomplete; unavailable full-grid means remain NA. Review this evidence before changing the method or running a final test.\n")
        write_json(output/"run_result.json",result);write_json(output/"artifact_manifest.json",artifact_manifest(output))
        write_json(root/"outputs/steps/step05_replication/LATEST.json",{"relative_path":output.relative_to(root).as_posix()})
        write_json(root/"state/step05_replication_result.json",result)
        log.info("STEP 5 PAUSED | saved five-seed parent results | remaining 30 baseline fits avoided | review required")
        log.info("READ FIRST | %s",output/"SUMMARY.md")
        return result,output
    original=root/read_json(panel/"source_neural_pilot_reference.json")["relative_path"]
    repaired=root/read_json(panel/"repaired_saved_source_reference.json")["relative_path"]
    mechanism=[]
    for seed in policy["seeds"]:
        c={"base":"neuromax","profile":"random_words","seed":seed}
        if seed in policy["reuse_seeds"]:
            evidence_folder=case_folder(original,c);readout_folder=case_folder(repaired,c)
            refinement=next(x for x in source_cases if x["seed"]==seed);refined_folder=source/"cases"/("seed"+str(seed))
        else:
            evidence_folder=readout_folder=case_folder(output/"refinement_sources",c)
            refined_folder=output/"proposed_cases"/("seed"+str(seed));refinement=read_json(refined_folder/"case_result.json")
        with np.load(readout_folder/"readouts_and_latents.npz",allow_pickle=False) as z: native=z["native"]
        with np.load(refined_folder/"readouts.npz",allow_pickle=False) as z:
            for role,field in [("native","native"),("proposed","candidate_primary"),("index_control","candidate_index"),("range_control","candidate_range"),("l2_control","candidate_l2"),("display_only","native")]:
                if field in z.files: mechanism.append(readout_mechanism(native,z[field],refinement["targets"],data["vocabulary"],role,seed))
        for variant in policy["new_acquisition_ablations"]:
            value=acquisition_ablation(root,seed,variant,evidence_folder,readout_folder,data,configs,
                output/"acquisition_ablations"/variant/("seed"+str(seed)),log,{"stage05_identity":key,"variant":variant,"seed":seed})
            ablations.append(value["row"])
            assert_ablation_grid(ablations,policy["retained_controls"]+policy["new_acquisition_ablations"],policy["seeds"],allow_partial=True)
            export_progress(output,rows,ablations,policy["seeds"])
            log.info("ABLATION SAVED | %s seed=%d | status=%s | dev NLL=%s",variant,seed,value["row"]["status"],value["row"]["prediction"].get("macro_source_nll"))
    write_csv(output/"readout_mechanism.csv",mechanism,list(mechanism[0]) if mechanism else ["role","seed"])
    projection_checks=[]
    for seed in policy["seeds"]:
        folder=source/"cases"/("seed"+str(seed)) if seed in policy["reuse_seeds"] else output/"proposed_cases"/("seed"+str(seed))
        cert=read_json(folder/"case_result.json")["certificates"]["primary"]
        movement=cert.get("relative_l1_movement")
        projection_checks.append({"seed":seed,"certified":bool(cert.get("success") and cert.get("certified")),
            "relative_l1_movement":movement,"within_declared_movement_budget":movement is not None and movement<=configs["step03_improvement"]["projection"]["max_relative_l1_movement"]})
    gates=final_reports(output,rows,ablations,policy,log,projection_checks)
    check_indexed_tree(panel)
    result={"status":"STEP05_REPLICATION_COMPLETE" if gates["ready_for_untouched_primary_test_review"] else "STEP05_REVIEW_REQUIRED",
        "engineering_passed":gates["engineering_passed"],"preparation_gate_passed":True,"model_quality_established":False,
        "protocol":PROTOCOL,"preparation_id":prepared["preparation_id"],"step05_id":key,"processed_relative_path":output.relative_to(root).as_posix(),
        "comparison_rows":len(rows),"ablation_rows":len(ablations),"training_seeds":policy["seeds"],
        "all_available_profile_means_win_or_tie":gates["all_available_profile_means_win_or_tie"],
        "ready_for_untouched_primary_test_review":gates["ready_for_untouched_primary_test_review"],
        "new_neural_fits":getattr(log,"new_neural_fits",0),"new_classical_fits":getattr(log,"new_classical_fits",0),
        "final_model_result":False,"full_experiment_gate_passed":False,"test_or_external_used":False}
    write_json(output/"run_result.json",result);write_json(output/"artifact_manifest.json",artifact_manifest(output))
    write_json(root/"outputs/steps/step05_replication/LATEST.json",{"relative_path":output.relative_to(root).as_posix()})
    write_json(root/"state/step05_replication_result.json",result)
    log.info("STEP 5 COMPLETE | engineering=%s | mean win/tie=%s | untouched-test review ready=%s",result["engineering_passed"],result["all_available_profile_means_win_or_tie"],result["ready_for_untouched_primary_test_review"])
    log.info("READ FIRST | %s",output/"SUMMARY.md");return result,output
