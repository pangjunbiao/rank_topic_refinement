"""Reviewed Step 5 freeze and one untouched-primary-test evaluation.

The old every-metric-win gate is preserved. A separately named review decision
accepts the observed small diversity tradeoff for test evaluation, not as a
claim of universal superiority. Models, seeds, profiles and rho stay fixed.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from ..runtime.artifacts import artifact_manifest, atomic_bytes, digest, file_hash, read_json, write_json, write_text
from ..runtime.state import require_preparation
from ..runtime.logging import utc_now
from ..models.neural_training import load_canonical
from ..models.neural_sources import resolve_sources as native_sources
from ..models.competitor_sources import resolve_sources as competitor_sources
from ..models.neural_prediction import log_predictions, backoff, moment_parameters, score_log_predictions
from ..models.frozen_inference import (arrays, assert_replay, checkpoint_index, restore_neural,
    saved_embeddings, frozen_encoder_folder, encode_test_observed, context_inputs, native_latents, competitor_probabilities, classical_probabilities)
from ..evaluation.replication import assert_grid, assert_ablation_grid, hierarchical_gain, readout_mechanism
from ..evaluation.competitor_evidence import summarize
from ..evaluation.frozen_test import load_test_view, frozen_probability_scores, heldout_descriptor_audit, summarize_test
from ..reporting.tables import export_table
from .step02_completion import check_indexed_tree
from .step02_native_refinement import save_arrays
from .step04_primary_benchmark import safe_step_folder
from .step05_ablations import profiles_to_run, profile_settings, LABEL_TO_METHOD
from .step05_recovery import validate_protocol as validate_recovery, seed_sources

PROTOCOL='STEP06_FROZEN_PRIMARY_TEST_R1_20261010'
POLICY_SHA256='9c5c4a7f6cea0c25fe9710b71b0648da84723279d926e4c01d3da46aafa4d7f1'


def validate_protocol(configs, root=None):
    _,old=validate_recovery(configs,root)
    p=configs.get('step06_frozen_test',{})
    if p.get('protocol')!=PROTOCOL or digest(p)!=POLICY_SHA256:
        raise ValueError('The test protocol changed. The frozen test may not be used to select a replacement method.')
    if (p['seeds']!=old['seeds'] or p['freeze']['refit_neural_models']
            or p['freeze']['reselect_graph_projection_profiles_seeds_or_rho']
            or p['freeze']['external_evaluation'] or p['freeze']['correction_in_proposed']
            or p['review_decision']['change_original_step05_gates']):
        raise ValueError('Invalid frozen-test scope.')
    return p,old


def review_step05(rows, controls, gates, policy, old):
    expected=[('Proposed','random_words')]+profiles_to_run(old)
    assert_grid(rows,expected,policy['seeds'])
    if len(controls)!=60 or len({(r['role'],r['seed']) for r in controls})!=60:
        raise ValueError('Require all 60 Step 5 control slots.')
    if (not gates.get('engineering_passed') or not gates.get('complete_five_seed_grid')
            or not gates.get('complete_ablation_grid') or not gates.get('actual_readout_mechanism_passed')
            or not gates.get('fixed_k_descriptor_panel_complete') or not gates.get('compatible_prediction_panel_complete')
            or not gates.get('global_recovery_selection',{}).get('passed')
            or gates['global_recovery_selection']['selected_configuration']!=policy['selected_configuration']
            or gates.get('test_or_external_used') or any(r['status']!='scored' for r in rows)):
        raise ValueError('Step 5 is not complete/certified on the unchanged development protocol.')
    missing=[(r['role'],r['seed']) for r in controls if r['status']!='scored']
    if missing!=[('strict_word_disjointness',812)]:
        raise ValueError('Unexpected unavailable controls; the reviewed exception was only strict disjointness seed 812.')
    means=summarize(rows,policy['seeds']);pro=next(r for r in means if r['method']=='Proposed')
    failures=[];tradeoffs=[]
    for mean in means:
        if mean['method']=='Proposed':continue
        for field in ['fitting_c_v','fitting_c_npmi']:
            if pro[field+'_mean']<mean[field+'_mean']-1e-12:failures.append((mean['method'],mean['profile'],field))
        pred=mean['development_macro_nll_mean']
        if pred is not None and pro['development_macro_nll_mean']>pred+1e-12:failures.append((mean['method'],mean['profile'],'development_macro_nll'))
        gap=mean['topic_diversity_mean']-pro['topic_diversity_mean']
        if gap>1e-12:
            tradeoffs.append({'method':mean['method'],'profile':mean['profile'],'mean_td_gap':gap})
            if gap>policy['review_decision']['maximum_baseline_mean_td_gap']:failures.append((mean['method'],mean['profile'],'topic_diversity_gap'))
    indexed={r['seed']:r for r in rows if r['method']=='NeuroMax'}
    if any(r['topic_diversity']+1e-12<indexed[r['seed']]['topic_diversity'] for r in rows if r['method']=='Proposed'):
        failures.append(('Proposed','random_words','native_diversity_not_preserved'))
    if failures:raise ValueError('Observed Step 5 does not satisfy the separately declared test-review policy: '+str(failures))
    return {'decision':'FREEZE_FOR_UNTOUCHED_TEST_WITH_EXPLICIT_DIVERSITY_TRADEOFF',
        'step05_experiments_complete':True,'original_status_preserved':'STEP05_RECOVERY_REVIEW_REQUIRED',
        'original_all_metric_win_gate':gates['all_available_profile_means_win_or_tie'],
        'original_untouched_test_ready_gate':gates['ready_for_untouched_primary_test_review'],
        'diversity_tradeoffs':tradeoffs,'unavailable_control_slots':missing,
        'all_ablation_metric_superiority_claim':False,'statistical_superiority_over_native_established':False,
        'step05_metrics_or_gates_changed':False,'test_or_external_used_for_review':False,
        'scope':'engineering complete; promising development point estimates; freeze before test; no 98-percent success claim'}


def accepted_source(root, preparation_id, policy, configs):
    root=Path(root).resolve()
    pointer=root/'outputs/steps/step05_recovery/LATEST.json';state=root/'state/step05_recovery_result.json'
    if pointer.is_file():relative=read_json(pointer)['relative_path']
    elif state.is_file():relative=read_json(state)['processed_relative_path']
    else:raise RuntimeError('Keep your completed Step 5 --recover outputs/state; Step 6 does not repeat it.')
    source=safe_step_folder(root,relative,'step05_recovery');check_indexed_tree(source)
    result=read_json(source/'run_result.json')
    if (file_hash(source/'artifact_manifest.json')!=policy['source_index_sha256']
            or result.get('step05_recovery_id')!=policy['source_recovery_id']
            or result.get('protocol')!=policy['source_protocol']
            or result.get('preparation_id')!=preparation_id or preparation_id!=policy['preparation_id']
            or result.get('test_or_external_used') or result.get('training_seeds')!=policy['seeds']):
        raise ValueError('Require the exact reviewed Step 5 R2 source and accepted preparation; no source replacement.')
    rows=read_json(source/'all_rows.json');controls=read_json(source/'ablation_rows.json')
    _,old=validate_recovery(configs,root)
    assert_ablation_grid(controls,configs['step05_recovery']['controls'],policy['seeds'])
    review=review_step05(rows,controls,read_json(source/'quality_gates.json'),policy,old)
    return source,rows,controls,review


def source_folders(root, source):
    stages={'completed_step05_reference.json':'step05_replication',
        'completed_step04_reference.json':'step04_competitor_pilot',
        'source_neural_pilot_reference.json':'step03_neural_pilot',
        'completed_pilot_reference.json':'step03_completion'}
    result={}
    for name,stage in stages.items():
        ref=read_json(source/name);folder=safe_step_folder(root,ref['relative_path'],stage)
        if file_hash(folder/'artifact_manifest.json')!=ref['artifact_manifest_sha256']:raise ValueError('Frozen upstream source changed.')
        check_indexed_tree(folder);result[stage]=folder
    return result


def native_folder(sources, source, method, profile, seed):
    if seed in [809,810]:return sources['step03_neural_pilot']/'cases'/method/profile/f'seed{seed}'
    panel=sources['step05_replication'] if method=='neuromax' else source
    return panel/'baseline_cases'/method/profile/f'seed{seed}'


def other_folder(sources, source, method, profile, seed):
    if seed not in [809,810]:return source/'baseline_cases'/method/profile/f'seed{seed}'
    if method in ['lda','nmf']:return sources['step03_completion']/'classical_cases'/method/profile/f'seed{seed}'
    return sources['step04_competitor_pilot']/'cases'/method/profile/f'seed{seed}'


def expected_arrays(folder):
    check_indexed_tree(folder)
    value=arrays(folder/'native_readout_and_predictions.npz')
    # Earlier classical completion used shorter field names. They are the same
    # saved native arrays; only the interface names are mapped here.
    if 'weights' in value:value['native_descriptor_weights']=value.pop('weights')
    if 'probabilities' in value:value['native_decoded_probabilities']=value.pop('probabilities')
    return value


def preflight(root, source, sources, rows, data, configs, policy, device, versions):
    snapshot=read_json(source/'protocol_resolved_before_scoring.json')
    if snapshot['device']!=device:raise ValueError('Keep the original Step 5 device for exact frozen inference.')
    for name,value in snapshot['versions'].items():
        if name in versions and versions[name]!=value:raise ValueError('Keep the original frozen dependency version: '+name)
    manifest=read_json(source/'shared_encoder_manifest.json');vectors=saved_embeddings(root,manifest,data)
    checkpoints=[]
    for row in rows:
        method=LABEL_TO_METHOD.get(row['method'])
        if method in ['neuromax','encot']:
            folder=native_folder(sources,source,method,row['profile'],row['seed'])
            check_indexed_tree(folder);details=read_json(folder/'native_training.json')
            path,_=checkpoint_index(root,details,('data','cache','step03_native'))
            checkpoints.append({'method':row['method'],'seed':row['seed'],'path':details['checkpoint_relative_path'],'sha256':file_hash(path)})
        elif method in ['combinedtm','fastopic','glocom']:
            folder=other_folder(sources,source,method,row['profile'],row['seed'])
            check_indexed_tree(folder);details=read_json(folder/'case_result.json')['training_details']
            path,_=checkpoint_index(root,details,('data','cache','step04_training'))
            checkpoints.append({'method':row['method'],'seed':row['seed'],'path':details['checkpoint_relative_path'],'sha256':file_hash(path)})
    # Check all encoder files before exposing any test outcome.
    frozen_encoder_folder(root, manifest['assets'])
    return vectors,manifest,checkpoints


def freeze_test_exposure(root, identity):
    path=Path(root)/'state/step06_test_exposure.json';key=digest(identity)
    if path.is_file():
        original=read_json(path)
        if original.get('freeze_id')!=key or original.get('identity')!=identity:
            raise ValueError('Primary test was already exposed under another frozen protocol. Retuning and rescoring it is blocked.')
        return key,True
    write_json(path,{'freeze_id':key,'identity':identity,'utc_first_exposure':utc_now(),
        'allow_identical_restart_only':True,'test_metrics_must_not_select_a_new_method':True})
    return key,False


def prepare_development_interfaces(root,source,sources,rows,data,vectors,configs,policy,device,output,nf,cf,log):
    """Verify every frozen inference path before any primary-test exposure."""
    checks=[]
    for row in rows:
        method=LABEL_TO_METHOD.get(row['method'])
        if method is None or method=='bertopic':continue
        seed=row['seed']
        if method in ['neuromax','encot']:
            folder=native_folder(sources,source,method,row['profile'],seed)
            training=read_json(folder/'native_training.json')
            if method=='neuromax':
                _,_,expected=seed_sources(root,sources['step05_replication'],seed,configs['step05_replication'])
                inputs=data['dev_observed']
            else:
                expected=arrays(folder/'readouts_and_latents.npz')
                context=arrays(sources['step03_neural_pilot']/'fitting_global_context.npz')
                inputs=context_inputs(data['dev_observed'],vectors['development_observed'],context['bank'],context['centers'])
            model=restore_neural(root,method,row['profile'],seed,training,nf[method],vectors,configs,device)
            _,errors=native_latents(model,inputs,inputs,expected,policy['inference'],device)
            del model
        else:
            folder=other_folder(sources,source,method,row['profile'],seed)
            expected=expected_arrays(folder);details=read_json(folder/'case_result.json')['training_details']
            profile=profile_settings(configs,method,row['profile'])
            if method in ['combinedtm','fastopic','glocom']:
                model=restore_neural(root,method,profile,seed,details,cf[method],vectors,configs,device)
                _,_,errors=competitor_probabilities(method,model,data,vectors,data['dev_observed'],
                    vectors['development_observed'],expected,seed,policy['inference']);del model
            else:
                destination=output/'baseline_cases'/method/row['profile']/f'seed{seed}'
                destination.mkdir(parents=True,exist_ok=True)
                _,_,errors=classical_probabilities(method,data,None,seed,profile,expected,details,policy['inference'],destination,log)
        checks.append({'method':row['method'],'profile':row['profile'],'seed':seed,'development_replay_checks':errors})
        write_json(output/'pretest_development_replay.json',{'cases':checks,'primary_test_used':False,'new_neural_fits':0})
        log.info('PRETEST REPLAY | %s %s seed=%d | verified | primary test not exposed',row['method'],row['profile'],seed)
    if len(checks)!=45:raise ValueError('Require all 45 compatible native/baseline development interfaces before test exposure.')
    return checks


def test_row(old, view, sources, logp=None, probabilities=None):
    row={k:copy.deepcopy(old[k]) for k in ['method','profile','seed','role','topics','fitting','topic_diversity','rho','prediction_interface'] if k in old}
    row.update(status=old['status'],scope='frozen_untouched_primary_test',final_primary_test_result=True,
        final_paper_claim_approved=False,model_configuration_selected_on='development_only_before_test',
        rho_selected_on='development_only_before_test',test_used_for_training_or_selection=False,
        moment_correction=old.get('moment_correction',False),projection_deployed=old.get('projection_deployed',False),
        original_development_prediction=copy.deepcopy(old['prediction']))
    if old['status']!='scored' or (logp is None and probabilities is None):
        row.update(prediction={'macro_source_nll':None,'micro_nll':None},document_losses=[],
            prediction_unavailable_reason=old.get('failure',old.get('prediction_unavailable_reason','native_word_likelihood_not_defined')))
    elif logp is not None:
        row['prediction'],row['document_losses']=score_log_predictions(logp,view['records'],view['masks'],sources)
    else:
        row['prediction'],row['document_losses']=frozen_probability_scores(probabilities,old['rho'],view['prior'],view,sources)
    return row


def save_test_case(folder, identity, rows, replay, **values):
    save_arrays(folder/'frozen_test_arrays.npz',**values)
    result={'identity':identity,'rows':rows,'development_replay_checks':replay,'neural_training_performed':False,
        'test_used_for_fit_or_parameter_selection':False}
    write_json(folder/'case_result.json',result)
    write_json(folder/'artifact_manifest.json',artifact_manifest(folder))
    return result


def cached_case(folder, identity):
    if not (folder/'artifact_manifest.json').is_file():return None
    check_indexed_tree(folder);result=read_json(folder/'case_result.json')
    if result['identity']!=identity:raise ValueError('Frozen test case cache identity changed.')
    return result


def proposed_and_controls(root,source,sources,seed,old_rows,controls,data,view,vectors,configs,policy,device,folder,key,native_folders):
    identity={'freeze_id':key,'seed':seed,'kind':'all_twelve_proposed_control_slots'}
    cached=cached_case(folder,identity)
    if cached:return cached
    folder.mkdir(parents=True,exist_ok=True)
    native_case=native_folder(sources,source,'neuromax','random_words',seed)
    training=read_json(native_case/'native_training.json')
    r1=sources['step05_replication'];_,_,a=seed_sources(root,r1,seed,configs['step05_replication'])
    model=restore_neural(root,'neuromax','random_words',seed,training,native_folders['neuromax'],vectors,configs,device)
    theta,replay=native_latents(model,data['dev_observed'],view['observed_counts'],a,policy['inference'],device)
    del model
    b=arrays(source/'matched_controls'/f'seed{seed}'/'readouts.npz')
    targets=read_json(source/'matched_controls'/f'seed{seed}'/'case_result.json')['targets']
    mechanism=readout_mechanism(a['native'],b['proposed'],targets,data['vocabulary'],'proposed',seed)
    if (mechanism['target_top10_membership_recall']<1.-1e-12
            or mechanism['maximum_topic_total_error_scaled']>1e-5 or mechanism['maximum_word_total_error_scaled']>1e-5):
        raise ValueError('Frozen Proposed readout failed its realization/conservation certificate.')
    results=[];stored={'theta_test':theta,'native':a['native'],'proposed':b['proposed'],
        'decoder_gain':a['decoder_gain'],'decoder_offset':a['decoder_offset'],'prior':a['prior']}
    expected_roles=configs['step05_recovery']['controls']
    for role in expected_roles:
        row=next(r for r in controls if r['seed']==seed and r['role']==role)
        if row['status']!='scored':results.append(test_row(row,view,view['sources']));continue
        if role in ['positive_only_association','strict_word_disjointness']:
            phi=arrays(source/'acquisition_controls'/role/f'seed{seed}'/'readouts.npz')['range']
        elif role=='alternate_calibration':phi=arrays(source/'candidate_cases/signed_full'/f'seed{seed}'/'readouts.npz')['range']
        else:
            field={'primary_control':'primary','range_control':'range','moment_control':'proposed','no_backoff':'proposed','display_only':'native'}.get(role,role)
            phi=b[field]
        moment=moment_parameters(a['theta_fit'],a['native'],phi) if role=='moment_control' else None
        prediction=backoff(log_predictions(theta,phi,a['decoder_gain'],a['decoder_offset'],moment),a['prior'],row['rho'])
        result=test_row(row,view,view['sources'],logp=prediction)
        result['actual_readout']='native' if role in ['native','display_only'] else role
        result['frozen_affinity_sha256']=__import__('hashlib').sha256(phi.tobytes()).hexdigest()
        if role=='proposed':result['mechanism']=mechanism
        results.append(result);stored['readout_'+role]=phi
        if moment is not None:stored['moment_gain'],stored['moment_offset']=moment
    for method,role in [('Proposed','proposed'),('NeuroMax','native')]:
        source_row=next(r for r in old_rows if r['method']==method and r['seed']==seed)
        result=next(r for r in results if r['role']==role)
        if result['topics']!=source_row['topics'] or result['rho']!=source_row['rho']:raise ValueError('Proposed/parent comparison and control paths disagree.')
    return save_test_case(folder,identity,results,replay,**stored)


def baseline_case(root,source,sources,old,data,view,vectors,test_vectors,configs,policy,device,folder,key,native_folders,comp_folders,log):
    identity={'freeze_id':key,'method':old['method'],'profile':old['profile'],'seed':old['seed']}
    cached=cached_case(folder,identity)
    if cached:return cached
    folder.mkdir(parents=True,exist_ok=True)
    method=LABEL_TO_METHOD[old['method']];seed=old['seed'];stored={};replay={}
    if method=='bertopic':
        row=test_row(old,view,view['sources'])
        row['prediction_unavailable_reason']='BERTopic native word likelihood is not defined; no fabricated word probability'
    elif method=='encot':
        src=native_folder(sources,source,method,old['profile'],seed);check_indexed_tree(src)
        training=read_json(src/'native_training.json');expected=arrays(src/'readouts_and_latents.npz')
        context=arrays(sources['step03_neural_pilot']/'fitting_global_context.npz')
        dev_input=context_inputs(data['dev_observed'],vectors['development_observed'],context['bank'],context['centers'])
        test_input=context_inputs(view['observed_counts'],test_vectors,context['bank'],context['centers'])
        model=restore_neural(root,method,old['profile'],seed,training,native_folders[method],vectors,configs,device)
        theta,replay=native_latents(model,dev_input,test_input,expected,policy['inference'],device);del model
        lp=backoff(log_predictions(theta,expected['native'],expected['decoder_gain'],expected['decoder_offset']),view['prior'],old['rho'])
        row=test_row(old,view,view['sources'],logp=lp)
        stored={'theta_test':theta,'native':expected['native'],'decoder_gain':expected['decoder_gain'],'decoder_offset':expected['decoder_offset']}
    else:
        src=other_folder(sources,source,method,old['profile'],seed);expected=expected_arrays(src)
        details=read_json(src/'case_result.json')['training_details'];profile=profile_settings(configs,method,old['profile'])
        if method in ['combinedtm','fastopic','glocom']:
            model=restore_neural(root,method,profile,seed,details,comp_folders[method],vectors,configs,device)
            prob,theta,replay=competitor_probabilities(method,model,data,vectors,view['observed_counts'],test_vectors,expected,seed,policy['inference']);del model
        else:
            prob,theta,replay=classical_probabilities(method,data,view['observed_counts'],seed,profile,expected,details,policy['inference'],folder,log)
        row=test_row(old,view,view['sources'],probabilities=prob)
        stored={'native_decoded_probabilities':prob,'theta_test':theta,'frozen_source_descriptor_weights':expected['native_descriptor_weights']}
    return save_test_case(folder,identity,[row],replay,**stored)


def add_descriptor_audits(output, rows, view):
    for row in rows:
        if row['status']!='scored' or not row.get('topics'):
            row['test_coherence']={'c_v':None,'c_npmi':None,'failure':'readout_unavailable_no_slot_removal'}
            continue
        scores,audit=heldout_descriptor_audit(row['topics'],view,output/'heldout_coherence_audits')
        row['test_coherence']=scores
    return rows


def export_progress(output, rows, controls, policy, complete=False):
    write_json(output/'all_test_rows.json',rows);write_json(output/'all_test_control_rows.json',controls)
    write_json(output/'progress.json',{'comparison_slots':len(rows),'expected_comparison_slots':60,
        'control_slots':len(controls),'expected_control_slots':60,'complete':complete,'test_parameters_selected':False})
    flat=[]
    for panel,panel_rows in [('comparisons',rows),('controls',controls)]:
        for r in panel_rows:
            flat.append({'panel':panel,'method':r['method'],'profile':r['profile'],
                'seed':r['seed'],'role':r['role'],'status':r['status'],'fitting_c_v':r['fitting'].get('c_v'),
                'fitting_c_npmi':r['fitting'].get('c_npmi'),'topic_diversity':r.get('topic_diversity'),
                'test_c_v':r.get('test_coherence',{}).get('c_v'),'test_c_npmi':r.get('test_coherence',{}).get('c_npmi'),
                'test_macro_nll':r['prediction']['macro_source_nll'],'test_micro_nll':r['prediction']['micro_nll'],'frozen_rho':r.get('rho')})
    if flat:export_table(output,'incremental_test_case_metrics',flat,list(flat[0]))


def report(root,output,rows,controls,data,policy,old,review):
    assert_grid(rows,[('Proposed','random_words')]+profiles_to_run(old),policy['seeds'])
    means=summarize_test(rows,policy['seeds']);am=summarize_test(controls,policy['seeds'])
    export_table(output,'primary_test_comparison_mean_sd',means,list(means[0]))
    export_table(output,'primary_test_controls_mean_sd',am,list(am[0]))
    p=next(r for r in means if r['method']=='Proposed');comparisons=[];intervals=[]
    proposed=[r for r in rows if r['method']=='Proposed']
    for mean in means:
        if mean['method']=='Proposed':continue
        entry={'method':mean['method'],'profile':mean['profile'],'likelihood_defined':mean['test_macro_nll_mean'] is not None}
        for field in ['fitting_c_v','fitting_c_npmi','topic_diversity','test_c_v','test_c_npmi','test_macro_nll']:
            a,b=p[field+'_mean'],mean[field+'_mean'];gain=None if a is None or b is None else (b-a if field=='test_macro_nll' else a-b)
            entry[field+'_proposed_gain']=gain;entry[field+'_win_or_tie']=None if gain is None else gain>=-1e-12
        comparisons.append(entry)
        part=[r for r in rows if (r['method'],r['profile'])==(mean['method'],mean['profile'])]
        if all(r.get('document_losses') for r in proposed+part):
            interval=hierarchical_gain(proposed,part,policy['seeds'],policy['reporting']['bootstrap_resamples'],policy['reporting']['bootstrap_seed'])
            interval.update(scope='frozen_primary_test_paired_seed_group_bootstrap_unadjusted_descriptive',
                primary_contrast=mean['method']=='NeuroMax',multiplicity_adjusted=False)
            intervals.append({'method':mean['method'],'profile':mean['profile'],**interval})
    for role in sorted({r['role'] for r in controls}):
        part=[r for r in controls if r['role']==role]
        if role not in ['proposed','native'] and all(r.get('document_losses') for r in proposed+part):
            value=hierarchical_gain(proposed,part,policy['seeds'],policy['reporting']['bootstrap_resamples'],policy['reporting']['bootstrap_seed'])
            value.update(scope='frozen_primary_test_control_bootstrap_unadjusted_descriptive',primary_contrast=False,multiplicity_adjusted=False)
            intervals.append({'method':part[0]['method'],'profile':'control/'+role,**value})
    write_json(output/'paired_prediction_intervals.json',intervals)
    export_table(output,'paired_prediction_intervals',intervals,list(intervals[0]))
    export_table(output,'primary_test_profile_comparisons',comparisons,list(comparisons[0]));write_json(output/'primary_test_profile_comparisons.json',comparisons)
    from ..reporting.frozen_test_figures import make_figures, fitting_cases
    make_figures(output,rows,controls,policy['seeds'])
    fitting_cases(output,controls,data)
    defined=[e['test_macro_nll_win_or_tie'] for e in comparisons if e['test_macro_nll_win_or_tie'] is not None]
    gate={'complete_comparison_slots':True,'complete_control_slots':True,'new_neural_fits':0,
        'test_or_external_used_for_model_selection':False,'frozen_test_wins_all_defined_mean_prediction_comparisons':all(defined),
        'all_metric_and_all_control_win_claim':False,'statistical_superiority_claim_automatically_approved':False,
        'heldout_coherence_finite_for_all_comparison_rows':all(r['test_coherence']['c_v'] is not None and r['test_coherence']['c_npmi'] is not None for r in rows),
        'comparison_rows':len(rows),'control_slots':len(controls),
        'unavailable_control_slots':[{'role':r['role'],'seed':r['seed'],'status':r['status']} for r in controls if r['status']!='scored']}
    write_json(output/'quality_gates.json',gate)
    def fmt(row,k):
        v,s=row[k+'_mean'],row[k+'_sd'];return 'NA' if v is None else f'{v:.6f} ± {s:.6f}'
    lines=['# Step 6: frozen primary test evaluation','',
        'C_v/NPMI in the main comparison use the unchanged common fitting reference and are descriptive. Test NLL uses only observed test evidence for inference. Full test-reference coherence is a separately labelled diagnostic with all slots retained.','',
        '| Model | Profile | Fitting C_v | Fitting NPMI | TD | Test macro NLL |',
        '|---|---|---:|---:|---:|---:|']
    for row in sorted(means,key=lambda r:(r['method']!='Proposed',r['method'],r['profile'])):
        lines.append('| '+' | '.join([row['method'],row['profile']]+[fmt(row,k) for k in ['fitting_c_v','fitting_c_npmi','topic_diversity','test_macro_nll']])+' |')
    lines+=['','The original Step 5 all-metric-win gate remains false. Its small diversity tradeoffs and strict-disjointness failure were accepted explicitly for frozen testing, not erased.',
        'Neural checkpoints, descriptors, graphs, projection, seeds and development-selected rho were frozen before test exposure. LDA/NMF use declared fitting-only replay because earlier stages did not persist estimators; saved development predictions must match before test inference. GSDMM is restored from saved sufficient statistics.',
        'All five seeds and both challenger profiles remain. BERTopic word NLL is NA because its native word likelihood is undefined.',
        'Intervals jointly resample training seeds and independent duplicate groups within source. They are unadjusted descriptive intervals; they do not automatically authorize statistical-superiority claims.',
        'Automatic fitting-document examples are illustrative candidate themes, not measured requirement-detection accuracy or human interpretability.',
        'Further model selection using these test outcomes would invalidate the untouched-test interpretation. Next: external generalization with these same frozen models; then manuscript assembly.','']
    write_text(output/'SUMMARY.md','\n'.join(lines))
    for name,values in [('T2_frozen_primary_test_comparison',means),('T3_frozen_primary_test_controls',am)]:
        # Compact presentation tables; full numeric fields remain in the
        # stage and supplement exports rather than a dozens-of-columns table.
        display=[]
        for row in sorted(values,key=lambda r:(r['method']!='Proposed',r['method'],r['profile'])):
            display.append({'Model':row['method'],'Profile':row['profile'],
                'Fitting C_v':fmt(row,'fitting_c_v'),'Fitting NPMI':fmt(row,'fitting_c_npmi'),
                'TD':fmt(row,'topic_diversity'),'Test macro NLL':fmt(row,'test_macro_nll')})
        export_table(root/'outputs/paper/main/tables',name,display,list(display[0]))
    for name,values in [('S6_primary_test_all_profiles',means),('S6_primary_test_all_controls',am),('S6_primary_test_prediction_intervals',intervals)]:
        export_table(root/'outputs/paper/supplementary/tables',name,values,list(values[0]))
    return gate


def run(root, configs, versions, hardware, log):
    root=Path(root).resolve();policy,old=validate_protocol(configs,root);prepared=require_preparation(root)
    source,source_rows,source_controls,review=accepted_source(root,prepared['preparation_id'],policy,configs)
    sources=source_folders(root,source);data=load_canonical(root/prepared['processed_relative_path']);device=hardware['future_model_device']
    log.info('STEP 5 REVIEW | 60 comparisons and 60 control slots complete | preserved all-metric-win=False | frozen test tradeoff review accepted')
    vectors,manifest,checkpoints=preflight(root,source,sources,source_rows,data,configs,policy,device,versions)
    nf,ns=native_sources(root,log);cf,cs=competitor_sources(root,log)
    # Identity covers code/config/data/source/checkpoints before any test score.
    identity={'protocol':policy,'preparation_id':prepared['preparation_id'],'source_index_sha256':file_hash(source/'artifact_manifest.json'),
        'versions':versions,'device':device,'checkpoints':checkpoints,
        'implementation':{p.relative_to(root).as_posix():file_hash(p) for p in sorted((root/'src/rank_refinement').rglob('*.py'))},
        'configs_sha256':digest(configs),'native_sources':ns,'competitor_sources':cs}
    key=digest(identity);output=root/'outputs/steps/step06_frozen_test'/key[:20];output.mkdir(parents=True,exist_ok=True)
    write_json(log.folder/'preparation_working_path.json',{'relative_path':output.relative_to(root).as_posix()})
    if (output/'artifact_manifest.json').is_file():
        check_indexed_tree(output)
        if read_json(output/'protocol_resolved_before_test.json')!=identity:raise ValueError('Completed test cache identity changed.')
        freeze_test_exposure(root,identity)
        result=read_json(output/'run_result.json');log.info('FROZEN TEST CACHE | identical completed result reused | new fits=0')
        return {**result,'cache_replay':True,'new_classical_fits':0},output
    write_json(output/'protocol_resolved_before_test.json',identity);write_json(output/'step05_review_decision.json',review)
    write_json(output/'reviewed_step05_recovery_reference.json',{'relative_path':source.relative_to(root).as_posix(),'artifact_manifest_sha256':file_hash(source/'artifact_manifest.json')})
    for name in ['completed_step05_reference.json','completed_step04_reference.json','source_neural_pilot_reference.json','repaired_saved_source_reference.json','completed_pilot_reference.json','frozen_proposed_reference.json']:
        write_json(output/name,read_json(source/name))
    # Estimators and all native interfaces are verified using fitting and
    # development only. A dependency/replay problem cannot expose test results.
    prepare_development_interfaces(root,source,sources,source_rows,data,vectors,configs,policy,device,output,nf,cf,log)
    development_sensitivity=read_json(source/'global_recovery_selection.json')['means']
    export_table(output,'pretest_development_graph_projection_sensitivity',development_sensitivity,list(development_sensitivity[0]))
    write_json(output/'pretest_sensitivity_scope.json',{'source':'completed_step05_global_four_configuration_search',
        'reference':'fitting_coherence_and_development_completion','new_search':False,'test_used':False})
    # Do not evaluate test outcomes during --self-check or --doctor-only.
    freeze_test_exposure(root,identity)
    view=load_test_view(root/prepared['processed_relative_path'],data['fit_ids'],data['dev_ids'])
    view['prior']=data['fit'].astype(np.float64).sum(0)+.5;view['prior']/=view['prior'].sum()
    view['sources']=sorted({r['source'] for r in data['fit_records']})
    write_json(output/'test_population.json',{'group_ids':view['ids'],'vocabulary_size':len(view['vocabulary']),
        'observed_text_only_inference':True,'sources':view['sources'],'no_test_used_for_training_or_selection':True})
    log.info('PRIMARY TEST FREEZE | configuration=%s | five seeds | groups=%d | neural refits=0 | up to ten fitting-only LDA/NMF replays',policy['selected_configuration'],len(view['ids']))
    test_vectors=encode_test_observed(root,manifest,view['ids'],view['observed_texts'],device,output,log)
    rows=[];controls=[]
    for seed in policy['seeds']:
        value=proposed_and_controls(root,source,sources,seed,source_rows,source_controls,data,view,vectors,configs,policy,device,
            output/'proposed_control_cases'/f'seed{seed}',key,nf)
        controls.extend(value['rows'])
        for method,role in [('Proposed','proposed'),('NeuroMax','native')]:
            r=copy.deepcopy(next(r for r in value['rows'] if r['role']==role));r['method']=method;rows.append(r)
            log.info('TEST SAVED | %s seed=%d | macro NLL=%s | frozen rho=%s',method,seed,r['prediction']['macro_source_nll'],r['rho'])
        export_progress(output,rows,controls,policy)
    for oldrow in source_rows:
        if oldrow['method'] in ['Proposed','NeuroMax']:continue
        folder=output/'baseline_cases'/LABEL_TO_METHOD[oldrow['method']]/oldrow['profile']/f"seed{oldrow['seed']}"
        value=baseline_case(root,source,sources,oldrow,data,view,vectors,test_vectors,configs,policy,device,folder,key,nf,cf,log)
        rows.extend(value['rows']);r=value['rows'][0]
        log.info('TEST SAVED | %s %s seed=%d | NLL=%s | rows=%d/60',r['method'],r['profile'],r['seed'],r['prediction']['macro_source_nll'],len(rows))
        export_progress(output,rows,controls,policy)
    log.info('HELDOUT COHERENCE | full frozen test reference | all ten words/topic retained | undefined values remain NA')
    add_descriptor_audits(output,rows+controls,view)
    export_progress(output,rows,controls,policy,complete=True)
    assert_ablation_grid(controls,configs['step05_recovery']['controls'],policy['seeds'])
    gates=report(root,output,rows,controls,data,policy,old,review)
    write_json(output/'data_firewall.json',{'fitting_group_ids':data['fit_ids'],'development_observed_group_ids':data['dev_ids'],
        'test_observed_group_ids':view['ids'],'test_full_text_encoded':False,'test_parameters_selected':False,
        'external_data_used':False,'neural_training_performed':False,'coherence_test_reference_used_only_after_freeze':True,
        'human_labels_or_assessments':False,'development_rho_frozen_every_case':True})
    check_indexed_tree(source)
    result={'status':'STEP06_FROZEN_PRIMARY_TEST_COMPLETE','protocol':PROTOCOL,'step06_freeze_id':key,
        'preparation_id':prepared['preparation_id'],'source_step05_recovery_id':policy['source_recovery_id'],
        'processed_relative_path':output.relative_to(root).as_posix(),'engineering_passed':True,'preparation_gate_passed':True,
        'model_quality_established':False,'final_primary_test_result':True,'final_paper_claim_approved':False,
        'training_seeds':policy['seeds'],'comparison_rows':len(rows),'control_slots':len(controls),'test_completion_groups':len(view['ids']),
        'test_evaluated':True,'external_evaluated':False,'test_used_for_model_or_rho_selection':False,
        'new_neural_fits':0,'new_classical_fits':getattr(log,'new_classical_fits',0),
        'frozen_test_wins_all_defined_mean_prediction_comparisons':gates['frozen_test_wins_all_defined_mean_prediction_comparisons'],
        'original_step05_all_metric_win_gate_preserved':False,'selected_configuration':policy['selected_configuration']}
    write_json(output/'run_result.json',result);write_json(output/'artifact_manifest.json',artifact_manifest(output))
    write_json(root/'state/step06_result.json',result)
    write_json(root/'outputs/steps/step06_frozen_test/LATEST.json',{'relative_path':result['processed_relative_path'],'step06_freeze_id':key})
    log.info('STEP 6 COMPLETE | frozen test comparison and controls saved | new neural fits=0 | test mean prediction win/tie=%s',result['frozen_test_wins_all_defined_mean_prediction_comparisons'])
    log.info('READ FIRST | %s',output/'SUMMARY.md')
    return result,output
