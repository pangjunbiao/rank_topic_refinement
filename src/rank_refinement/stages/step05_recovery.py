"""Named development recovery of Step 5; no parent retraining or test scoring.

Four GLOBAL configurations (two graphs by two certified projections) retain
every seed. The selected configuration is then compared with unchanged locked
competitors. Original R1 results remain immutable and separately referenced.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from ..runtime.artifacts import artifact_manifest, digest, file_hash, read_json, write_json, write_csv, write_text
from ..runtime.state import require_preparation
from ..models.neural_training import load_canonical, encode_views, global_context
from ..models.neural_sources import resolve_sources as native_sources
from ..models.competitor_sources import resolve_sources as competitor_sources
from ..models.adapters.pilot_anchors import topics_from_weights
from ..acquisition.native_diversity_targets import recovery_targets
from ..projection.replay_solver import primary_tv, range_secondary, index_secondary
from ..projection.predictive_controls import l2_projection
from ..projection.minimum_tv import certify
from ..evaluation.diversity import topic_diversity
from ..evaluation.competitor_evidence import reference_scores, summarize, availability_gates
from ..evaluation.replication import assert_grid, assert_ablation_grid, hierarchical_gain, readout_mechanism
from ..reporting.tables import export_table
from ..reporting.recovery_figures import plot_recovery
from .step02_completion import check_indexed_tree
from .step02_native_refinement import save_arrays
from .step03_improvement import load_evidence, failed_row
from .step03_neural_pilot import prediction_row
from .step03_saved_replay import case_folder
from .step04_primary_benchmark import safe_step_folder, enriched, compare_profiles, competitor_case
from .step05_ablations import (validate_protocol as validate_r1, profiles_to_run, profile_settings,
                              native_case, classical_case, export_progress, LABEL_TO_METHOD, POLICY_SHA256 as R1_SHA256)

PROTOCOL = 'STEP05_NATIVE_DIVERSITY_RECOVERY_R2_20261010'
POLICY_SHA256 = 'e7fe1f635010b06dfc81c8c403ae67b16149c0280362635afad6f50f209c2fa0'
LABELS = {'native':'NeuroMax', 'proposed':'Proposed', 'primary_control':'Primary TV control',
    'range_control':'Prediction-aware range control', 'index_control':'Index secondary control',
    'l2_control':'L2 control', 'moment_control':'Proposed with moment correction',
    'no_backoff':'Proposed without backoff', 'display_only':'Same-core display only',
    'alternate_calibration':'Alternate signed graph', 'positive_only_association':'Positive-only associations',
    'strict_word_disjointness':'Absolute 100-word disjointness control'}


def validate_protocol(configs, root=None):
    old = validate_r1(configs, root)
    p = configs.get('step05_recovery', {})
    if p.get('protocol') != PROTOCOL or digest(p) != POLICY_SHA256:
        raise ValueError('Recovery policy changed; declare a new named revision before scoring.')
    if p['seeds'] != old['seeds'] or p['test_evaluation'] or p['external_evaluation'] or p['per_seed_method_selection']:
        raise ValueError('Recovery must retain all five seeds, one global configuration and the data firewall.')
    return p, old


def accepted_source(root, preparation_id, policy):
    root = Path(root).resolve()
    pointer = root/'outputs/steps/step05_replication/LATEST.json'
    state = root/'state/step05_replication_result.json'
    if pointer.is_file(): relative = read_json(pointer)['relative_path']
    elif state.is_file(): relative = read_json(state)['processed_relative_path']
    else: raise RuntimeError('Keep your original Step 5 outputs/state. Recovery uses the five saved parent models.')
    source = safe_step_folder(root, relative, 'step05_replication')
    check_indexed_tree(source)
    r = read_json(source/'run_result.json')
    snapshot = read_json(source/'protocol_resolved_before_scoring.json')
    if (r.get('protocol') != policy['source_protocol'] or r.get('preparation_id') != preparation_id
            or not r.get('engineering_passed') or r.get('test_or_external_used')
            or r.get('training_seeds') != policy['seeds'] or digest(snapshot['policy']) != R1_SHA256):
        raise ValueError('Require the indexed original five-seed Step 5 parent panel on the accepted preparation.')
    rows = read_json(source/'all_rows.json')
    parents = [r for r in rows if r['method'] == 'NeuroMax']
    assert_grid(parents, [('NeuroMax', policy['profile'])], policy['seeds'])
    if any(r['status'] != 'scored' for r in parents): raise ValueError('All five saved native parents must be available.')
    return source, rows, snapshot


def seed_sources(root, source, seed, old):
    c = {'base':'neuromax', 'profile':'random_words', 'seed':seed}
    if seed in old['reuse_seeds']:
        folders = []
        for name, stage in [('source_neural_pilot_reference.json','step03_neural_pilot'),
                            ('repaired_saved_source_reference.json','step03_saved_replay')]:
            ref = read_json(source/name)
            f = safe_step_folder(root, ref['relative_path'], stage)
            if file_hash(f/'artifact_manifest.json') != ref['artifact_manifest_sha256']:
                raise ValueError('Saved parent source changed.')
            check_indexed_tree(f); folders.append(case_folder(f,c))
        evidence_folder, arrays_folder = folders
    else:
        if seed not in old['new_seeds']: raise ValueError('Foreign recovery seed.')
        evidence_folder = arrays_folder = case_folder(source/'refinement_sources',c)
    check_indexed_tree(evidence_folder); check_indexed_tree(arrays_folder)
    with np.load(arrays_folder/'readouts_and_latents.npz',allow_pickle=False) as z:
        arrays = {n:z[n] for n in z.files}
    required = {'native','theta_fit','theta_development','decoder_gain','decoder_offset','prior'}
    if not required.issubset(arrays): raise ValueError('Incomplete saved decoder/latent interface.')
    return evidence_folder, arrays_folder, arrays


def score_readout(root, arrays, affinity, data, configs, seed, role, label, correction=False, backoff=True):
    topics = topics_from_weights(affinity, data['vocabulary'])
    fit, dev, audit = reference_scores(topics, data, root/'data/cache/step05_recovery_coherence')
    row = prediction_row(role,label,affinity,topics,fit,dev,topic_diversity(topics),
        arrays['theta_fit'],arrays['theta_development'],arrays['native'],arrays['decoder_gain'],
        arrays['decoder_offset'],arrays['prior'],data,configs['step03_neural']['projection_pilot'],
        'neuromax','random_words',seed,correction=correction,use_backoff=backoff,deployed=role!='native')
    row.update(scope='exploratory_five_seed_development_recovery',final_model_result=False,
               prediction_interface='frozen_native_affine_decoder',development_selected_revision=True)
    return enriched(row), audit


def unavailable(seed, role, failure, configs):
    row = failed_row(role,LABELS.get(role,role),seed,failure,configs['step03_improvement'])
    row.update(failure_kind='readout_unavailable_no_native_substitution',final_model_result=False,
               scope='exploratory_five_seed_development_recovery')
    return enriched(row)


def saved_contexts(evidence_folder, arrays, data):
    group_ids = read_json(evidence_folder/'range_contexts.json')['group_ids']
    positions = {g:i for i,g in enumerate(data['fit_observed_ids'])}
    if len(group_ids) != 64 or len(set(group_ids)) != 64 or not set(group_ids).issubset(positions):
        raise ValueError('Range contexts must be the same 64 saved independent fitting-observed groups.')
    return arrays['theta_fit'][[positions[g] for g in group_ids]]


def projected_case(root, source, seed, graph, data, configs, old, folder, log, identity,
                   modes=('primary','range'), strict=False):
    if (folder/'artifact_manifest.json').is_file():
        check_indexed_tree(folder); value = read_json(folder/'case_result.json')
        if value['identity'] != identity: raise ValueError('Recovery case identity changed.')
        return value
    folder.mkdir(parents=True,exist_ok=True)
    ef, af, a = seed_sources(root,source,seed,old)
    evidence = load_evidence(ef,data['fit_records'])
    target, selection = recovery_targets(a['native'],data['vocabulary'],data['metadata'],evidence,
        configs['step03_improvement']['acquisition'],graph,strict=strict)
    projection = configs['step03_improvement']['projection']; controls = configs['step03_improvement']['controls']
    readouts = {}; certificates = {}; rows = {}; audits = {}; mechanism = []
    phi = None; certificate = {'certified':False,'failure':'no_feasible_supported_edit'}
    if selection['required_diversity_achieved'] and selection['selected']:
        phi, certificate = primary_tv(a['native'],target,projection['rank_margin_scaled'],projection['primary_seconds'])
    certificates['primary'] = certificate
    if phi is not None:
        readouts['primary'] = phi
        if 'range' in modes:
            secondary, cert = range_secondary(a['native'],target,certificate['solver_tv_scaled'],
                saved_contexts(ef,a,data),a['decoder_gain'],phi,projection['rank_margin_scaled'],
                controls['tv_cap_slack_scaled'],controls['secondary_seconds'],
                controls['range_maximum_iterations'],controls['range_tolerance'],log)
            certificates['range'] = cert
            if secondary is not None: readouts['range'] = secondary
    for mode in modes:
        if mode in readouts:
            check = certify(a['native'],readouts[mode],target,margin=projection['rank_margin_scaled'])
            if not check['certified']: raise ValueError('Fresh recovery readout failed independent certification.')
            row,audit = score_readout(root,a,readouts[mode],data,configs,seed,'candidate',graph+'/'+mode)
            mech = readout_mechanism(a['native'],readouts[mode],target,data['vocabulary'],mode,seed)
            if mech['target_top10_membership_recall'] < 1.-1e-12:
                raise ValueError('Actual readout does not realize every selected TOP10 target.')
            row.update(recovery_graph=graph,recovery_projection=mode,projection_certificate_passed=True,
                       relative_l1_movement=mech['relative_l1_affinity_movement'])
            mechanism.append(mech); audits[mode]=audit
        else:
            row=unavailable(seed,'candidate',certificates.get(mode,certificate).get('failure','projection_not_certified'),configs)
            row.update(method=graph+'/'+mode,recovery_graph=graph,recovery_projection=mode,
                       projection_certificate_passed=False,relative_l1_movement=None)
        rows[mode]=row
    save_arrays(folder/'readouts.npz',**readouts)
    value={'identity':identity,'seed':seed,'graph':graph,'targets':target,'selection':selection,
        'certificates':certificates,'rows':rows,'mechanism':mechanism,'reference_audits':audits,
        'source_arrays_manifest_sha256':file_hash(af/'artifact_manifest.json')}
    write_json(folder/'case_result.json',value);write_json(folder/'artifact_manifest.json',artifact_manifest(folder))
    return value


def select_global(rows, parents, policy):
    profiles=[(g+'/'+p,'random_words') for g in policy['recovery_graphs'] for p in policy['projection_candidates']]
    assert_grid(rows,profiles,policy['seeds'],allow_partial=True)
    assert_grid(parents,[('NeuroMax','random_words')],policy['seeds'])
    means=summarize(rows,policy['seeds']); native=summarize([enriched(r) for r in parents],policy['seeds'])[0]
    indexed={r['seed']:r for r in parents}; decisions=[]
    for method,profile in profiles:
        part=[r for r in rows if r['method']==method]
        mean=next((m for m in means if m['method']==method),None)
        complete=(len(part)==len(policy['seeds']) and {r['seed'] for r in part}==set(policy['seeds'])
                  and all(r['status']=='scored' for r in part))
        gain=None if not complete else native['development_macro_nll_mean']-mean['development_macro_nll_mean']
        diversity=complete and all(r['topic_diversity']+1e-12>=indexed[r['seed']]['topic_diversity'] for r in part)
        descriptors=complete and all(mean[k+'_mean']>native[k+'_mean'] for k in ['fitting_c_v','fitting_c_npmi'])
        certified=complete and all(r.get('projection_certificate_passed') and
            r.get('relative_l1_movement',float('inf'))<=policy['acceptance']['maximum_relative_l1_movement'] for r in part)
        passed=bool(complete and diversity and descriptors and certified and gain is not None
                    and gain>=policy['acceptance']['minimum_mean_native_nll_gain'])
        decisions.append({'configuration':method,'complete_scored_five_seed_grid':complete,'passed':passed,
            'mean_native_minus_candidate_nll':gain,'native_diversity_preserved_every_seed':bool(diversity),
            'descriptor_gain_passed':bool(descriptors),'certificates_and_movement_passed':bool(certified),
            'mean_development_nll':mean['development_macro_nll_mean'] if mean else None})
    eligible=[d for d in decisions if d['passed']]
    selected=min(eligible,key=lambda d:(d['mean_development_nll'],d['configuration']))['configuration'] if eligible else None
    return {'passed':selected is not None,'selected_configuration':selected,'decisions':decisions,'means':means,
        'selection_unit':'one_global_configuration_for_all_five_seeds','per_seed_method_selection':False,
        'exploratory_post_selection':True,'test_or_external_used':False}


def matched_controls(root, source, seed, selected, cases, data, configs, old, output, log, identity):
    destination=output/'matched_controls'/('seed'+str(seed))
    if (destination/'artifact_manifest.json').is_file():
        check_indexed_tree(destination); value=read_json(destination/'case_result.json')
        if value['identity'] != identity: raise ValueError('Matched-control identity changed.')
        return value
    destination.mkdir(parents=True,exist_ok=True)
    graph,mode=selected.split('/'); c=cases[(seed,graph)]; target=c['targets']
    _,_,a=seed_sources(root,source,seed,old)
    with np.load(output/'candidate_cases'/graph/('seed'+str(seed))/'readouts.npz',allow_pickle=False) as z:
        readouts={k:z[k] for k in z.files}
    readouts['proposed']=readouts[mode]; readouts['native']=a['native']
    pp=configs['step03_improvement']['projection']; cp=configs['step03_improvement']['controls']
    certificates={}
    for role,solver in [('index_control',index_secondary),('l2_control',l2_projection)]:
        if role=='index_control':
            phi,cert=solver(a['native'],target,c['certificates']['primary']['solver_tv_scaled'],
                           pp['rank_margin_scaled'],cp['tv_cap_slack_scaled'],cp['secondary_seconds'])
        else: phi,cert=solver(a['native'],target,pp['rank_margin_scaled'],cp['l2_seconds'],cp['l2_epsilon'])
        certificates[role]=cert
        if phi is not None: readouts[role]=phi
    rows=[]; mechanism=[]
    for role in ['native','proposed','primary_control','range_control','index_control','l2_control','moment_control','no_backoff','display_only']:
        field={'primary_control':'primary','range_control':'range','moment_control':'proposed',
               'no_backoff':'proposed','display_only':'proposed'}.get(role,role)
        if field not in readouts: row=unavailable(seed,role,'projection_not_certified',configs)
        elif role=='display_only':
            row=copy.deepcopy(next(r for r in rows if r['role']=='native'))
            display=c['rows'][mode]
            row.update(role=role,method=LABELS[role],topics=display['topics'],fitting=display['fitting'],
                development=display['development'],topic_diversity=display['topic_diversity'],projection_deployed=False,
                display_scores_from_projected_topics=True,readout_is_native=True)
        else:
            row,_=score_readout(root,a,readouts[field],data,configs,seed,role,LABELS[role],
                               correction=role=='moment_control',backoff=role!='no_backoff')
        rows.append(row)
        if field in readouts:
            actual=a['native'] if role=='display_only' else readouts[field]
            mechanism.append(readout_mechanism(a['native'],actual,target,data['vocabulary'],role,seed))
    other=next(g for g in configs['step05_recovery']['recovery_graphs'] if g!=graph)
    alternate=copy.deepcopy(cases[(seed,other)]['rows'][mode]);alternate.update(role='alternate_calibration',method=LABELS['alternate_calibration'])
    rows.append(alternate)
    for role,changed_graph,strict in [('positive_only_association','positive_only_stable' if graph=='signed_stable' else 'positive_only_full',False),
                                    ('strict_word_disjointness',graph,True)]:
        v=projected_case(root,source,seed,changed_graph,data,configs,old,output/'acquisition_controls'/role/('seed'+str(seed)),
            log,{**identity,'role':role},modes=('primary',mode) if mode!='primary' else ('primary',),strict=strict)
        row=copy.deepcopy(v['rows'][mode]);row.update(role=role,method=LABELS[role]);rows.append(row)
        for m in v['mechanism']:
            if m['role']==mode: mechanism.append({**m,'role':role})
    save_arrays(destination/'readouts.npz',**readouts)
    value={'identity':identity,'rows':rows,'mechanism':mechanism,'targets':target,'certificates':certificates}
    write_json(destination/'case_result.json',value);write_json(destination/'artifact_manifest.json',artifact_manifest(destination))
    return value


def completion_jobs(old, rows):
    profiles=profiles_to_run(old)
    assert_grid(rows,[('Proposed','random_words')]+profiles,old['seeds'],allow_partial=True)
    have={(r['method'],r['profile'],r['seed']) for r in rows}
    jobs=[(s,m,p) for m,p in profiles if m!='NeuroMax' for s in old['new_seeds'] if (m,p,s) not in have]
    if len(jobs)>30: raise ValueError('Unexpected expansion of the locked completion budget.')
    return jobs


def complete_baselines(root,output,key,rows,ablations,data,configs,old,device,log):
    jobs=completion_jobs(old,rows)
    if not jobs: return rows
    # Only reached after the saved-parent recovery gate passes.
    nf,ns=native_sources(root,log);cf,cs=competitor_sources(root,log)
    vectors,embedding=encode_views(root,data,configs['acquisition_anchors']['encoder'],device,log)
    write_json(output/'shared_encoder_manifest.json',embedding)
    contexts=global_context(data,vectors,configs['step03_neural']['global_context'])
    prior=data['fit'].astype(np.float64).sum(0)+.5;prior/=prior.sum()
    for seed,label,name in jobs:
        method=LABEL_TO_METHOD[label];profile=profile_settings(configs,method,name)
        folder=output/'baseline_cases'/method/name/('seed'+str(seed))
        identity={'recovery_id':key,'method':method,'profile':profile,'seed':seed,'embedding_sha256':embedding['vectors_sha256']}
        if method=='encot':
            value=native_case(root,method,name,seed,data,vectors,contexts,nf,ns,configs['step03_neural'],device,prior,folder,log,identity)
        elif method in {'lda','nmf'}:
            value=classical_case(root,method,profile,seed,data,prior,folder,log,identity,configs['step04_competitors']['prediction']['shared_backoff_grid'])
        else:
            value=competitor_case(root,method,profile,seed,data,vectors,cf,cs,configs['step04_competitors'],device,prior,folder,log,identity)
        rows.append(value['row']);export_progress(output,rows,ablations,old['seeds'])
        r=value['row'];log.info('BASELINE SAVED | %s %s seed=%d | status=%s | Cv=%s NPMI=%s NLL=%s | rows=%d/60',
            label,name,seed,r['status'],r['fitting'].get('c_v'),r['fitting'].get('c_npmi'),r['prediction'].get('macro_source_nll'),len(rows))
    return rows


def report_tables(output, rows, ablations, policy, old, selection, mechanism):
    expected=[('Proposed','random_words')]+profiles_to_run(old)
    assert_grid(rows,expected,policy['seeds'])
    assert_ablation_grid(ablations,policy['controls'],policy['seeds'])
    means,am=export_progress(output,rows,ablations,policy['seeds'])
    comparisons=compare_profiles(means);write_json(output/'all_profile_comparisons.json',comparisons)
    export_table(output,'all_profile_comparisons',comparisons,list(comparisons[0]))
    availability=availability_gates(means,list(old['primary_profiles']))
    observed=[v for r in comparisons for k,v in r.items() if k.endswith('_win_or_tie') and v is not None]
    allwin=bool(observed) and all(observed)
    intervals=[]; proposed=[r for r in rows if r['method']=='Proposed']
    for method,profile in profiles_to_run(old):
        part=[r for r in rows if (r['method'],r['profile'])==(method,profile)]
        if all(r.get('document_losses') and r['status']=='scored' for r in proposed+part):
            intervals.append({'method':method,'profile':profile,**hierarchical_gain(proposed,part,policy['seeds'],old['bootstrap']['resamples'],old['bootstrap']['seed'])})
    write_json(output/'hierarchical_prediction_bootstrap.json',intervals)
    write_csv(output/'readout_mechanism.csv',mechanism,list(mechanism[0]))
    pm=next(r for r in means if r['method']=='Proposed');ac=[]
    for r in am:
        entry={'method':r['method'],'complete_seed_grid':r['complete_seed_grid']}
        for k in ['fitting_c_v','fitting_c_npmi','topic_diversity','development_macro_nll']:
            x,y=pm[k+'_mean'],r[k+'_mean'];entry[k+'_proposed_gain']=None if x is None or y is None else (y-x if k=='development_macro_nll' else x-y)
        ac.append(entry)
    for entry in ac:
        for k in ['fitting_c_v','fitting_c_npmi','topic_diversity','development_macro_nll']:
            gain=entry[k+'_proposed_gain'];entry[k+'_win_or_tie']=None if gain is None else gain>=-1e-12
    write_json(output/'ablation_comparisons.json',ac);export_table(output,'ablation_comparisons',ac,list(ac[0]))
    plot_recovery(output,ablations,policy['seeds'])
    scored=all(r['status']=='scored' for r in rows)
    unavailable_controls=[{'role':r['role'],'seed':r['seed'],'status':r['status'],'failure':r.get('failure')} for r in ablations if r['status']!='scored']
    primary_audit=[m for m in mechanism if m['role']=='proposed']
    mechanism_passed=len(primary_audit)==5 and all(m['target_top10_membership_recall']>=1.-1e-12 and
        m['maximum_topic_total_error_scaled']<=1e-5 and m['maximum_word_total_error_scaled']<=1e-5 for m in primary_audit)
    ready=bool(scored and allwin and selection['passed'] and mechanism_passed and
               availability['fixed_k_descriptor_panel_complete'] and availability['compatible_prediction_panel_complete'])
    gates={'engineering_passed':bool(scored and mechanism_passed),'complete_five_seed_grid':True,
        'complete_ablation_grid':True,'all_available_profile_means_win_or_tie':allwin,
        'all_ablation_readouts_available':not unavailable_controls,'unavailable_ablation_cases':unavailable_controls,
        'actual_readout_mechanism_passed':mechanism_passed,'ready_for_untouched_primary_test_review':ready,
        'all_available_ablation_means_win_or_tie':all(v for entry in ac for k,v in entry.items() if k.endswith('_win_or_tie') and v is not None),
        'global_recovery_selection':selection,'coherence_scope':'common_fitting_reference_in_sample_descriptive',
        'test_or_external_used':False,'final_model_result':False,**availability}
    write_json(output/'quality_gates.json',gates)
    def formatted(r,key):
        v,sd=r[key+'_mean'],r[key+'_sd'];return 'NA' if v is None else f'{v:.6f} ± {sd:.6f}'
    def table(title, values):
        lines=[title,'','C_v/NPMI use the common fitting reference; NLL is development completion. These are not final test results.','',
            '| Model | Profile | C_v | NPMI | TD | Dev macro NLL |','|---|---|---:|---:|---:|---:|']
        for r in sorted(values,key=lambda r:(r['method']!='Proposed',r['method'])):
            lines.append('| '+' | '.join([r['method'],r['profile']]+[formatted(r,k) for k in ['fitting_c_v','fitting_c_npmi','topic_diversity','development_macro_nll']])+' |')
        return lines
    lines=table('# Step 5 R2: recovery and locked competitor completion',means)
    lines+=['',f"One global selected configuration: {selection['selected_configuration']}.",
        f'Available-profile mean win/tie, including diversity: {allwin}. Untouched-test review ready: {ready}.',
        'Preserving native diversity replaces the absolute 100-word requirement; promoted cores remain disjoint.',
        'The original R1 seed-812 failure and every original metric remain in historical_step05_r1_rows.json.',
        'The four-configuration search and any uncertainty are exploratory after development selection.',
        'Ties with same-target controls/display-only are legitimate; neither a missing control nor a missing likelihood is a win.',
        'Review all_profile_comparisons.json, ABLATION_SUMMARY.md and quality_gates.json before Step 6.','']
    write_text(output/'SUMMARY.md','\n'.join(lines))
    write_text(output/'ABLATION_SUMMARY.md','\n'.join(table('# Step 5 R2 matched controls',am)+['',
        'Unavailable controls retain all five slots; their full-grid means are NA. Strict disjointness is a feasibility control.',
        'Display-only has the same descriptors but the native readout and native predictor; see readout_mechanism.csv.','']))
    return gates


def run(root, configs, versions, hardware, log):
    root=Path(root).resolve();policy,old=validate_protocol(configs,root);prepared=require_preparation(root)
    source,source_rows,snapshot=accepted_source(root,prepared['preparation_id'],policy)
    device=hardware['future_model_device']
    if snapshot['device']!=device: raise ValueError('Keep the original Step 5 CPU/CUDA device for remaining competitor fits.')
    for name,value in snapshot['versions'].items():
        if name in versions and versions[name]!=value: raise ValueError('Keep original replication dependency version: '+name)
    identity={'protocol':policy,'preparation_id':prepared['preparation_id'],'source_index_sha256':file_hash(source/'artifact_manifest.json'),
        'versions':versions,'device':device,'implementation':{p.relative_to(root).as_posix():file_hash(p) for p in sorted((root/'src/rank_refinement').rglob('*.py'))}}
    key=digest(identity);output=root/'outputs/steps/step05_recovery'/key[:20];output.mkdir(parents=True,exist_ok=True)
    write_json(log.folder/'preparation_working_path.json',{'relative_path':output.relative_to(root).as_posix()})
    if (output/'artifact_manifest.json').is_file():
        check_indexed_tree(output);r=read_json(output/'run_result.json')
        if r['step05_recovery_id']!=key or read_json(output/'protocol_resolved_before_scoring.json')!=identity: raise ValueError('Recovery cache identity changed.')
        log.info('STEP 5 RECOVERY CACHE | indexed results reused | new fits=0');return {**r,'cache_replay':True,'new_neural_fits':0,'new_classical_fits':0},output
    write_json(output/'protocol_resolved_before_scoring.json',identity)
    write_json(output/'completed_step05_reference.json',{'relative_path':source.relative_to(root).as_posix(),'artifact_manifest_sha256':file_hash(source/'artifact_manifest.json')})
    for name in ['completed_step04_reference.json','frozen_proposed_reference.json','source_neural_pilot_reference.json','repaired_saved_source_reference.json','completed_pilot_reference.json']:
        if (source/name).is_file(): write_json(output/name,read_json(source/name))
    write_json(output/'historical_step05_r1_rows.json',source_rows)
    write_json(output/'historical_step05_r1_ablations.json',read_json(source/'ablation_rows.json'))
    data=load_canonical(root/prepared['processed_relative_path']);parents=[copy.deepcopy(r) for r in source_rows if r['method']=='NeuroMax']
    write_json(output/'data_firewall.json',{'fitting_group_ids':data['fit_ids'],
        'fitting_observed_group_ids':data['fit_observed_ids'],'development_observed_group_ids':data['dev_ids'],
        'acquisition_uses_fitting_only':True,'full_development_text_encoded':False,'encoder_fitted':False,
        'test_or_external_scored':False,'human_labels_or_assessments':False,
        'global_method_selected_on_development':True,'parent_training_seeds_unchanged':policy['seeds']})
    cases={};candidate_rows=[]
    log.info('RECOVERY | five saved parents | new neural fits=0 | encoder calls=0 | four global configurations | original R1 retained')
    for seed in policy['seeds']:
        for graph in policy['recovery_graphs']:
            value=projected_case(root,source,seed,graph,data,configs,old,output/'candidate_cases'/graph/('seed'+str(seed)),log,
                {'recovery_id':key,'seed':seed,'graph':graph})
            cases[(seed,graph)]=value;candidate_rows.extend(value['rows'].values())
            write_json(output/'recovery_candidate_rows.json',candidate_rows)
            for mode,row in value['rows'].items():
                log.info('RECOVERY SAVED | %s/%s seed=%d | status=%s | Cv=%s NPMI=%s NLL=%s TD=%s',
                    graph,mode,seed,row['status'],row['fitting'].get('c_v'),row['fitting'].get('c_npmi'),row['prediction'].get('macro_source_nll'),row.get('topic_diversity'))
    selection=select_global(candidate_rows,parents,policy);write_json(output/'global_recovery_selection.json',selection)
    export_table(output,'recovery_configurations_mean_sd',selection['means'],list(selection['means'][0]))
    log.info('RECOVERY GATE | passed=%s | selected=%s | one configuration for all seeds',selection['passed'],selection['selected_configuration'])
    gates={'engineering_passed':True,'complete_five_seed_grid':False,'ready_for_untouched_primary_test_review':False,
           'all_available_profile_means_win_or_tie':False}
    rows=[];ablations=[]
    if selection['passed']:
        selected=selection['selected_configuration'];graph,mode=selected.split('/')
        rows=[copy.deepcopy(r) for r in source_rows if r['method']!='Proposed']
        # Never mix new and old Proposed rows; every seed uses the selected revision.
        mechanism=[]
        for seed in policy['seeds']:
            p=copy.deepcopy(cases[(seed,graph)]['rows'][mode]);p.update(method='Proposed',role='proposed')
            rows.append(p)
            log.info('PROPOSED SAVED | seed=%d | configuration=%s | Cv=%s NPMI=%s NLL=%s TD=%s',seed,selected,
                p['fitting']['c_v'],p['fitting']['c_npmi'],p['prediction']['macro_source_nll'],p['topic_diversity'])
            v=matched_controls(root,source,seed,selected,cases,data,configs,old,output,log,{'recovery_id':key,'seed':seed,'selected':selected})
            ablations.extend(v['rows']);mechanism.extend(v['mechanism']);export_progress(output,rows,ablations,policy['seeds'])
        assert_ablation_grid(ablations,policy['controls'],policy['seeds'])
        log.info('RECOVERY COMPLETE | saved 60 matched control slots | remaining baseline fits <=30 | no parent retraining')
        rows=complete_baselines(root,output,key,rows,ablations,data,configs,old,device,log)
        gates=report_tables(output,rows,ablations,policy,old,selection,mechanism)
    else:
        write_json(output/'quality_gates.json',{**gates,'global_recovery_selection':selection})
        write_text(output/'SUMMARY.md','# Step 5 R2 recovery requires review\n\nEvery candidate/seed slot is saved. No global configuration passed the declared parent gain, diversity and certification gate. Thirty additional baseline fits were avoided. No final test or external data was used. Read global_recovery_selection.json.\n')
    check_indexed_tree(source)
    result={'status':'STEP05_RECOVERY_COMPLETE' if gates['ready_for_untouched_primary_test_review'] else 'STEP05_RECOVERY_REVIEW_REQUIRED',
        'protocol':PROTOCOL,'preparation_id':prepared['preparation_id'],'step05_recovery_id':key,
        'processed_relative_path':output.relative_to(root).as_posix(),'engineering_passed':gates['engineering_passed'],
        'preparation_gate_passed':True,'model_quality_established':False,'training_seeds':policy['seeds'],
        'complete_five_seed_grid':gates['complete_five_seed_grid'],'comparison_rows':len(rows),'ablation_rows':len(ablations),
        'selected_configuration':selection['selected_configuration'],'all_available_profile_means_win_or_tie':gates['all_available_profile_means_win_or_tie'],
        'ready_for_untouched_primary_test_review':gates['ready_for_untouched_primary_test_review'],
        'new_neural_fits':getattr(log,'new_neural_fits',0),'new_classical_fits':getattr(log,'new_classical_fits',0),
        'recovery_new_parent_fits':0,'recovery_new_encoder_calls':0,'test_or_external_used':False,'final_model_result':False}
    write_json(output/'run_result.json',result);write_json(output/'artifact_manifest.json',artifact_manifest(output))
    write_json(root/'outputs/steps/step05_recovery/LATEST.json',{'relative_path':output.relative_to(root).as_posix()})
    write_json(root/'state/step05_recovery_result.json',result)
    log.info('READ FIRST | %s',output/'SUMMARY.md');return result,output
