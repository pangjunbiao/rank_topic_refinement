"""Complete the reviewed Step 6 reports from saved, frozen predictions only.

The exposure ledger remains byte-identical; source checks allow only the recorded vendored documentation edit.
No encoder, model constructor, optimizer, fit or new prediction is called.
The JSON coherence-cache identity is canonicalized in this reporting module.
"""
from __future__ import annotations
import copy
import json
import shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from rank_refinement.runtime.artifacts import (artifact_manifest, digest, file_hash,
    read_json, stable_json, verify_artifacts, write_json, write_text)
from rank_refinement.models.neural_training import load_canonical
from rank_refinement.models.frozen_inference import arrays
from rank_refinement.models.neural_prediction import log_predictions, backoff, score_log_predictions
from rank_refinement.evaluation.frozen_test import load_test_view, frozen_probability_scores
from rank_refinement.evaluation.competitor_evidence import audit_native_reference
from rank_refinement.evaluation.replication import assert_grid, assert_ablation_grid, readout_mechanism
from rank_refinement.stages.step02_completion import check_indexed_tree
from rank_refinement.stages.step05_ablations import profiles_to_run, LABEL_TO_METHOD
from rank_refinement.stages.step06_frozen_test import (validate_protocol, accepted_source,
    source_folders, export_progress, report, PROTOCOL)

EXPECTED_FREEZE_ID = '88b9b4de3e1f7d877709c0d104f377b26507093b0849fdbfb10d09f1658ba852'
REPAIR = 'STEP06_REPORT_ONLY_JSON_CACHE_REPAIR_R4_20261010'


def frozen_preparation(root, identity, configs):
    """Verify the prepared bytes already used by the exposed test.

    Reports consume only these indexed artifacts and saved predictions. They
    do not decode raw CSVs or create another preparation. The original raw
    input hashes remain in the verified input manifest.
    """
    root = Path(root).resolve()
    state = read_json(root / 'state/step01_accepted.json')
    if (state.get('preparation_id') != identity['preparation_id']
            or not state.get('engineering_passed')
            or not state.get('preparation_gate_passed')):
        raise ValueError('The accepted preparation differs from the frozen test.')
    relative = 'data/processed/primary/' + identity['preparation_id'][:20]
    if state.get('processed_relative_path') != relative:
        raise ValueError('Unexpected frozen preparation path.')
    folder = (root / relative).resolve()
    if not folder.is_relative_to(root):
        raise ValueError('Unsafe preparation path.')
    verify_artifacts(folder, state['artifacts'])
    snapshot = read_json(folder / 'configuration_snapshot.json')
    if any(configs[key] != snapshot[key] for key in ['default', 'data', 'evaluation']):
        raise ValueError('Frozen preparation configuration changed.')
    return state


def verify_exposure(root, configs):
    root = Path(root).resolve()
    ledger_path = root / 'state/step06_test_exposure.json'
    ledger = read_json(ledger_path)
    identity = ledger['identity']; key = ledger['freeze_id']
    if key != EXPECTED_FREEZE_ID or digest(identity) != key:
        raise ValueError('Require the exact reviewed Step 6 exposure; no replacement test run is authorized.')
    if digest(configs) != identity['configs_sha256']:
        raise ValueError('Frozen configurations changed; report completion cannot reselect a model.')
    from step08_assembly.integrity import verify_code
    verify_code(root, identity['implementation'], 'src/rank_refinement')
    output = root / 'outputs/steps/step06_frozen_test' / key[:20]
    if read_json(output/'protocol_resolved_before_test.json') != identity:
        raise ValueError('Saved test protocol differs from its original exposure.')
    return output, identity, file_hash(ledger_path)


def corrected_descriptor_audit(topics, view, cache):
    # JSON encodes tuples as lists. Canonicalize BEFORE both key creation and
    # comparison, so the pre-existing cache's key/bytes remain valid.
    identity = json.loads(stable_json({'topics':topics,
        'reference':[(r['group_id'],r['tokens']) for r in view['coherence_records']],
        'vocabulary':view['vocabulary'], 'windows':[110,10], 'retain_all_slots':True}))
    path = Path(cache) / (digest(identity)+'.json')
    index = path.with_suffix('.index.json')
    if path.is_file() != index.is_file():
        raise ValueError('Incomplete coherence cache; preserve it for inspection rather than silently replace it.')
    if path.is_file():
        saved = read_json(index)
        if saved['identity'] != identity or file_hash(path) != saved['audit_sha256']:
            raise ValueError('Frozen coherence cache content or identity genuinely changed.')
        audit = read_json(path)
    else:
        audit = audit_native_reference(topics,view['coherence_records'],view['vocabulary'])
        write_json(path,audit)
        write_json(index,{'identity':identity,'audit_sha256':file_hash(path)})
    return {'c_v':audit['metrics']['c_v']['mean'],'c_npmi':audit['metrics']['c_npmi']['mean'],
        'reference_groups':audit['reference_groups'],'supported_slots':audit['supported_slots'],
        'total_slots':audit['total_slots'],'reference_absent_words':audit['reference_absent_words'],
        'all_descriptor_slots_retained':True,'scope':'frozen_test_reference_diagnostic_no_model_selection',
        'failure':None if all(audit['metrics'][k]['complete'] for k in ['c_v','c_npmi'])
            else 'standard_metric_nonfinite_no_slot_removal'}, audit


def numeric_equal(actual, expected, label):
    if isinstance(actual,dict) and isinstance(expected,dict):
        if set(actual)!=set(expected):raise ValueError('Saved score fields differ: '+label)
        for k in actual:numeric_equal(actual[k],expected[k],label+'/'+k)
    elif isinstance(actual,(int,float)) and isinstance(expected,(int,float)):
        if not np.isfinite(actual) or not np.isfinite(expected) or not np.isclose(actual,expected,atol=1e-11,rtol=1e-12):
            raise ValueError('Saved score replay failed: '+label)
    elif actual!=expected:raise ValueError('Saved score contract differs: '+label)


def verify_losses(row, scores, losses, view):
    numeric_equal(scores,row['prediction'],row['method']+'/aggregate')
    if len(losses)!=len(view['ids']) or len(row['document_losses'])!=len(view['ids']):
        raise ValueError('Missing test document losses; no partial population is allowed.')
    maximum = 0.
    for actual,saved in zip(losses,row['document_losses']):
        numeric_equal(actual,saved,row['method']+'/document')
        maximum=max(maximum,abs(actual['negative_log_probability']-saved['negative_log_probability']))
    return maximum


def predictive_digest(rows):
    return digest([{k:v for k,v in r.items() if k!='test_coherence'} for r in rows])


def frozen_case(folder, identity):
    check_indexed_tree(folder)
    value=read_json(folder/'case_result.json')
    if value['identity']!=identity or value.get('neural_training_performed') or value.get('test_used_for_fit_or_parameter_selection'):
        raise ValueError('Saved prediction case differs from the frozen experiment.')
    return value


def verify_case_metadata(row, original, comparison_from_control=False):
    for key in ['method','profile','seed','role','topics','topic_diversity','rho','status']:
        if row.get(key)!=original.get(key):raise ValueError('Frozen case metadata changed: '+key)
    if comparison_from_control:
        # The original native comparison stores a full per-topic audit while
        # the matched native control stores its compact summary. Their metric
        # values/reference/population must agree; presentation fields differ.
        for key in ['c_v','c_npmi','reference_groups','scored','failure']:
            if row['fitting'].get(key)!=original['fitting'].get(key):
                raise ValueError('Frozen comparison fitting metric changed: '+key)
    elif row['fitting']!=original['fitting']:
        raise ValueError('Frozen case fitting audit changed.')
    if row['original_development_prediction']!=original['prediction']:
        raise ValueError('Original development scores changed.')
    if row.get('test_used_for_training_or_selection') or row.get('model_configuration_selected_on')!='development_only_before_test':
        raise ValueError('Test-dependent model selection is not permitted.')


def verify_saved_predictions(root, output, source, rows, controls, data, view, policy, configs):
    original_rows=read_json(source/'all_rows.json');original_controls=read_json(source/'ablation_rows.json')
    assert_grid(rows,[('Proposed','random_words')]+profiles_to_run(configs['step05_replication']),policy['seeds'])
    assert_ablation_grid(controls,configs['step05_recovery']['controls'],policy['seeds'])
    unavailable=[(r['role'],r['seed'])for r in controls if r['status']!='scored']
    if unavailable!=[('strict_word_disjointness',812)]:raise ValueError('Unavailable control slots changed.')
    if any(r['status']!='scored' for r in rows):raise ValueError('Comparison grid is incomplete.')
    raw_comparisons=[];raw_controls=[];folders=[];max_loss_error=0.;recomputed=0;certificates=[]
    for seed in policy['seeds']:
        folder=output/'proposed_control_cases'/f'seed{seed}'
        value=frozen_case(folder,{'freeze_id':EXPECTED_FREEZE_ID,'seed':seed,'kind':'all_twelve_proposed_control_slots'})
        saved=arrays(folder/'frozen_test_arrays.npz');folders.append(folder)
        prior=saved['prior']
        if not np.array_equal(prior,view['prior']):raise ValueError('Fitting-derived unigram prior changed.')
        for row in value['rows']:
            old=next(r for r in original_controls if (r['role'],r['seed'])==(row['role'],seed))
            verify_case_metadata(row,old)
            raw_controls.append(copy.deepcopy(row))
            if row['status']!='scored':continue
            phi=saved['readout_'+row['role']]
            if __import__('hashlib').sha256(phi.tobytes()).hexdigest()!=row['frozen_affinity_sha256']:
                raise ValueError('Saved actual affinity hash changed.')
            moment=(saved['moment_gain'],saved['moment_offset']) if row['role']=='moment_control' else None
            lp=backoff(log_predictions(saved['theta_test'],phi,saved['decoder_gain'],saved['decoder_offset'],moment),prior,row['rho'])
            scores,losses=score_log_predictions(lp,view['records'],view['masks'],view['sources'])
            max_loss_error=max(max_loss_error,verify_losses(row,scores,losses,view));recomputed+=1
        for method,role in [('Proposed','proposed'),('NeuroMax','native')]:
            row=copy.deepcopy(next(r for r in value['rows']if r['role']==role));row['method']=method
            verify_case_metadata(row,next(r for r in original_rows if(r['method'],r['seed'])==(method,seed)),comparison_from_control=True)
            raw_comparisons.append(row)
        targets=read_json(source/'matched_controls'/f'seed{seed}'/'case_result.json')['targets']
        certificate=readout_mechanism(saved['native'],saved['proposed'],targets,data['vocabulary'],'proposed',seed)
        if certificate['target_top10_membership_recall']<1.-1e-12 or max(certificate['maximum_topic_total_error_scaled'],certificate['maximum_word_total_error_scaled'])>1e-5:
            raise ValueError('Actual Proposed readout certificate failed.')
        numeric_equal(certificate,next(r for r in raw_comparisons if r['method']=='Proposed'and r['seed']==seed)['mechanism'],'actual_readout')
        certificates.append(certificate)
    for original in original_rows:
        if original['method']in['Proposed','NeuroMax']:continue
        method=LABEL_TO_METHOD[original['method']]
        folder=output/'baseline_cases'/method/original['profile']/f"seed{original['seed']}"
        value=frozen_case(folder,{'freeze_id':EXPECTED_FREEZE_ID,'method':original['method'],
            'profile':original['profile'],'seed':original['seed']})
        if len(value['rows'])!=1:raise ValueError('Unexpected baseline case cardinality.')
        row=value['rows'][0];verify_case_metadata(row,original);raw_comparisons.append(copy.deepcopy(row));folders.append(folder)
        if method=='bertopic':
            if row['document_losses'] or any(v is not None for v in row['prediction'].values()):
                raise ValueError('BERTopic must retain undefined native word likelihood.')
            continue
        saved=arrays(folder/'frozen_test_arrays.npz')
        if method=='encot':
            lp=backoff(log_predictions(saved['theta_test'],saved['native'],saved['decoder_gain'],saved['decoder_offset']),view['prior'],row['rho'])
            scores,losses=score_log_predictions(lp,view['records'],view['masks'],view['sources'])
        else:scores,losses=frozen_probability_scores(saved['native_decoded_probabilities'],row['rho'],view['prior'],view,view['sources'])
        max_loss_error=max(max_loss_error,verify_losses(row,scores,losses,view));recomputed+=1
    order=lambda r:(r['method'],r['profile'],r['seed'])
    if predictive_digest(sorted(rows,key=order))!=predictive_digest(sorted(raw_comparisons,key=order)):
        raise ValueError('Incremental comparison scores differ from their immutable saved cases.')
    order=lambda r:(r['role'],r['seed'])
    if predictive_digest(sorted(controls,key=order))!=predictive_digest(sorted(raw_controls,key=order)):
        raise ValueError('Incremental control scores differ from their immutable saved cases.')
    immutable={p.relative_to(root).as_posix():file_hash(p)for f in folders for p in sorted(f.rglob('*'))if p.is_file()}
    return {'recomputed_from_saved_arrays':recomputed,'maximum_document_loss_error':max_loss_error,
        'comparison_slots':len(rows),'control_slots':len(controls),'test_groups':len(view['ids']),
        'actual_readout_certificates':certificates,'unchanged_case_payloads':immutable,
        'new_model_predictions':0,'new_neural_fits':0,'new_classical_fits':0,'new_encoder_calls':0}


def run(root,configs,versions,hardware,log):
    root=Path(root).resolve();policy,old=validate_protocol(configs,root)
    output,identity,ledger_sha=verify_exposure(root,configs)
    write_json(log.folder/'preparation_working_path.json',{'relative_path':output.relative_to(root).as_posix()})
    prepared=frozen_preparation(root,identity,configs)
    source,original_rows,original_controls,review=accepted_source(root,prepared['preparation_id'],policy,configs)
    source_folders(root,source)
    if identity['source_index_sha256']!=file_hash(source/'artifact_manifest.json') or identity['preparation_id']!=prepared['preparation_id']:
        raise ValueError('Original source or preparation identity changed.')
    if (output/'artifact_manifest.json').is_file():
        check_indexed_tree(output);result=read_json(output/'run_result.json')
        if result.get('status')!='STEP06_FROZEN_PRIMARY_TEST_COMPLETE':raise ValueError('Unexpected completed cache status.')
        audit=read_json(output/'report_completion_audit.json')
        if audit['original_exposure_ledger_sha256']!=ledger_sha:
            raise ValueError('Original test exposure ledger changed.')
        for relative,sha in audit['unchanged_case_payloads'].items():
            path=(root/relative).resolve()
            if not path.is_relative_to(output.resolve()) or file_hash(path)!=sha:
                raise ValueError('Saved prediction-case payload changed after report completion.')
        log.info('STEP 6 REPORT CACHE | verified completed result reused | new model predictions=0')
        return {**result,'report_cache_replay':True,'new_classical_fits':0},output
    data=load_canonical(root/prepared['processed_relative_path'])
    view=load_test_view(root/prepared['processed_relative_path'],data['fit_ids'],data['dev_ids'])
    view['prior']=data['fit'].astype(np.float64).sum(0)+.5;view['prior']/=view['prior'].sum()
    view['sources']=sorted({r['source']for r in data['fit_records']})
    if read_json(output/'test_population.json')['group_ids']!=view['ids']:raise ValueError('Frozen test population changed.')
    pretest=read_json(output/'pretest_development_replay.json')
    if len(pretest['cases'])!=45 or pretest['primary_test_used'] or pretest['new_neural_fits']:
        raise ValueError('Require the completed original 45-interface pretest replay.')
    rows=read_json(output/'all_test_rows.json');controls=read_json(output/'all_test_control_rows.json')
    log.info('REPORT-ONLY COMPLETION | verify all 60 comparison and 60 control slots | no model/encoder calls')
    audit=verify_saved_predictions(root,output,source,rows,controls,data,view,policy,configs)
    before= predictive_digest(rows+controls)
    for i,row in enumerate(rows+controls):
        if row['status']!='scored' or not row.get('topics'):
            row['test_coherence']={'c_v':None,'c_npmi':None,'failure':'readout_unavailable_no_slot_removal'}
        else:row['test_coherence'],_=corrected_descriptor_audit(row['topics'],view,output/'heldout_coherence_audits')
        if (i+1)%20==0:log.info('COHERENCE REPORT CACHE | %d/120 slots | all descriptor slots retained',i+1)
    if predictive_digest(rows+controls)!=before:raise ValueError('Report completion changed a prediction or descriptor.')
    export_progress(output,rows,controls,policy,complete=True)
    log.info('REPORT EXPORT | comparisons, controls, fixed bootstrap intervals and figures | no parameter selection')
    gates=report(root,output,rows,controls,data,policy,old,review)
    write_json(output/'data_firewall.json',{'fitting_group_ids':data['fit_ids'],'development_observed_group_ids':data['dev_ids'],
        'test_observed_group_ids':view['ids'],'test_full_text_encoded':False,'test_parameters_selected':False,
        'external_data_used':False,'neural_training_performed':False,'coherence_test_reference_used_only_after_freeze':True,
        'human_labels_or_assessments':False,'development_rho_frozen_every_case':True})
    for relative,sha in audit['unchanged_case_payloads'].items():
        if file_hash(root/relative)!=sha:raise ValueError('Immutable saved prediction payload changed during reporting.')
    if file_hash(root/'state/step06_test_exposure.json')!=ledger_sha:raise ValueError('Original test exposure ledger changed.')
    check_indexed_tree(source)
    audit.update(repair=REPAIR,original_freeze_id=EXPECTED_FREEZE_ID,original_exposure_ledger_sha256=ledger_sha,
        prediction_rows_digest_before=before,prediction_rows_digest_after=predictive_digest(rows+controls),
        old_prediction_implementation_retained=True,test_or_external_used_for_selection=False,
        completion_module_sha256=file_hash(Path(__file__)),completion_environment=versions)
    write_json(output/'report_completion_audit.json',audit)
    result={'status':'STEP06_FROZEN_PRIMARY_TEST_COMPLETE','protocol':PROTOCOL,'step06_freeze_id':EXPECTED_FREEZE_ID,
        'preparation_id':prepared['preparation_id'],'source_step05_recovery_id':policy['source_recovery_id'],
        'processed_relative_path':output.relative_to(root).as_posix(),'engineering_passed':True,'preparation_gate_passed':True,
        'model_quality_established':False,'final_primary_test_result':True,'final_paper_claim_approved':False,
        'training_seeds':policy['seeds'],'comparison_rows':60,'control_slots':60,'test_completion_groups':len(view['ids']),
        'test_evaluated':True,'external_evaluated':False,'test_used_for_model_or_rho_selection':False,
        'new_neural_fits':0,'new_classical_fits':0,'new_model_predictions':0,'new_encoder_calls':0,
        'completion_mode':REPAIR,'frozen_test_wins_all_defined_mean_prediction_comparisons':gates['frozen_test_wins_all_defined_mean_prediction_comparisons'],
        'original_step05_all_metric_win_gate_preserved':False,'selected_configuration':policy['selected_configuration']}
    write_json(output/'run_result.json',result);write_json(output/'artifact_manifest.json',artifact_manifest(output))
    write_json(root/'state/step06_result.json',result)
    write_json(root/'outputs/steps/step06_frozen_test/LATEST.json',{'relative_path':result['processed_relative_path'],'step06_freeze_id':EXPECTED_FREEZE_ID})
    log.info('STEP 6 COMPLETE | original frozen predictions preserved | reports saved | new model/encoder calls=0')
    log.info('READ FIRST | %s',output/'SUMMARY.md')
    return result,output


def make_completion_inspection(root,run,processed_folder=None,kind='step06'):
    """Include this reporting repair while preserving original prediction code."""
    from rank_refinement.reporting.export import make_inspection_zip
    root=Path(root);destination=make_inspection_zip(root,run,processed_folder,kind)
    temporary=destination.with_suffix('.report_completion.zip')
    extras=[root/'step06_finalize.py',root/'README_STEP06_REPORT_FIX.md']
    with ZipFile(destination) as original,ZipFile(temporary,'w',compression=ZIP_DEFLATED)as updated:
        records={r['path']:r for r in json.loads(original.read('INSPECTION_CONTENTS.json'))}
        for info in original.infolist():
            if info.filename=='INSPECTION_CONTENTS.json':continue
            with original.open(info.filename)as a,updated.open(info.filename,'w')as b:shutil.copyfileobj(a,b,1024*1024)
        for path in extras:
            if path.is_file():
                name=path.relative_to(root).as_posix();updated.write(path,name)
                records[name]={'path':name,'bytes':path.stat().st_size,'sha256':file_hash(path)}
        updated.writestr('INSPECTION_CONTENTS.json',json.dumps([records[k]for k in sorted(records)],ensure_ascii=False,indent=2))
    with ZipFile(temporary)as z:
        if z.testzip()is not None:raise ValueError('Completion inspection ZIP failed CRC validation.')
    temporary.replace(destination)
    return destination
