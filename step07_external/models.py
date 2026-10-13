"""Fixed-profile external fits and certified readout refinement, shared across seeds."""
from __future__ import annotations
import copy
import io
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from rank_refinement.runtime.artifacts import atomic_bytes, artifact_manifest, digest, file_hash, read_json, write_json
from rank_refinement.stages.step02_completion import check_indexed_tree
from rank_refinement.models.neural_prediction import (affine_parameters, log_predictions, moment_parameters,
    score_log_predictions, select_backoff, verify_native_interface, backoff)
from rank_refinement.models.neural_training import train_native, export_latents, global_context
from rank_refinement.models.adapters.pilot_anchors import topics_from_weights, UnusableAnchor
from rank_refinement.models.classical_completion import fit_and_predict, select_probability_backoff
from rank_refinement.models.competitor_classical import fit_gsdmm, fit_bertopic
from rank_refinement.models.competitor_neural import train_model, decode_model, cluster_contexts
from rank_refinement.acquisition.native_targets import calibrate, native_indices
from rank_refinement.acquisition.native_diversity_targets import recovery_targets
from rank_refinement.projection.replay_solver import primary_tv, range_secondary
from rank_refinement.evaluation.replication import readout_mechanism
from rank_refinement.stages.step03_neural_pilot import fitting_context_ids
from .protocol import LABELS
from .numerics import safe_primary,safe_range

def save_arrays(path,**arrays):
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);atomic_bytes(path,stream.getvalue())

def encode(root,folder,data,policy,device,log,offline=False):
    """Full text only on fit. Every held-out input uses observed occurrences only."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    from rank_refinement.models.encoder_assets import resolve_encoder
    if offline:
        asset_dir=root/policy['local_directory']/policy['revision']
        absent=[n for n in policy['file_sha256'] if not (asset_dir/n).is_file()]
        if absent:raise RuntimeError('Offline English encoder assets missing: '+', '.join(absent))
    try:
        asset_dir,assets=resolve_encoder(root,policy,log)
    except RuntimeError as exc:
        raise RuntimeError('Pinned English BGE assets could not be resolved. Keep the cache and retry; '
                           'both Hugging Face and hf-mirror are tried, without encoder substitution. '+str(exc))from exc
    views={'fit_full':[r['clean_text']for r in data['fit_records']],
        'fit_observed':[data['masks'][g]['observed_text']for g in data['fit_observed_ids']],
        'development_observed':[data['masks'][g]['observed_text']for g in data['dev_ids']],
        'test_observed':[data['masks'][g]['observed_text']for g in data['test_ids']],
        'words':[w.replace('_',' ')for w in data['vocabulary']]}
    identity={'assets':assets,'texts':views,'pooling':'cls_l2','device':device,
              'torch':torch.__version__,'max_length':policy['max_length']}
    key=digest(identity);cache=folder/'workspace/data/cache/english_encoder'/key[:20]
    index=cache/'manifest.json';values=cache/'vectors.npz'
    if index.is_file():
        saved=read_json(index)
        if saved['identity']!=identity or saved['vectors_sha256']!=file_hash(values):
            raise ValueError('English encoder cache was changed.')
        with np.load(values,allow_pickle=False)as z:vectors={n:z[n]for n in views}
        log.info('ENGLISH ENCODER | verified embeddings cache reused')
    else:
        model=AutoModel.from_pretrained(str(asset_dir),local_files_only=True,trust_remote_code=False,
            use_safetensors=True).to(device).eval()
        tokenizer=AutoTokenizer.from_pretrained(str(asset_dir),local_files_only=True,trust_remote_code=False)
        vectors={};truncated={}
        with torch.inference_mode():
            for name,texts in views.items():
                chunks=[];truncated[name]=0
                for start in range(0,len(texts),policy['batch_size']):
                    selected=texts[start:start+policy['batch_size']]
                    truncated[name]+=sum(len(t)>policy['max_length']for t in tokenizer(selected,truncation=False)['input_ids'])
                    inputs=tokenizer(selected,padding=True,truncation=True,max_length=policy['max_length'],return_tensors='pt')
                    output=model(**{k:v.to(device)for k,v in inputs.items()}).last_hidden_state[:,0]
                    chunks.append(torch.nn.functional.normalize(output,dim=1).cpu().numpy().astype(np.float32))
                    log.info('ENGLISH EMBEDDINGS | %s | %d/%d | %s',name,min(start+policy['batch_size'],len(texts)),len(texts),device)
                vectors[name]=np.concatenate(chunks)
        save_arrays(values,**vectors)
        write_json(index,{'identity':identity,'vectors_sha256':file_hash(values),'truncations':truncated,
            'dimension':vectors['fit_full'].shape[1],'heldout_full_text_encoded':False})
        del model
        if device=='cuda':torch.cuda.empty_cache()
    for name,texts in views.items():
        if vectors[name].shape!=(len(texts),384) or not np.isfinite(vectors[name]).all() or not np.allclose(np.linalg.norm(vectors[name],axis=1),1,atol=1e-5):
            raise ValueError('Invalid English embedding values, width or normalization: '+name)
    write_json(folder/'encoder_manifest.json',{'repository':policy['repository'],'revision':policy['revision'],
        'assets':assets,'cache_relative_path':cache.relative_to(folder).as_posix(),
        'vectors_sha256':file_hash(values),'heldout_full_text_encoded':False,'instruction':'none','pooling':'cls_l2'})
    return vectors

def combined_views(data,vectors):
    """Adapters' development key contains only observed inputs, dev then test.

    No held-out target or full-text reference is passed to a fitting routine.
    Backoff selection slices the development rows exclusively.
    """
    all_observed=np.concatenate([data['dev_observed'],data['test_observed']])
    adapter_data={**data,'dev_observed':all_observed}
    adapter_vectors={**vectors,'development_observed':np.concatenate([vectors['development_observed'],vectors['test_observed']])}
    return adapter_data,adapter_vectors

def development_choice(raw_logp,prior,data,grid):
    # The development loss function needs finite target probabilities but permits
    # genuine zero probabilities elsewhere in classical model distributions.
    probabilities=np.exp(raw_logp)
    return select_probability_backoff(probabilities[:len(data['dev_ids'])],prior,grid,
        data['dev_records'],data['masks'],['external_transit'])

def result_row(method,profile,seed,topics,raw_logp,prior,data,grid,interface,details=None):
    chosen,curve=(development_choice(raw_logp,prior,data,grid)if raw_logp is not None else (None,[]))
    return {'method':method,'profile':profile,'seed':int(seed),'scope':'external_frozen_recipe_refit',
        'status':'scored','topics':topics,'fixed_k10':topics is not None and len(topics)==10,
        'native_topic_count':len(topics)if topics is not None else (details or {}).get('native_topic_count'),
        'rho':chosen['rho']if chosen else None,'development_prediction':chosen['scores']if chosen else {'macro_source_nll':None,'micro_nll':None},
        'development_backoff_curve':curve,'prediction_interface':interface,
        'labels_used':False,'test_used_for_training_or_selection':False,
        'details':details or {},'prediction_unavailable_reason':None if chosen else 'native_word_likelihood_not_defined'}

def cached(folder,identity):
    if not (folder/'artifact_manifest.json').is_file():return None
    check_indexed_tree(folder);value=read_json(folder/'case_result.json')
    if value['identity']!=identity:raise ValueError('External case identity changed; cached results were not overwritten.')
    return value

def prior_from_fit(data,policy):
    prior=data['fit'].sum(axis=0,dtype=np.float64)+policy['prediction']['unigram_pseudocount']
    return prior/prior.sum()

def refine(native,theta_fit,gain,data,policy,seed,log):
    acq=policy['acquisition'];top=native_indices(native)
    order=[np.lexsort((np.arange(len(data['vocabulary'])),-row))[:acq['neighborhood']].tolist()for row in native]
    frequency=data['metadata']['document_frequency']
    needed=sorted({data['vocabulary'][i]for indices in top+order for i in indices
                   if i in {w for row in top for w in row} or frequency[i]>=acq['word_group_support']})
    # Same seed convention as the existing complete-set acquisition.
    evidence=calibrate(data['fit_records'],needed,seed,policy['calibration'])
    target,selection=recovery_targets(native,data['vocabulary'],data['metadata'],evidence,acq,'signed_stable')
    p=policy['projection_settings'];s=policy['secondary_settings'];pred=policy['prediction']
    indices=fitting_context_ids(data['fit_observed_records'],pred['context_count'],pred['context_seed'])
    certificates={};readouts={};mechanism=None
    if not selection['selected']:
        # This is the acquisition's explicit native option, not a solver fallback.
        readouts={'proposed':native.copy(),'primary_tv':native.copy()}
        deployment='abstained_no_supported_edit'
        certificates={'abstention':{'native_readout_retained':True,'selected_edits':0,'certification_required':False}}
    else:
        primary,cert=safe_primary(native,target,p['rank_margin_scaled'],p['primary_seconds'])
        certificates['primary_tv']=cert
        deployment='unavailable_projection_not_certified'
        if primary is not None:
            readouts['primary_tv']=primary
            refined,certificate=safe_range(native,target,cert['solver_tv_scaled'],theta_fit[indices],gain,primary,
                p['rank_margin_scaled'],s['tv_cap_slack_scaled'],s['secondary_seconds'],
                s['range_maximum_iterations'],s['range_tolerance'],log)
            certificates['range']=certificate
            if refined is not None:
                mechanism=readout_mechanism(native,refined,target,data['vocabulary'],'proposed',seed)
                if mechanism['target_top10_membership_recall']<1-1e-12:
                    raise ValueError('External projected readout did not realize the selected target sets.')
                if mechanism['relative_l1_affinity_movement']<=p['max_relative_l1_movement']:
                    readouts['proposed']=refined;deployment='certified_projection_deployed'
                else:deployment='unavailable_declared_movement_budget_exceeded'
    return readouts,{'selection':selection,'targets':target,'certificates':certificates,'mechanism':mechanism,
        'deployment':deployment,'context_group_ids':[data['fit_observed_ids'][int(i)]for i in indices],
        'contexts_fit_only':True,'development_and_test_used_for_acquisition':False},evidence

def fit_native_case(root,output,seed,data,vectors,policy,device,folders,specs,log,identity):
    folder=output/'cases/proposed_parent'/f'seed{seed}'
    previous=cached(folder,identity)
    if previous is not None:return previous
    workspace=output/'workspace';adapter_data,adapter_vectors=combined_views(data,vectors)
    model,training=train_native(workspace,'neuromax','random_words',seed,policy['native_training'],folders,specs,
        adapter_data,adapter_vectors,None,device,log)
    # Retain author training identity; contextual encoder language is separately documented.
    training['external_language']='English';training['external_fit_only']=True
    log.new_neural_fits=getattr(log,'new_neural_fits',0)+int(training['completed_before_run']==0)
    import torch
    replay=verify_native_interface(model,torch.from_numpy(data['fit_observed'][:4]).to(device))
    native=model.get_beta().detach().cpu().double().numpy()
    theta_fit=export_latents(model,data['fit_observed'],device)
    theta_observed=export_latents(model,adapter_data['dev_observed'],device)
    gain,offset=affine_parameters(model);prior=prior_from_fit(data,policy)
    readouts,refinement,evidence=refine(native,theta_fit,gain,data,policy,seed,log)
    readouts={'native':native,**readouts}
    rows=[];raw={};grid=policy['prediction']['backoff_grid']
    native_logp=log_predictions(theta_observed,native,gain,offset)
    native_topics=topics_from_weights(native,data['vocabulary'])
    row=result_row('NeuroMax','random_words',seed,native_topics,native_logp,prior,data,grid,'neuromax_author_affine_bn')
    rows.append(row);raw['native']=native_logp
    names={'proposed':('Proposed','signed_stable/range'),
           'primary_tv':('Primary TV control','same_acquired_targets'),
           'display_only':('Same-core display only','same_core_native_predictor'),
           'moment_correction':('Moment correction control','same_projection_with_moments')}
    for role,(label,profile)in names.items():
        if role in ['display_only','moment_correction']and 'proposed'not in readouts or role in ['proposed','primary_tv']and role not in readouts:
            rows.append({'method':label,'profile':profile,'seed':seed,'status':'unavailable','topics':None,
                'fixed_k10':False,'rho':None,'development_prediction':{'macro_source_nll':None,'micro_nll':None},
                'prediction_interface':'unavailable_uncertified_projection','details':{},'deployment':refinement['deployment']})
            continue
        phi=readouts['proposed']if role in ['display_only','moment_correction']else readouts[role]
        topics=topics_from_weights(phi,data['vocabulary'])
        if role=='display_only':values=native_logp.copy()
        elif role=='moment_correction':values=log_predictions(theta_observed,phi,gain,offset,
                moment_parameters(theta_fit,native,phi,policy['prediction']['moment_std_floor']))
        else:values=log_predictions(theta_observed,phi,gain,offset)
        current=result_row(label,profile,seed,topics,values,prior,data,grid,'neuromax_author_affine_bn')
        current.update(role=role,deployment=refinement['deployment'],moment_correction=role=='moment_correction',
            projection_deployed=role!='display_only'and refinement['deployment']=='certified_projection_deployed')
        if role=='display_only'and (current['rho']!=row['rho']or current['development_prediction']!=row['development_prediction']):
            raise AssertionError('Display-only changed the native predictive interface.')
        rows.append(current);raw[role]=values
    folder.mkdir(parents=True,exist_ok=True)
    save_arrays(folder/'predictive_arrays.npz',**{'logp_'+k:v for k,v in raw.items()},
        **{'readout_'+k:v for k,v in readouts.items()},theta_fit=theta_fit,theta_observed=theta_observed,
        decoder_gain=gain,decoder_offset=offset,prior=prior)
    save_arrays(folder/'association_evidence.npz',**{k:v for k,v in evidence.items()if isinstance(v,np.ndarray)})
    write_json(folder/'association_words_and_samples.json',{'words':evidence['words'],'sample_group_ids':evidence['sample_group_ids']})
    write_json(folder/'training.json',training);write_json(folder/'native_interface_certificate.json',replay)
    write_json(folder/'refinement.json',refinement)
    result={'identity':identity,'rows':rows,'refinement':refinement,'training':training,'raw_logp_roles':list(raw)}
    write_json(folder/'case_result.json',result);write_json(folder/'artifact_manifest.json',artifact_manifest(folder))
    log.info('PROPOSED EXTERNAL FIT | seed=%d | %s | acquired edited topics=%d | no moment correction',
             seed,refinement['deployment'],len(refinement['selection']['selected']))
    del model
    if device=='cuda':torch.cuda.empty_cache()
    return result

def fit_baseline_case(root,output,item,seed,data,vectors,policy,device,folders,native_specs,comp_folders,comp_specs,log,identity):
    method=item['method'];profile=item['profile'];name=profile['name']
    folder=output/'cases/competitors'/method/name/f'seed{seed}'
    previous=cached(folder,identity)
    if previous is not None:return previous
    workspace=output/'workspace';adapter_data,adapter_vectors=combined_views(data,vectors)
    observed=adapter_data['dev_observed'];prior=prior_from_fit(data,policy);training={};details={};topics=None
    if method=='encot':
        contexts=global_context(adapter_data,adapter_vectors,policy['native_training']['global_context'])
        model,training=train_native(workspace,method,name,seed,policy['native_training'],folders,native_specs,
            adapter_data,adapter_vectors,contexts,device,log)
        import torch
        details=verify_native_interface(model,torch.from_numpy(contexts['development_observed'][:4]).to(device))
        weights=model.get_beta().detach().cpu().double().numpy();theta=export_latents(model,contexts['development_observed'],device)
        gain,offset=affine_parameters(model);raw=log_predictions(theta,weights,gain,offset)
        topics=topics_from_weights(weights,data['vocabulary']);interface='encot_author_affine_bn'
        training['external_language']='English';training['external_fit_only']=True
        log.new_neural_fits=getattr(log,'new_neural_fits',0)+int(training['completed_before_run']==0)
        del model
    elif method in ['combinedtm','fastopic','glocom']:
        contexts=(cluster_contexts(data['fit'],vectors['fit_full'],observed,adapter_vectors['development_observed'],profile,seed)if method=='glocom'else None)
        model,training=train_model(workspace,method,profile,seed,comp_folders[method],data['fit'],adapter_vectors,
            data['fit_ids'],comp_specs,device,policy['native_training']['checkpoint_every_epochs'],log,contexts)
        weights,probabilities,theta,details=decode_model(method,model,observed,adapter_vectors,contexts,seed,20)
        with np.errstate(divide='ignore'):raw=np.log(probabilities)
        topics=topics_from_weights(weights,data['vocabulary']);interface=details['prediction_interface']
        del model
    elif method in ['lda','nmf']:
        topics,weights,probabilities,details=fit_and_predict(method,data['fit'],observed,data['vocabulary'],seed,profile)
        with np.errstate(divide='ignore'):raw=np.log(probabilities)
        interface=details['prediction_interface'];log.new_classical_fits=getattr(log,'new_classical_fits',0)+1
    elif method=='gsdmm':
        topics,weights,probabilities,theta,details=fit_gsdmm(data['fit'],observed,data['vocabulary'],seed,profile,log)
        with np.errstate(divide='ignore'):raw=np.log(probabilities)
        interface=details['prediction_interface'];log.new_classical_fits=getattr(log,'new_classical_fits',0)+1
    elif method=='bertopic':
        topics,weights,probabilities,theta,details=fit_bertopic(data['fit'],observed,data['vocabulary'],seed,profile,adapter_vectors,log)
        raw=None;interface=details['prediction_interface'];log.new_classical_fits=getattr(log,'new_classical_fits',0)+1
    else:raise ValueError('Unknown external comparator: '+method)
    reported_profile='english_words'if method=='encot'and name=='chinese_words'else name
    if reported_profile!=name:
        details.update(primary_profile_name=name,external_profile_name=reported_profile,
                       language_adaptation='same_word_initialization_route_with_shared_English_BGE_vectors')
    row=result_row(LABELS[method],reported_profile,seed,topics,raw,prior,data,policy['prediction']['backoff_grid'],interface,details)
    if method in ['gsdmm','bertopic']:
        row['fixed_k10']=details.get('fixed_k10',False)and topics is not None
        row['native_topic_count']=details.get('native_topic_count')
    folder.mkdir(parents=True,exist_ok=True)
    arrays={'weights':weights,'prior':prior}
    if raw is not None:arrays['logp_model']=raw
    save_arrays(folder/'predictive_arrays.npz',**arrays)
    write_json(folder/'training.json',training)
    value={'identity':identity,'rows':[row],'raw_logp_roles':['model']if raw is not None else []}
    write_json(folder/'case_result.json',value);write_json(folder/'artifact_manifest.json',artifact_manifest(folder))
    log.info('EXTERNAL COMPARATOR SAVED | %s | %s | seed=%d | native topics=%s | dev NLL=%s',
             row['method'],name,seed,row['native_topic_count'],row['development_prediction']['macro_source_nll'])
    if device=='cuda':
        import torch
        torch.cuda.empty_cache()
    return value

def frozen_scores(logp,rho,prior,data):
    """No zero clipping, backoff reselection, or filtering difficult target groups."""
    from rank_refinement.evaluation.likelihood import aggregate_nll
    values=backoff(logp,prior,rho)
    if np.isnan(values).any()or np.isposinf(values).any()or not np.allclose(logsumexp(values,axis=1),0,atol=1e-10):
        raise ValueError('Invalid frozen external test probability distribution.')
    losses=[];nonfinite=[]
    for i,r in enumerate(data['test_records']):
        ids=np.asarray(data['masks'][r['group_id']]['target_ids'],dtype=int)
        if not np.isfinite(values[i,ids]).all():nonfinite.append(r['group_id'])
        losses.append({'group_id':r['group_id'],'source':r['source'],'target_tokens':int(len(ids)),
            'negative_log_probability':float(-values[i,ids].sum())if np.isfinite(values[i,ids]).all()else None})
    if nonfinite:return {'macro_source_nll':None,'micro_nll':None,'failure':'zero_probability_on_frozen_test_targets',
                         'affected_group_ids':nonfinite},losses
    return aggregate_nll(losses,['external_transit']),losses
