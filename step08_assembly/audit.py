"""Saved-array replay, probability population checks, and projection evidence."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from rank_refinement.runtime.artifacts import digest,file_hash,read_json
from .integrity import SEEDS,EXTERNAL_ID,safe
from .metrics import grid,normalize,summaries,close,replay_losses,logits

def jsonl(path):
    return [json.loads(line)for line in Path(path).read_text(encoding='utf-8').splitlines()if line]

def arrays(path):
    with np.load(path,allow_pickle=False)as z:return {k:z[k]for k in z.files}

def descriptor_indices(phi):
    if phi.ndim!=2 or not np.isfinite(phi).all()or (phi<0).any():
        raise ValueError('Invalid actual readout.')
    return [np.lexsort((np.arange(phi.shape[1]),-r))[:10].tolist()for r in phi]

def mechanism(native,phi,vocabulary,words,seed,dataset,targets=None):
    native=np.asarray(native,dtype=np.float64);phi=np.asarray(phi,dtype=np.float64)
    if native.shape!=phi.shape or native.shape[0]!=10 or native.shape[1]!=len(vocabulary):
        raise ValueError('Actual readout shape changed.')
    idx=descriptor_indices(phi);actual=[[vocabulary[i]for i in t]for t in idx]
    if actual!=words:raise ValueError('Reported Proposed words differ from the actual readout.')
    target=idx if targets is None else targets
    if len(target)!=10 or any(len(t)!=10 or len(set(t))!=10 for t in target):
        raise ValueError('Missing target descriptors.')
    recall=float(np.mean([len(set(t)&set(a))/10 for t,a in zip(target,idx)]))
    scale=float(native.mean())
    if scale<=0 or not np.isfinite(native).all()or (native<0).any():raise ValueError('Invalid native affinities.')
    row_error=float(np.max(np.abs(phi.sum(1)-native.sum(1)))/scale)
    col_error=float(np.max(np.abs(phi.sum(0)-native.sum(0)))/scale)
    gaps=[]
    for r,t in zip(phi,target):
        other=np.setdiff1d(np.arange(len(vocabulary)),t)
        gaps.append(float((r[t].min()-r[other].max())/scale))
    if recall<1 or row_error>1e-7 or col_error>1e-7 or min(gaps)<1e-6-1e-9:
        raise ValueError('Actual rank/totals projection checks failed.')
    return {'dataset':dataset,'seed':seed,'membership_recall':recall,
        'target_source':'frozen_acquisition_targets'if targets is not None else 'frozen_reported_target_words',
        'relative_l1_movement':float(np.abs(phi-native).sum()/native.sum()),
        'max_topic_total_error_scaled':row_error,'max_word_total_error_scaled':col_error,
        'minimum_rank_gap_scaled':min(gaps),'nonnegative':True,
        'native_td':len({i for t in descriptor_indices(native)for i in t})/100,
        'proposed_td':len({i for t in idx for i in t})/100}

def check_masks(features,masks,ids,split,vocabulary):
    for g in ids:
        r,m=features[g],masks[g]
        if r['split']!=split or not m['eligible']:raise ValueError('Frozen split eligibility changed.')
        observed=m['observed_positions'];target=m['target_positions']
        if set(observed)&set(target)or len(set(observed))!=len(observed)or len(set(target))!=len(target):
            raise ValueError('Token positions overlap or repeat.')
        mapping={w:i for i,w in enumerate(vocabulary)}
        positions=[i for i,w in enumerate(r['tokens'])if w in mapping]
        if set(observed+target)!=set(positions):raise ValueError('Observed/target partitions lost in-vocabulary occurrences.')
        for role in ['observed','target']:
            if [mapping[r['tokens'][i]]for i in m[role+'_positions']]!=m[role+'_ids']:
                raise ValueError('Saved occurrence/vocabulary IDs changed.')

def external_replay(folder,rows,identity,log):
    prep=folder/'preparation';features={r['group_id']:r for r in read_json(prep/'feature_records.json')}
    masks=read_json(prep/'completion_masks.json');split=read_json(prep/'split_group_ids.json')
    vocabulary=read_json(prep/'vocabulary.json')['tokens']
    ids=[g for g in split['test']if masks[g]['eligible']]
    dev=[g for g in split['development']if masks[g]['eligible']]
    if any(set(split[a])&set(split[b])for a,b in [('fit','development'),('fit','test'),('development','test')]):
        raise ValueError('External split group overlap.')
    freeze=read_json(folder/'external_test_freeze.json')['identity']
    if digest(ids)!=freeze['test_group_ids_sha256']or digest({g:masks[g]for g in ids})!=freeze['test_masks_sha256']:
        raise ValueError('Frozen external test masks changed.')
    check_masks(features,masks,ids,'test',vocabulary)
    check_masks(features,masks,dev,'development',vocabulary)
    checks=[];certificates=[];cache={};fits=set(split['fit'])
    for row in rows:
        if row.get('test_used_for_training_or_selection')or row.get('labels_used'):
            raise ValueError('External test selection/label training was reported.')
        case=safe(folder,row['case_relative_path'])
        if case not in cache:cache[case]=arrays(case/'predictive_arrays.npz')
        a=cache[case];role=row.get('role','native'if row['method']=='NeuroMax'else 'model')
        key='logp_'+role
        if row['prediction']['macro_source_nll']is not None:
            if key not in a:raise ValueError('Predictive array missing for a reported external score.')
            replay=replay_losses(a[key][len(dev):],a['prior'],row,ids,masks,features)
            checks.append({'dataset':'External','method':row['method'],'profile':row['profile'],'seed':row['seed'],**replay})
        elif row['method']!='BERTopic'and row['status']=='scored':
            # Preserve genuine zero-target/undefined losses; never manufacture a score.
            if row.get('prediction',{}).get('failure')!='zero_probability_on_frozen_test_targets':
                raise ValueError('Unexpected missing external predictive result.')
        if role in ['native','proposed','primary_tv','display_only','moment_correction']and 'readout_native'in a:
            phi=a['readout_native']if role in ['native','display_only']else a.get('readout_proposed'if role=='moment_correction'else 'readout_'+role)
            if phi is not None and key in a:
                moment=None
                if role=='moment_correction':
                    before=a['theta_fit']@a['readout_native'];after=a['theta_fit']@phi
                    floor=identity['policy']['prediction']['moment_std_floor']
                    gain=np.maximum(before.std(0),floor)/np.maximum(after.std(0),floor)
                    moment=(gain,before.mean(0)-gain*after.mean(0))
                replay=logits(a['theta_observed'],phi,a['decoder_gain'],a['decoder_offset'],moment)
                if not np.allclose(replay,a[key],atol=1e-10,rtol=0):
                    raise ValueError('External saved log probabilities differ from the actual readout.')
        if row['method']=='Proposed':
            details=read_json(case/'refinement.json')
            if details['deployment']!='certified_projection_deployed':
                raise ValueError('Reviewed Step 7 was expected to deploy all five projections.')
            certificate=mechanism(a['readout_native'],a['readout_proposed'],vocabulary,row['topics'],row['seed'],'External',details['targets'])
            primary=details['certificates']['primary_tv'];secondary=details['certificates']['range']
            scale=float(a['readout_native'].max())
            close(scale,secondary['scale'],'LP max-affinity scale')
            movement=float(.5*np.abs(a['readout_proposed']-a['readout_native']).sum()/scale)
            if (not primary['certified']or not secondary['certified']or secondary['post_solve_clipping']
                    or movement>secondary['tv_cap_scaled']+1e-7):
                raise ValueError('External minimum-TV cap or certificates changed.')
            contexts=details['context_group_ids']
            fit_observed=sorted(g for g in split['fit']if masks[g]['eligible']and any(w in vocabulary for w in features[g]['tokens']))
            if len(contexts)!=64 or not set(contexts)<=fits:raise ValueError('Projection used non-fitting contexts.')
            index={g:i for i,g in enumerate(fit_observed)}
            perturb=(a['theta_fit'][[index[g]for g in contexts]]@(a['readout_proposed']-a['readout_native']))*a['decoder_gain']
            objective=float(np.ptp(perturb,axis=1).mean())
            close(objective,secondary['secondary_objective'],'range objective',1e-7)
            certificate.update(edited_topics=len(details['selection']['selected']),tv_scaled=movement,
                secondary_fit_range_objective=objective,solver_optimality_gap=secondary['optimality_gap'])
            sample=read_json(case/'association_words_and_samples.json')['sample_group_ids']
            if any(not set(part)<=fits or len(part)!=len(set(part))for part in sample):
                raise ValueError('Association subsamples used non-fitting or repeated evidence units.')
            certificates.append(certificate)
    log.info('EXTERNAL REPLAY | %d finite prediction slots | %d certified actual readouts',len(checks),len(certificates))
    return checks,certificates,{'features':features,'masks':masks,'ids':ids,'vocabulary':vocabulary,'split':split}

def primary_replay(folder,prep,rows,log):
    features={r['group_id']:r for r in jsonl(prep/'feature_records.jsonl')}
    masks={r['group_id']:r for r in jsonl(prep/'completion_masks.jsonl')}
    ids=read_json(folder/'test_population.json')['group_ids'];vocabulary=read_json(prep/'vocabulary.json')['tokens']
    if ids!=read_json(prep/'completion_group_ids_by_split.json')['test']:
        raise ValueError('Primary test completion population changed.')
    check_masks(features,masks,ids,'test',vocabulary)
    shared=arrays(folder/'proposed_control_cases/seed809/frozen_test_arrays.npz')['prior']
    checks=[];certificates=[];cache={}
    method_names={'EnCOT':'encot','NeuroMax':'neuromax','LDA':'lda','NMF':'nmf','GSDMM':'gsdmm',
                  'BERTopic':'bertopic','CombinedTM':'combinedtm','FASTopic':'fastopic','GloCOM':'glocom'}
    for row in rows:
        if row.get('test_used_for_training_or_selection')or row.get('model_configuration_selected_on')!='development_only_before_test':
            raise ValueError('Primary test selection contract changed.')
        if row['status']!='scored'or row['prediction']['macro_source_nll']is None:continue
        if row.get('prediction_interface')=='frozen_native_affine_decoder'and row['method']!='EnCOT':
            case=folder/'proposed_control_cases'/f"seed{row['seed']}"
        else:
            case=folder/'baseline_cases'/method_names[row['method']]/row['profile']/f"seed{row['seed']}"
        if case not in cache:cache[case]=arrays(case/'frozen_test_arrays.npz')
        a=cache[case]
        if 'native_decoded_probabilities'in a:
            with np.errstate(divide='ignore'):raw=np.log(a['native_decoded_probabilities'])
            prior=shared
        else:
            role=row.get('actual_readout','native');phi=a.get('readout_'+role,a.get('native'))
            if phi is None:raise ValueError('Missing frozen readout for primary prediction.')
            moment=(a['moment_gain'],a['moment_offset'])if row.get('moment_correction')else None
            raw=logits(a['theta_test'],phi,a['decoder_gain'],a['decoder_offset'],moment)
            prior=a.get('prior',shared)
        checks.append({'dataset':'Primary','method':row['method'],'profile':row['profile'],'seed':row['seed'],
                       **replay_losses(raw,prior,row,ids,masks,features)})
        if row['method']=='Proposed'and row.get('role')=='proposed'and not any(c['seed']==row['seed']for c in certificates):
            certificates.append(mechanism(a['native'],a['proposed'],vocabulary,row['topics'],row['seed'],'Primary'))
    log.info('PRIMARY REPLAY | %d finite prediction slots | %d verified actual readouts',len(checks),len(certificates))
    return checks,certificates,{'features':features,'masks':masks,'ids':ids,'vocabulary':vocabulary}

def coherence_replay(context,normalized,log):
    """Recompute Proposed/parent fitting scores without removing descriptor slots."""
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    checks=[]
    for dataset,data in [('Primary',context['primary_data']),('External',context['external_data'])]:
        texts=[r['tokens']for _,r in sorted(data['features'].items())if r['split']=='fit']
        dictionary=Dictionary([data['vocabulary']])
        for row in normalized:
            if row['dataset']!=dataset or row['control']or row['method']not in ['Proposed','NeuroMax']:continue
            for metric,name,width in [('fit_c_v','c_v',110),('fit_npmi','c_npmi',10)]:
                value=float(CoherenceModel(topics=row['topics'],texts=texts,dictionary=dictionary,
                    coherence=name,window_size=width,topn=10,processes=1).get_coherence())
                error=close(value,row[metric],dataset+'/'+row['method']+'/'+name,1e-9)
                checks.append({'dataset':dataset,'method':row['method'],'seed':row['seed'],'metric':name,'absolute_error':error})
            log.info('COHERENCE REPLAY | %s | %s | seed=%d | all 100 slots retained',dataset,row['method'],row['seed'])
    return checks

def interval_replay(rows,intervals):
    """Verify both parent intervals and both external GSDMM contrasts independently."""
    checks=[]
    selected=[r for r in intervals if r['method']=='NeuroMax'or(r['dataset']=='External'and r['method']=='GSDMM')]
    for interval in selected:
        dataset=interval['dataset']
        pro={r['seed']:r for r in rows if r['dataset']==dataset and r['method']=='Proposed'and not r['control']}
        comp={r['seed']:r for r in rows if r['dataset']==dataset and r['method']==interval['method']
              and r['profile']==interval['profile']and not r['control']}
        if set(pro)!=set(SEEDS)or set(comp)!=set(SEEDS):raise ValueError('Bootstrap seed pairing changed.')
        rng=np.random.default_rng(interval['seed']);n=interval['resamples']
        seed_weights=rng.multinomial(5,np.full(5,.2),size=n);draws=np.zeros(n)
        sources=sorted({x['source']for x in pro[809]['document_losses']})
        for source in sources:
            base=sorted([x for x in pro[809]['document_losses']if x['source']==source],key=lambda x:x['group_id'])
            groups=[x['group_id']for x in base];tokens=np.array([x['target_tokens']for x in base],dtype=float)
            group_weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=n)
            total=group_weights@tokens
            for j,seed in enumerate(SEEDS):
                p={x['group_id']:x for x in pro[seed]['document_losses']if x['source']==source}
                c={x['group_id']:x for x in comp[seed]['document_losses']if x['source']==source}
                if set(p)!=set(groups)or set(c)!=set(groups):raise ValueError('Bootstrap group pairing changed.')
                if any(p[g]['target_tokens']!=tokens[i]or c[g]['target_tokens']!=tokens[i]for i,g in enumerate(groups)):
                    raise ValueError('Bootstrap target tokens changed.')
                loss=np.array([c[g]['negative_log_probability']-p[g]['negative_log_probability']for g in groups])
                draws+=(group_weights@loss)/total*seed_weights[:,j]/5/len(sources)
        lo,hi=np.quantile(draws,[.025,.975])
        checks.append({'dataset':dataset,'method':interval['method'],'profile':interval['profile'],
            'low_error':close(float(lo),interval['percentile_95_low'],'bootstrap lower bound'),
            'high_error':close(float(hi),interval['percentile_95_high'],'bootstrap upper bound')})
    return checks

def audit(bundle,log,check_coherence=True):
    p,e=bundle['primary'],bundle['external']
    primary=read_json(p/'all_test_rows.json');primary_controls=read_json(p/'all_test_control_rows.json')
    external=read_json(e/'comparison_rows.json');external_controls=read_json(e/'control_rows.json')
    for rows,n in [(primary,60),(primary_controls,60),(external,60),(external_controls,15)]:grid(rows,n)
    if read_json(e/'all_rows.json')!=external+external_controls:
        # Stored all_rows interleaves comparisons/controls, so compare by unique key.
        saved=read_json(e/'all_rows.json')
        key=lambda r:(r['method'],r['profile'],r['seed'])
        if sorted(saved,key=key)!=sorted(external+external_controls,key=key):raise ValueError('External row panels differ.')
    expected_panel={(r['label'],'english_words'if r['method']=='encot'else r['profile']['name'])
                    for r in bundle['external_identity']['policy']['comparators']}
    expected_panel.add(('Proposed','signed_stable/range'))
    if {(r['method'],r['profile'])for r in external}!=expected_panel:
        raise ValueError('External comparator profile panel changed.')
    normalized=[normalize(r,'Primary',control)for control,rows in [(False,primary),(True,primary_controls)]for r in rows]
    normalized+=[normalize(r,'External',control)for control,rows in [(False,external),(True,external_controls)]for r in rows]
    checks1,certs1,pdata=primary_replay(p,bundle['preparation'],primary+primary_controls,log)
    checks2,certs2,edata=external_replay(e,external+external_controls,bundle['external_identity'],log)
    summary=summaries(normalized)
    # Compare independently derived five-seed numbers with the saved Step 7 summaries.
    old={(r['method'],r['profile']):r for r in read_json(e/'summary_metrics.json')}
    mapping={'fit_c_v':'fitting_c_v','fit_npmi':'fitting_c_npmi','td':'topic_diversity','test_nll':'test_nll',
             'test_c_v':'test_c_v','test_npmi':'test_c_npmi'}
    for r in summary:
        if r['dataset']!='External':continue
        source=old[(r['method'],r['profile'])]
        for field,previous in mapping.items():
            for suffix in ['mean','sd']:close(r[field+'_'+suffix],source[previous+'_'+suffix],'five-seed summary')
    context={'primary_data':pdata,'external_data':edata}
    coherence=coherence_replay(context,normalized,log)if check_coherence else []
    intervals=[]
    for dataset,name,folder in [('Primary','paired_prediction_intervals.json',p),('External','paired_intervals.json',e)]:
        for original in read_json(folder/name):
            interval={'dataset':dataset,**original}
            if original.get('available',True):
                pp=[r for r in normalized if r['dataset']==dataset and not r['control']and r['method']=='Proposed']
                cc=[r for r in normalized if r['dataset']==dataset and r['method']==original['method']and
                    (r['profile']==original['profile']or original['profile'].startswith('control/'))]
                # A parent appears in both comparison/control panels; use one immutable copy.
                by={}
                for r in cc:by.setdefault(r['seed'],r)
                cc=list(by.values())
                if len(cc)!=5:raise ValueError('Interval comparison panel is incomplete.')
                delta=sum(r['test_nll']for r in cc)/5-sum(r['test_nll']for r in pp)/5
                close(delta,original['mean_competitor_minus_proposed_macro_nll'],'interval point mean')
                if original['percentile_95_low']>original['percentile_95_high']:raise ValueError('Reversed confidence interval.')
            intervals.append(interval)
    bootstrap=interval_replay(normalized,intervals)
    log.info('PAIRED INTERVAL REPLAY | %d parent/GSDMM contrasts | bounds reproduced',len(bootstrap))
    return {'rows':normalized,'summary':summary,'prediction_checks':checks1+checks2,
        'certificates':certs1+certs2,'coherence_checks':coherence,'intervals':intervals,**context,
        'bootstrap_checks':bootstrap,
        'verified_model_slots':195,'new_fits':0,'new_encoder_calls':0,'saved_predictions_only':True,
        'max_prediction_error':max(r['max_loss_error']for r in checks1+checks2)}
