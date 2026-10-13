from __future__ import annotations
import copy
import datetime as dt
import hashlib
import importlib.metadata
import json
import logging
import platform
from pathlib import Path
import sys
import threading
import time
import traceback
import uuid
import zipfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .core import (read,write,sha,safe,verified_inputs,load_population,seed_bundle,replay_selected,
                   predict,acquire,coherence,topic_indices,summarize,csv_write)
from .vendor.projection.replay_solver import primary_tv,range_secondary
from .vendor.projection.minimum_tv import certify
from .report import tables,figures

PINS={'numpy':'1.26.4','scipy':'1.13.1','gensim':'4.4.0','PyYAML':'6.0.3','matplotlib':'3.9.2'}


def setup(root):
    import yaml
    module=root/'sensitivity_analysis';plan=read(module/'data/protocol.json')
    versions={k:importlib.metadata.version(k) for k in PINS}
    errors=[f'{k}: installed={versions[k]}, required={v}' for k,v in PINS.items() if versions[k]!=v]
    if errors:raise RuntimeError('Run python -m pip install -r requirements-sensitivity.txt\n'+'\n'.join(errors))
    paths,audit=verified_inputs(root,plan)
    cfg=yaml.safe_load(paths['configs/step03_improvement.yaml'].read_text(encoding='utf-8'))
    if cfg['acquisition']['pair_group_support']!=plan['selected_support'] or cfg['projection']['rank_margin_scaled']!=plan['selected_margin']:
        raise ValueError('The selected markers differ from the frozen original settings.')
    data=load_population(paths,plan)
    bundles={s['seed']:seed_bundle(paths,s,data) for s in plan['seed_sources']}
    return module,plan,versions,paths,audit,cfg,data,bundles


def cache_load(folder,key):
    path=folder/'result.json'
    if not path.is_file():return None
    result=read(path)
    if result['key']!=key:raise ValueError('Case cache protocol/input identity changed.')
    for name,digest in result['output_hashes'].items():
        if sha(folder/name)!=digest:raise ValueError('Case cache output changed: '+name)
    return result['row']


def run_case(folder,identity,study,support,margin,mode,seed,bundle,data,plan,cfg,baseline,log,
             primary_result=None,targets=None,selection=None):
    key=hashlib.sha256(json.dumps([identity,study,support,margin,mode,seed],sort_keys=True).encode()).hexdigest()
    old=cache_load(folder,key)
    if old is not None:
        detail=read(folder/'evidence.json')
        if (folder/'readout.npz').is_file():
            with np.load(folder/'readout.npz',allow_pickle=False) as z:phi=z['affinity']
            check=certify(bundle['arrays']['native'],phi,detail['targets'],margin=margin)
            if check['certified']!=old['certified']:
                raise ValueError('Cached sensitivity readout certificate does not reproduce.')
            scores,losses=predict(bundle,phi,data,plan['backoff_rho'])
            fit=coherence(phi,data)
            for name,value in [('development_nll',scores['macro_source_nll']),('cv',fit['cv']),('npmi',fit['npmi']),('td',fit['td'])]:
                expected=old[name]
                if (expected is None)!=(value is None) or (value is not None and abs(value-expected)>1e-10):
                    raise ValueError('Cached sensitivity metric does not reproduce: '+name)
            if abs(old['delta_native_nll']-(scores['macro_source_nll']-baseline))>1e-10:
                raise ValueError('Cached paired prediction difference does not reproduce.')
            oldlosses=detail['document_losses']
            if [(x['group_id'],x['source'],x['target_tokens']) for x in losses]!=[(x['group_id'],x['source'],x['target_tokens']) for x in oldlosses] or max(abs(x['negative_log_probability']-y['negative_log_probability']) for x,y in zip(losses,oldlosses))>1e-9:
                raise ValueError('Cached per-group development losses do not reproduce.')
            native=bundle['arrays']['native']
            movement=100*float(np.abs(phi-native).sum()/native.sum())
            logit_range=float(np.ptp((bundle['contexts']@(phi-native))*bundle['arrays']['decoder_gain'],axis=1).mean())
            if abs(movement-old['relative_l1_percent'])>1e-10 or abs(logit_range-old['fitting_logit_range'])>1e-10:
                raise ValueError('Cached affinity movement/logit objective does not reproduce.')
        old=copy.deepcopy(old);old['replayed_from_cache']=True
        log.info('CACHE | %s seed=%d support=%d margin=%g %s',study,seed,support,margin,mode)
        return old,None
    folder.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    selected_setting=is_original_setting(support,margin,plan,cfg)
    native=bundle['arrays']['native'];saved=bundle['selected']
    if targets is None:
        if support==plan['selected_support']:
            targets=copy.deepcopy(saved['targets']);selection=copy.deepcopy(saved['selection'])
        else: targets,selection=acquire(bundle,data,cfg['acquisition'],support)
    row={'study':study,'seed':seed,'support':support,'margin':margin,'objective':mode,
         'selected_setting':selected_setting,'status':'pending','certified':False,
         'development_nll':None,'delta_native_nll':None,'cv':None,'npmi':None,'td':None,
         'relative_l1_percent':None,'target_recall':None,'edited_topics':len(selection['selected']),
         'replacements':sum(len(x['core']) for x in selection['selected']),
         'candidate_count':selection['candidate_count'],'fitting_logit_range':None,
         'native_development_nll':baseline,'prediction_rho':plan['backoff_rho'],
         'moment_correction':False,'post_exposure_diagnostic':True,'final_recipe_changed':False}
    row['replayed_from_cache']=False
    phi=None;certificate={};primary=None;primary_certificate=None
    if selected_setting:
        phi=bundle['selected_readouts'][mode].copy();certificate=copy.deepcopy(saved['certificates'][mode])
        row['computation']='verified_saved_selected_readout_reuse'
    elif not selection['required_diversity_achieved']:
        row['status']='acquisition_diversity_infeasible'
    elif not selection['selected']:
        phi=native.copy();certificate=certify(native,phi,targets,margin=margin)
        row['computation']='explicit_native_abstention';row['status']='abstained_native'
    else:
        pp=cfg['projection'];cp=cfg['controls']
        if primary_result is None:
            primary,primary_certificate=primary_tv(native,targets,margin,pp['primary_seconds'])
        else: primary,primary_certificate=primary_result
        if mode=='primary':phi,certificate=primary,primary_certificate
        elif primary is not None:
            phi,certificate=range_secondary(native,targets,primary_certificate['solver_tv_scaled'],
                bundle['contexts'],bundle['arrays']['decoder_gain'],primary,margin,
                cp['tv_cap_slack_scaled'],cp['secondary_seconds'],cp['range_maximum_iterations'],cp['range_tolerance'],log)
        else:certificate={'failure':'primary_infeasible_or_not_certified','primary_certificate':primary_certificate}
        row['computation']='new_saved_parent_projection'
    losses=[];scores=None;fit=None
    if phi is not None:
        check=certify(native,phi,targets,margin=margin)
        if not check['certified'] and row['status']!='abstained_native':
            phi=None;certificate['independent_rejection']=check
        else:
            row['certified']=check['certified']
            fit=coherence(phi,data)
            scores,losses=predict(bundle,phi,data,plan['backoff_rho'])
            actual=topic_indices(phi)
            recall=sum(len(set(a)&set(b)) for a,b in zip(actual,targets))/100.
            if recall!=1. or fit['td']+1e-12<len({w for t in topic_indices(native) for w in t})/100.:
                raise ValueError('Actual readout failed target realization/diversity checks.')
            change=(bundle['contexts']@(phi-native))*bundle['arrays']['decoder_gain']
            row.update(cv=fit['cv'],npmi=fit['npmi'],td=fit['td'],development_nll=scores['macro_source_nll'],
                       delta_native_nll=scores['macro_source_nll']-baseline,
                       relative_l1_percent=100*float(np.abs(phi-native).sum()/native.sum()),
                       target_recall=recall,fitting_logit_range=float(np.ptp(change,axis=1).mean()))
            if row['status']=='pending':row['status']='certified_scored'
            certificate['independent_replay_certificate']=check
            if selected_setting:
                expected=saved['rows'][mode]
                for name,expectedname in [('cv','c_v'),('npmi','c_npmi')]:
                    if row[name] is None or abs(row[name]-expected['fitting'][expectedname])>1e-10:
                        raise ValueError('Selected fitting coherence failed source replay.')
                if abs(row['development_nll']-expected['prediction']['macro_source_nll'])>1e-10:
                    raise ValueError('Selected development loss failed source replay.')
    if phi is None:
        row['status']='projection_unavailable' if row['status']=='pending' else row['status']
        row['failure']=certificate.get('failure',certificate.get('message','No certified projected readout.'))
    row['wall_seconds']=time.perf_counter()-started
    detail={'targets':targets,'selection':selection,'certificate':certificate,
            'coherence':fit,'prediction':scores,'document_losses':losses}
    write(folder/'evidence.json',detail)
    if phi is not None:
        tmp=folder/'readout.npz.tmp'
        with tmp.open('wb') as stream:np.savez_compressed(stream,affinity=phi)
        tmp.replace(folder/'readout.npz')
    else:(folder/'readout.npz').unlink(missing_ok=True)
    names=['evidence.json']+(['readout.npz'] if phi is not None else [])
    write(folder/'result.json',{'key':key,'row':row,'output_hashes':{n:sha(folder/n) for n in names}})
    log.info('CASE | %s | seed=%d support=%d margin=%g %s | %s | delta dev NLL=%s | L1%%=%s',
             study,seed,support,margin,mode,row['status'],row['delta_native_nll'],row['relative_l1_percent'])
    return row,(primary,primary_certificate) if primary is not None else None


def is_original_setting(support,margin,plan,cfg):
    # The reviewed selected readouts must never be reused for a changed budget.
    return (support==plan['selected_support'] and margin==plan['selected_margin']
            and cfg['acquisition']['maximum_replacements']==4
            and cfg['acquisition']['retained_native_words_minimum']==6)


def make_zip(path,files,root):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in files:z.write(file,file.relative_to(root).as_posix())
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:raise ValueError('Inspection archive CRC validation failed.')


def seed_job(payload):
    root,cache,identity,seed,bundle,data,plan,cfg,baseline=payload
    logger=logging.getLogger('sensitivity_seed_'+str(seed));logger.setLevel(logging.INFO)
    handler=logging.StreamHandler(sys.stdout);handler.setFormatter(logging.Formatter('seed='+str(seed)+' | %(message)s'));logger.addHandler(handler)
    results=[]
    try:
        for support in plan['support_values']:
            row,_=run_case(cache/'support'/str(support)/str(seed),identity,'support',support,plan['selected_margin'],'range',seed,bundle,data,plan,cfg,baseline,logger)
            results.append(row)
        for margin in plan['margin_values']:
            primary_saved=None
            for mode in plan['projection_objectives']:
                folder=cache/'margin'/format(margin,'.0e')/str(seed)/mode
                row,computed=run_case(folder,identity,'margin',plan['selected_support'],margin,mode,seed,bundle,data,plan,cfg,baseline,logger,primary_result=primary_saved)
                if mode=='primary':
                    primary_saved=computed
                    if primary_saved is None and (folder/'readout.npz').is_file() and row['certified']:
                        with np.load(folder/'readout.npz',allow_pickle=False) as z:phi=z['affinity']
                        primary_saved=(phi,read(folder/'evidence.json')['certificate'])
                results.append(row)
        return results
    finally:logger.removeHandler(handler);handler.close()


def cli(args,root):
    root=Path(root).resolve();run=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]
    output=root/'outputs/steps/step08_sensitivity'/run;output.mkdir(parents=True,exist_ok=True)
    logdir=root/'logs';logdir.mkdir(exist_ok=True)
    logger=logging.getLogger('primary_sensitivity_'+run);logger.setLevel(logging.INFO)
    formatter=logging.Formatter('%(asctime)sZ | %(levelname)s | %(message)s',datefmt='%Y-%m-%dT%H:%M:%S')
    formatter.converter=time.gmtime
    for handler in [logging.StreamHandler(sys.stdout),logging.FileHandler(logdir/(run+'.log'),encoding='utf-8'),logging.FileHandler(logdir/'experiment_log.txt',encoding='utf-8')]:
        handler.setFormatter(formatter);logger.addHandler(handler)
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(30):logger.info('RUNNING | sensitivity | run=%s | saved-array LPs on CPU | no neural fits',run)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    started=time.perf_counter();result={};exit_code=1
    try:
        logger.info('START | primary sensitivity | requested_device=%s | saved-array compute=CPU | new model/encoder calls=0',args.device)
        module,plan,versions,paths,audit,cfg,data,bundles=setup(root)
        logger.info('POPULATION | primary | fit=%d fit_observed=%d development_completion=%d V=%d | sources=%s',
                    len(data['fit_ids']),len(data['fit_observed_ids']),len(data['dev_ids']),len(data['vocabulary']),','.join(data['sources']))
        write(output/'protocol.json',plan);write(output/'input_audit.json',audit)
        result={'run_id':run,'protocol':plan['protocol'],'versions':versions,'python':platform.python_version(),
                'new_neural_fits':0,'new_encoder_calls':0,'test_scoring_calls':0,'final_recipe_changed':False,
                'implementation_sha256':{p.relative_to(root).as_posix():sha(p) for p in module.rglob('*.py')}}
        if args.self_check:
            from .tests import run_checks
            checks=run_checks(root,plan,data,bundles,cfg)
            write(output/'self_check.json',checks)
            logger.info('SELF CHECK | tests=%d | failures=%d | errors=%d',checks['tests'],checks['failures'],checks['errors'])
            result.update(status='SELF_CHECK_PASSED' if checks['passed'] else 'SELF_CHECK_FAILED',checks=checks)
            exit_code=0 if checks['passed'] else 1
        else:
            replay={s:replay_selected(bundles[s],data,plan) for s in plan['seeds']}
            write(output/'selected_source_replay.json',replay)
            logger.info('SOURCE REPLAY PASS | five seeds | selected primary/range rows | max group loss error=%.3g',max(replay.values()))
            identity=hashlib.sha256(json.dumps({'plan':plan,'versions':versions},sort_keys=True).encode()).hexdigest()
            cache=root/'data/cache/primary_sensitivity'/identity[:20];cache.mkdir(parents=True,exist_ok=True)
            if getattr(args,'sensitivity_recompute',False):
                cache=cache/'independent_recompute'/run;cache.mkdir(parents=True,exist_ok=True)
            write(output/'sweep_declared_before_computation.json',{'identity':identity,'plan':plan,'versions':versions,
                  'maximum_slots':65,'source':'reviewed fitting/development arrays','selected_recipe_remains_locked':True})
            rows=[];baselines={}
            for seed in plan['seeds']:
                bundle=bundles[seed];baselines[seed]=predict(bundle,bundle['arrays']['native'],data,plan['backoff_rho'])[0]['macro_source_nll']
            write(output/'native_development_nll.json',baselines)
            logger.info('EXECUTION | two independent seed workers | original LP budgets | completed cases resume from verified cache')
            with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as executor:
                jobs=[executor.submit(seed_job,(root,cache,identity,seed,bundles[seed],data,plan,cfg,baselines[seed])) for seed in plan['seeds']]
                for future in as_completed(jobs):
                    rows.extend(future.result());write(output/'individual_results.json',rows)
                    logger.info('SEED COMPLETE | saved slots=%d/65',len(rows))
            rows.sort(key=lambda r:(0 if r['study']=='support' else 1,r['support'],r['margin'],r['seed'],plan['projection_objectives'].index(r['objective'])))
            aggregates=summarize(rows)
            for a in aggregates:
                logger.info('MEAN RESULTS | %s | support=%d margin=%g %s | certified=%d/5 | Cv=%s NPMI=%s | delta dev NLL=%s | affinity L1%%=%s | selected_parameters=%s',
                            a['study'],a['support'],a['margin'],a['objective'],a['certified_seeds'],a['cv_mean'],a['npmi_mean'],a['delta_native_nll_mean'],a['relative_l1_percent_mean'],a['selected_setting'])
            if len(rows)!=65 or len({(r['study'],r['support'],r['margin'],r['objective'],r['seed']) for r in rows})!=65:
                raise ValueError('The full sensitivity grid is incomplete or duplicated.')
            write(output/'mean_sd_results.json',aggregates)
            tables(output,rows,aggregates,module);figures(output,aggregates,rows,plan)
            success=sum(r['certified'] for r in rows)
            text=['Primary sensitivity study complete','',f'Computed/reused all 65 declared slots; {success}/65 independently certified. Every failure or abstention is retained.',
                  'No neural training, encoder calls, test evaluation or changes to frozen final settings.',
                  'This is a post-test-exposure descriptive study on fitting acquisition and development completion. It cannot retrospectively establish pre-test selection or justify choosing a replacement final model.',
                  '', 'Open tables/matched_ablations_complete.md for the full existing ablation table.',
                  'Open tables/sensitivity_table.md and figures/R1_primary_support_sensitivity.pdf / R2_primary_margin_tradeoff.pdf.',
                  'Star/dashed markers identify the original selected support=5 and scaled margin=1e-6. They do not assert that these settings maximize every metric.',
                  'Fitting C_v/NPMI and seed SD are not held-out interpretability evidence or confidence intervals.',
                  'The logit-range objective concerns 64 fitting contexts. Development NLL is independently evaluated through the fixed final decoder (moment off; rho=0).',
                  '',f'Case cache: {cache.relative_to(root).as_posix()}',
                  'Rerunning verifies and resumes this exact grid. Completed case readouts, solver histories, losses and checksums are preserved in the cache.']
            (output/'SUMMARY.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
            result.update(status='SENSITIVITY_GRID_COMPLETE',completed_slots=65,certified_slots=success,
                          all_sensitivity_readouts_certified=success==65,full_five_seed_summary=True,
                          selected_support=plan['selected_support'],selected_margin=plan['selected_margin'],cache_relative_path=cache.relative_to(root).as_posix(),
                          independently_replayed_cache_slots=sum(r.get('replayed_from_cache',False) for r in rows))
            exit_code=0
            (output.parent/'LATEST.txt').write_text(output.relative_to(root).as_posix()+'\n',encoding='utf-8')
            logger.info('COMPLETE | slots=65 | certified=%d/65 | final model unchanged',success)
            logger.info('READ FIRST | %s',output/'SUMMARY.md')
    except KeyboardInterrupt:
        logger.warning('INTERRUPTED | completed case cache retained; rerun the same command to resume')
        result.update(status='INTERRUPTED');exit_code=130
    except Exception as exc:
        logger.exception('FAILED | %s: %s',type(exc).__name__,exc)
        result.update(status='FAILED',failure=str(exc),traceback=traceback.format_exc())
    finally:
        stop.set();thread.join(timeout=1)
        result['wall_seconds']=time.perf_counter()-started
        write(output/'run_result.json',result)
        files=[p for p in output.rglob('*') if p.is_file()]+[logdir/(run+'.log')]
        if not args.self_check and result.get('status')=='SENSITIVITY_GRID_COMPLETE':
            files += [p for p in cache.rglob('*') if p.is_file() and p.suffix in ['.json','.npz']]
        inspection=root/'outputs/inspection'/('sensitivity_step08_'+run+'_inspection.zip');inspection.parent.mkdir(parents=True,exist_ok=True)
        make_zip(inspection,files,root)
        logger.info('INSPECTION ZIP | %s',inspection)
        logger.info('REPORT | %s',output/'run_result.json')
        for h in list(logger.handlers):h.close();logger.removeHandler(h)
    return exit_code
