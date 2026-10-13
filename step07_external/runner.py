"""Step 7 orchestration; isolated outputs, fixed recipe, resumable fits and audit ZIP."""
from __future__ import annotations
import argparse
import datetime as dt
import io
import logging
import os
import platform
import sys
import threading
import time
import traceback
import unittest
import uuid
import zipfile
from pathlib import Path
import numpy as np
from rank_refinement.runtime.artifacts import artifact_manifest, digest, file_hash, read_json, write_json, write_text
from rank_refinement.runtime.config import load_configs
from rank_refinement.runtime.hardware import inspect_hardware
from rank_refinement.runtime.state import ProjectLock
from rank_refinement.evaluation.resource_usage import ResourceMonitor
from rank_refinement.stages.step02_completion import check_indexed_tree
from .protocol import (PROTOCOL,DATA_SHA,primary_guard,assert_primary_unchanged,recipe,
                       implementation,input_file,versions)
from .data import prepare

class ExternalLog:
    def __init__(self,root):
        self.root=root;self.step=7
        self.run_id=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]
        self.started=time.perf_counter();self.folder=root/'outputs/runs'/self.run_id/'step07_external'
        self.folder.mkdir(parents=True);(root/'logs').mkdir(exist_ok=True)
        self.log_path=root/'logs'/(self.run_id+'.log')
        self.logger=logging.getLogger('rank_external.'+self.run_id);self.logger.setLevel(logging.INFO);self.logger.propagate=False
        fmt=logging.Formatter('%(asctime)sZ | %(levelname)s | %(message)s',datefmt='%Y-%m-%dT%H:%M:%S');fmt.converter=time.gmtime
        for handler in [logging.StreamHandler(sys.stdout),logging.FileHandler(self.log_path,encoding='utf-8')]:
            handler.setFormatter(fmt);self.logger.addHandler(handler)
        self.history('start',{})
    def info(self,message,*args):self.logger.info(message,*args)
    def history(self,event,details):
        from rank_refinement.runtime.artifacts import stable_json
        value={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'run_id':self.run_id,'step':7,'event':event,
            'elapsed_seconds':time.perf_counter()-self.started,**details}
        with(root_file:=self.root/'run_history.txt').open('a',encoding='utf-8')as f:f.write(stable_json(value)+'\n')
    def close(self):
        for handler in list(self.logger.handlers):handler.flush();handler.close();self.logger.removeHandler(handler)

def inspect_zip(root,log,output):
    """Results+model indices, not encoder weights, optimizer states or old ZIPs."""
    paths=list(log.folder.rglob('*'))+[log.log_path,root/'run_history.txt',root/'main.py',root/'step07_settings.json']
    paths+=list((root/'step07_external').rglob('*.py'))+list(root.glob('requirements*.txt'))
    paths+=list((root/'configs').glob('*'))
    paths+=list((root/'src').rglob('*.py'))+[root/'step06_finalize.py']
    if output is not None:
        paths += [p for p in output.rglob('*')if p.is_file()and p.suffix not in {'.pt','.safetensors','.zip','.download','.pyc'}]
    paths += [p for p in [root/'state/step07_external_exposure.json',root/'state/step07_external_result.json']if p.is_file()]
    paths += [root/'outputs/steps/step07_external/LATEST.json']
    selected=sorted({p.resolve()for p in paths if p.is_file()})
    metadata=[{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':file_hash(p)}for p in selected]
    destination=root/'outputs/inspection'/('step07_'+log.run_id+'_inspection.zip');destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6)as z:
        for p in selected:z.write(p,p.relative_to(root).as_posix())
        from rank_refinement.runtime.artifacts import stable_json
        z.writestr('INSPECTION_CONTENTS.json',stable_json({'files':metadata,'large_checkpoints_excluded':True,'raw_csv_excluded':True})+'\n')
    temporary.replace(destination);return destination

def freeze_exposure(root,identity):
    path=root/'state/step07_external_exposure.json'
    frozen={'freeze_id':digest(identity),'identity':identity}
    if path.is_file():
        prior=read_json(path)
        if prior.get('freeze_id')!=digest(identity)or prior.get('identity')!=identity:
            raise RuntimeError('External test was already exposed under a different recipe, input, implementation or device. '
                               'Preserve the saved run; do not delete the exposure ledger or retune this test.')
        return prior
    write_json(path,{**frozen,'first_exposure_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    return read_json(path)

def run_experiment(root,configs,policy,version_info,hardware,log,args,primary):
    from .models import encode,fit_native_case,fit_baseline_case
    from .report import score_cases,export
    path=input_file(root,args.external_input)
    identity={'protocol':PROTOCOL,'policy':policy,'versions':version_info,
        'implementation':implementation(root),'input_sha256':DATA_SHA,
        'primary_freeze_id':primary['freeze_id'],'device':hardware['future_model_device']}
    key=digest(identity);output=root/'outputs/steps/step07_external'/key[:20]
    output.mkdir(parents=True,exist_ok=True)
    if (output/'artifact_manifest.json').is_file():
        check_indexed_tree(output)
    write_json(log.folder/'working_path.json',{'relative_path':output.relative_to(root).as_posix()})
    protocol_path=output/'protocol_resolved_before_training.json'
    if protocol_path.is_file()and read_json(protocol_path)!=identity:raise ValueError('External recipe output identity changed.')
    write_json(protocol_path,identity)
    write_json(output/'primary_result_protection.json',primary)
    log.info('PHASE 1/5 | English data preparation | no model training or test scoring')
    data=prepare(path,policy,output/'preparation',log)
    if not data['audit']['engineering_data_gate_passed']:
        write_text(output/'SUMMARY.md','External preparation stopped before training.\n\n'+str(data['audit']['data_gate_failures']))
        raise RuntimeError('External preparation cannot support the declared run: '+', '.join(data['audit']['data_gate_failures']))
    if args.prepare_only:
        result={'status':'STEP07_EXTERNAL_PREPARATION_COMPLETE','engineering_passed':True,'preparation_gate_passed':True,
            'model_quality_established':False,'external_models_fitted':False,'test_losses_scored':False,
            'data_audit':data['audit'],'primary_results_unchanged':True,'experiment_id':key}
        write_json(output/'preparation_result.json',result)
        write_text(output/'PREPARATION_SUMMARY.md',
            '# External preparation only\n\nNo model results were generated.\n\n'+
            f"Rows: {data['audit']['nonempty_text_rows']}; independent groups: {data['audit']['independent_groups']}; vocabulary: {len(data['vocabulary'])}; test completion groups: {len(data['test_ids'])}.\n\n"+
            'The fixed fitting vocabulary covers '+f"{data['audit']['coverage'][-1]['in_vocabulary_fraction']:.1%}"+' of test lexical tokens.\n\nRun `python .\\main.py --step 7 --device auto` to fit the complete frozen external recipe.\n')
        return result,output
    # Reject recipe changes before fitting if this external test has already been exposed.
    exposure=root/'state/step07_external_exposure.json'
    if exposure.is_file()and read_json(exposure)['identity']['experiment_id']!=key:
        raise RuntimeError('External test already exposed under another experiment identity; no new fits or retuning were performed.')
    from rank_refinement.models.neural_sources import resolve_sources as native_sources
    from rank_refinement.models.competitor_sources import resolve_sources as competitor_sources
    folders,native_specs=native_sources(root,log);comp_folders,comp_specs=competitor_sources(root,log)
    log.info('PHASE 2/5 | pinned English encoder | held-out inputs are observed occurrences only')
    vectors=encode(root,output,data,policy['encoder'],hardware['future_model_device'],log,args.offline)
    log.info('PHASE 3/5 | five fixed seeds | 50 other-profile fits at most, plus 5 shared Proposed/parent fits | no external profile search')
    paths=[];device=hardware['future_model_device']
    for seed in policy['seeds']:
        case_identity={'experiment_id':key,'kind':'proposed_parent_controls','seed':seed}
        fit_native_case(root,output,seed,data,vectors,policy,device,folders,native_specs,log,case_identity)
        paths.append(output/'cases/proposed_parent'/f'seed{seed}')
        for item in policy['comparators']:
            if item['method']=='neuromax':continue
            case_identity={'experiment_id':key,'kind':'competitor','method':item['method'],'profile':item['profile']['name'],'seed':seed}
            fit_baseline_case(root,output,item,seed,data,vectors,policy,device,folders,native_specs,comp_folders,comp_specs,log,case_identity)
            paths.append(output/'cases/competitors'/item['method']/item['profile']['name']/f'seed{seed}')
    if len(paths)!=55:raise ValueError('External model-fit grid is incomplete.')
    # All backoff choices, acquired targets and projected readouts exist before test losses are used.
    case_hashes={p.relative_to(output).as_posix():file_hash(p/'artifact_manifest.json')for p in paths}
    for p in paths:check_indexed_tree(p)
    frozen=freeze_exposure(root,{'experiment_id':key,'dataset_sha256':DATA_SHA,'policy_sha256':digest(policy),
        'implementation':identity['implementation'],'fit_cases':case_hashes,
        'test_group_ids_sha256':digest(data['test_ids']),
        'test_masks_sha256':digest({g:data['masks'][g]for g in data['test_ids']}),
        'test_used_for_model_selection':False})
    write_json(output/'external_test_freeze.json',frozen)
    log.info('PHASE 4/5 | recipe locked | score held-out external test; retain all NA and losses')
    rows=score_cases(output,paths,data,policy,log)
    assert_primary_unchanged(root,primary)
    log.info('PHASE 5/5 | tables, controls, paired uncertainty, figures and projection certificates')
    result=export(output,rows,data,policy,True,log)
    result.update(experiment_id=key,external_test_freeze_id=frozen['freeze_id'],processed_relative_path=output.relative_to(root).as_posix())
    write_json(output/'run_result.json',result)
    write_json(output/'artifact_manifest.json',artifact_manifest(output))
    write_json(root/'outputs/steps/step07_external/LATEST.json',{'experiment_id':key,'relative_path':output.relative_to(root).as_posix()})
    write_json(root/'state/step07_external_result.json',result)
    log.info('STEP 7 COMPLETE | Proposed labeled explicitly | primary results unchanged | full external grid saved')
    log.info('READ FIRST | %s',output/'SUMMARY.md')
    return result,output

def cli(args,root):
    root=Path(root).resolve();log=ExternalLog(root);monitor=ResourceMonitor();output=None;code=2;result=None
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(30):log.info('RUNNING | Step 7 | run=%s | detailed log=%s',log.run_id,log.log_path)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    primary=None
    try:
        log.info('START | Step 7 | run=%s | requested_device=%s',log.run_id,args.device)
        version_info=versions(full=not args.prepare_only)
        hardware=inspect_hardware(args.device)
        import torch
        torch.set_num_threads(min(8,os.cpu_count()or 1))
        configs=load_configs(root)
        primary=primary_guard(root,configs)
        log.info('PRIMARY PROTECTION | completed Step 6 verified | exact source and YAML retained')
        log.info('DEVICE | external neural training/English encoder=%s | text/graph/LP=CPU',hardware['future_model_device'])
        for gpu in hardware['gpus']:log.info('GPU | %s | VRAM=%.2f GiB',gpu['name'],gpu['memory_gib'])
        write_json(log.folder/'hardware.json',hardware);write_json(log.folder/'versions.json',version_info)
        fixed=recipe(configs);settings=read_json(root/'step07_settings.json')
        if settings.get('protocol')!=PROTOCOL or settings.get('dataset_sha256')!=DATA_SHA or settings.get('seeds')!=fixed['seeds']:
            raise ValueError('External provenance settings changed; require the supplied Step 7 files.')
        if args.doctor_only:
            path=input_file(root,args.external_input)
            result={'status':'STEP07_DOCTOR_COMPLETE','engineering_passed':True,'preparation_gate_passed':False,
                    'primary_results_unchanged':True,'input_sha256':file_hash(path),'model_quality_established':False};code=0
        elif args.self_check:
            from . import tests
            tests.PROJECT_ROOT=root
            suite=unittest.defaultTestLoader.loadTestsFromModule(tests);stream=io.StringIO()
            checked=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
            write_text(log.folder/'self_checks.txt',stream.getvalue())
            log.info('SELF CHECK | tests=%d | failures=%d | errors=%d | skipped=%d',checked.testsRun,len(checked.failures),len(checked.errors),len(checked.skipped))
            if checked.testsRun<15 or not checked.wasSuccessful()or checked.skipped:
                log.info('%s',stream.getvalue());raise AssertionError('Step 7 checks failed. See self_checks.txt in inspection ZIP.')
            result={'status':'SELF_CHECK_PASSED','engineering_passed':True,'preparation_gate_passed':False,
                'model_quality_established':False,'tests':checked.testsRun,'synthetic_fixture_only':True,
                'real_external_models_fitted':False,'real_external_test_scored':False};code=0
        else:
            with ProjectLock(root):result,output=run_experiment(root,configs,fixed,version_info,hardware,log,args,primary)
            code=0
        assert_primary_unchanged(root,primary)
        log.info('VERDICT | %s | engineering=%s | external quality must be assessed from saved results',result['status'],result['engineering_passed'])
    except (Exception,KeyboardInterrupt)as exc:
        result={'status':'INTERRUPTED'if isinstance(exc,KeyboardInterrupt)else 'FAILED','engineering_passed':False,
            'preparation_gate_passed':False,'model_quality_established':False,'error_type':type(exc).__name__,'error':str(exc)}
        write_text(log.folder/'failure_traceback.txt',traceback.format_exc());log.logger.error('FAILED | %s: %s',type(exc).__name__,exc)
        if (log.folder/'working_path.json').is_file():output=root/read_json(log.folder/'working_path.json')['relative_path']
        code=2
    finally:
        stop.set();thread.join(timeout=1)
        if primary is not None:
            try:assert_primary_unchanged(root,primary)
            except Exception as exc:
                result={'status':'FAILED_PRIMARY_PROTECTION','engineering_passed':False,'error':str(exc)};code=2
        resources=monitor.finish()
        try:
            import torch
            resources['gpu_peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available()else 0.
        except ImportError:resources['gpu_peak_allocated_gib']=0.
        resources.update(new_neural_fits=getattr(log,'new_neural_fits',0),new_classical_fits=getattr(log,'new_classical_fits',0))
        write_json(log.folder/'resource_usage.json',resources)
        log.info('RESOURCES | wall=%.2f sec | peak CPU RSS=%.3f GiB | GPU peak=%.3f GiB | new neural fits=%d | new classical fits=%d',
            resources['wall_seconds'],resources['peak_cpu_rss_gib'],resources['gpu_peak_allocated_gib'],resources['new_neural_fits'],resources['new_classical_fits'])
        write_json(log.folder/'run_result.json',result or {'status':'FAILED_UNKNOWN'})
        log.history('finish',{'status':(result or {}).get('status'),'exit_code':code})
        for handler in log.logger.handlers:handler.flush()
        try:
            bundle=inspect_zip(root,log,output);log.info('INSPECTION ZIP | %s',bundle)
        except Exception as exc:log.logger.error('INSPECTION ZIP FAILED | %s',exc);code=2
        log.info('REPORT | %s',log.folder/'run_result.json');log.close()
    return code
