"""Read-only report entrypoint, timestamped log, manifests and review ZIP."""
from __future__ import annotations
import datetime as dt
import io
import json
import logging
import os
import sys
import threading
import time
import traceback
import unittest
import uuid
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from rank_refinement.runtime.artifacts import artifact_manifest,digest,file_hash,read_json,write_json,write_text
from rank_refinement.runtime.state import ProjectLock
from rank_refinement.evaluation.resource_usage import ResourceMonitor
from .integrity import PROTOCOL,PRIMARY_ID,EXTERNAL_ID,EXTERNAL_FREEZE,protect,unchanged,implementation,verify_tree

class ReportLog:
    def __init__(self,root):
        self.root=root;self.step=8;self.started=time.perf_counter()
        self.run_id=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]
        self.folder=root/'outputs/runs'/self.run_id/'step08_assembly';self.folder.mkdir(parents=True)
        (root/'logs').mkdir(exist_ok=True);self.log_path=root/'logs'/(self.run_id+'.log')
        self.logger=logging.getLogger('rank_report.'+self.run_id);self.logger.setLevel(logging.INFO);self.logger.propagate=False
        fmt=logging.Formatter('%(asctime)sZ | %(levelname)s | %(message)s',datefmt='%Y-%m-%dT%H:%M:%S');fmt.converter=time.gmtime
        for h in [logging.StreamHandler(sys.stdout),logging.FileHandler(self.log_path,encoding='utf-8')]:
            h.setFormatter(fmt);self.logger.addHandler(h)
        self.history('start',{})
    def info(self,message,*args):self.logger.info(message,*args)
    def history(self,event,details):
        value={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'step':8,'run_id':self.run_id,'event':event,
               'elapsed_seconds':time.perf_counter()-self.started,**details}
        with(self.root/'run_history.txt').open('a',encoding='utf-8')as f:
            f.write(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
    def close(self):
        for h in list(self.logger.handlers):h.flush();h.close();self.logger.removeHandler(h)

def zip_files(destination,root,files,extra=None):
    destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_suffix('.tmp.zip')
    selected=sorted({p.resolve()for p in files if p.is_file()and p.resolve()!=destination.resolve()})
    records=[{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':file_hash(p)}for p in selected]
    with ZipFile(temporary,'w',ZIP_DEFLATED,compresslevel=6)as z:
        for p in selected:z.write(p,p.relative_to(root).as_posix())
        z.writestr('PACKAGE_CONTENTS.json',json.dumps({'files':records,**(extra or {})},indent=2)+'\n')
    temporary.replace(destination)
    return destination

def run(root,log):
    from .audit import audit
    from .export import export
    settings=read_json(root/'step08_settings.json')
    if settings!={'protocol':PROTOCOL,'primary_freeze_id':PRIMARY_ID,'external_experiment_id':EXTERNAL_ID,
                  'external_freeze_id':EXTERNAL_FREEZE,'new_model_fits':0,'test_parameter_selection':False}:
        raise ValueError('Require the supplied reviewed Step 8 settings.')
    log.info('PHASE 1/4 | verify completed primary/external source, exposure, case and report bytes')
    bundle=protect(root)
    identity={'protocol':PROTOCOL,'primary_manifest_sha256':file_hash(bundle['primary']/'artifact_manifest.json'),
        'external_manifest_sha256':file_hash(bundle['external']/'artifact_manifest.json'),
        'implementation':implementation(root),'new_fits':0,'new_encoder_calls':0,'new_test_selection':False}
    key=digest(identity);folder=root/'outputs/steps/step08_assembly'/key[:20]
    write_json(log.folder/'working_path.json',{'relative_path':folder.relative_to(root).as_posix()})
    if (folder/'artifact_manifest.json').is_file():
        verify_tree(folder);unchanged(root,bundle['snapshot']);result=read_json(folder/'run_result.json')
        if read_json(folder/'assembly_identity.json')!=identity:raise ValueError('Reporting identity changed.')
        log.info('VERIFIED REUSE | earlier Step 8 package unchanged | no fits, encoder calls or metric changes')
        return result,folder
    folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'assembly_identity.json',identity)
    write_json(folder/'input_result_hashes.json',bundle['snapshot'])
    log.info('PHASE 2/4 | independently replay saved probabilities, loss populations, readout constraints and fitting coherence')
    evaluated=audit(bundle,log,check_coherence=True)
    log.info('REPLAY PASS | prediction slots=%d | coherence checks=%d | max loss error=%.3g',
        len(evaluated['prediction_checks']),len(evaluated['coherence_checks']),evaluated['max_prediction_error'])
    log.info('PHASE 3/4 | exact tables, LaTeX, figures, controls, examples and restrained manuscript claims')
    decisions=export(folder,bundle,evaluated)
    unchanged(root,bundle['snapshot'])
    result={'status':'STEP08_REPORTING_COMPLETE','protocol':PROTOCOL,'engineering_passed':True,
        'reporting_ready':True,'submission_quality_automatically_approved':False,'all_metric_win_established':False,
        'model_quality_established':False,'primary_results_unchanged':True,'external_results_unchanged':True,
        'primary_freeze_id':PRIMARY_ID,'external_experiment_id':EXTERNAL_ID,'external_freeze_id':EXTERNAL_FREEZE,
        'new_model_fits':0,'new_encoder_calls':0,'new_parameter_selections':0,'verified_slots':195,
        'finite_predictive_slots_replayed':len(evaluated['prediction_checks']),
        'fitting_coherence_replay_checks':len(evaluated['coherence_checks']),
        'parent_and_gsdmm_interval_replay_checks':len(evaluated['bootstrap_checks']),
        'certified_proposed_readouts_verified':len(evaluated['certificates']),
        'maximum_loss_replay_error':evaluated['max_prediction_error'],
        'assembly_id':key,'processed_relative_path':folder.relative_to(root).as_posix()}
    write_json(folder/'run_result.json',result)
    write_text(folder/'SUMMARY.md','# Step 8: verified paper results package\n\n'
        'Experiment assembly is complete. Earlier predictions, choices and reported outcomes are unchanged. No model fitting or encoder call occurred.\n\n'
        'Main exports are T1–T4 and F1–F2; supplementary exports preserve all controls, unavailable cells, profiles, descriptor words and uncertainty. CSV values retain full precision. LaTeX requires booktabs.\n\n'
        'Read RESULTS_TEXT.md, SUPPORTED_CLAIMS.md and METHOD_IMPLEMENTATION_ALIGNMENT.md. The primary panel has strong point means, while external superiority is mixed and parent intervals cross zero on both corpora. This package does not approve journal acceptance, universal superiority, or requirement-detection accuracy.\n\n'
        'paper_package.zip contains the generated publication exports. Model/data artifacts remain in the existing project. The separate outputs/inspection ZIP contains the report package, source and numerical audit for review.\n')
    log.info('PHASE 4/4 | preserve result hashes and package main/supplementary exports')
    publication=zip_files(folder/'paper_package.zip',folder,[p for p in folder.rglob('*')if p.is_file()],
        {'scope':'publication report exports; earlier model checkpoints remain in the project','assembly_id':key})
    write_json(folder/'artifact_manifest.json',artifact_manifest(folder))
    write_json(root/'state/step08_assembly_result.json',result)
    write_json(root/'outputs/steps/step08_assembly/LATEST.json',{'assembly_id':key,'relative_path':folder.relative_to(root).as_posix()})
    for r in decisions['point_winners']:
        log.info('POINT-MEAN LEADER | %s | %s | %s / %s | %.6f',r['dataset'],r['metric'],r['method'],r['profile'],r['mean'])
    log.info('STEP 8 COMPLETE | reporting ready=True | universal superiority=False | journal approval not established')
    log.info('READ FIRST | %s',folder/'SUMMARY.md');log.info('PAPER PACKAGE | %s',publication)
    return result,folder

def cli(args,root):
    root=Path(root).resolve();log=ReportLog(root);monitor=ResourceMonitor();output=None;result=None;code=2
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(30):log.info('RUNNING | Step 8 | report verification/export | detailed log=%s',log.log_path)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        import importlib.metadata
        versions={k:importlib.metadata.version(k)for k in ['numpy','scipy','gensim','matplotlib','PyYAML','psutil']}
        if versions['numpy']!='1.26.4'or versions['scipy']!='1.13.1'or versions['gensim']!='4.4.0':
            raise RuntimeError('Run python -m pip install -r requirements-step08.txt using the same project interpreter.')
        write_json(log.folder/'versions.json',versions)
        log.info('START | Step 8 | run=%s | requested device=%s | report replay/export=CPU | neural fits=0',log.run_id,args.device)
        if args.self_check:
            from . import tests
            stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromModule(tests)
            checked=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
            write_text(log.folder/'self_checks.txt',stream.getvalue())
            log.info('SELF CHECK | tests=%d | failures=%d | errors=%d | skipped=%d',checked.testsRun,len(checked.failures),len(checked.errors),len(checked.skipped))
            if checked.testsRun<15 or not checked.wasSuccessful()or checked.skipped:
                log.info('%s',stream.getvalue());raise AssertionError('Step 8 self-check failed.')
            result={'status':'SELF_CHECK_PASSED','engineering_passed':True,'tests':checked.testsRun,
                    'new_model_fits':0,'synthetic_checks_only':True,'reporting_ready':False};code=0
        elif args.doctor_only:
            b=protect(root);unchanged(root,b['snapshot'])
            result={'status':'STEP08_DOCTOR_COMPLETE','engineering_passed':True,'new_model_fits':0,'reporting_ready':False};code=0
        else:
            with ProjectLock(root):result,output=run(root,log)
            code=0
    except (Exception,KeyboardInterrupt)as exc:
        result={'status':'INTERRUPTED'if isinstance(exc,KeyboardInterrupt)else'FAILED','engineering_passed':False,
                'reporting_ready':False,'new_model_fits':0,'error_type':type(exc).__name__,'error':str(exc)}
        write_text(log.folder/'failure_traceback.txt',traceback.format_exc());log.logger.error('FAILED | %s: %s',type(exc).__name__,exc)
        if (log.folder/'working_path.json').is_file():output=root/read_json(log.folder/'working_path.json')['relative_path']
        code=2
    finally:
        stop.set();thread.join(timeout=1);resources=monitor.finish()
        resources.update(new_model_fits=0,new_encoder_calls=0,GPU_compute=False)
        write_json(log.folder/'resource_usage.json',resources);write_json(log.folder/'run_result.json',result or {'status':'FAILED'})
        log.history('finish',{'status':(result or {}).get('status'),'exit_code':code,'new_model_fits':0})
        log.info('RESOURCES | wall=%.2f sec | peak CPU RSS=%.3f GiB | model/encoder calls=0',resources['wall_seconds'],resources['peak_cpu_rss_gib'])
        try:
            files=list(log.folder.rglob('*'))+[log.log_path,root/'run_history.txt',root/'main.py',root/'step08_settings.json',root/'requirements-step08.txt']
            files+=list((root/'step08_assembly').glob('*.py'))
            if output is not None:files+=list(output.rglob('*'))
            review=zip_files(root/'outputs/inspection'/('step08_'+log.run_id+'_inspection.zip'),root,files,
                {'scope':'Step 8 reporting audit; no duplicated earlier checkpoints or raw inputs'})
            log.info('INSPECTION ZIP | %s',review)
        except Exception as exc:log.logger.error('Inspection packaging failed: %s',exc);code=2
        log.info('REPORT | %s',log.folder/'run_result.json');log.close()
    return code
