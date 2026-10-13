"""Declared, descriptive replacement-budget sensitivity on saved primary models."""
from __future__ import annotations
import copy
import csv
import datetime as dt
import hashlib
import io
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import threading
import time
import traceback
import unittest
import uuid
import numpy as np
from .core import acquire, csv_write, predict, read, replay_selected, sha, summarize, topic_indices, write
from .runner import cli as original_cli, setup, run_case, make_zip, is_original_setting
from .report import fmt, latex_table, ablation_readout_recall

VALUES = [2, 4, 6]
SELECTED = 4


def budget_config(cfg, value):
    if type(value) is not int or value not in VALUES:
        raise ValueError('Replacement budget must be one of the declared values: 2, 4, 6.')
    result = copy.deepcopy(cfg)
    result['acquisition']['maximum_replacements'] = value
    # This is the same constraint: replacing at most R of ten words retains at least 10-R.
    result['acquisition']['retained_native_words_minimum'] = 10-value
    return result


def validate_targets(native, targets, value):
    original = topic_indices(native)
    if len(targets) != 10 or any(len(t) != 10 or len(set(t)) != 10 for t in targets):
        raise ValueError('Replacement-budget study requires ten complete ten-word topics.')
    if any(len(set(t)-set(n)) > value for t, n in zip(targets, original)):
        raise ValueError('Target set exceeds its declared replacement budget.')


def budget_means(rows):
    result = []
    for value in VALUES:
        group = [copy.deepcopy(r) for r in rows if r['maximum_replacements']==value]
        if not group:
            raise ValueError('A declared budget has no report slots.')
        for r in group:
            r['study']='budget_'+str(value)
        entries=summarize(group)
        if len(entries)!=1:
            raise ValueError('Unexpected variation within a budget condition.')
        entry=entries[0];entry.update(study='replacement_budget',maximum_replacements=value)
        result.append(entry)
    return result


def seed_job(payload):
    cache, identity, seed, bundle, data, plan, cfg, baseline = payload
    logger=logging.getLogger('budget_seed_'+str(seed));logger.setLevel(logging.INFO)
    handler=logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('seed='+str(seed)+' | %(message)s'));logger.addHandler(handler)
    rows=[]
    try:
        for value in VALUES:
            changed=budget_config(cfg,value);folder=cache/str(value)/str(seed)
            if (folder/'result.json').is_file():
                detail=read(folder/'evidence.json');targets=detail['targets'];selection=detail['selection']
            elif value==SELECTED:
                targets=copy.deepcopy(bundle['selected']['targets']);selection=copy.deepcopy(bundle['selected']['selection'])
            else:
                logger.info('ACQUISITION | R=%d | support=5 | retain >=%d native words',value,10-value)
                targets,selection=acquire(bundle,data,changed['acquisition'],plan['selected_support'])
            validate_targets(bundle['arrays']['native'],targets,value)
            row,_=run_case(folder,identity,'budget_'+str(value),plan['selected_support'],plan['selected_margin'],
                           'range',seed,bundle,data,plan,changed,baseline,logger,targets=targets,selection=selection)
            row['maximum_replacements']=value
            rows.append(row)
        return rows
    finally:
        logger.removeHandler(handler);handler.close()


def projection_panel(module, folder):
    """Change-sensitive controls; do not repeat fixed descriptor metrics."""
    source=module/'data/T3_matched_ablations.csv'
    if sha(source)!='3bf33a675c00f31d2289ad3700860de831927789848a318f6952b86af39c8da9':
        raise ValueError('Reviewed ablation source changed.')
    with source.open(encoding='utf-8-sig',newline='') as stream:
        ablations=list(csv.DictReader(stream))
    recall=ablation_readout_recall(module);movement={}
    for entry in read(module/'data/ablation_inputs.json'):
        from .core import safe
        path=safe(module,entry['packed_path'])
        if sha(path)!=entry['sha256']:
            raise ValueError('Projection mechanism input changed.')
        for row in read(path)['mechanism']:
            movement.setdefault(row['role'],[]).append(100*row['relative_l1_affinity_movement'])
    roles={'Proposed (prediction-aware range secondary)':'proposed','Primary minimum-TV only':'primary_control',
           'Index secondary':'index_control','L2 projection':'l2_control','Same-core display only':'display_only'}
    rows=[]
    for row in ablations:
        if row['panel']!='B: projection':continue
        values=movement[roles[row['variant']]]
        if len(values)!=5:raise ValueError('Incomplete five-seed projection mechanism source.')
        rows.append({'variant':row['variant'],'actual_target_inclusion_percent':recall[row['variant']],
                     'affinity_L1_percent':f'{np.mean(values):.5f} ± {np.std(values,ddof=1):.5f}',
                     'frozen_test_nll':row['nll']})
    headers=['Projection control','Actual target inclusion (%) ↑','Affinity L1 movement (%) ↓','Frozen-test NLL ↓']
    keys=['variant','actual_target_inclusion_percent','affinity_L1_percent','frozen_test_nll']
    lines=['Panel B: realizing fixed targets in the decoder','',
           'Five matched seeds, mean ± sample SD. Descriptor words are fixed across controls; their fitting-reference C_v, NPMI and TD are therefore omitted from this panel. For display-only, these are displayed descriptors, not the native decoder top-word sets. Actual target inclusion is measured in each decoder affinity row; it is a constraint metric, not requirement-detection accuracy.','',
           '| '+' | '.join(headers)+' |','| --- | ---: | ---: | ---: |']
    lines += ['| '+' | '.join(r[k] for k in keys)+' |' for r in rows]
    lines += ['', 'Projection increases target inclusion from 81% to 100%. Proposed has the lowest mean NLL among these controls, but the Proposed-minus-native paired 95% interval [-0.002268, 0.001263] includes zero: predictive superiority is not established.',
              'These descriptors share C_v=0.5935 ± 0.0551, NPMI=0.0619 ± 0.0308 and TD=0.9940 ± 0.0089. Component effects belong in Panel A. Recommended main rows: Proposed, minimum-TV only, L2 and display-only. The index secondary can remain in the supplement. L2 provides a distinct movement-geometry control; never remove display-only.']
    (folder/'projection_controls_revised.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    csv_write(folder/'projection_controls_revised.csv',rows)
    latex_table(folder/'projection_controls_revised.tex',headers,[[r[k] for k in keys] for r in rows],
                'Matched projection controls on the primary corpus. Fixed descriptor metrics are reported once; actual decoder target inclusion, movement and frozen-test loss isolate projection effects.')


def reports(output, rows, aggregate, support, module):
    folder=output/'tables';folder.mkdir(exist_ok=True)
    csv_write(folder/'replacement_budget_individual.csv',rows)
    csv_write(folder/'replacement_budget_mean_sd.csv',aggregate)
    headers=['Maximum replacements R','Certified','Fit C_v ↑','Fit NPMI ↑','Δ development NLL vs native ↓',
             'Affinity L1 movement (%) ↓','Actual replacements per topic']
    values=[]
    for r in aggregate:
        replace='NA' if r['replacements_mean'] is None else f"{r['replacements_mean']/10:.2f} ± {r['replacements_sd']/10:.2f}"
        values.append([str(r['maximum_replacements'])+(' (selected)' if r['maximum_replacements']==SELECTED else ''),
                       f"{r['certified_seeds']}/5",fmt(r,'cv',4),fmt(r,'npmi',4),fmt(r,'delta_native_nll',6),fmt(r,'relative_l1_percent',5),replace])
    lines=['Primary replacement-budget sensitivity','',
           'Five matched saved models. Pair support=5; scaled rank margin=1e-6; range secondary; moment off; rho=0. All other acquisition thresholds, search budgets and fitting contexts remain fixed. R bounds replacements; actual replacements may be fewer. Retaining at least 10-R native words expresses the same constraint. This is a post-test-exposure descriptive fitting/development diagnostic, not retrospective final-model selection.','',
           '| '+' | '.join(headers)+' |','| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    lines += ['| '+' | '.join(row)+' |' for row in values]
    finite=[r for r in aggregate if r['development_nll_mean'] is not None]
    if len(finite)==len(VALUES):
        best=min(finite,key=lambda r:r['development_nll_mean'])
        lines += ['',f"Among these three conditions, R={best['maximum_replacements']} has the lowest mean development NLL. This is a conditional point-mean comparison, not proof of statistical or global optimality."]
    lines += ['', 'The selected R=4 is marked because it belongs to the frozen recipe. The marker does not assert a unique or universal optimum. Coherence uses the fitting reference; uncertainty bars are sample SD across five saved training seeds, not confidence intervals. All failed/NA slots are retained.']
    (folder/'replacement_budget_sensitivity.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    latex_table(folder/'replacement_budget_sensitivity.tex',headers,values,
                'Replacement-budget sensitivity on the primary corpus. Mean and sample SD over five matched seeds; fixed support and rank margin. Post-exposure descriptive diagnostic.')
    projection_panel(module,folder)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'pdf.fonttype':42,
                         'ps.fonttype':42,'svg.hashsalt':'PRIMARY_BUDGET_DIAGNOSTIC_V1'})
    fdir=output/'figures';fdir.mkdir(exist_ok=True)
    def export(fig,name):
        for ext in ['pdf','svg','png']:
            metadata={'Creator':'Primary hyperparameter sensitivity','CreationDate':None,'ModDate':None} if ext=='pdf' else ({'Date':None} if ext=='svg' else None)
            fig.savefig(fdir/(name+'.'+ext),dpi=300,bbox_inches='tight',metadata=metadata)
        plt.close(fig)
    def panel(ax,entries,key,parameter,selected,title):
        entries=sorted(entries,key=lambda r:r[parameter]);x=[r[parameter] for r in entries]
        y=[np.nan if r[key+'_mean'] is None else r[key+'_mean'] for r in entries]
        sd=[np.nan if r[key+'_sd'] is None else r[key+'_sd'] for r in entries]
        ax.errorbar(x,y,yerr=sd,fmt='o-',capsize=3,color='#2166ac',label='Mean ± seed SD')
        ax.axvline(selected,color='#b2182b',lw=1,ls='--',label='Frozen selected value')
        if key=='delta_native_nll':ax.axhline(0,color='gray',lw=.8);ax.set_ylabel('Δ NLL (nats/token)')
        ax.set(title=title,xticks=x,xlabel='Pair support q' if parameter=='support' else 'Maximum replacements R')
        ax.grid(alpha=.2)
    fig,axes=plt.subplots(2,2,figsize=(8,5.8),layout='constrained')
    for ax,key,title in zip(axes.ravel(),['cv','npmi','delta_native_nll','relative_l1_percent'],
                            ['(a) Fitting $C_v$ ↑','(b) Fitting NPMI ↑','(c) Development loss change ↓','(d) Affinity movement (%) ↓']):
        panel(ax,aggregate,key,'maximum_replacements',SELECTED,title)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Primary corpus: replacement-budget sensitivity | R = 4 selected',fontsize=11)
    export(fig,'R3_primary_replacement_budget_sensitivity')
    fig,axes=plt.subplots(2,3,figsize=(10,5.7),layout='constrained')
    for row,entries,parameter,selected in [(0,support,'support',5),(1,aggregate,'maximum_replacements',4)]:
        for col,key,title in [(0,'cv','Fitting $C_v$ ↑'),(1,'npmi','Fitting NPMI ↑'),(2,'delta_native_nll','Development loss change ↓')]:
            panel(axes[row,col],entries,key,parameter,selected,f'({chr(97+row*3+col)}) '+title)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Primary corpus: sensitivity to two method-defining hyperparameters',fontsize=11)
    export(fig,'R_primary_core_hyperparameter_sensitivity')


def extra_checks(cfg,plan,data,bundles):
    class Checks(unittest.TestCase):
        def test_declared_parameter_values(self):self.assertEqual(VALUES,[2,4,6]);self.assertIn(SELECTED,VALUES)
        def test_change_is_the_single_replacement_constraint(self):
            for r in VALUES:
                changed=budget_config(cfg,r);self.assertEqual(changed['acquisition']['retained_native_words_minimum'],10-r)
                changed['acquisition']['maximum_replacements']=4;changed['acquisition']['retained_native_words_minimum']=6
                self.assertEqual(changed,cfg)
        def test_invalid_budget_rejected(self):
            for r in [0,3,10,True]:
                with self.assertRaises(ValueError):budget_config(cfg,r)
        def test_changed_budget_cannot_reuse_selected_readout(self):
            for r in VALUES:self.assertEqual(is_original_setting(5,1e-6,plan,budget_config(cfg,r)),r==4)
        def test_selected_target_bounds_all_seeds(self):
            for bundle in bundles.values():validate_targets(bundle['arrays']['native'],bundle['selected']['targets'],4)
        def test_budget_bound_rejection(self):
            native=np.tile(np.arange(100,0,-1),(10,1));targets=[list(range(3,13)) for _ in range(10)]
            with self.assertRaises(ValueError):validate_targets(native,targets,2)
        def test_original_input_config_not_mutated(self):
            before=copy.deepcopy(cfg);budget_config(cfg,6);self.assertEqual(cfg,before)
        def test_budget_acquisition_bounds(self):
            bundle=bundles[809];changed=budget_config(cfg,2)
            targets,selection=acquire(bundle,data,changed['acquisition'],5)
            validate_targets(bundle['arrays']['native'],targets,2)
            self.assertTrue(selection['only_fitting_groups_used']);self.assertFalse(selection['development_inputs_used'])
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    return {'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'passed':r.wasSuccessful(),'details':stream.getvalue()}


def cli(args,root):
    root=Path(root).resolve()
    # Recheck every existing support/margin cell before building a combined main figure.
    if not args.self_check:
        if original_cli(args,root)!=0:return 1
    run=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]
    output=root/'outputs/steps/step08_sensitivity_budget'/run;output.mkdir(parents=True,exist_ok=True)
    logdir=root/'logs';logdir.mkdir(exist_ok=True)
    logger=logging.getLogger('budget_'+run);logger.setLevel(logging.INFO)
    formatter=logging.Formatter('%(asctime)sZ | %(levelname)s | %(message)s',datefmt='%Y-%m-%dT%H:%M:%S');formatter.converter=time.gmtime
    for h in [logging.StreamHandler(sys.stdout),logging.FileHandler(logdir/(run+'.log'),encoding='utf-8'),logging.FileHandler(logdir/'experiment_log.txt',encoding='utf-8')]:
        h.setFormatter(formatter);logger.addHandler(h)
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(30):logger.info('RUNNING | replacement budgets 2/4/6 | original CPU LP budgets | no neural fits')
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    started=time.perf_counter();result={};exit_code=1;cache=None
    try:
        logger.info('START | replacement-budget sensitivity | device=%s | CPU saved-array compute',args.device)
        module,plan,versions,paths,audit,cfg,data,bundles=setup(root)
        result={'run_id':run,'new_neural_fits':0,'new_encoder_calls':0,'test_scoring_calls':0,'final_recipe_changed':False,
                'post_test_exposure':True,'implementation_sha256':{p.relative_to(root).as_posix():sha(p) for p in module.rglob('*.py')}}
        if args.self_check:
            from .tests import run_checks
            original=run_checks(root,plan,data,bundles,cfg);extra=extra_checks(cfg,plan,data,bundles)
            checks={'tests':original['tests']+extra['tests'],'failures':original['failures']+extra['failures'],
                    'errors':original['errors']+extra['errors'],'passed':original['passed'] and extra['passed'],
                    'original':original,'budget':extra}
            write(output/'self_check.json',checks);result.update(status='SELF_CHECK_PASSED' if checks['passed'] else 'SELF_CHECK_FAILED',checks=checks)
            logger.info('SELF CHECK | tests=%d failures=%d errors=%d',checks['tests'],checks['failures'],checks['errors'])
            exit_code=0 if checks['passed'] else 1
        else:
            write(output/'input_audit.json',audit)
            replay={s:replay_selected(bundles[s],data,plan) for s in plan['seeds']};write(output/'selected_source_replay.json',replay)
            declared={'protocol':'PRIMARY_POST_EXPOSURE_BUDGET_V1','source_plan':plan,'versions':versions,'fixed_cfg':cfg,
                      'replacement_values':VALUES,'selected_replacements':SELECTED,'maximum_report_slots':15,
                      'changed_constraint':'replace at most R native words; equivalently retain at least 10-R',
                      'fixed_pair_support':5,'fixed_scaled_margin':1e-6,'fixed_objective':'range',
                      'post_test_exposure':True,'final_recipe_changed':False}
            identity=hashlib.sha256(json.dumps(declared,sort_keys=True).encode()).hexdigest()
            write(output/'declared_before_budget_computation.json',declared)
            cache=root/'data/cache/primary_sensitivity_budget'/identity[:20];cache.mkdir(parents=True,exist_ok=True)
            baselines={s:predict(bundles[s],bundles[s]['arrays']['native'],data,0.)[0]['macro_source_nll'] for s in plan['seeds']}
            rows=[]
            with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as executor:
                jobs=[executor.submit(seed_job,(cache,identity,s,bundles[s],data,plan,cfg,baselines[s])) for s in plan['seeds']]
                for future in as_completed(jobs):
                    rows.extend(future.result());write(output/'individual_results.json',rows)
                    logger.info('SEED COMPLETE | budget slots=%d/15',len(rows))
            rows.sort(key=lambda r:(r['maximum_replacements'],r['seed']))
            if len(rows)!=15 or len({(r['maximum_replacements'],r['seed']) for r in rows})!=15:
                raise ValueError('The declared budget grid is incomplete or duplicated.')
            aggregate=budget_means(rows);write(output/'mean_sd_results.json',aggregate)
            for r in aggregate:
                logger.info('MEAN | R=%d | certified=%d/5 | Cv=%s NPMI=%s | delta dev NLL=%s movement%%=%s',
                            r['maximum_replacements'],r['certified_seeds'],r['cv_mean'],r['npmi_mean'],r['delta_native_nll_mean'],r['relative_l1_percent_mean'])
            previous=root/(root/'outputs/steps/step08_sensitivity/LATEST.txt').read_text(encoding='utf-8').strip()
            old=read(previous/'mean_sd_results.json');support=[r for r in old if r['study']=='support']
            if len(support)!=3:raise ValueError('Combined figure requires all three verified support conditions.')
            reports(output,rows,aggregate,support,module)
            from reproduce_s5_figures import verify, plot
            paper_records,paper_check=verify(root/'data/supplementary_s5')
            expected=paper_records['replacement_budget_mean_sd.csv']
            for actual, archived in zip(sorted(aggregate,key=lambda x:x['maximum_replacements']), expected):
                for metric in ['cv','npmi','td','delta_native_nll','relative_l1_percent','fitting_logit_range']:
                    for suffix in ['_mean','_sd']:
                        if abs(actual[metric+suffix]-float(archived[metric+suffix]))>1e-10:
                            raise ValueError('Paper CSV and independently replayed budget differ: '+metric+suffix)
            old_expected=paper_records['sensitivity_mean_sd.csv']
            for actual in old:
                archived=[x for x in old_expected if x['study']==actual['study'] and int(x['support'])==actual['support'] and float(x['margin'])==actual['margin'] and x['objective']==actual['objective']]
                if len(archived)!=1:raise ValueError('Missing paper support/margin setting')
                for metric in ['cv','npmi','td','delta_native_nll','relative_l1_percent','fitting_logit_range']:
                    for suffix in ['_mean','_sd']:
                        if abs(actual[metric+suffix]-float(archived[0][metric+suffix]))>1e-10:
                            raise ValueError('Paper CSV and independently replayed support/margin differ')
            paper_check['independent_array_replay_passed']=True
            paper_check['output_files']=plot(paper_records,output/'figures',['pdf','png'])
            write(output/'figures/S5_reproduction_report.json',paper_check)
            logger.info('PAPER FIGURES | SF1_parameter_sensitivity.pdf; SF2_rank_margin_tradeoff.pdf | publication style | saved values unchanged')

            text=['Primary method-parameter sensitivity complete','',
                  'All 65 original support/rank-margin report slots were independently replayed before this 15-slot replacement-budget study.',
                  'The new main figure varies q={3,5,10} and R={2,4,6} one at a time. Red lines show the frozen q=5 and R=4, not newly optimized values. Rank-margin stability remains in the previous R2 figure.',
                  'All data and figures are primary-corpus results. Coherence is fitting-reference coherence; loss is actual development completion NLL. This is post-test-exposure descriptive robustness analysis. No frozen-test rescoring, training, encoder calls or final-model changes.',
                  'Open tables/replacement_budget_sensitivity.md, tables/projection_controls_revised.md and figures/R_primary_core_hyperparameter_sensitivity.pdf.',
                  'Projection controls omit repeated descriptor metrics and report actual decoder target inclusion, affinity movement and frozen-test NLL instead. Full unchanged descriptor results remain in the earlier complete ablation export.',
                  f'Original sweep outputs: {previous.relative_to(root).as_posix()}',f'Budget cache: {cache.relative_to(root).as_posix()}']
            finite=[r for r in aggregate if r['development_nll_mean'] is not None]
            if len(finite)==len(VALUES):
                best=min(finite,key=lambda r:r['development_nll_mean'])
                text += ['',f"Lowest mean development NLL among the three tested budgets: R={best['maximum_replacements']}. This is a descriptive comparison, not a new final-model selection."]
            (output/'SUMMARY.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
            success=sum(r['certified'] for r in rows)
            result.update(status='BUDGET_SENSITIVITY_COMPLETE',completed_slots=15,certified_slots=success,
                          all_budget_readouts_certified=success==15,independently_replayed_cache_slots=sum(r.get('replayed_from_cache',False) for r in rows),
                          original_sweep_output=previous.relative_to(root).as_posix(),cache_relative_path=cache.relative_to(root).as_posix())
            (output.parent/'LATEST.txt').write_text(output.relative_to(root).as_posix()+'\n',encoding='utf-8')
            logger.info('COMPLETE | budget slots=15 | certified=%d/15 | final recipe unchanged',success)
            logger.info('READ FIRST | %s',output/'SUMMARY.md');exit_code=0
    except KeyboardInterrupt:
        logger.warning('INTERRUPTED | completed cache retained; rerun same command to resume');result.update(status='INTERRUPTED');exit_code=130
    except Exception as exc:
        logger.exception('FAILED | %s',exc);result.update(status='FAILED',failure=str(exc),traceback=traceback.format_exc())
    finally:
        stop.set();thread.join(timeout=1);result['wall_seconds']=time.perf_counter()-started
        write(output/'run_result.json',result)
        files=[p for p in output.rglob('*') if p.is_file()]+[logdir/(run+'.log')]
        if cache is not None and result.get('status')=='BUDGET_SENSITIVITY_COMPLETE':files += [p for p in cache.rglob('*') if p.is_file()]
        inspection=root/'outputs/inspection'/('budget_sensitivity_step08_'+run+'_inspection.zip');inspection.parent.mkdir(parents=True,exist_ok=True)
        make_zip(inspection,files,root);logger.info('INSPECTION ZIP | %s',inspection)
        for h in list(logger.handlers):h.close();logger.removeHandler(h)
    return exit_code
