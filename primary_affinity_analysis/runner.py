"""Read-only export from frozen primary data, with explicit source provenance."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import logging
import sys
import time
import traceback
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import numpy as np
from . import metrics

PACKAGE = Path(__file__).resolve().parent
SEEDS = [809,810,811,812,813]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def safe(root, relative):
    rel=Path(relative)
    if not isinstance(relative,str) or '\\' in relative or rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe saved artifact path.')
    path=(Path(root)/rel).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Saved artifact leaves the project.')
    return path


class Inputs:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.reference=read_json(PACKAGE/'data/reference_inputs.json')
        self.expected={r['path']:r for r in self.reference['files']}
        self.used={}

    def path(self,relative):
        if relative not in self.expected:
            raise ValueError('Unregistered input: '+relative)
        path=safe(self.root,relative);record=self.expected[relative]
        if not path.is_file():
            raise RuntimeError('Required frozen input missing: '+relative+'. Keep the completed Step 6 and compact Step 8 outputs; do not refit the model.')
        if relative not in self.used:
            if path.stat().st_size != record['bytes'] or file_hash(path) != record['sha256']:
                raise ValueError('Frozen input bytes differ from the inspected run: '+relative)
            self.used[relative]=record
        return path

    def json(self,relative):
        return read_json(self.path(relative))

    def csv(self,relative):
        with self.path(relative).open(encoding='utf-8-sig',newline='') as stream:
            return list(csv.DictReader(stream))

    def jsonl(self,relative):
        rows=[json.loads(line) for line in self.path(relative).read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(rows) != len({r['group_id'] for r in rows}):
            raise ValueError('Duplicate evidence-group rows.')
        return {r['group_id']:r for r in rows}

    def arrays(self,relative):
        with np.load(self.path(relative),allow_pickle=False) as z:
            return {name:z[name] for name in z.files}

    def unchanged(self):
        for relative,record in self.used.items():
            path=safe(self.root,relative)
            if not path.is_file() or file_hash(path) != record['sha256']:
                raise ValueError('A read-only input changed during export: '+relative)


def close(actual,expected,label,tolerance=1e-10):
    if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual-expected)>tolerance:
        raise ValueError('Independent replay mismatch: '+label)
    return abs(actual-expected)


def check_masks(group_ids,masks,features,vocabulary):
    index={word:i for i,word in enumerate(vocabulary)}
    if len(group_ids)!=410 or len(set(group_ids))!=410:
        raise ValueError('The locked primary completion population changed.')
    for gid in group_ids:
        m,f=masks[gid],features[gid]
        if f['split']!='test' or not m['eligible'] or not m['observed_ids'] or not m['target_ids']:
            raise ValueError('Invalid frozen observed/target document.')
        obs,tar=m['observed_positions'],m['target_positions']
        if set(obs)&set(tar) or len(set(obs))!=len(obs) or len(set(tar))!=len(tar):
            raise ValueError('Frozen observed and target positions overlap or repeat.')
        expected={i for i,w in enumerate(f['tokens']) if w in index}
        if set(obs+tar)!=expected:
            raise ValueError('Completion mask lost vocabulary occurrences.')
        for role in ['observed','target']:
            if [index[f['tokens'][i]] for i in m[role+'_positions']]!=m[role+'_ids']:
                raise ValueError('Vocabulary order or completion IDs changed.')


def replay(root,log):
    inputs=Inputs(root);ref=inputs.reference
    compact,primary,prep=ref['compact_prefix'],ref['primary_prefix'],ref['preparation_prefix']
    status=inputs.json(primary+'/run_result.json')
    if status['status']!='STEP06_FROZEN_PRIMARY_TEST_COMPLETE' or not status['engineering_passed']:
        raise ValueError('Require the completed frozen primary test.')
    state=inputs.json('state/step01_accepted.json')
    if state['processed_relative_path']!=prep:
        raise ValueError('Primary preparation reference changed.')
    exposure=inputs.json('state/step06_test_exposure.json')
    if exposure['freeze_id']!=ref['primary_freeze_id']:
        raise ValueError('Primary test exposure changed.')
    vocabulary=inputs.json(prep+'/vocabulary.json')['tokens']
    if len(vocabulary)!=1472 or len(set(vocabulary))!=1472:
        raise ValueError('Unexpected primary vocabulary.')
    group_ids=inputs.json(primary+'/test_population.json')['group_ids']
    masks=inputs.jsonl(prep+'/completion_masks.jsonl')
    features=inputs.jsonl(prep+'/feature_records.jsonl')
    check_masks(group_ids,masks,features,vocabulary)
    target_payload=inputs.json(compact+'/numerical_evidence/primary_fitting_targets.json')
    targets={int(r['seed']):r for r in target_payload['rows']}
    if not target_payload['selected_before_test'] or set(targets)!=set(SEEDS):
        raise ValueError('Require all independent, pre-test target definitions.')
    old_seed_rows=[r for r in inputs.csv(compact+'/numerical_evidence/divergence_per_seed.csv') if r['dataset']=='Primary']
    if len(old_seed_rows)!=5 or {int(r['seed']) for r in old_seed_rows}!=set(SEEDS):
        raise ValueError('The recorded primary seed population is incomplete.')
    old_by_seed={int(r['seed']):r for r in old_seed_rows}
    old_documents=[r for r in inputs.csv(compact+'/numerical_evidence/divergence_per_document.csv') if r['dataset']=='Primary']
    old_doc={(int(r['seed']),r['group_id']):r for r in old_documents}
    if len(old_documents)!=2050 or len(old_doc)!=2050:
        raise ValueError('The primary document-divergence cache is incomplete or duplicated.')
    rows=[];documents=[];topics=[];cases={};example=None;max_error=0.
    for seed in SEEDS:
        path=primary+'/proposed_control_cases/seed'+str(seed)
        case=inputs.json(path+'/case_result.json')
        if case['test_used_for_fit_or_parameter_selection'] or case['neural_training_performed']:
            raise ValueError('The frozen readout comparison changed scope.')
        z=inputs.arrays(path+'/frozen_test_arrays.npz')
        a,b=z['native'],z['proposed']
        if a.shape!=b.shape or a.shape!=(10,1472):
            raise ValueError('Matched readout dimensions changed.')
        np.testing.assert_array_equal(a,z['readout_native'])
        np.testing.assert_array_equal(b,z['readout_proposed'])
        cached=inputs.arrays(compact+f'/numerical_evidence/primary/seed{seed}/affinity_before_after.npz')
        np.testing.assert_array_equal(a,cached['native']);np.testing.assert_array_equal(b,cached['proposed'])
        np.testing.assert_array_equal(b-a,cached['delta'])
        if inputs.json(compact+f'/numerical_evidence/primary/seed{seed}/vocabulary.json')['tokens']!=vocabulary:
            raise ValueError('Affinity vocabulary order differs from the primary vocabulary.')
        target=targets[seed]['targets']
        if hashlib.sha256(b.astype(np.float64).tobytes()).hexdigest()!=targets[seed]['proposed_float64_bytes_sha256']:
            raise ValueError('The refined readout differs from its pre-test reference.')
        np.testing.assert_array_equal(np.asarray(target),cached['targets'])
        nr,pr=metrics.target_inclusion(a,target),metrics.target_inclusion(b,target)
        if not np.all(pr==1):
            raise ValueError('Proposed does not realize every independently acquired target.')
        affinity=metrics.divergence(a,b)
        profiles_a=np.exp(metrics.log_softmax(a*z['decoder_gain']+z['decoder_offset']))
        profiles_b=np.exp(metrics.log_softmax(b*z['decoder_gain']+z['decoder_offset']))
        np.testing.assert_allclose(profiles_a,cached['unit_topic_decoder_native'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(profiles_b,cached['unit_topic_decoder_proposed'],atol=1e-12,rtol=0)
        prototype=metrics.divergence(profiles_a,profiles_b)
        roles={r['role']:r for r in case['rows']}
        cases[seed]={};predictions={};scores={}
        for role,readout in [('native',a),('proposed',b)]:
            r=roles[role]
            if r['moment_correction'] or r['test_used_for_training_or_selection'] or r['model_configuration_selected_on']!='development_only_before_test':
                raise ValueError('Prediction correction/selection contract changed.')
            if r['actual_readout']!=role or r['rho']!=0:
                raise ValueError('Expected the locked primary uncorrected zero-backoff path.')
            actual_words=[[vocabulary[i] for i in t] for t in metrics.top_words(readout)]
            if actual_words!=r['topics']:
                raise ValueError('Displayed descriptors differ from the actual affinity readout.')
            logp=metrics.predict(z['theta_test'],readout,z['decoder_gain'],z['decoder_offset'],z['prior'],float(r['rho']))
            predictions[role]=np.exp(logp)
            loss=metrics.losses(logp,group_ids,masks,features)
            scores[role]=metrics.aggregate(loss);cases[seed][role]=loss
            saved={d['group_id']:d for d in r['document_losses']}
            if len(saved)!=len(loss) or len(saved)!=len(r['document_losses']) or set(saved)!=set(group_ids):
                raise ValueError('Saved target-loss population differs.')
            for d in loss:
                old=saved[d['group_id']]
                if old['source']!=d['source'] or old['target_tokens']!=d['target_tokens']:
                    raise ValueError('Saved source or target count differs.')
                max_error=max(max_error,close(d['negative_log_probability'],old['negative_log_probability'],'document loss',1e-9))
            for field in ['macro_source_nll','micro_nll']:
                close(scores[role][field],r['prediction'][field],'completion '+field)
        final=metrics.divergence(predictions['native'],predictions['proposed'])
        scale=float(a.max())
        row_error=float(np.max(np.abs(b.sum(axis=1)-a.sum(axis=1)))/scale)
        col_error=float(np.max(np.abs(b.sum(axis=0)-a.sum(axis=0)))/scale)
        if (a<0).any() or (b<0).any() or max(row_error,col_error)>1e-6:
            raise ValueError('Nonnegativity or conservation failed.')
        result={'dataset':'Primary','seed':seed,'affinity_js_nats':float(affinity['js'].mean()),
            'affinity_tv':float(affinity['tv'].mean()),'prototype_js_nats':float(prototype['js'].mean()),
            'prototype_tv':float(prototype['tv'].mean()),'final_document_js_nats':float(final['js'].mean()),
            'final_document_tv':float(final['tv'].mean()),'relative_l1_percent':float(100*np.abs(b-a).sum()/a.sum()),
            'native_target_recall':float(nr.mean()),'proposed_target_recall':float(pr.mean()),
            'native_test_nll':scores['native']['macro_source_nll'],'proposed_test_nll':scores['proposed']['macro_source_nll'],
            'proposed_minus_native_nll':scores['proposed']['macro_source_nll']-scores['native']['macro_source_nll'],
            'max_row_error_native_max_scaled':row_error,'max_column_error_native_max_scaled':col_error,
            'conservation_and_nonnegativity_pass':True}
        for field in ['affinity_js_nats','affinity_tv','prototype_js_nats','prototype_tv','final_document_js_nats','final_document_tv','relative_l1_percent','native_target_recall','proposed_target_recall']:
            close(result[field],float(old_by_seed[seed][field]),field,1e-12)
        for i,gid in enumerate(group_ids):
            old=old_doc[(seed,gid)]
            close(float(final['js'][i]),float(old['final_js_nats']),'document divergence',1e-12)
            close(float(final['tv'][i]),float(old['final_tv']),'document TV',1e-12)
            documents.append({'dataset':'Primary','seed':seed,'group_id':gid,'js_nats':float(final['js'][i]),'tv':float(final['tv'][i])})
        for k in range(10):
            topics.append({'dataset':'Primary','seed':seed,'topic_index':k,'affinity_js_nats':float(affinity['js'][k]),
                'decoder_profile_js_nats':float(prototype['js'][k]),'native_target_inclusion':float(nr[k]),'proposed_target_inclusion':float(pr[k])})
        rows.append(result)
        if seed==809:
            example=(a.copy(),b.copy(),vocabulary)
        log.info('VERIFIED | Primary seed=%d | target inclusion %.0f%% -> %.0f%% | JS affinity=%.6g | JS prediction=%.6g',seed,100*nr.mean(),100*pr.mean(),result['affinity_js_nats'],result['final_document_js_nats'])
    benchmark=inputs.csv(compact+'/main/tables/T2_primary_competitors.csv')
    for label,field in [('NeuroMax','native_test_nll'),('Proposed','proposed_test_nll')]:
        matches=[r for r in benchmark if r['method']==label and r['profile']=='random_words']
        if len(matches)!=1 or int(matches[0]['test_nll_available_seeds'])!=5:
            raise ValueError('The matched primary benchmark summary is incomplete.')
        values=np.array([r[field] for r in rows])
        close(float(values.mean()),float(matches[0]['test_nll_mean']),'mean NLL')
        close(float(values.std(ddof=1)),float(matches[0]['test_nll_sd']),'NLL sample SD')
    intervals=inputs.json(primary+'/paired_prediction_intervals.json')
    matched=[r for r in intervals if r['method']=='NeuroMax' and r['profile']=='random_words']
    if len(matched)!=1:
        raise ValueError('Missing frozen parent paired interval.')
    interval=matched[0]
    low,high=metrics.paired_interval(cases,interval,SEEDS)
    close(low,interval['percentile_95_low'],'paired interval lower')
    close(high,interval['percentile_95_high'],'paired interval upper')
    interval_pro_minus_native=[-high,-low]
    old_lookup=inputs.csv(compact+'/main/figures/F1_primary_word_lookup.csv')
    np.testing.assert_array_equal(metrics.changed_columns(example[0],example[1]),[int(r['word_id']) for r in old_lookup])
    examples=inputs.csv(compact+'/main/tables/T5_transport_text_examples.csv')
    inputs.unchanged()
    return {'inputs':inputs,'rows':rows,'documents':documents,'topics':topics,'example':example,
            'interval':interval_pro_minus_native,'max_loss_error':max_error,'examples':examples}


def write_csv(path,rows):
    if not rows:raise ValueError('No rows to export.')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def mean_sd(rows,field,factor=1):
    values=np.array([r[field] for r in rows])*factor
    return float(values.mean()),float(values.std(ddof=1))


def pm(rows,field,factor=1,scientific=False):
    mean,sd=mean_sd(rows,field,factor)
    fmt='.3e' if scientific else ('.2f' if factor==100 else '.6f')
    return format(mean,fmt)+' ± '+format(sd,fmt)


def export(folder,result,log):
    from .figure import draw
    rows=result['rows'];interval=result['interval']
    a,b,vocabulary=result['example']
    lookup=draw(folder/'main/figures',a,b,vocabulary,read_json(PACKAGE/'data/labels_en.json'))
    write_csv(folder/'supplementary/F1_English_Chinese_word_mapping.csv',lookup)
    write_csv(folder/'supplementary/divergence_per_seed.csv',rows)
    write_csv(folder/'supplementary/divergence_per_topic.csv',result['topics'])
    write_csv(folder/'supplementary/divergence_per_document.csv',result['documents'])
    lo,hi=interval
    panel_a=[{'measure':'Selected-target top-10 inclusion (%)','native':pm(rows,'native_target_recall',100),
        'proposed':pm(rows,'proposed_target_recall',100),'proposed_minus_native':f"+{mean_sd(rows,'proposed_target_recall',100)[0]-mean_sd(rows,'native_target_recall',100)[0]:.2f} percentage points"},
        {'measure':'Frozen test macro-source NLL (nats/token)','native':pm(rows,'native_test_nll'),
         'proposed':pm(rows,'proposed_test_nll'),'proposed_minus_native':f"{mean_sd(rows,'proposed_minus_native_nll')[0]:+.6f}; paired 95% CI [{lo:+.6f}, {hi:+.6f}]"}]
    panel_b=[{'measure':'Affinity-share JS divergence (nats)','native_to_proposed':pm(rows,'affinity_js_nats',scientific=True),'meaning':'Change in row-normalized affinity; not decoder word probabilities.'},
        {'measure':'Unit-topic decoder-profile JS divergence (nats)','native_to_proposed':pm(rows,'prototype_js_nats',scientific=True),'meaning':'Change in decoder word distributions with unit-topic input.'},
        {'measure':'Final document-prediction JS divergence (nats)','native_to_proposed':pm(rows,'final_document_js_nats',scientific=True),'meaning':'Change in predictions for all 410 eligible frozen test groups.'},
        {'measure':'Relative total affinity L1 movement (%)','native_to_proposed':pm(rows,'relative_l1_percent'),'meaning':'Total absolute adjustment divided by native total affinity, times 100.'},
        {'measure':'Nonnegativity and row/column conservation','native_to_proposed':'5/5 seeds passed','meaning':'Numerical invariants; detailed residuals retained in supplementary data.'}]
    write_csv(folder/'main/tables/T6A_primary_before_after.csv',panel_a)
    write_csv(folder/'main/tables/T6B_primary_divergence.csv',panel_b)
    text=['# Primary-only native-versus-Proposed analysis','',
        'Matched native NeuroMax versus its Proposed refinement, seeds 809–813. Entries are mean ± sample SD of five seed-level results. Affinity/decoder statistics first average the ten matched topics per seed; document divergence first averages all 410 eligible independent test groups per seed. NLL is the frozen macro-source completion loss, not that unweighted document-divergence average.','',
        '**Panel A: actual constraint realization and prediction**','',
        '| Measure | Native NeuroMax | Proposed | Proposed minus native |','|---|---:|---:|---|']
    text += [f"| {r['measure']} | {r['native']} | {r['proposed']} | {r['proposed_minus_native']} |" for r in panel_a]
    text += ['','**Panel B: measured adjustment of the same model**','','| Measure | Native-to-Proposed change | What is measured |','|---|---:|---|']
    text += [f"| {r['measure']} | {r['native_to_proposed']} | {r['meaning']} |" for r in panel_b]
    text += ['','JS uses natural logarithms and denotes divergence, not square-root distance. Low divergence measures proximity, not superiority. Target inclusion is a constraint-realization statistic, not requirement-detection accuracy. Its Proposed SD is zero because every seed realizes every selected target. The NLL interval crosses zero and establishes neither superiority nor equivalence.',
        '', '**Suggested manuscript statement.** Proposed realizes all selected top-ten target words, increasing mean target inclusion from 81% to 100%, with a mean relative affinity movement of 0.0346%. The mean final document-prediction JS divergence is 2.879e-4 nats. Nonnegativity and topic-row/word-column totals are preserved in all five seeds. The small NLL point-mean reduction is not a statistically established improvement over the native parent.',
        '', '**Figure caption.** Matched native NeuroMax and Proposed topic-word affinity on the primary Chinese transit corpus, fixed seed 809. Panels (a) and (b) share a raw-affinity colour scale; panel (c) shows Proposed minus native on a separate symmetric scale. All ten topic rows and the 24 vocabulary columns with the largest summed absolute affinity changes are shown, from a total of 1,472 words. Selection uses fitting-derived readouts, with word-ID tie breaking; it is not based on test performance. English glosses are explanatory labels; exact Chinese tokens and IDs are in the supplementary mapping. Affinity values are not decoder probabilities. Five-seed aggregate evidence is reported in the table.',
        '', 'Source/provenance hashes and independently recomputed per-seed, per-topic and per-document values accompany the export. No model fits, encoder calls, optimisation, test-based selection or edits to earlier results occur.']
    table_path=folder/'main/tables/T6_primary_affinity_analysis.md';table_path.parent.mkdir(parents=True,exist_ok=True)
    table_path.write_text('\n'.join(text)+'\n',encoding='utf-8')
    def esc(value):
        return str(value).replace('\\',r'\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('±',r'$\pm$')
    latex=[r'\begin{table*}[!t]',r'\centering',r'\caption{Primary-corpus native-versus-Proposed adjustment over five matched seeds (mean $\pm$ sample SD). JS uses natural logarithms. The paired NLL interval is unadjusted and crosses zero.}',r'\label{tab:primary_refinement_divergence}',r'\begingroup\footnotesize',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{l l l}',r'\hline',r'\multicolumn{3}{l}{\textbf{Panel A: realization and prediction}} \\',r'Measure & Native NeuroMax & Proposed \\',r'\hline']
    latex += [' & '.join([esc(r['measure']),esc(r['native']),esc(r['proposed'])])+r' \\' for r in panel_a]
    latex += [r'\hline',r'\multicolumn{3}{l}{\textbf{Panel B: matched adjustment}} \\',r'Measure & \multicolumn{2}{l}{Native-to-Proposed change} \\',r'\hline']
    latex += [esc(r['measure'])+' & '+r'\multicolumn{2}{l}{'+esc(r['native_to_proposed'])+r'} \\' for r in panel_b]
    latex += [r'\hline',r'\end{tabular}',r'\par\smallskip',f"Proposed minus native NLL: {mean_sd(rows,'proposed_minus_native_nll')[0]:+.6f}; paired 95\\% CI [{lo:+.6f}, {hi:+.6f}].",r'\endgroup',r'\end{table*}']
    (table_path.with_suffix('.tex')).write_text('\n'.join(latex)+'\n',encoding='utf-8')
    # Preserve the six original example choices; translation never reselects posts.
    translations=read_json(PACKAGE/'data/transport_examples_en.json')
    english=[]
    for original in result['examples']:
        gid=original['group_id'];translated=translations[gid]
        if hashlib.sha256(original['text'].encode('utf-8')).hexdigest()!=translated['original_excerpt_sha256']:
            raise ValueError('A transport example changed; its translation no longer matches.')
        english.append({'topic':f"T{int(original['topic'])+1:02d}",'source':original['source'],
            'candidate_theme':translated['candidate_theme'],'english_excerpt':translated['english_excerpt'],
            'evidence_group':gid,'original_chinese_excerpt':original['text']})
    write_csv(folder/'supplementary/transport_examples_English_with_provenance.csv',english)
    application=['# Source-grounded transportation examples (English)','',
        'The same six automatically selected fitting examples are retained. Topic identifiers T01–T10 correspond to original indices 0–9. English theme descriptions are illustrative summaries, not annotated requirement labels; posts include complaints, announcements and news. Main-table English excerpts retain their first 45 words; ellipses mark omitted text. Full translations, original Chinese excerpts and evidence IDs are retained in supplementary data.','',
        '| Topic | Collection source | Illustrative service theme | English excerpt |','|---|---|---|---|']
    for r in english:
        words=r['english_excerpt'].split()
        displayed=' '.join(words[:45])+(' ...' if len(words)>45 else '')
        application.append(f"| {r['topic']} | {r['source']} | {r['candidate_theme']} | {displayed} |")
    (folder/'main/tables/T5_transport_examples_English.md').write_text('\n'.join(application)+'\n',encoding='utf-8')
    log.info('EXPORTED | primary-only figure PDF/SVG/PNG; two-panel divergence table; six unchanged transport examples translated')


def bundle(root,folder,inputs):
    """Include exact read-only inputs, not only screenshots, for reproduction."""
    paths=[Path(root)/'main.py',Path(root)/'README_PRIMARY_AFFINITY_ANALYSIS.md',Path(root)/'requirements-primary-affinity.txt']
    paths += [p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [safe(root,r) for r in inputs.used]
    destination=folder/'primary_affinity_reproducibility.zip'
    with ZipFile(destination,'w',compression=ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(set(paths)):
            z.write(path,path.relative_to(root).as_posix())
    with ZipFile(destination,'r') as check:
        expected={p.relative_to(root).as_posix() for p in paths}
        if len(check.namelist())!=len(expected) or set(check.namelist())!=expected or check.testzip() is not None:
            raise ValueError('The affinity analysis reproduction ZIP failed its contents/CRC check.')
    return destination


def cli(args,root):
    now=datetime.now(timezone.utc)
    run_id=now.strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]
    log_path=root/'logs'/(run_id+'.log');log_path.parent.mkdir(parents=True,exist_ok=True)
    log=logging.getLogger('primary_affinity_analysis.'+run_id);log.setLevel(logging.INFO);log.propagate=False
    formatter=logging.Formatter('%(asctime)sZ | %(levelname)s | %(message)s',datefmt='%Y-%m-%dT%H:%M:%S')
    formatter.converter=time.gmtime
    for handler in [logging.FileHandler(log_path,encoding='utf-8'),logging.StreamHandler(sys.stdout)]:
        handler.setFormatter(formatter);log.addHandler(handler)
    run_folder=root/'outputs/runs'/run_id/'step08_primary_affinity_analysis';run_folder.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();status='FAILED';result=None
    try:
        import matplotlib
        log.info('START | primary-only affinity analysis export | requested_device=%s | saved-array compute=CPU | new fits=0',args.device)
        log.info('ENVIRONMENT | Python=%s | NumPy=%s | Matplotlib=%s',sys.version.split()[0],np.__version__,matplotlib.__version__)
        if args.self_check:
            from . import tests
            stream=io.StringIO()
            test_result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
            (run_folder/'self_check.txt').write_text(stream.getvalue(),encoding='utf-8')
            if not test_result.wasSuccessful():
                raise RuntimeError('Primary export self-check failed; see '+str(run_folder/'self_check.txt'))
            status='PRIMARY_AFFINITY_ANALYSIS_SELF_CHECK_PASSED'
            log.info('SELF CHECK | tests=%d | failures=0 | errors=0',test_result.testsRun)
        else:
            log.info('PHASE 1/3 | verify primary frozen inputs and replay all five matched seeds')
            result=replay(root,log)
            log.info('PHASE 2/3 | primary English figure, two-panel affinity table and unchanged case examples')
            folder=root/'outputs/steps/step08_primary_affinity_analysis'/run_id;folder.mkdir(parents=True,exist_ok=False)
            export(folder,result,log)
            write_json(folder/'source_provenance.json',{'corpus':'Primary','seeds':SEEDS,'inputs':list(result['inputs'].used.values()),
                'code_sha256':{p.relative_to(root).as_posix():file_hash(p) for p in sorted(PACKAGE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts},
                'max_replayed_document_loss_error':result['max_loss_error'],'all_2050_seed_group_divergences_reproduced':True,
                'paired_nll_interval_proposed_minus_native':result['interval'],'models_fitted':0,'encoder_calls':0,'new_parameter_selections':0})
            log.info('PHASE 3/3 | verify unchanged inputs and package exact reproduction inputs')
            result['inputs'].unchanged();reproduction=bundle(root,folder,result['inputs']);result['inputs'].unchanged()
            status='PRIMARY_AFFINITY_ANALYSIS_EXPORT_COMPLETE'
            write_json(folder/'run_result.json',{'status':status,'corpus':'Primary','seeds':SEEDS,'new_model_fits':0,
                'inputs_unchanged':True,'paired_nll_superiority_established':False,'numeric_sensitivity_sweep_complete':False})
            parent=folder.parent;(parent/'LATEST.txt').write_text(folder.name+'\n',encoding='utf-8')
            log.info('READ FIRST | %s',folder/'main/tables/T6_primary_affinity_analysis.md')
            log.info('FIGURE | %s',folder/'main/figures/F1_primary_affinity_before_after_English.pdf')
            log.info('REPRODUCTION PACKAGE | %s',reproduction)
        log.info('VERDICT | %s | scientific claims remain bounded',status)
        return 0
    except Exception as error:
        log.error('FAILED | %s: %s',type(error).__name__,str(error))
        (run_folder/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        return 1
    finally:
        elapsed=time.perf_counter()-start
        write_json(run_folder/'run_result.json',{'status':status,'corpus':'Primary','self_check':bool(args.self_check),
            'new_model_fits':0,'encoder_calls':0,'wall_seconds':elapsed,'timestamp_utc':now.isoformat()})
        with (root/'run_history.txt').open('a',encoding='utf-8') as stream:
            stream.write(f"{now.isoformat()} | primary affinity export | run={run_id} | status={status} | wall={elapsed:.2f}s\n")
        log.info('SAVED | run report=%s | wall=%.2f seconds',run_folder/'run_result.json',elapsed)
        for handler in list(log.handlers):
            handler.close();log.removeHandler(handler)
