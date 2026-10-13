"""Six short main tables; four short supplementary tables, with exact CSVs."""
from __future__ import annotations
import csv
import json
import shutil
import statistics
from pathlib import Path
from rank_refinement.runtime.artifacts import write_json,write_text
from step08_assembly.tables import plain_table,fmt,escape


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def summary_rows(path):
    rows=read_csv(path)
    for r in rows:
        for k,v in list(r.items()):
            if k.endswith(('_mean','_sd')):
                r[k]=None if v=='' else float(v)
    return rows


def table(folder,name,rows,columns,caption):
    keys=[k for k,_ in columns]
    plain_table(folder,name,[{k:r[k]for k in keys}for r in rows],columns,caption=caption)
    # Wrap text-heavy columns instead of exporting a single overwide line.
    path=folder/(name+'.tex');tex=path.read_text(encoding='utf-8')
    if name=='T3_matched_ablations':
        tex=tex.replace(r'\begin{tabular}{lrrrrr}',r'\begin{tabularx}{\textwidth}{lXrrrr}')
    elif name=='T5_transport_text_examples':
        tex=tex.replace(r'\begin{tabular}{lrrrr}',r'\begin{tabularx}{\textwidth}{rp{0.18\textwidth}lp{0.18\textwidth}X}')
        for row in rows:
            g=row['group_id']
            tex=tex.replace(escape(g),r'\shortstack[l]{'+escape(g[:14])+r'\\'+escape(g[14:])+'}')
    elif name=='S4_reproducibility_summary':
        tex=tex.replace(r'\begin{tabular}{lr}',r'\begin{tabularx}{\textwidth}{lX}')
    if r'\begin{tabularx}' in tex:
        tex=tex.replace(r'\end{tabular}',r'\end{tabularx}')
        write_text(path,'% Requires booktabs and tabularx; use a CJK-enabled engine for Chinese text.\n'+tex)


def exhibit_tables(folder,assembly,seed_results):
    main=folder/'main/tables';supp=folder/'supplementary/tables'
    main.mkdir(parents=True);supp.mkdir(parents=True)
    for old,new in [('T1_dataset_characteristics','T1_data'),('T2_primary_comparison','T2_primary_competitors'),
                    ('T4_external_replication','T4_external_comparison')]:
        for ext in ['csv','md','tex']:
            shutil.copy2(assembly/'main'/(old+'.'+ext),main/(new+'.'+ext))
    controls=summary_rows(assembly/'main/T3_primary_controls.csv')
    by_name={r['method']:r for r in controls}
    component=[('Proposed','Proposed'),('NeuroMax','Without acquisition/projection (native parent)'),
        ('Alternate signed graph','Without association stability discount (signed full graph)'),
        ('Positive-only associations','Without signed negative associations')]
    projection=[('Proposed','Proposed (prediction-aware range secondary)'),
        ('Primary TV control','Primary minimum-TV only'),('Index secondary control','Index secondary'),
        ('L2 control','L2 projection'),('Same-core display only','Same-core display only')]
    rows=[]
    for panel,choices in [('A: components',component),('B: projection',projection)]:
        for key,name in choices:
            r=by_name[key];rows.append(dict(panel=panel,variant=name,
                cv=fmt(r['fit_c_v_mean'],r['fit_c_v_sd']),npmi=fmt(r['fit_npmi_mean'],r['fit_npmi_sd']),
                td=fmt(r['td_mean'],r['td_sd']),nll=fmt(r['test_nll_mean'],r['test_nll_sd'],6)))
    table(main,'T3_matched_ablations',rows,[('panel','Panel'),('variant','Variant'),('cv','Fit C_v ↑'),
        ('npmi','Fit NPMI ↑'),('td','TD ↑'),('nll','Test NLL ↓')],
        'Matched primary controls, five seeds (sample SD). Coherence is measured on fitting data. '+
        'Retain improvements and losses: removal of stability does not worsen every metric. '+
        'Display-only descriptors equal Proposed; its actual predictive readout remains native. '+
        'The range-control alias of Proposed and the zero-rho no-backoff alias are not repeated.')
    examples=read_csv(assembly/'supplementary/S09_automatic_fitting_examples.csv')
    # Purely automatic screening for service-related words, no assigned labels.
    anchors={'发车','排队','刷_码','拥挤_度','车门','停靠站','公交网','支付','刷卡机',
             '安全','司机','开_空调','首车','车','路'}
    candidates=[];seen=set()
    for r in examples:
        if r['dataset']!='Primary' or r['topic_index'] in seen:
            continue
        seen.add(r['topic_index']);words=r['descriptor_words'].split(' | ')
        candidates.append((len(set(words)&anchors),int(r['topic_index']),r))
    chosen=sorted(sorted(candidates,key=lambda x:(-x[0],x[1]))[:6],key=lambda x:x[1])
    app=[]
    for score,topic,r in chosen:
        text=r['fitting_text'];excerpt=text if len(text)<=150 else text[:150]+'…'
        app.append(dict(topic=topic,words='; '.join(r['descriptor_words'].split(' | ')[:6]),
            source=r['source'],group_id=r['group_id'],text=excerpt,service_anchor_count=score))
    table(main,'T5_transport_text_examples',app,[('topic','Topic'),('words','Leading words'),
        ('source','Source'),('group_id','Evidence group'),('text','Post excerpt')],
        'Automatically screened fitting examples, fixed seed 809. From each topic, use its first '+
        'previously selected highest-overlap post. Rank topics by overlap with a fixed service-word '+
        'list (ties: topic index), retain six, and display in topic order. Excerpts are at most 150 '+
        'Unicode characters; ellipses mark truncation. These are text-grounded themes, not validated '+
        'passenger-requirement labels. Exact original text and the selection list are saved separately.')
    write_json(folder/'numerical_evidence/application_selection.json',
        {'seed':809,'anchors':sorted(anchors),'selection_rule':'top six topic anchor counts, topic-index tie break; first highest-overlap post',
         'selected_original_rows':[r for _,_,r in chosen],'human_labels_or_accuracy':False})
    metrics=[('affinity_js_nats','Row-normalized affinity JS (nats)',True),
        ('affinity_tv','Row-normalized affinity TV',True),
        ('prototype_js_nats','Unit-topic decoder-profile JS (nats)',True),
        ('final_document_js_nats','Deployed document-prediction JS (nats)',True),
        ('final_document_tv','Deployed document-prediction TV',True),
        ('relative_l1_percent','Relative affinity L1 movement (%)',False),
        ('native_target_recall','Native actual-readout target recall',False),
        ('proposed_target_recall','Proposed actual-readout target recall',False)]
    results=[]
    for key,label,scientific in metrics:
        row={'measure':label}
        for d in ['Primary','External']:
            v=[r[key]for r in seed_results if r['dataset']==d]
            row[d]=f'{statistics.mean(v):.3e} ± {statistics.stdev(v):.3e}' if scientific else fmt(statistics.mean(v),statistics.stdev(v),6)
        results.append(row)
    table(main,'T6_native_proposed_divergence',results,[('measure','Measure'),('Primary','Primary'),('External','External')],
        'Matched native NeuroMax versus Proposed: mean ± sample SD of five seed-level means. '+
        'Affinity and prototype means average ten matched topic rows; predictive means average all '+
        'eligible frozen test groups (410 primary, 112 external), each weighted equally. Natural-log '+
        'JS divergence is reported, not its square-root distance. Affinity shares are not decoder '+
        'probabilities. Decoder profiles use unit-topic input through the frozen affine decoder. '+
        'Deployed prediction includes each method’s frozen backoff (rho=0 primary, 0.2 external). '+
        'Small divergence indicates proximity, not superior prediction. All ten readouts pass '+
        'nonnegativity and row/column conservation checks; exact errors are in the numerical evidence.')
    external=summary_rows(assembly/'supplementary/S02_external_controls.csv')
    extra=[by_name[n]for n in ['Proposed with moment correction','Absolute 100-word disjointness control']]
    sr=[]
    for r in external+extra:
        sr.append(dict(dataset=r['dataset'],variant=r['method'],cv=fmt(r['fit_c_v_mean'],r['fit_c_v_sd']),
            npmi=fmt(r['fit_npmi_mean'],r['fit_npmi_sd']),td=fmt(r['td_mean'],r['td_sd']),
            nll=fmt(r['test_nll_mean'],r['test_nll_sd'],6)))
    table(supp,'S1_additional_controls',sr,[('dataset','Corpus'),('variant','Control'),('cv','Fit C_v'),
        ('npmi','Fit NPMI'),('td','TD'),('nll','Test NLL')],
        'Additional controls, all five seeds. Moment correction is excluded from Proposed. '+
        'Absolute word-disjointness is a stricter control, not a removed Proposed component. '+
        'NA denotes unavailable complete-seed summaries, not a zero result.')
    intervals=read_csv(assembly/'supplementary/S03_paired_uncertainty.csv')
    uncertainty=[]
    for r in intervals:
        if r['method']not in {'NeuroMax','GSDMM'}:continue
        uncertainty.append(dict(dataset=r['dataset'],comparator=r['method']+' / '+r['profile'],
            delta=f"{float(r['comparator_minus_proposed_nll']):+.6f}" if r['comparator_minus_proposed_nll'] else 'NA',
            interval=f"[{float(r['percentile_95_low']):+.6f}, {float(r['percentile_95_high']):+.6f}]" if r['percentile_95_low'] else 'NA'))
    table(supp,'S2_paired_prediction_uncertainty',uncertainty,[('dataset','Corpus'),('comparator','Comparator'),
        ('delta','Comparator − Proposed NLL'),('interval','Paired 95% interval')],
        'Frozen paired seed/group percentile-bootstrap intervals, unadjusted for multiplicity. '+
        'Positive differences favor Proposed. Zero-crossing intervals establish neither superiority '+
        'nor equivalence. All comparisons remain in the earlier audit CSV.')
    availability=[]
    for d in ['Primary','External']:
        r=next(x for x in summary_rows(assembly/('main/T2_primary_comparison.csv' if d=='Primary' else 'main/T4_external_replication.csv'))if x['method']=='Proposed')
        availability.append(dict(dataset=d,fit_cv=r['fit_c_v_available_seeds']+'/5',
            fit_npmi=r['fit_npmi_available_seeds']+'/5',heldout_cv=r['test_c_v_available_seeds']+'/5',
            heldout_npmi=r['test_npmi_available_seeds']+'/5',prediction=r['test_nll_available_seeds']+'/5'))
    table(supp,'S3_metric_availability',availability,[('dataset','Corpus'),('fit_cv','Fit C_v'),
        ('fit_npmi','Fit NPMI'),('heldout_cv','Held-out C_v'),('heldout_npmi','Held-out NPMI'),
        ('prediction','Completion NLL')],
        'Number of available Proposed seeds. Sparse fixed held-out coherence references do not '+
        'support complete descriptor scores. Keep the fitting/held-out distinction in the manuscript. '+
        'No unsupported words, topics or unsuccessful competitor runs were removed.')
    reproducibility=[
        {'item':'Matched runs','value':'Seeds 809–813; one fixed group split per corpus'},
        {'item':'Final parent','value':'NeuroMax, random-word initialization; EnCOT remains a competitor'},
        {'item':'Acquisition','value':'Signed local-window associations; duplicate-group support; subsample stability; native neighborhoods'},
        {'item':'Rank projection','value':'Minimum TV; range secondary within the declared TV cap; row and column totals conserved'},
        {'item':'Final prediction','value':'Moment correction disabled; development-selected frozen backoff'},
        {'item':'External scope','value':'Same recipe refitted on a separate English corpus; not zero-shot Chinese-weight transfer'},
        {'item':'Text example scope','value':'Automatic fitting examples; no human labels or requirement-detection accuracy'},
        {'item':'Reporting execution','value':'CPU, saved arrays only; zero new fits, encoder calls or parameter selections'}]
    table(supp,'S4_reproducibility_summary',reproducibility,[('item','Item'),('value','Fixed setting')],
        'Compact reproducibility summary. Exact profiles, dependencies, hashes and raw per-run '+
        'results remain in the audit files, not in long supplementary manuscript tables.')
    # One readable main-only document, never concatenate raw words/configuration dumps.
    text=['# Compact main results','',
          'Six tables. Fit coherence is descriptive; five seeds use one fixed split per corpus.','']
    for p in sorted(main.glob('*.md')):text+=['## '+p.stem,'',p.read_text(encoding='utf-8'),'']
    write_text(folder/'MAIN_RESULTS.md','\n'.join(text))
    text=['# Compact supplementary results','']
    for p in sorted(supp.glob('*.md')):text+=['## '+p.stem,'',p.read_text(encoding='utf-8'),'']
    write_text(folder/'SUPPLEMENTARY_RESULTS.md','\n'.join(text))
