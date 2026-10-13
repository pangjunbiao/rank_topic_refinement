"""External held-out scoring, complete-slot summaries and paired uncertainty."""
from __future__ import annotations
import copy
import statistics
from pathlib import Path
import numpy as np
from rank_refinement.runtime.artifacts import digest, file_hash, read_json, write_json, write_csv, write_text
from rank_refinement.evaluation.competitor_evidence import audit_native_reference
from rank_refinement.evaluation.diversity import topic_diversity
from rank_refinement.evaluation.replication import hierarchical_gain
from .models import frozen_scores

def audit_topics(topics,records,vocabulary):
    if topics is None:return {'c_v':None,'c_npmi':None,'supported_slots':0,'total_slots':0,
        'reason':'native_topics_not_available_without_padding','audit':None}
    audit=audit_native_reference(topics,records,vocabulary)
    return {'c_v':audit['metrics']['c_v']['mean'],'c_npmi':audit['metrics']['c_npmi']['mean'],
        'supported_slots':audit['supported_slots'],'total_slots':audit['total_slots'],'audit':audit,
        'reason':None if all(audit['metrics'][m]['complete']for m in ['c_v','c_npmi'])else 'undefined_standard_metric_all_slots_retained'}

def score_cases(output,case_paths,data,policy,log):
    rows=[];dev_count=len(data['dev_ids'])
    coherence_cache=output/'coherence';coherence_cache.mkdir(exist_ok=True)
    for n,path in enumerate(case_paths,1):
        saved=read_json(path/'case_result.json')
        with np.load(path/'predictive_arrays.npz',allow_pickle=False)as z:arrays={name:z[name]for name in z.files}
        for original in saved['rows']:
            row=copy.deepcopy(original)
            for split,records in [('fitting',data['fit_coherence_records']),('development',data['dev_coherence_records']),('test',data['test_coherence_records'])]:
                identity={'topics':row['topics'],'reference':[(r['group_id'],r['tokens'])for r in records],
                    'vocabulary':data['vocabulary'],'windows':[110,10]}
                file=coherence_cache/(digest(identity)+'.json')
                index=file.with_suffix('.index.json')
                if file.is_file():
                    if not index.is_file()or read_json(index).get('sha256')!=file_hash(file):
                        raise ValueError('External coherence cache checksum changed or its index is missing.')
                    entry=read_json(file)
                    if entry['identity_sha256']!=digest(identity):raise ValueError('External coherence cache identity changed.')
                    metric=entry['metric']
                else:
                    metric=audit_topics(row['topics'],records,data['vocabulary'])
                    write_json(file,{'identity_sha256':digest(identity),'metric':metric})
                    write_json(index,{'sha256':file_hash(file)})
                row[split+'_coherence']=metric
            row['topic_diversity']=topic_diversity(row['topics'])if row['topics']else None
            role=row.get('role','native'if row['method']=='NeuroMax'else 'model')
            key='logp_'+role
            if row['status']=='scored'and key in arrays:
                row['prediction'],row['document_losses']=frozen_scores(arrays[key][dev_count:],row['rho'],arrays['prior'],data)
            else:
                row['prediction']={'macro_source_nll':None,'micro_nll':None};row['document_losses']=[]
            row['external_test_result']=True;row['test_used_for_training_or_selection']=False
            row['case_relative_path']=path.relative_to(output).as_posix()
            rows.append(row)
        log.info('EXTERNAL REPORT | verified/scored case %d/%d | %s',n,len(case_paths),path.relative_to(output).as_posix())
    return rows

def summary_rows(rows,seeds):
    output=[]
    for method,profile in sorted({(r['method'],r['profile'])for r in rows}):
        selected=sorted([r for r in rows if(r['method'],r['profile'])==(method,profile)],key=lambda r:r['seed'])
        if len(selected)!=len(seeds)or {r['seed']for r in selected}!=set(seeds):
            raise ValueError('Missing or duplicate seed slot for '+method+'/'+profile)
        item={'method':method,'profile':profile,'n_seeds':len(seeds),'fixed_k10_all_seeds':all(r['fixed_k10']for r in selected),
            'scored_seeds':sum(r['status']=='scored'for r in selected),
            'native_topic_counts':','.join(str(r.get('native_topic_count'))for r in selected),
            'rho_by_seed':','.join(str(r.get('rho'))for r in selected),
            'deployed_seeds':sum(r.get('deployment')=='certified_projection_deployed'for r in selected),
            'abstained_seeds':sum(r.get('deployment')=='abstained_no_supported_edit'for r in selected)}
        fields={'test_nll':[r['prediction']['macro_source_nll']for r in selected],
            'development_nll':[r['development_prediction']['macro_source_nll']for r in selected],
            'topic_diversity':[r['topic_diversity']if r['fixed_k10']else None for r in selected]}
        for split in ['fitting','development','test']:
            for metric in ['c_v','c_npmi']:
                fields[split+'_'+metric]=[r[split+'_coherence'][metric]if r['fixed_k10']else None for r in selected]
        for name,values in fields.items():
            complete=all(v is not None and np.isfinite(v)for v in values)
            item[name+'_mean']=statistics.mean(values)if complete else None
            item[name+'_sd']=statistics.stdev(values)if complete and len(values)>1 else None
            item[name+'_available_seeds']=int(sum(v is not None and np.isfinite(v)for v in values))
        output.append(item)
    return output

def paired_intervals(rows,policy):
    proposed=[r for r in rows if r['method']=='Proposed'];result=[]
    for method,profile in sorted({(r['method'],r['profile'])for r in rows if r['method']!='Proposed'}):
        other=[r for r in rows if(r['method'],r['profile'])==(method,profile)]
        valid=all(r['prediction']['macro_source_nll']is not None for r in proposed+other)
        if valid:
            # Preserve the paired hierarchical arithmetic; name the external scope.
            gain=hierarchical_gain(proposed,other,policy['seeds'],policy['bootstrap_resamples'],policy['bootstrap_seed'])
            gain['scope']='fixed_external_recipe_single_split; seed_and_duplicate_group_bootstrap; multiple_comparisons_not_adjusted'
            result.append({'method':method,'profile':profile,'available':True,**gain})
        else:result.append({'method':method,'profile':profile,'available':False,
            'mean_competitor_minus_proposed_macro_nll':None,'percentile_95_low':None,'percentile_95_high':None,
            'reason':'at_least_one_missing_native_predictive_slot; not_a_win_or_loss'})
    return result

def format_value(mean,sd):
    return 'NA'if mean is None else f'{mean:.6f} ± {sd:.6f}'if sd is not None else f'{mean:.6f}'

def markdown_table(rows):
    lines=['| Method / profile | Fit C_v ↑ | Fit NPMI ↑ | TD ↑ | External test NLL ↓ | Test C_v ↑ | Test NPMI ↑ |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values=[format_value(row[n+'_mean'],row[n+'_sd'])for n in ['fitting_c_v','fitting_c_npmi','topic_diversity','test_nll','test_c_v','test_c_npmi']]
        lines.append('| '+row['method']+' / '+row['profile']+' | '+' | '.join(values)+' |')
    return '\n'.join(lines)+'\n'

def figures(output,intervals,rows,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=output/'figures';folder.mkdir(exist_ok=True)
    valid=[x for x in intervals if x['available']]
    if valid:
        fig,ax=plt.subplots(figsize=(9,max(3,.38*len(valid)+1)))
        for i,x in enumerate(valid):
            mean=x['mean_competitor_minus_proposed_macro_nll'];lo=x['percentile_95_low'];hi=x['percentile_95_high']
            ax.plot([lo,hi],[i,i],color='#307a91',lw=1.7);ax.scatter([mean],[i],color='#153b50',s=20)
        ax.axvline(0,color='#666666',ls='--',lw=1);ax.set_yticks(range(len(valid)),[x['method']+' / '+x['profile']for x in valid],fontsize=8)
        ax.set_xlabel('Comparator minus Proposed test NLL (positive favors Proposed)')
        ax.set_title('External corpus: paired seed/group 95% intervals');ax.grid(axis='x',alpha=.2);fig.tight_layout()
        fig.savefig(folder/'F7_external_prediction_intervals.png',dpi=220);fig.savefig(folder/'F7_external_prediction_intervals.pdf');plt.close(fig)
    proposed=sorted([r for r in rows if r['method']=='Proposed'],key=lambda r:r['seed'])
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    axes[0].bar(['Fit','Development','Test'],[r['in_vocabulary_fraction']for r in data['audit']['coverage']],color=['#255d79','#579c9e','#87bbaa'])
    axes[0].set_ylim(0,1);axes[0].set_ylabel('Fraction of lexical token occurrences');axes[0].set_title('Fitting-vocabulary coverage')
    support=[r['test_coherence']['supported_slots']for r in proposed]
    axes[1].bar([str(r['seed'])for r in proposed],support,color='#255d79');axes[1].axhline(100,color='#666666',ls='--',lw=1)
    axes[1].set_ylim(0,105);axes[1].set_xlabel('Pipeline seed');axes[1].set_ylabel('Descriptor slots seen in test / 100');axes[1].set_title('No absent descriptors removed')
    fig.tight_layout();fig.savefig(folder/'S7_external_reference_support.png',dpi=220);fig.savefig(folder/'S7_external_reference_support.pdf');plt.close(fig)

def export(output,rows,data,policy,primary_unchanged,log):
    control_methods={'Primary TV control','Same-core display only','Moment correction control'}
    comparison=[r for r in rows if r['method']not in control_methods];controls=[r for r in rows if r['method']in control_methods]
    if len(comparison)!=60 or len(controls)!=15:raise ValueError('Require all 60 comparison and 15 control slots, including explicit NA.')
    summary=summary_rows(rows,policy['seeds']);intervals=paired_intervals(rows,policy)
    comparison_summary=[r for r in summary if r['method']not in control_methods]
    control_summary=[r for r in summary if r['method']in control_methods]
    tables=output/'tables';tables.mkdir(exist_ok=True)
    write_json(output/'all_rows.json',rows);write_json(output/'comparison_rows.json',comparison);write_json(output/'control_rows.json',controls)
    write_json(output/'summary_metrics.json',summary);write_json(output/'paired_intervals.json',intervals)
    for name,values in [('T7_external_comparison',comparison_summary),('S7_external_controls',control_summary)]:
        write_csv(tables/(name+'.csv'),values,list(values[0]));write_text(tables/(name+'.md'),markdown_table(values))
    paired_columns=['method','profile','available','mean_competitor_minus_proposed_macro_nll','percentile_95_low','percentile_95_high']
    write_csv(tables/'S7_external_paired_uncertainty.csv',[{k:r.get(k)for k in paired_columns}for r in intervals],paired_columns)
    slot_rows=[{'method':r['method'],'profile':r['profile'],'seed':r['seed'],'fixed_k10':r['fixed_k10'],
        'test_c_v':r['test_coherence']['c_v'],'test_c_npmi':r['test_coherence']['c_npmi'],
        'supported_slots':r['test_coherence']['supported_slots'],'total_slots':r['test_coherence']['total_slots'],
        'absent_words':';'.join((r['test_coherence'].get('audit')or {}).get('reference_absent_words',[]))}for r in rows]
    write_csv(tables/'S7_external_metric_availability.csv',slot_rows,list(slot_rows[0]))
    topic_rows=[{'method':r['method'],'profile':r['profile'],'seed':r['seed'],'topic_index':t,
                'words':' | '.join(words)}for r in rows for t,words in enumerate(r['topics']or [])]
    write_csv(tables/'S7_external_topics.csv',topic_rows,['method','profile','seed','topic_index','words'])
    mechanisms=[]
    for p in sorted((output/'cases/proposed_parent').glob('seed*/refinement.json')):
        value=read_json(p);m=value['mechanism']or {}
        mechanisms.append({'seed':int(p.parent.name[4:]),'deployment':value['deployment'],
            'edited_topics':len(value['selection']['selected']),
            'target_membership_recall':m.get('target_top10_membership_recall'),
            'relative_l1_movement':m.get('relative_l1_affinity_movement'),
            'max_topic_total_error_scaled':m.get('maximum_topic_total_error_scaled'),
            'max_word_total_error_scaled':m.get('maximum_word_total_error_scaled')})
    write_csv(tables/'S7_external_projection_certificates.csv',mechanisms,list(mechanisms[0]))
    proposed=next(r for r in summary if r['method']=='Proposed');native=next(r for r in summary if r['method']=='NeuroMax')
    parent_interval=next(r for r in intervals if r['method']=='NeuroMax')
    points={}
    for metric,direction in [('fitting_c_v',1),('fitting_c_npmi',1),('topic_diversity',1),('test_nll',-1)]:
        p=proposed[metric+'_mean'];valid=[r for r in comparison_summary if r['method']!='Proposed'and r[metric+'_mean']is not None]
        points[metric]={'defined_comparators':len(valid),'wins_or_ties_all_defined_point_means':
            p is not None and bool(valid)and all(direction*(p-r[metric+'_mean'])>=-1e-12 for r in valid),
            'unavailable_comparators':len(comparison_summary)-1-len(valid)}
    unavailable=sum(r['status']!='scored'for r in rows)
    result={'status':'STEP07_EXTERNAL_REPLICATION_COMPLETE','protocol':policy['protocol'],
        'engineering_passed':True,'preparation_gate_passed':True,'model_quality_established':False,
        'external_mode':'frozen_recipe_refit','comparison_slots':len(comparison),'control_slots':len(controls),
        'unavailable_execution_slots':unavailable,'all_five_seeds_retained':True,
        'primary_results_unchanged':primary_unchanged,'primary_freeze_id':policy['primary_freeze_id'],
        'dataset_sha256':data['audit']['input_sha256'],'heldout_test_completion_groups':len(data['test_ids']),
        'external_test_lexical_coverage':data['audit']['coverage'][-1]['in_vocabulary_fraction'],
        'proposed_deployed_seeds':proposed['deployed_seeds'],'proposed_abstained_seeds':proposed['abstained_seeds'],
        'point_mean_comparisons':points,'parent_paired_nll_interval':parent_interval,
        'all_metric_win_established':False,'test_used_for_model_or_parameter_selection':False,
        'claim_scope':'recipe replication on this anonymized English transit-experience corpus; not Chinese weight transfer, requirement accuracy, or universal superiority'}
    write_json(output/'run_result.json',result)
    text=['# Step 7: external recipe replication','',
        f"Status: {result['status']}. All 60 comparison and 15 control slots retained.",
        f"Dataset: {data['audit']['nonempty_text_rows']} tweets; {data['audit']['independent_groups']} independent groups; fitting vocabulary {len(data['vocabulary'])} words.",
        f"Proposed: signed_stable/range on the single selected NeuroMax parent; moment correction disabled; certified projection deployed in {proposed['deployed_seeds']}/5 seeds, explicit native abstention in {proposed['abstained_seeds']}/5.",
        '', '## Comparison results', '',markdown_table(comparison_summary),'## Matched external controls','',markdown_table(control_summary),
        '## Interpretation','',
        'Fit C_v/NPMI describe the fitting corpus, where association acquisition was performed. They are not held-out estimates. Test coherence retains every topic and every ten-word descriptor; undefined cells remain NA.',
        f"Only {data['audit']['coverage'][-1]['in_vocabulary_fraction']:.1%} of external test lexical occurrences are in the fitting vocabulary. Completion NLL is conditional on these in-vocabulary targets; it is not a full-text likelihood and must not be compared numerically with the primary corpus NLL.",
        'With one external source, macro-source and micro NLL coincide. Five seeds measure training/acquisition variation on one fixed split, not five independent datasets or splits.',
        'BERTopic has no native word likelihood. A variable number of native topics is retained in detailed files and marked NA in fixed-K descriptor comparisons; no topic is padded or removed to improve a score.',
        'Bootstrap intervals resample paired duplicate groups and pipeline seeds. They are unadjusted for multiple comparisons and do not justify a universal-win claim.',
        'Same-core display only shares the Proposed words but retains the native predictor. The projection certificates, not descriptor gains alone, demonstrate the actual-readout mechanism.',
        'This small, already curated corpus emphasizes interpersonal transit experiences. It supports a bounded external replication, not population representativeness or labeled requirement-detection accuracy.',
        'Original primary results and configurations were verified unchanged. The external recipe was locked before any external test loss was scored.',
        '', '## Files','',
        '`tables/T7_external_comparison.csv`: all fixed profiles; `tables/S7_external_controls.csv`: matched controls; `paired_intervals.json`: paired uncertainty; `tables/S7_external_metric_availability.csv`: support/NA details; `tables/S7_external_topics.csv`: every genuine descriptor; `tables/S7_external_projection_certificates.csv`: readout checks; `preparation/data_audit.json`: coverage and data provenance.',
        'Model checkpoints and encoder cache remain in workspace/data/cache. The inspection ZIP includes their indices and hashes rather than duplicating large checkpoint or encoder-weight files.']
    write_text(output/'SUMMARY.md','\n\n'.join(text)+'\n')
    figures(output,intervals,rows,data)
    for r in comparison_summary:
        log.info('EXTERNAL RESULT | %s | %s | fit Cv=%s | fit NPMI=%s | TD=%s | test NLL=%s',r['method'],r['profile'],
            format_value(r['fitting_c_v_mean'],r['fitting_c_v_sd']),format_value(r['fitting_c_npmi_mean'],r['fitting_c_npmi_sd']),
            format_value(r['topic_diversity_mean'],r['topic_diversity_sd']),format_value(r['test_nll_mean'],r['test_nll_sd']))
    if parent_interval['available']:
        log.info('MATCHED PARENT | native minus Proposed NLL=%.6f | paired 95%% interval=[%.6f, %.6f]',
            parent_interval['mean_competitor_minus_proposed_macro_nll'],parent_interval['percentile_95_low'],parent_interval['percentile_95_high'])
    return result
