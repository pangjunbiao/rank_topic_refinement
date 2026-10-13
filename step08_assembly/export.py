"""Paper/supplement assembly from immutable completed results."""
from __future__ import annotations
import shutil
import statistics
from pathlib import Path
from rank_refinement.runtime.artifacts import read_json,write_json,write_csv,write_text
from .metrics import point_winners
from .tables import comparison_table,plain_table,fmt

def preparation_table(bundle):
    prep=bundle['preparation'];audit=read_json(prep/'raw_audit.json');coverage=read_json(prep/'coverage.json')
    allc={r['split']:r for r in coverage if r['source']=='ALL'}
    ext=read_json(bundle['external']/'preparation/data_audit.json');ec={r['split']:r for r in ext['coverage']}
    p={'dataset':'Primary Chinese transit posts','raw_records':audit['raw_records'],
       'independent_text_groups':read_json(prep/'preparation_report.json')['eligible_representatives'],
       'fit_groups':allc['fit']['representatives'],'development_groups':allc['development']['representatives'],
       'test_groups':allc['test']['representatives'],'test_completion_groups':allc['test']['completion_eligible_documents'],
       'fitting_vocabulary':len(read_json(prep/'vocabulary.json')['tokens']),
       'test_lexical_coverage':allc['test']['coverage'],'sources':3}
    e={'dataset':'External English transit tweets','raw_records':ext['raw_rows'],
       'independent_text_groups':ext['independent_groups'],'fit_groups':ec['fit']['independent_groups'],
       'development_groups':ec['development']['independent_groups'],'test_groups':ec['test']['independent_groups'],
       'test_completion_groups':ec['test']['completion_eligible_groups'],'fitting_vocabulary':ext['vocabulary'],
       'test_lexical_coverage':ec['test']['in_vocabulary_fraction'],'sources':1}
    return [p,e]

def paired_table(audit):
    result=[]
    for r in audit['intervals']:
        available=r.get('available',True);lo=r.get('percentile_95_low');hi=r.get('percentile_95_high')
        sign='unavailable'if not available else 'crosses or touches zero'if lo<=0<=hi else 'favors Proposed'if lo>0 else 'favors comparator'
        result.append({'dataset':r['dataset'],'method':r['method'],'profile':r['profile'],'available':available,
            'comparator_minus_proposed_nll':r.get('mean_competitor_minus_proposed_macro_nll'),
            'percentile_95_low':lo,'percentile_95_high':hi,'interval_direction':sign,
            'multiplicity_adjusted':False,'scope':'paired seed/group descriptive interval; not a global significance declaration'})
    return result

def claims(audit):
    summary=audit['summary'];decisions=[]
    for dataset in ['Primary','External']:
        p=next(r for r in summary if r['dataset']==dataset and r['method']=='Proposed'and not r['control'])
        native=next(r for r in summary if r['dataset']==dataset and r['method']=='NeuroMax'and not r['control'])
        interval=next(r for r in audit['intervals']if r['dataset']==dataset and r['method']=='NeuroMax')
        decisions.append({'dataset':dataset,'fit_c_v_delta_vs_parent':p['fit_c_v_mean']-native['fit_c_v_mean'],
            'fit_npmi_delta_vs_parent':p['fit_npmi_mean']-native['fit_npmi_mean'],
            'test_nll_proposed_minus_parent':p['test_nll_mean']-native['test_nll_mean'],
            'parent_minus_proposed_interval_low':interval['percentile_95_low'],
            'parent_minus_proposed_interval_high':interval['percentile_95_high'],
            'parent_prediction_superiority_supported':interval['percentile_95_low']>0,
            'all_five_heldout_coherence_available':p['test_c_v_available_seeds']==5 and p['test_npmi_available_seeds']==5,
            'actual_readout_constraints_verified':True})
    return {'reporting_ready':True,'submission_quality_automatically_approved':False,
        'universal_superiority_supported':False,'human_interpretability_or_requirement_accuracy_measured':False,
        'zero_shot_transfer_measured':False,'primary_and_external_test_tuning_performed':False,
        'external_scope':'fixed recipe refitted on one small curated corpus, one fixed split, five pipeline seeds',
        'parent_comparisons':decisions,'point_winners':point_winners(summary),
        'necessary_claim_limits':['Fitting coherence is descriptive and uses the acquisition corpus.',
            'A confidence interval crossing zero establishes neither superiority nor equivalence.',
            'Same-core display only obtains the same descriptor scores without changing the predictor.',
            'An LP range-bound improvement does not establish improved unseen prediction loss.',
            'Native variable-topic models retain their predictions; fixed-K descriptor summaries remain NA.',
            'NMF reconstruction cross-entropy is not a native generative likelihood.',
            'No neural training, encoder inference, parameter search, or new test selection occurs in Step 8.']}

def write_claims(folder,decisions):
    text=['# Evidence and manuscript claims','',
        '**The verified experiment package is ready for reporting. Journal acceptance and universal superiority are not established.**','',
        'Supported wording: post-training rank refinement realized the selected lexical requirements in the actual readout while preserving native topic/word totals. Both corpora show higher fitting coherence than the matched parent, with small observed changes in completion NLL.','',
        'Fitting coherence evaluates the same corpus used for lexical acquisition; call it descriptive coherence, not held-out topic generalization. Automatic examples illustrate discovered text themes and do not establish requirement-detection accuracy or human interpretability.','']
    for r in decisions['parent_comparisons']:
        text += ['## '+r['dataset'],'',
            f"Fitting C_v gain: {r['fit_c_v_delta_vs_parent']:+.6f}; fitting NPMI gain: {r['fit_npmi_delta_vs_parent']:+.6f}.",
            f"Proposed minus native test NLL: {r['test_nll_proposed_minus_parent']:+.6f} nats/target token.",
            f"Native minus Proposed 95% interval: [{r['parent_minus_proposed_interval_low']:.6f}, {r['parent_minus_proposed_interval_high']:.6f}].",
            'The parent interval crosses zero: prediction superiority and statistical equivalence are both unproven.','']
    text += ['## Claims to avoid','',
        '- Winning all competitors or all controls on both corpora.',
        '- Positive external NPMI: the measured Proposed external mean is negative.',
        '- Held-out coherence superiority: the complete five-seed held-out descriptor scores are unavailable.',
        '- Improved requirement-detection accuracy, human interpretability, population representativeness, or Chinese-weight transfer.',
        '- A unique minimum-TV solution: the solver finds a minimum-TV feasible affinity; uniqueness was not established.',
        '- Significant gains inferred only from best means, small SDs, or unadjusted pairwise intervals.','',
        'Same-core display only shares the Proposed descriptors. Its identical coherence is expected. The projection contributes faithful realization in the actual predictive readout, not an additional descriptor-score improvement beyond that display control.','',
        'The two supplied test sets have already been exposed. Any future method revision needs a new evaluation protocol and new untouched evaluation evidence; these results cannot be relabeled as untouched tests after test-guided optimization.']
    write_text(folder/'SUPPORTED_CLAIMS.md','\n'.join(text)+'\n')

def paper_text(folder,audit):
    summary=audit['summary']
    find=lambda d,m:next(r for r in summary if r['dataset']==d and r['method']==m and not r['control'])
    p=find('Primary','Proposed');e=find('External','Proposed');n=find('External','NeuroMax')
    text=f'''# Results text grounded in the verified outputs

On the primary corpus, Proposed obtained fitting C_v of {fmt(p['fit_c_v_mean'],p['fit_c_v_sd'])}, fitting NPMI of {fmt(p['fit_npmi_mean'],p['fit_npmi_sd'])}, and macro-source completion NLL of {fmt(p['test_nll_mean'],p['test_nll_sd'],6)} across five matched seeds. It had the highest fitting coherence point means and the lowest defined completion-loss point mean in this panel. The parent comparison's paired interval crossed zero, so the small prediction gain does not establish predictive superiority over NeuroMax. EnCOT and FASTopic retained slightly higher topic-diversity point means.

The same selected recipe was refitted on an independent English transit-experience corpus with 559 text groups and 112 eligible test completion groups. Proposed obtained fitting C_v of {fmt(e['fit_c_v_mean'],e['fit_c_v_sd'])}, fitting NPMI of {fmt(e['fit_npmi_mean'],e['fit_npmi_sd'])}, and completion NLL of {fmt(e['test_nll_mean'],e['test_nll_sd'],6)}. Relative to the matched NeuroMax parent, fitting C_v increased by {e['fit_c_v_mean']-n['fit_c_v_mean']:.6f} and NPMI by {e['fit_npmi_mean']-n['fit_npmi_mean']:.6f}, while NLL increased by {e['test_nll_mean']-n['test_nll_mean']:.6f}. The paired native-minus-Proposed NLL interval crossed zero. EnCOT had the largest C_v and diversity means, LDA the largest NPMI mean, and low-beta GSDMM the smallest NLL mean. GSDMM occupied fewer than ten topics in most runs; its native prediction is retained while fixed-K descriptor scores are marked unavailable.

All five Proposed projections were deployed on each corpus and independently verified from their saved affinity arrays. Selected leading-word membership was fully realized, and topic/word totals were conserved to numerical tolerance. Same-core display only had identical descriptors but retained the native predictor, separating descriptor acquisition from realization in the actual readout.

Fitting coherence is descriptive because lexical constraints were acquired on the fitting corpus. Complete five-seed held-out coherence was unavailable on both corpora and no absent words or topics were removed. Only 49.5% of external test lexical occurrences were in the fitting vocabulary; completion loss is conditional on those common in-vocabulary targets. The external evaluation therefore supports a limited recipe replication, not universal superiority, zero-shot parameter transfer, or requirement-detection accuracy. Five seeds used one fixed split per corpus. Pairwise bootstrap intervals are unadjusted descriptive summaries.
'''
    write_text(folder/'RESULTS_TEXT.md',text)
    write_text(folder/'METHOD_IMPLEMENTATION_ALIGNMENT.md','''# Align the manuscript with the executed method

The projection contribution remains: enforce corpus-derived TOP10 membership in the actual trained affinity, preserve each native topic-row and word-column total, and compute a minimum-TV feasible affinity without gradient updates of neural parameters. A prediction-aware range secondary is solved within the declared near-minimum-TV cap. Do not claim a unique primary optimum or an exact minimum-TV secondary solution when the declared slack is nonzero.

The executed acquisition is a methodological revision. It uses local-window signed associations, fitting-only independent duplicate-group support, subsample stability, native-topic neighborhoods, complete ten-word-set scoring and constrained replacements. It preserves the native union-diversity floor. It does not use the original positive-only document graph, spectral partition into ten disjoint communities, or enforce absolute 100-word disjointness in Proposed. Do not describe these substantive changes as implementation fixes.

The final base is NeuroMax with random-word initialization. EnCOT remains a competitor. Proposed disables moment correction; the separate control retains it. External refitting uses English surface tokenization and the analogous pinned English BGE encoder; the selected model/graph/projection settings and budgets are shared. This is recipe replication, not transfer of Chinese parameters or vocabulary.

Primary topic diversity below 1.0 and external diversity of 0.824 mean absolute word disjointness is not a property of the executed final model. The strict-disjointness control has a failed seed and must remain visible. The alternate signed graph has better primary fitting coherence, whereas the selected variant has a smaller prediction-loss mean; describe this as a tradeoff.

Backoff was selected on development inputs/targets only. The primary selected coefficient is zero and the external selected coefficient is 0.2. Fitting coherence uses windows 110 for C_v and 10 for NPMI. Test NLL uses frozen occurrence-completion masks and observed-only inference. NMF reports normalized reconstruction cross-entropy; BERTopic has no native word likelihood.
''')

def examples(audit):
    result=[]
    for dataset,key in [('Primary','primary_data'),('External','external_data')]:
        row=next(r for r in audit['rows']if r['dataset']==dataset and not r['control']and r['method']=='Proposed'and r['seed']==809)
        fitting=[r for _,r in sorted(audit[key]['features'].items())if r['split']=='fit']
        for topic,words in enumerate(row['topics']):
            scored=sorted([(len(set(words)&set(r['tokens'])),r['group_id'],r)for r in fitting],key=lambda x:(-x[0],x[1]))
            for count,g,r in scored[:2]:
                result.append({'dataset':dataset,'seed':809,'topic_index':topic,'descriptor_words':' | '.join(words),
                    'group_id':g,'source':r['source'],'descriptor_words_present':count,
                    'fitting_text':r['clean_text'],'selection':'first fixed seed; highest distinct-word overlap; group-ID tie break',
                    'scope':'fitting-only illustration; no validated semantic label, accuracy, or representativeness claim'})
    return result

def export(folder,bundle,audit):
    main=folder/'main';supp=folder/'supplementary';prov=folder/'provenance'
    for p in [main,supp,prov,main/'figures',supp/'figures']:p.mkdir(parents=True,exist_ok=True)
    datasets=preparation_table(bundle)
    # Exact quantities are also preserved in an unabridged supplementary CSV.
    write_csv(supp/'S01_dataset_characteristics.csv',datasets,list(datasets[0]))
    readable=[{'dataset':'Primary (Chinese)'if r['sources']==3 else 'External (English)','groups':r['independent_text_groups'],'fit':r['fit_groups'],
        'development':r['development_groups'],'test':r['test_groups'],'completion_test':r['test_completion_groups'],
        'vocabulary':r['fitting_vocabulary'],'test_IV':f"{100*r['test_lexical_coverage']:.1f}%"}for r in datasets]
    plain_table(main,'T1_dataset_characteristics',readable,[(k,h)for k,h in [('dataset','Corpus'),('groups','Groups'),
        ('fit','Fit'),('development','Dev'),('test','Test'),('completion_test','Test completion'),('vocabulary','V'),('test_IV','Test IV')]],
        'Corpus characteristics after independent-group preparation. City names denote collection sources. IV is fitting-vocabulary lexical coverage.')
    summary=audit['summary']
    comparison_table(main,'T2_primary_comparison',[r for r in summary if r['dataset']=='Primary'and not r['control']],
        'Primary corpus comparison across five matched seeds (mean and sample SD). Coherence uses the common fitting reference; completion loss uses the frozen test.')
    comparison_table(main,'T3_primary_controls',[r for r in summary if r['dataset']=='Primary'and r['control']],
        'Primary matched controls. The range control aliases Proposed; the zero-backoff control also coincides because the selected primary backoff is zero. The failed strict-disjointness seed remains NA.')
    comparison_table(main,'T4_external_replication',[r for r in summary if r['dataset']=='External'and not r['control']],
        'Fixed-recipe external replication on English transit tweets. Native variable-topic profiles remain; undefined fixed-K descriptors are NA. This is refitting, not Chinese-weight transfer.')
    comparison_table(supp,'S02_external_controls',[r for r in summary if r['dataset']=='External'and r['control']],
        'Matched external controls. All three controls have the same selected descriptors as Proposed; their native predictive paths differ.')
    write_csv(supp/'S03_paired_uncertainty.csv',paired_table(audit),list(paired_table(audit)[0]))
    write_json(supp/'S03_paired_uncertainty.json',audit['intervals'])
    availability=[{k:r[k]for k in ['dataset','method','profile','seed','control','status','fixed_k10','native_topic_count',
        'fit_c_v','fit_npmi','td','test_nll','test_c_v','test_npmi','supported_slots','total_slots']}for r in audit['rows']]
    write_csv(supp/'S04_metric_availability_all_slots.csv',availability,list(availability[0]))
    topic_rows=[{'dataset':r['dataset'],'method':r['method'],'profile':r['profile'],'seed':r['seed'],'control':r['control'],
        'topic_index':i,'words':' | '.join(words)}for r in audit['rows']for i,words in enumerate(r['topics']or [])]
    write_csv(supp/'S05_all_descriptors.csv',topic_rows,list(topic_rows[0]))
    certs=audit['certificates'];columns=sorted({k for r in certs for k in r})
    write_csv(supp/'S06_actual_projection_certificates.csv',[{k:r.get(k)for k in columns}for r in certs],columns)
    write_json(prov/'array_and_coherence_replay.json',{k:audit[k]for k in ['prediction_checks','coherence_checks','bootstrap_checks','max_prediction_error','verified_model_slots','new_fits','new_encoder_calls']})
    losses=[{'dataset':r['dataset'],'method':r['method'],'profile':r['profile'],'seed':r['seed'],'control':r['control'],**loss}
            for r in audit['rows']for loss in r['document_losses']]
    write_csv(supp/'S07_all_document_losses.csv',losses,list(losses[0]))
    source=[]
    for r in audit['rows']:
        for name,value in r['original'].get('prediction',{}).get('by_source',{}).items():
            source.append({'dataset':r['dataset'],'method':r['method'],'profile':r['profile'],'seed':r['seed'],
                'control':r['control'],'collection_source':name,'nll':value})
    write_csv(supp/'S08_source_level_prediction.csv',source,list(source[0]))
    illustrative=examples(audit)
    write_csv(supp/'S09_automatic_fitting_examples.csv',illustrative,list(illustrative[0]))
    sensitivity=bundle['primary']/'pretest_development_graph_projection_sensitivity.csv'
    shutil.copy2(sensitivity,supp/'S10_pretest_four_recipe_comparison.csv')
    shutil.copy2(bundle['primary']/'pretest_sensitivity_scope.json',supp/'S10_sensitivity_scope.json')
    write_text(supp/'S10_sensitivity_scope.md','This is the saved four-configuration graph/projection comparison on fitting/development data, made before test exposure. It is not a new hyperparameter sweep and does not establish a parameter optimum. No test-based sensitivity curve is generated.\n')
    write_json(supp/'S11_fixed_profiles_and_hyperparameters.json',{'primary_configs':bundle['configs'],
        'external_recipe':bundle['external_identity']['policy'],'external_encoder':read_json(bundle['external']/'encoder_manifest.json')})
    # Preserve earlier source exports rather than overwriting their records.
    for name in ['quality_gates.json','step05_review_decision.json','report_completion_audit.json']:
        shutil.copy2(bundle['primary']/name,prov/('primary_'+name))
    for dataset,src,selected in [('primary',bundle['primary'],['run_result.json','protocol_resolved_before_test.json']),
        ('external',bundle['external'],['run_result.json','protocol_resolved_before_training.json','external_test_freeze.json'])]:
        for name in selected:shutil.copy2(src/name,prov/(dataset+'_'+name))
    write_json(prov/'integrity_verification.json',bundle['integrity'])
    decision=claims(audit);write_json(folder/'claim_decisions.json',decision)
    write_claims(folder,decision);paper_text(folder,audit)
    write_json(supp/'S12_point_mean_winners.json',decision['point_winners'])
    write_text(folder/'RESULTS_INDEX.md','''# Recommended results placement

Main: T1 corpus characteristics; T2 primary comparison; T3 matched primary controls; T4 bounded external replication; F1 source-level prediction heterogeneity; F2 actual rank realization and affinity/prediction disturbance.

Supplement: exact corpus counts and vocabulary coverage; external controls; paired intervals; all metric availability slots; all descriptors; projection certificates; document losses; source losses; automatic fitting examples; the already completed four-recipe development comparison; fixed profiles/hyperparameters; point-mean winners; source provenance.

Figures expose source distributions, actual ranks and per-seed disturbance, rather than repeating aggregate table values. Fit coherence is labeled everywhere. All complete-seed held-out coherence remains NA; its per-seed support/availability is in S04. Review METHOD_IMPLEMENTATION_ALIGNMENT.md and SUPPORTED_CLAIMS.md before writing the final manuscript.

Step 8 completes experiment assembly. It does not automatically approve journal submission or add missing human/application accuracy evidence. Booktabs is required by the LaTeX tables. Reproduce full calculations with the saved model/data artifacts, which remain in the existing project; this paper package is an export, not a replacement for checkpoints.
''')
    from .figures import figures
    figures(folder,bundle,audit)
    return decision
