"""Scientific test diagnostics that expose heterogeneity beyond mean tables."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..runtime.artifacts import write_csv, write_json
from ..evaluation.competitor_evidence import paired_groups


def make_figures(output, rows, controls, seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);folder=output/'figures';folder.mkdir(parents=True,exist_ok=True)
    pro=[r for r in rows if r['method']=='Proposed'];parent=[r for r in rows if r['method']=='NeuroMax']
    paired=paired_groups(pro,parent,seeds);write_csv(folder/'source_group_prediction_gains.csv',paired,list(paired[0]))
    sources=sorted({r['source'] for r in paired});values=[]
    for source in sources:
        groups=sorted({r['group_id'] for r in paired if r['source']==source})
        values.append([float(np.mean([r['gain_per_token'] for r in paired if r['group_id']==g])) for g in groups])
    fig,ax=plt.subplots(figsize=(7,4.4))
    ax.boxplot(values,tick_labels=sources,showfliers=True,medianprops={'color':'#157c69'})
    ax.axhline(0,color='black',linestyle='--',linewidth=1)
    ax.set_ylabel('Native NeuroMax − Proposed\nmean paired group NLL per token')
    ax.set_xlabel('Primary test source; each point represents one independent group')
    ax.set_title('Held-out prediction gains vary across independent groups')
    fig.tight_layout()
    for extension in ['png','pdf','svg']:fig.savefig(folder/('F6_source_prediction_heterogeneity.'+extension),dpi=300)
    plt.close(fig)
    mechanism=[]
    for seed in seeds:
        p=next(r for r in pro if r['seed']==seed);n=next(r for r in parent if r['seed']==seed)
        audit=p['mechanism']
        mechanism.append({'seed':seed,'relative_l1_affinity_movement':audit['relative_l1_affinity_movement'],
            'test_native_minus_proposed_macro_nll':n['prediction']['macro_source_nll']-p['prediction']['macro_source_nll'],
            'top10_membership_recall':audit['target_top10_membership_recall'],
            'maximum_topic_total_error_scaled':audit['maximum_topic_total_error_scaled'],
            'maximum_word_total_error_scaled':audit['maximum_word_total_error_scaled']})
    write_csv(folder/'actual_projection_seed_tradeoffs.csv',mechanism,list(mechanism[0]))
    fig,ax=plt.subplots(figsize=(6.6,4.4))
    for row in mechanism:
        x,y=100*row['relative_l1_affinity_movement'],row['test_native_minus_proposed_macro_nll']
        ax.scatter(x,y,s=65,color='#157c69');ax.annotate(str(row['seed']),(x,y),xytext=(5,5),textcoords='offset points')
    ax.axhline(0,color='black',linestyle='--',linewidth=1)
    ax.set_xlabel('Actual readout L1 movement / native total (%)')
    ax.set_ylabel('Native − Proposed test macro NLL')
    ax.set_title('Realized constraints and their held-out predictive cost')
    fig.tight_layout()
    for extension in ['png','pdf','svg']:fig.savefig(folder/('S6_actual_projection_tradeoff.'+extension),dpi=300)
    plt.close(fig)
    write_json(folder/'FIGURE_CAPTIONS.json',{
        'F6_source_prediction_heterogeneity':'Each distribution contains independent test duplicate groups; a group value averages the five paired training seeds. Boxes show group distributions, not five independent document replications. Positive values favor Proposed. This diagnostic exposes source-level heterogeneity beyond the macro mean.',
        'S6_actual_projection_tradeoff':'Each point is one frozen training seed. Horizontal position measures the actual certified affinity movement, not a display-only descriptor edit. Vertical position is native minus Proposed macro-source test NLL. All five seeds remain; negative values favor native. Constraint certificates are provided separately. No test-driven selection was performed.'})


def fitting_cases(output, controls, data):
    row=next(r for r in controls if r['role']=='proposed' and r['seed']==809)
    examples=[]
    for k,topic in enumerate(row['topics']):
        words=set(topic)
        ranked=sorted(data['fit_records'],key=lambda r:(-len(words&set(r['tokens'])),r['group_id']))
        for record in ranked[:3]:
            examples.append({'topic_id':k+1,'descriptor_words':' '.join(topic),'source':record['source'],
                'group_id':record['group_id'],'distinct_descriptor_words_present':len(words&set(record['tokens'])),
                'fitting_text':record['clean_text'],'scope':'automatic_fitting_only_illustration_no_accuracy_claim'})
    export=Path(output)/'automatic_fitting_cases'
    write_csv(export/'topic_examples.csv',examples,list(examples[0]))
    write_json(export/'topic_examples.json',examples)
    from ..runtime.artifacts import write_text
    write_text(export/'README.md','These deterministic examples come from fitting groups only and rank documents by distinct descriptor-word coverage. They illustrate candidate topics. They are not labels, independent validation, requirement-detection accuracy, or a human interpretability assessment. Seed 809 was fixed before test evaluation; no favorable examples were selected from test or external data.\n')
