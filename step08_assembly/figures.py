"""Publication figures with per-group/per-seed evidence beyond table means."""
from __future__ import annotations
import shutil
import numpy as np
from rank_refinement.runtime.artifacts import write_csv,write_json
from .audit import arrays,descriptor_indices

def figures(folder,bundle,audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main=folder/'main/figures';supp=folder/'supplementary/figures'
    captions={}
    original=bundle['primary']/'figures'
    for suffix in ['.pdf','.png','.svg']:
        shutil.copy2(original/('F6_source_prediction_heterogeneity'+suffix),main/('F1_primary_source_heterogeneity'+suffix))
    captions['F1_primary_source_heterogeneity']=('Independent primary test-group prediction differences by collection source. '
        'Each group averages the five paired seeds; boxes show group distributions, not five independent document replications. '
        'Positive values favor Proposed. Reuses the verified Step 6 figure without score or sample changes.')
    ranks=[]
    for dataset,case,native_key,proposed_key in [
        ('Primary',bundle['primary']/'proposed_control_cases/seed809','native','proposed'),
        ('External',bundle['external']/'cases/proposed_parent/seed809','readout_native','readout_proposed')]:
        a=arrays(case/('frozen_test_arrays.npz'if dataset=='Primary'else 'predictive_arrays.npz'))
        vocabulary=audit['primary_data'if dataset=='Primary'else 'external_data']['vocabulary']
        for topic,(native,phi)in enumerate(zip(a[native_key],a[proposed_key])):
            order0=np.lexsort((np.arange(len(native)),-native));order1=np.lexsort((np.arange(len(phi)),-phi))
            inv0=np.empty(len(native),dtype=int);inv1=np.empty(len(phi),dtype=int)
            inv0[order0]=np.arange(1,len(native)+1);inv1[order1]=np.arange(1,len(phi)+1)
            entrants=sorted(set(order1[:10])-set(order0[:10]))
            for w in entrants:ranks.append({'dataset':dataset,'seed':809,'topic_index':topic,'word':vocabulary[w],
                                            'native_rank':int(inv0[w]),'proposed_rank':int(inv1[w])})
    if not ranks:raise ValueError('Reviewed projections were expected to edit genuine TOP10 membership.')
    write_csv(supp/'F2_exact_rank_changes.csv',ranks,list(ranks[0]))
    disturbances=[]
    for cert in audit['certificates']:
        pro=next(r for r in audit['rows']if r['dataset']==cert['dataset']and not r['control']and r['method']=='Proposed'and r['seed']==cert['seed'])
        native=next(r for r in audit['rows']if r['dataset']==cert['dataset']and not r['control']and r['method']=='NeuroMax'and r['seed']==cert['seed'])
        disturbances.append({'dataset':cert['dataset'],'seed':cert['seed'],
            'relative_l1_movement_percent':100*cert['relative_l1_movement'],
            'proposed_minus_native_test_nll':pro['test_nll']-native['test_nll'],
            'fitting_c_v_gain':pro['fit_c_v']-native['fit_c_v'],
            'fitting_npmi_gain':pro['fit_npmi']-native['fit_npmi']})
    write_csv(supp/'F2_exact_seed_disturbance.csv',disturbances,list(disturbances[0]))
    with plt.rc_context({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42}):
        fig,axes=plt.subplots(1,2,figsize=(10.6,4.1))
        colors={'Primary':'#225b79','External':'#b36b24'};markers={'Primary':'o','External':'^'}
        ax=axes[0];maxrank=max(r['native_rank']for r in ranks)+2
        ax.plot([0,maxrank],[0,maxrank],color='#888888',ls=':',lw=1)
        ax.axhline(10.5,color='#888888',ls='--',lw=1);ax.axvline(10.5,color='#888888',ls='--',lw=1)
        for dataset in ['Primary','External']:
            selected=[r for r in ranks if r['dataset']==dataset]
            ax.scatter([r['native_rank']for r in selected],[r['proposed_rank']for r in selected],
                       color=colors[dataset],marker=markers[dataset],s=34,alpha=.75,
                       label=f'{dataset}: {len(selected)} entrants')
        ax.set_xlim(0,maxrank);ax.set_ylim(0,12);ax.set_xlabel('Native word rank');ax.set_ylabel('Projected word rank')
        ax.set_title('(a) Actual TOP10 entrants, fixed seed 809');ax.legend(fontsize=8,loc='lower right');ax.grid(alpha=.16)
        ax=axes[1]
        for dataset in ['Primary','External']:
            selected=[r for r in disturbances if r['dataset']==dataset]
            ax.scatter([r['relative_l1_movement_percent']for r in selected],
                [r['proposed_minus_native_test_nll']for r in selected],color=colors[dataset],marker=markers[dataset],s=38,label=dataset)
            for r in selected:ax.annotate(str(r['seed']),(r['relative_l1_movement_percent'],r['proposed_minus_native_test_nll']),
                xytext=(4,3),textcoords='offset points',fontsize=7)
        ax.axhline(0,color='#777777',ls='--',lw=1);ax.set_xscale('log')
        ax.margins(x=.15,y=.12)
        ax.set_xlabel('Actual relative L1 affinity movement (%)')
        ax.set_ylabel('Proposed − native test NLL (nats/token)')
        ax.set_title('(b) All five seeds in each corpus');ax.legend(fontsize=8);ax.grid(alpha=.16)
        fig.tight_layout()
        for suffix in ['.pdf','.png','.svg']:
            kwargs={'dpi':240}if suffix=='.png'else {}
            fig.savefig(main/('F2_actual_rank_and_prediction_disturbance'+suffix),**kwargs)
        plt.close(fig)
    captions['F2_actual_rank_and_prediction_disturbance']=('Panel (a) shows every newly entering TOP10 word in the predeclared first seed, 809, '
        'without selecting topics by score. Identical coordinates can overlap; the CSV retains every entry. Panel (b) retains all five seeds per corpus '
        'and compares conditional prediction changes with actual affinity movement. Negative vertical values favor Proposed. '
        'The horizontal axis is logarithmic. These are per-seed observations, not confidence intervals or a new parameter sweep.')
    for src,dest in [('F7_external_prediction_intervals','S1_external_paired_intervals'),
                     ('S7_external_reference_support','S2_external_coverage_and_descriptor_support')]:
        for suffix in ['.pdf','.png']:
            shutil.copy2(bundle['external']/'figures'/(src+suffix),supp/(dest+suffix))
    captions['S1_external_paired_intervals']='Saved external paired seed/group 95% intervals. Positive comparator-minus-Proposed values favor Proposed. Unadjusted descriptive intervals; unavailable BERTopic NLL is not a win.'
    captions['S2_external_coverage_and_descriptor_support']='Saved fitting-vocabulary coverage and retained descriptor-slot support. No absent words or topics were removed to improve coherence.'
    write_json(folder/'FIGURE_CAPTIONS.json',captions)
