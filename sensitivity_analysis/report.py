from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from .core import csv_write,read,sha,safe


def ablation_readout_recall(module):
    values={}
    for entry in read(module/'data/ablation_inputs.json'):
        path=safe(module,entry['packed_path'])
        if sha(path)!=entry['sha256']:raise ValueError('Ablation mechanism source changed.')
        c=read(path)
        for r in c['mechanism']:
            values.setdefault(r['role'],[]).append(100*r['target_top10_membership_recall'])
    roles={'Proposed (prediction-aware range secondary)':'proposed','Primary minimum-TV only':'primary_control',
           'Index secondary':'index_control','L2 projection':'l2_control','Same-core display only':'display_only'}
    result={}
    for label,role in roles.items():
        x=values[role]
        if len(x)!=5:raise ValueError('Incomplete five-seed ablation mechanism source.')
        result[label]=f'{np.mean(x):.2f} ± {np.std(x,ddof=1):.2f}'
    return result


def fmt(row, metric, digits=6):
    x,y=row.get(metric+'_mean'),row.get(metric+'_sd')
    return 'NA' if x is None else f'{x:.{digits}f} ± {y:.{digits}f}'


def latex_text(value):
    return str(value).replace('\\','\\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('↑','').replace('↓','').replace('±',r'$\pm$')


def latex_table(path,headers,values,caption):
    text=['\\begin{table*}[t]','\\centering','\\small','\\caption{'+latex_text(caption)+'}',
          '\\begin{tabular}{'+'l'*len(headers)+'}','\\hline',
          ' & '.join(latex_text(h) for h in headers)+r' \\',r'\hline']
    text += [' & '.join(latex_text(x) for x in row)+r' \\' for row in values]
    text += [r'\hline',r'\end{tabular}',r'\end{table*}']
    path.write_text('\n'.join(text)+'\n',encoding='utf-8')


def tables(output, rows, aggregate, module):
    folder=output/'tables';folder.mkdir(exist_ok=True)
    csv_write(folder/'sensitivity_individual_runs.csv',rows)
    csv_write(folder/'sensitivity_mean_sd.csv',aggregate)
    lines=['Primary sensitivity diagnostics — five matched seeds','',
           'Fitting C_v/NPMI; fixed-recipe development completion NLL in nats/token. Values are mean ± sample SD. A star marks the original selected parameter values; range is the selected projection objective. This post-exposure diagnostic does not select or replace the frozen final model. NA and failed slots are retained.','',
           '| Study | Support | Scaled margin | Objective | Certified | Fit C_v ↑ | Fit NPMI ↑ | Δ dev NLL vs native ↓ | Affinity L1 (%) ↓ |',
           '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in aggregate:
        lines.append('| '+' | '.join([r['study']+(' *' if r['selected_setting'] else ''),str(r['support']),f"{r['margin']:.0e}",r['objective'],f"{r['certified_seeds']}/5",fmt(r,'cv',4),fmt(r,'npmi',4),fmt(r,'delta_native_nll',6),fmt(r,'relative_l1_percent',5)])+' |')
    (folder/'sensitivity_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    latex_table(folder/'sensitivity_table.tex',['Study','Support','Margin','Objective','Certified','Fit C_v','Fit NPMI','Delta dev NLL','Affinity L1 (%)'],
        [[r['study']+(' *' if r['selected_setting'] else ''),r['support'],f"{r['margin']:.0e}",r['objective'],f"{r['certified_seeds']}/5",fmt(r,'cv',4),fmt(r,'npmi',4),fmt(r,'delta_native_nll',6),fmt(r,'relative_l1_percent',5)] for r in aggregate],
        'Primary sensitivity: five seeds, mean and sample SD; star marks frozen selected values. Coherence uses fitting data; NLL uses development completion.')
    # The reported test-ablation table is copied unchanged from the reviewed Step 8 export.
    source=module/'data/T3_matched_ablations.csv'
    if sha(source)!='3bf33a675c00f31d2289ad3700860de831927789848a318f6952b86af39c8da9':
        raise ValueError('The reviewed frozen-test ablation table changed.')
    with source.open(encoding='utf-8-sig',newline='') as f: ablations=list(csv.DictReader(f))
    recall=ablation_readout_recall(module)
    for r in ablations:r['actual_target_recall_percent']=recall.get(r['variant'],'')
    csv_write(folder/'matched_ablations_complete.csv',ablations)
    lines=['Complete matched ablations — primary corpus','',
           'Five seeds, mean ± sample SD. Coherence is fitting-reference coherence; NLL is frozen-test completion. These existing results are not recomputed or changed by sensitivity analysis.','']
    for panel in ['A: components','B: projection']:
        lines += [panel,'','| Variant | Fit C_v ↑ | Fit NPMI ↑ | TD ↑ | Test NLL ↓ |'+(' Targets in actual readout (%) ↑ |' if panel=='B: projection' else ''),
                  '| --- | ---: | ---: | ---: | ---: |'+(' ---: |' if panel=='B: projection' else '')]
        for r in ablations:
            if r['panel']==panel: lines.append('| '+' | '.join(r[k] for k in ['variant','cv','npmi','td','nll']+(['actual_target_recall_percent'] if panel=='B: projection' else []))+' |')
        lines.append('')
    lines+=['Interpretation: removing stability slightly improves fitting coherence but worsens mean NLL. Display-only shares Proposed descriptors and retains the native predictor. Identical descriptor scores cannot establish an additional coherence benefit from projection. Native-parent versus Proposed is a joint removal, not an isolated acquisition ablation. Selected-word inclusion refers to the top ten affinities in each actual decoder readout row.','',
            'Main-paper recommendation: keep all four component rows and Proposed, primary-TV, index-secondary and display-only in the projection panel. L2 can remain in the supplement when space is limited. Retain all rows in the complete export.']
    (folder/'matched_ablations_complete.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for panel,name in [('A: components','A_components'),('B: projection','B_projection')]:
        fields=['variant','cv','npmi','td','nll']+(['actual_target_recall_percent'] if panel=='B: projection' else [])
        headers=['Variant','Fit C_v','Fit NPMI','TD','Test NLL']+(['Actual target inclusion (%)'] if panel=='B: projection' else [])
        latex_table(folder/('matched_ablations_'+name+'.tex'),headers,[[r[k] for k in fields] for r in ablations if r['panel']==panel],
                    panel+'; primary corpus, five matched seeds, mean and sample SD. Coherence uses fitting data; NLL uses frozen test completion.')
    main=[r for r in ablations if r['variant']!='L2 projection']
    short={'Without acquisition/projection (native parent)':'Native NTM (no refinement)',
           'Without association stability discount (signed full graph)':'w/o stability discount',
           'Without signed negative associations':'w/o signed negative associations',
           'Proposed (prediction-aware range secondary)':'Proposed','Primary minimum-TV only':'Minimum-TV only'}
    lines=['Recommended main-paper ablation table','',
           'Five matched seeds, mean ± sample SD. C_v/NPMI use the common fitting reference; NLL uses frozen-test completion. The native row jointly removes acquisition and projection.','']
    for panel in ['A: components','B: projection']:
        lines += [panel,'','| Variant | Fit C_v ↑ | Fit NPMI ↑ | TD ↑ | Test NLL ↓ |'+(' Actual target inclusion (%) ↑ |' if panel=='B: projection' else ''),
                  '| --- | ---: | ---: | ---: | ---: |'+(' ---: |' if panel=='B: projection' else '')]
        for r in main:
            if r['panel']==panel:
                lines.append('| '+' | '.join([short.get(r['variant'],r['variant'])]+[r[k] for k in ['cv','npmi','td','nll']]+([r['actual_target_recall_percent']] if panel=='B: projection' else []))+' |')
        lines.append('')
    lines+=['Removing stability trades slightly better coherence for worse mean prediction loss. Removing signed negative associations also slightly increases diversity. Preserve these tradeoffs in the manuscript.',
            'Display-only shares the descriptors but retains the native predictor. The actual-inclusion column is computed against the same pre-test selected target sets. L2 remains in the complete supplementary export.']
    (folder/'main_matched_ablations.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def figures(output, aggregate, rows, plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.hashsalt':plan['protocol']})
    folder=output/'figures';folder.mkdir(exist_ok=True)
    def export(fig,name):
        for ext in ['pdf','svg','png']:
            metadata={'Creator':'Frozen primary sensitivity analysis','CreationDate':None,'ModDate':None} if ext=='pdf' else ({'Date':None} if ext=='svg' else None)
            fig.savefig(folder/(name+'.'+ext),dpi=300,bbox_inches='tight',metadata=metadata)
        plt.close(fig)
    support=sorted((r for r in aggregate if r['study']=='support'),key=lambda r:r['support'])
    fig,axes=plt.subplots(2,2,figsize=(8,5.8),layout='constrained')
    for ax,metric,title in zip(axes.ravel(),['cv','npmi','delta_native_nll','relative_l1_percent'],
                             ['(a) Fitting coherence: $C_v$ ↑','(b) Fitting coherence: NPMI ↑','(c) Development NLL change vs native ↓','(d) Total affinity L1 movement (%)']):
        x=[r['support'] for r in support]
        y=[np.nan if r[metric+'_mean'] is None else r[metric+'_mean'] for r in support]
        ax.errorbar(x,y,yerr=[np.nan if r[metric+'_sd'] is None else r[metric+'_sd'] for r in support],fmt='o-',color='#2166ac',capsize=3,label='Mean ± seed SD')
        ax.axvline(plan['selected_support'],color='#b2182b',ls='--',lw=1,label='Selected support = 5')
        for r in support:
            ax.annotate(f"{r['certified_seeds']}/5",(r['support'],r[metric+'_mean'] if r[metric+'_mean'] is not None else 0.),xytext=(0,9),textcoords='offset points',ha='center',fontsize=7)
        if metric=='delta_native_nll':ax.axhline(0,color='gray',lw=.8)
        ax.set(title=title,xlabel='Minimum independent-group pair support',xticks=plan['support_values']);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    axes[1,0].set_ylabel('Δ NLL (nats/token)')
    fig.suptitle('Primary corpus: pair-support sensitivity | labels show certified seeds',fontsize=11)
    export(fig,'R1_primary_support_sensitivity')
    fig,axes=plt.subplots(2,2,figsize=(8,5.8),layout='constrained')
    colors={'range':'#2166ac','primary':'#e08214'}
    for mode in plan['projection_objectives']:
        subset=sorted((r for r in aggregate if r['study']=='margin' and r['objective']==mode),key=lambda r:r['margin'])
        label='Proposed: range secondary' if mode=='range' else 'Minimum TV only'
        for ax,metric in zip(axes.ravel()[:3],['relative_l1_percent','delta_native_nll','fitting_logit_range']):
            x=[r['margin'] for r in subset];y=[np.nan if r[metric+'_mean'] is None else r[metric+'_mean'] for r in subset]
            ax.plot(x,y,'o-' if mode=='range' else '^--',color=colors[mode],label=label)
            # Faint matched-seed traces show variation without obscuring the small between-margin changes.
            for seed in plan['seeds']:
                seedrows=sorted((r for r in rows if r['study']=='margin' and r['objective']==mode and r['seed']==seed),key=lambda r:r['margin'])
                ax.plot([r['margin'] for r in seedrows],[np.nan if r.get(metric) is None else r[metric] for r in seedrows],color=colors[mode],alpha=.15,lw=.6)
            ax.set_xscale('log');ax.axvline(plan['selected_margin'],color='#b2182b',ls='--',lw=.8);ax.grid(alpha=.2)
        axes[1,1].plot([r['margin'] for r in subset],[r['certified_seeds'] for r in subset],'o-' if mode=='range' else '^--',color=colors[mode],label=label)
    titles=['(a) Affinity movement (%)','(b) Development NLL change vs native ↓','(c) Fitting-context logit-range objective ↓','(d) Independently certified readouts']
    for ax,title in zip(axes.ravel(),titles):ax.set(title=title,xlabel='Scaled rank margin')
    axes[0,1].axhline(0,color='gray',lw=.8)
    axes[1,1].set_xscale('log');axes[1,1].set_ylim(-.2,5.4);axes[1,1].set_yticks(range(6));axes[1,1].axvline(plan['selected_margin'],color='#b2182b',ls='--',lw=.8);axes[1,1].grid(alpha=.2)
    axes[0,0].legend(fontsize=7,title='Markers: means; faint lines: individual seeds',title_fontsize=7)
    axes[0,1].set_ylabel('Δ NLL (nats/token)')
    fig.suptitle('Primary corpus: rank-margin tradeoff | selected margin = 10⁻⁶ (red)',fontsize=11)
    export(fig,'R2_primary_margin_tradeoff')
