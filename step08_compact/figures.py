"""Exact scientific figures with shared scales and saved selection provenance."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from rank_refinement.runtime.artifacts import write_json,write_csv,read_json


def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','pdf','svg']:
        fig.savefig(path.with_suffix('.'+suffix),dpi=300,bbox_inches='tight')
    plt.close(fig)


def draw(folder,details,recipe_path,assembly):
    captions={}
    for dataset in ['Primary','External']:
        c=details[(dataset,809)];a,b=c['a'],c['b'];delta=b-a
        # Display selection uses the first fixed seed's fitting-derived change,
        # no coherence score or test-loss ranking. Full matrices are exported.
        movement=np.abs(delta).sum(axis=0)
        ids=np.lexsort((np.arange(a.shape[1]),-movement))[:24]
        chinese_font=None
        if dataset=='Primary':
            for family in ['Microsoft YaHei','SimHei','SimSun','Noto Sans CJK SC','Noto Sans CJK JP','WenQuanYi Zen Hei']:
                try:
                    found=font_manager.findfont(family,fallback_to_default=False)
                    chinese_font=font_manager.FontProperties(fname=found)
                    break
                except ValueError:
                    continue
        labels=[f'w{i:04d}'for i in ids] if dataset=='Primary' and chinese_font is None else [c['vocabulary'][i]for i in ids]
        lookup=[{'display_order':j,'word_id':int(i),'label':labels[j],'word':c['vocabulary'][i],
                 'summed_absolute_affinity_change':float(movement[i])}for j,i in enumerate(ids)]
        target_dir=folder/('main/figures'if dataset=='Primary'else'supplementary/figures')
        write_csv(target_dir/('F1_'+dataset.lower()+'_word_lookup.csv'),lookup,list(lookup[0]))
        maxvalue=max(float(a[:,ids].max()),float(b[:,ids].max()));dmax=float(np.abs(delta[:,ids]).max())
        fig,axes=plt.subplots(3,1,figsize=(11.4,8.4),layout='constrained')
        names=['Native NeuroMax affinity','Proposed affinity','Proposed minus native affinity']
        for j,(ax,data,title)in enumerate(zip(axes,[a,b,delta],names)):
            im=ax.imshow(data[:,ids],aspect='auto',cmap='viridis'if j<2 else'RdBu_r',
                vmin=0 if j<2 else-dmax,vmax=maxvalue if j<2 else dmax,interpolation='nearest')
            ax.set_title(title,loc='left',fontsize=11);ax.set_ylabel('Matched topic row')
            ax.set_yticks(range(10),range(10));ax.set_xticks(range(len(ids)),labels,rotation=55,ha='right',fontsize=8)
            if chinese_font:
                for label in ax.get_xticklabels():label.set_fontproperties(chinese_font);label.set_fontsize(8)
            fig.colorbar(im,ax=ax,fraction=.027,pad=.012,label='Raw affinity'if j<2 else'Raw change')
        fig.suptitle(dataset+' corpus · fixed seed 809 · 24 fitting-derived changed word columns',fontsize=13)
        axes[-1].set_xlabel('Before/after use the same color scale; delta uses a separate symmetric scale')
        name='F1_affinity_before_after' if dataset=='Primary' else 'S_F1_external_affinity_before_after'
        save(fig,target_dir/name)
        captions[name]=dict(seed=809,display_word_count=24,full_vocabulary=a.shape[1],
            word_labels='Chinese words using installed CJK font'if chinese_font else'word IDs for primary; English words for external',
            before_after_shared_color_scale=True,delta_symmetric_color_scale=True,
            word_selection='highest fitting-derived column L1 change; word-ID tie break; no test ranking',
            caption='Matched native and Proposed affinity with signed difference, first fixed seed. '+
            'Only 24 word columns are displayed; every full matrix and exact vocabulary mapping is saved. '+
            'Chinese words use an installed CJK font when available; otherwise stable word IDs map to the supplied exact lookup. '+
            'Topic rows are matched within the parent; rows across different seeds are not aligned or averaged.')
    payload=read_json(recipe_path);rows=payload['rows']
    from .report import read_csv
    historical=read_csv(assembly/'supplementary/S10_pretest_four_recipe_comparison.csv')
    for h in historical:
        graph,mode=h['method'].split('/')
        matches=[r for r in rows if r['graph']==graph and r['projection']==mode]
        for new,old in [('development_nll','development_macro_nll_mean'),('fit_c_v','fitting_c_v_mean'),('fit_npmi','fitting_c_npmi_mean')]:
            if len(matches)!=5 or not np.isclose(np.mean([r[new]for r in matches]),float(h[old]),atol=1e-12,rtol=0):
                raise ValueError('Pre-test per-seed figure values disagree with the verified original means.')
    expected={(s,g,m)for s in payload['seeds']for g in ['signed_stable','signed_full']for m in ['primary','range']}
    if len(rows)!=20 or {(r['seed'],r['graph'],r['projection'])for r in rows}!=expected or any(r['status']!='scored'for r in rows):
        raise ValueError('Pre-test graph/solver diagnostic must retain all twenty cases.')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4),layout='constrained')
    colors={'signed_stable':'#2472A4','signed_full':'#DB7843'}
    for graph in ['signed_stable','signed_full']:
        for i,seed in enumerate(payload['seeds']):
            r=sorted([r for r in rows if r['graph']==graph and r['seed']==seed],key=lambda r:r['projection'])
            x=np.array([0,1])+(-.04 if graph=='signed_stable'else.04)
            for ax,field,factor in [(axes[0],'development_nll',1),(axes[1],'relative_l1_movement',100)]:
                ax.plot(x,[v[field]*factor for v in r],color=colors[graph],alpha=.65,lw=1,
                    marker='o'if graph=='signed_stable'else's',ms=4,
                    label=('Stable signed graph'if graph=='signed_stable'else'Full signed graph')if i==0 else None)
    for ax in axes:
        ax.set_xticks([0,1],['Primary TV','Range secondary']);ax.grid(axis='y',alpha=.22)
    axes[0].set_ylabel('Development macro-source NLL (nats/token)');axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Relative affinity L1 movement (%)')
    fig.suptitle('Graph/solver sensitivity · five matched seeds · fixed rank margin 10⁻⁶',fontsize=12)
    save(fig,folder/'main/figures/F2_graph_solver_sensitivity')
    write_json(folder/'numerical_evidence/pretest_recipe_cases.json',payload)
    captions['F2_graph_solver_sensitivity']=dict(
        caption='Twenty pre-test development cases: two association graphs, two projections and all five seeds. '+
        'Each line joins one matched seed under the same graph. The plots show development prediction and '+
        'affinity movement, unlike the final test ablation table. Rank margin is fixed at 10^-6. '+
        'This is graph/solver sensitivity, not a support-threshold, gamma, rank-margin or data-perturbation sweep.',
        scope='pretest_development_diagnostic',source_sha256=payload['source_sha256'],new_parameter_selection=False)
    write_json(folder/'FIGURE_CAPTIONS.json',captions)
