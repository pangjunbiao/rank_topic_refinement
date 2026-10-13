"""Exact primary affinity heatmaps; English glosses never change vocabulary."""
from __future__ import annotations
import numpy as np
from .metrics import changed_columns


def draw(folder, native, proposed, vocabulary, glossary, seed=809):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    ids = changed_columns(native, proposed)
    lookup = []
    for j, wid in enumerate(ids):
        word = vocabulary[int(wid)]
        gloss = glossary.get(word, {})
        lookup.append({'display_column': j+1, 'word_id': int(wid), 'original_token': word,
                       'english_label': gloss.get('label', f'w{wid:04d}'),
                       'translation_note': gloss.get('note', 'No English gloss supplied; exact word ID retained.'),
                       'summed_absolute_affinity_change': float(np.abs(proposed[:,wid]-native[:,wid]).sum())})
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7.8,
        'ytick.labelsize': 7.8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.linewidth': .6, 'svg.hashsalt': 'primary-affinity-fixed-seed-809'})
    fig = plt.figure(figsize=(8.25,5.8), layout='constrained')
    grid = fig.add_gridspec(3,2,width_ratios=[1,.02],hspace=.08,wspace=.025)
    axes = [fig.add_subplot(grid[j,0]) for j in range(3)]
    delta = proposed-native
    max_raw = float(max(native[:,ids].max(), proposed[:,ids].max()))
    max_delta = float(np.abs(delta[:,ids]).max())
    if max_delta == 0:
        raise ValueError('There are no adjustments to display; do not invent a change scale.')
    titles = ['(a) Native NeuroMax: before refinement', '(b) Proposed: after refinement', '(c) Signed change: Proposed minus native']
    for j, (ax, matrix, title) in enumerate(zip(axes,[native,proposed,delta],titles)):
        if j < 2:
            im = ax.imshow(matrix[:,ids],aspect='auto',cmap='viridis',vmin=0,vmax=max_raw,interpolation='nearest')
        else:
            im = ax.imshow(matrix[:,ids],aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-max_delta,vcenter=0,vmax=max_delta),interpolation='nearest')
        ax.set_title(title,loc='left',pad=5)
        ax.set_yticks(range(10),[f'T{k+1:02d}' for k in range(10)])
        ax.set_ylabel('Matched topic')
        ax.set_xticks(range(len(ids)))
        if j == 2:
            ax.set_xticklabels([r['english_label'] for r in lookup],rotation=55,ha='right',rotation_mode='anchor')
            ax.set_xlabel('English glosses of the same 24 Chinese tokens',labelpad=4)
        else:
            ax.tick_params(axis='x',bottom=False,labelbottom=False)
        ax.tick_params(length=2.5,pad=3)
        if j == 1:
            bar=fig.colorbar(im,cax=fig.add_subplot(grid[:2,1]));bar.set_label('Raw affinity (shared scale)')
        elif j == 2:
            bar=fig.colorbar(im,cax=fig.add_subplot(grid[2,1]));bar.set_label('Affinity change')
    fig.suptitle('Primary Chinese corpus: matched affinity adjustment',fontsize=11,fontweight='medium')
    fig.get_layout_engine().set(rect=(0,.052,1,.955))
    fig.text(.05,.027,f'Fixed seed {seed}; 24 most-changed fitting-derived columns of {len(vocabulary):,}. All 10 topic rows are shown.',fontsize=7)
    fig.text(.05,.007,'Red / blue indicate increases / decreases, not improvements / deterioration. Exact Chinese tokens are mapped separately.',fontsize=6.9)
    folder.mkdir(parents=True,exist_ok=True)
    name=folder/'F1_primary_affinity_before_after_English'
    metadata={'Title':'Primary native and Proposed affinity: English-labelled comparison', 'CreationDate':None,'ModDate':None}
    fig.savefig(name.with_suffix('.pdf'),metadata=metadata)
    fig.savefig(name.with_suffix('.png'),dpi=300)
    fig.savefig(name.with_suffix('.svg'),metadata={'Date':None})
    plt.close(fig)
    return lookup
