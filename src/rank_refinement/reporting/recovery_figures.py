"""Per-seed differences, rather than another rendering of aggregate tables."""
from pathlib import Path
from ..runtime.artifacts import write_csv


def plot_recovery(output, rows, seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    points=[]
    roles={'proposed':'Proposed','primary_control':'Primary TV','range_control':'Range secondary',
           'alternate_calibration':'Alternate graph','positive_only_association':'Positive only'}
    for seed in seeds:
        native=next(r for r in rows if r['role']=='native' and r['seed']==seed)
        for role,label in roles.items():
            row=next(r for r in rows if r['role']==role and r['seed']==seed)
            if row['status']!='scored':continue
            points.append({'seed':seed,'role':role,'label':label,
                'npmi_gain_over_native':row['fitting']['c_npmi']-native['fitting']['c_npmi'],
                'development_nll_gain_over_native':native['prediction']['macro_source_nll']-row['prediction']['macro_source_nll']})
    folder=Path(output)/'figures';folder.mkdir(parents=True,exist_ok=True)
    write_csv(folder/'seed_tradeoffs.csv',points,list(points[0]))
    merged=False
    pairs={seed:{r['role']:r for r in points if r['seed']==seed} for seed in seeds}
    if all('proposed' in pair and 'range_control' in pair and all(abs(pair['proposed'][k]-pair['range_control'][k])<1e-12
        for k in ['npmi_gain_over_native','development_nll_gain_over_native']) for pair in pairs.values()):
        roles.pop('range_control');roles['proposed']='Proposed (= range control)';merged=True
    fig,ax=plt.subplots(figsize=(8,5))
    for role,label in roles.items():
        part=[p for p in points if p['role']==role]
        if not part:continue
        ax.scatter([p['npmi_gain_over_native'] for p in part],[p['development_nll_gain_over_native'] for p in part],label=label,s=40)
        for p in part:
            offset=(4,-12) if role=='alternate_calibration' or (role=='proposed' and p['development_nll_gain_over_native']<0.) else (4,7)
            ax.annotate(str(p['seed']),(p['npmi_gain_over_native'],p['development_nll_gain_over_native']),xytext=offset,textcoords='offset points',fontsize=7)
    ax.axhline(0,color='gray',lw=.8);ax.axvline(0,color='gray',lw=.8)
    ax.set_xlabel('Fitting NPMI gain over the same-seed native parent')
    ax.set_ylabel('Development NLL gain over the same-seed native parent')
    ax.set_title('Step 5 recovery: every scored seed and retained losses')
    ax.legend(fontsize=8,frameon=False);ax.spines[['top','right']].set_visible(False)
    note='Exploratory development selection. Failed/NA cases remain in tables; no final test results.'
    if merged:note+=' Identical range control merged.'
    fig.text(.02,.01,note,fontsize=7)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(folder/'seed_tradeoffs.png',dpi=220);fig.savefig(folder/'seed_tradeoffs.pdf');plt.close(fig)
