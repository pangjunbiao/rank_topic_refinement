"""Readable Markdown, exact CSV, and two-column LaTeX tables."""
from __future__ import annotations
import math
from rank_refinement.runtime.artifacts import write_csv,write_text

def label(row):
    if row['method']=='BERTopic':return 'BERTopic (K-means)'if row['profile'].startswith('k10')else 'BERTopic (HDBSCAN)'
    if row['method']=='GSDMM':return 'GSDMM (low beta)'if 'low_beta'in row['profile']else 'GSDMM (default)'
    return row['method']

def fmt(mean,sd,digits=4):
    if mean is None:return 'NA'
    if not math.isfinite(mean)or sd is None or not math.isfinite(sd):raise ValueError('Incomplete reported mean/SD.')
    return f'{mean:.{digits}f} ± {sd:.{digits}f}'

def escape(text):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_',
             '{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(mapping.get(c,c)for c in str(text))

def plain_table(folder,name,rows,columns,caption='',latex=True):
    """columns: (data key, readable heading); no imputed blanks or fake scores."""
    keys=[k for k,_ in columns];headings=[h for _,h in columns]
    write_csv(folder/(name+'.csv'),rows,keys)
    lines=['| '+' | '.join(headings)+' |','| '+' | '.join('---'for _ in keys)+' |']
    for row in rows:
        values=[str(row[k])if row.get(k)is not None else 'NA'for k in keys]
        lines.append('| '+' | '.join(v.replace('|',r'\|').replace('\n',' ')for v in values)+' |')
    write_text(folder/(name+'.md'),(caption+'\n\n'if caption else '')+'\n'.join(lines)+'\n')
    if latex:
        text=[r'\begin{table*}[!t]',r'\centering',r'\caption{'+escape(caption)+r'}',
              r'\label{tab:'+name.lower()+r'}',r'\begingroup',r'\footnotesize',r'\setlength{\tabcolsep}{4pt}',
              r'\begin{tabular}{l'+'r'*(len(keys)-1)+'}',r'\toprule',
              ' & '.join(escape(h)for h in headings)+r' \\',r'\midrule']
        for row in rows:text.append(' & '.join(escape(row[k])if row.get(k)is not None else 'NA'for k in keys)+r' \\')
        text += [r'\bottomrule',r'\end{tabular}',r'\endgroup',r'\end{table*}']
        write_text(folder/(name+'.tex'),'\n'.join(text)+'\n')

def comparison_table(folder,name,rows,caption):
    """Exact numeric CSV plus compact display; best values refer to defined means."""
    ordered=sorted(rows,key=lambda r:(r['method']!='Proposed',label(r)))
    fields=['fit_c_v','fit_npmi','td','test_nll'];direction=[1,1,1,-1]
    best={field:max(sign*r[field+'_mean']for r in ordered if r[field+'_mean']is not None)
          for field,sign in zip(fields,direction)if any(r[field+'_mean']is not None for r in ordered)}
    raw=[];md=['| Model / control | Fit C_v ↑ | Fit NPMI ↑ | TD ↑ | Test NLL / recon. CE ↓ |',
              '|---|---:|---:|---:|---:|']
    tex=[r'\begin{table*}[!t]',r'\centering',r'\caption{'+escape(caption)+r'}',r'\label{tab:'+name.lower()+r'}',
         r'\begingroup',r'\footnotesize',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrr}',r'\toprule',
         r'Model / control & Fit $C_v$ $\uparrow$ & Fit NPMI $\uparrow$ & TD $\uparrow$ & Test NLL / CE $\downarrow$ \\',r'\midrule']
    for row in ordered:
        raw.append({k:v for k,v in row.items()})
        mdvals=[];texvals=[]
        for field,sign in zip(fields,direction):
            value=fmt(row[field+'_mean'],row[field+'_sd'],6 if field=='test_nll'else 4)
            highlight=row[field+'_mean']is not None and abs(sign*row[field+'_mean']-best[field])<=1e-12
            mdvals.append('**'+value+'**'if highlight else value)
            literal='NA'if value=='NA'else r'$'+value.replace(' ± ',r'\,\pm\,')+'$'
            texvals.append(r'$\mathbf{'+value.replace(' ± ',r'\,\pm\,')+'}$'if highlight else literal)
        md.append('| '+label(row)+' | '+' | '.join(mdvals)+' |')
        tex.append(escape(label(row))+' & '+' & '.join(texvals)+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\par\vspace{3pt}',r'\begin{minipage}{0.97\textwidth}',r'\scriptsize',
        r'Five seeds; sample SD. Fitting coherence is descriptive, not held-out coherence. NA is retained. '+
        r'NMF uses normalized reconstruction cross-entropy; BERTopic has no native word NLL. '+
        r'Bold marks the best available point mean, not statistical superiority.',
        r'\end{minipage}',r'\endgroup',r'\end{table*}']
    write_csv(folder/(name+'.csv'),raw,list(raw[0]))
    write_text(folder/(name+'.md'),caption+'\n\n'+'\n'.join(md)+'\n')
    write_text(folder/(name+'.tex'),'\n'.join(tex)+'\n')
