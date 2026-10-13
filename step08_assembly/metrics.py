"""Complete-seed summaries and independent saved-array likelihood arithmetic."""
from __future__ import annotations
import math
import statistics
from collections import defaultdict
import numpy as np
from scipy.special import logsumexp
from .integrity import SEEDS

def grid(rows, count):
    keys=[(r['method'],r['profile'],r['seed'])for r in rows]
    if len(rows)!=count or len(keys)!=len(set(keys)):
        raise ValueError('Missing, duplicate or extra model slots.')
    for method,profile in {(r['method'],r['profile'])for r in rows}:
        part=[r for r in rows if(r['method'],r['profile'])==(method,profile)]
        if len(part)!=5 or {r['seed']for r in part}!=set(SEEDS):
            raise ValueError('Do not replace seeds or compute partial-seed means.')

def normalize(row, dataset, control=False):
    fit=row.get('fitting_coherence',row.get('fitting',{}))
    test=row.get('test_coherence',{})
    fixed=row.get('fixed_k10',row.get('topics')is not None and len(row['topics'])==10)
    return {'dataset':dataset,'method':row['method'],'profile':row['profile'],'seed':int(row['seed']),
        'control':control,'role':row.get('role','model'),'status':row['status'],
        'fixed_k10':bool(fixed),'native_topic_count':row.get('native_topic_count',len(row['topics'])if row.get('topics')else None),
        'fit_c_v':fit.get('c_v')if fixed else None,'fit_npmi':fit.get('c_npmi')if fixed else None,
        'td':row.get('topic_diversity')if fixed else None,
        'test_nll':row.get('prediction',{}).get('macro_source_nll'),
        'test_micro_nll':row.get('prediction',{}).get('micro_nll'),
        'test_c_v':test.get('c_v')if fixed else None,'test_npmi':test.get('c_npmi')if fixed else None,
        'supported_slots':test.get('supported_slots',0),'total_slots':test.get('total_slots',0),
        'rho':row.get('rho'),'prediction_interface':row.get('prediction_interface'),
        'projection_deployed':row.get('projection_deployed',False),
        'deployment':row.get('deployment'),'topics':row.get('topics'),
        'document_losses':row.get('document_losses',[]),'original':row}

def summaries(rows):
    result=[]
    for key in sorted({(r['dataset'],r['method'],r['profile'],r['control'])for r in rows}):
        part=sorted([r for r in rows if(r['dataset'],r['method'],r['profile'],r['control'])==key],key=lambda x:x['seed'])
        if len(part)!=5 or {r['seed']for r in part}!=set(SEEDS):raise ValueError('Incomplete five-seed summary.')
        dataset,method,profile,control=key
        item={'dataset':dataset,'method':method,'profile':profile,'control':control,
              'seeds':5,'scored_seeds':sum(r['status']=='scored'for r in part),
              'native_topics_by_seed':','.join(str(r['native_topic_count'])for r in part),
              'rho_by_seed':','.join(str(r['rho'])for r in part)}
        for field in ['fit_c_v','fit_npmi','td','test_nll','test_micro_nll','test_c_v','test_npmi']:
            vals=[r[field]for r in part];finite=[v for v in vals if v is not None and math.isfinite(v)]
            item[field+'_available_seeds']=len(finite)
            item[field+'_mean']=statistics.mean(vals)if len(finite)==5 else None
            item[field+'_sd']=statistics.stdev(vals)if len(finite)==5 else None
        result.append(item)
    return result

def close(actual,expected,label,tolerance=1e-10):
    if actual is None or expected is None:
        if actual!=expected:raise ValueError('Unavailable value changed: '+label)
        return 0.
    if not math.isfinite(actual)or not math.isfinite(expected)or abs(actual-expected)>tolerance:
        raise ValueError('Saved score does not replay: '+label)
    return abs(actual-expected)

def aggregate(losses):
    sums=defaultdict(float);counts=defaultdict(int);seen=set()
    for r in losses:
        if r['group_id']in seen:raise ValueError('Duplicate completion group.')
        seen.add(r['group_id']);n=r['target_tokens'];v=r['negative_log_probability']
        if type(n)is not int or n<=0 or v is None or not math.isfinite(v)or v<0:
            raise ValueError('Invalid target count or document loss.')
        sums[r['source']]+=v;counts[r['source']]+=n
    if not counts:raise ValueError('Empty predictive population.')
    by={s:sums[s]/counts[s]for s in sorted(counts)}
    return {'macro_source_nll':statistics.mean(by.values()),'micro_nll':sum(sums.values())/sum(counts.values()),
            'by_source':by,'target_tokens_by_source':dict(counts)}

def mix_logp(logp,prior,rho):
    raw=np.asarray(logp,dtype=np.float64);prior=np.asarray(prior,dtype=np.float64)
    if (raw.ndim!=2 or raw.shape[1]!=len(prior)or np.isnan(raw).any()or np.isposinf(raw).any()
            or not np.allclose(logsumexp(raw,axis=1),0,atol=1e-10,rtol=0)
            or not np.isfinite(prior).all()or (prior<=0).any()or abs(prior.sum()-1)>1e-10 or not 0<=rho<1):
        raise ValueError('Invalid saved predictive distribution or backoff.')
    return raw.copy()if rho==0 else np.logaddexp(np.log1p(-rho)+raw,np.log(rho)+np.log(prior)[None,:])

def replay_losses(raw,prior,row,ids,masks,features):
    values=mix_logp(raw,prior,float(row['rho']))
    if values.shape[0]!=len(ids):raise ValueError('Predictive row population differs.')
    actual=[]
    for i,g in enumerate(ids):
        m=masks[g];target=np.asarray(m['target_ids'],dtype=np.int64)
        if (not m['eligible']or target.size==0 or target.min()<0 or target.max()>=values.shape[1]
                or set(m['observed_positions'])&set(m['target_positions'])):
            raise ValueError('Invalid or overlapping frozen completion mask.')
        selected=values[i,target]
        if not np.isfinite(selected).all():raise ValueError('Zero probability on a frozen target; do not clip.')
        actual.append({'group_id':g,'source':features[g]['source'],'target_tokens':int(len(target)),
                       'negative_log_probability':float(-selected.sum())})
    saved=row['document_losses']
    if len(saved)!=len(actual)or {r['group_id']for r in saved}!=set(ids):
        raise ValueError('Missing or substituted saved completion groups.')
    indexed={r['group_id']:r for r in saved};error=0.
    for r in actual:
        old=indexed[r['group_id']]
        if old['source']!=r['source']or old['target_tokens']!=r['target_tokens']:
            raise ValueError('Saved document target/source differs.')
        error=max(error,close(r['negative_log_probability'],old['negative_log_probability'],'document loss',1e-9))
    scores=aggregate(actual)
    for field in ['macro_source_nll','micro_nll']:
        error=max(error,close(scores[field],row['prediction'][field],field))
    for source,value in scores['by_source'].items():
        error=max(error,close(value,row['prediction']['by_source'][source],source))
    return {'max_loss_error':error,'groups':len(actual),'target_tokens':sum(r['target_tokens']for r in actual)}

def logits(theta,readout,gain,offset,moment=None):
    value=np.asarray(theta,dtype=np.float64)@np.asarray(readout,dtype=np.float64)
    if moment is not None:value=value*moment[0]+moment[1]
    value=value*gain+offset
    if not np.isfinite(value).all():raise ValueError('Nonfinite frozen affine decoder.')
    return value-logsumexp(value,axis=1,keepdims=True)

def point_winners(summary):
    winners=[]
    for dataset in ['Primary','External']:
        panel=[r for r in summary if r['dataset']==dataset and not r['control']]
        for metric,direction in [('fit_c_v',1),('fit_npmi',1),('td',1),('test_nll',-1)]:
            available=[r for r in panel if r[metric+'_mean']is not None]
            best=max(direction*r[metric+'_mean']for r in available)if available else None
            for r in available:
                if abs(direction*r[metric+'_mean']-best)<=1e-12:
                    winners.append({'dataset':dataset,'metric':metric,'method':r['method'],'profile':r['profile'],
                                    'mean':r[metric+'_mean'],'scope':'point_mean_only; availability-specific; no significance claim'})
    return winners
