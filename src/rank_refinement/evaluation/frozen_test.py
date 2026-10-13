"""Untouched-test views and complete-grid summaries; no test-time selection."""
from __future__ import annotations
import json
import math
import statistics
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import sparse
from ..runtime.artifacts import read_json, write_json
from ..models.neural_prediction import backoff, score_log_predictions
from .competitor_evidence import audit_native_reference


def load_test_view(preparation, fit_ids, dev_ids):
    preparation=Path(preparation)
    def jsonl(name):
        return [json.loads(line) for line in (preparation/name).read_text(encoding='utf-8').splitlines() if line]
    features={r['group_id']:r for r in jsonl('feature_records.jsonl')}
    masks={r['group_id']:r for r in jsonl('completion_masks.jsonl')}
    ids=read_json(preparation/'completion_group_ids_by_split.json')['test']
    vocabulary=read_json(preparation/'vocabulary.json')['tokens']
    all_ids=read_json(preparation/'matrix_rows.json');index={g:i for i,g in enumerate(all_ids)}
    if (not ids or len(ids)!=len(set(ids)) or set(ids)&(set(fit_ids)|set(dev_ids))
            or len(all_ids)!=len(set(all_ids))):
        raise ValueError('Test completion groups are empty, duplicated or overlap fitting/development.')
    observed=sparse.load_npz(preparation/'observed_counts.npz')[[index[g] for g in ids]].toarray().astype(np.float32)
    records,texts=[],[]
    for i,g in enumerate(ids):
        r,m=features[g],masks[g]
        if (r['split']!='test' or not m['eligible'] or not m['observed_ids'] or not m['target_ids']
                or set(m['observed_positions'])&set(m['target_positions'])):
            raise ValueError('Test mask population or token-position firewall changed.')
        hist=np.zeros(len(vocabulary),dtype=np.float32)
        for j,n in Counter(m['observed_ids']).items():
            if not isinstance(j,int) or not 0<=j<len(vocabulary):raise ValueError('Invalid observed vocabulary index.')
            hist[j]=n
        if not np.array_equal(hist,observed[i]):raise ValueError('Observed test counts differ from frozen masks.')
        # Full text and target words are deliberately absent from model inputs.
        records.append({'group_id':g,'source':r['source'],'split':'test'})
        texts.append(m['observed_text'])
    coherence_records=[r for _,r in sorted(features.items()) if r['split']=='test']
    return {'ids':ids,'observed_counts':observed,'observed_texts':texts,'records':records,
        'masks':{g:masks[g] for g in ids},'coherence_records':coherence_records,
        'vocabulary':vocabulary,'full_test_text_used_for_inference':False}


def frozen_probability_scores(probabilities, rho, prior, view, sources):
    p=np.asarray(probabilities,dtype=np.float64)
    if (p.shape!=(len(view['ids']),len(prior)) or not np.isfinite(p).all() or (p<0).any()
            or not np.allclose(p.sum(1),1.,rtol=0,atol=1e-10)):
        raise ValueError('Invalid frozen test prediction distribution.')
    with np.errstate(divide='ignore'):logp=np.log(p)
    mixed=backoff(logp,prior,float(rho))
    # A zero target probability is a real failure, never epsilon-clipped.
    # Zero non-target probabilities remain valid for sparse reconstruction.
    losses=[]
    from .likelihood import aggregate_nll
    for i,r in enumerate(view['records']):
        targets=np.asarray(view['masks'][r['group_id']]['target_ids'],dtype=np.int64)
        if targets.size==0 or targets.min()<0 or targets.max()>=mixed.shape[1]:raise ValueError('Invalid frozen targets.')
        selected=mixed[i,targets]
        if not np.isfinite(selected).all():raise ValueError('Frozen test decoder assigned zero probability to a target; no clipping or new rho selection.')
        losses.append({'group_id':r['group_id'],'source':r['source'],'target_tokens':int(targets.size),
                      'negative_log_probability':float(-selected.sum())})
    scores=aggregate_nll(losses,sources)
    return scores,losses


def heldout_descriptor_audit(topics, view, cache):
    from ..runtime.artifacts import digest
    identity={'topics':topics,'reference':[(r['group_id'],r['tokens']) for r in view['coherence_records']],
              'vocabulary':view['vocabulary'],'windows':[110,10],'retain_all_slots':True}
    path=Path(cache)/(digest(identity)+'.json');index=path.with_suffix('.index.json')
    from ..runtime.artifacts import file_hash
    if path.is_file() and index.is_file():
        saved=read_json(index)
        if saved['identity']!=identity or file_hash(path)!=saved['audit_sha256']:
            raise ValueError('Frozen heldout coherence audit cache changed.')
        audit=read_json(path)
    else:
        audit=audit_native_reference(topics,view['coherence_records'],view['vocabulary'])
        write_json(path,audit)
        write_json(index,{'identity':identity,'audit_sha256':file_hash(path)})
    return {'c_v':audit['metrics']['c_v']['mean'],'c_npmi':audit['metrics']['c_npmi']['mean'],
        'reference_groups':audit['reference_groups'],'supported_slots':audit['supported_slots'],
        'total_slots':audit['total_slots'],'reference_absent_words':audit['reference_absent_words'],
        'all_descriptor_slots_retained':True,'scope':'frozen_test_reference_diagnostic_no_model_selection',
        'failure':None if all(audit['metrics'][k]['complete'] for k in ['c_v','c_npmi']) else 'standard_metric_nonfinite_no_slot_removal'},audit


def summarize_test(rows, seeds):
    fields={'fitting_c_v':('fitting','c_v'),'fitting_c_npmi':('fitting','c_npmi'),
        'test_c_v':('test_coherence','c_v'),'test_c_npmi':('test_coherence','c_npmi'),
        'test_macro_nll':('prediction','macro_source_nll'),'test_micro_nll':('prediction','micro_nll')}
    out=[]
    for method,profile in sorted({(r['method'],r['profile']) for r in rows}):
        part=[r for r in rows if (r['method'],r['profile'])==(method,profile)]
        if len(part)!=len(seeds) or {r['seed'] for r in part}!=set(seeds):raise ValueError('Incomplete frozen-test seed grid.')
        entry={'method':method,'profile':profile,'n_seeds':len(seeds),'complete_seed_grid':True,
            'scored_seed_count':sum(r['status']=='scored' for r in part),'fitting_coherence_scope':'common_fitting_reference_descriptive',
            'test_coherence_scope':'all_slots_diagnostic_no_filtering','final_primary_test_result':True}
        for name,(parent,child) in fields.items():
            values=[r.get(parent,{}).get(child) for r in part]
            good=all(x is not None and np.isfinite(x) for x in values)
            entry[name+'_mean']=statistics.mean(values) if good else None
            entry[name+'_sd']=statistics.stdev(values) if good else None
            entry[name+'_available_seeds']=sum(x is not None and np.isfinite(x) for x in values)
        values=[r.get('topic_diversity') for r in part]
        good=all(x is not None and np.isfinite(x) for x in values)
        entry['topic_diversity_mean']=statistics.mean(values) if good else None
        entry['topic_diversity_sd']=statistics.stdev(values) if good else None
        nll=[r.get('prediction',{}).get('macro_source_nll') for r in part]
        ppx=[math.exp(x) for x in nll] if all(x is not None and np.isfinite(x) for x in nll) else None
        entry['test_macro_perplexity_mean']=statistics.mean(ppx) if ppx else None
        entry['test_macro_perplexity_sd']=statistics.stdev(ppx) if ppx else None
        out.append(entry)
    return out
