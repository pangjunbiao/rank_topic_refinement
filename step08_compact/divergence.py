"""Separate affinity shares, decoder prototypes and deployed predictions.

Natural logarithms give divergence in nats. Topic identity is preserved within
each matched parent; topic matrices from different seeds are never averaged.
No epsilon smoothing or clipping of saved readouts is performed.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from scipy.special import logsumexp, rel_entr
from scipy.spatial.distance import jensenshannon
from rank_refinement.runtime.artifacts import read_json, write_csv, write_json
from step08_assembly.integrity import SEEDS
from step08_assembly.audit import arrays, descriptor_indices
from step08_assembly.metrics import logits, mix_logp


def probabilities(value):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all() or (a < 0).any() or (a.sum(1) <= 0).any():
        raise ValueError('Require finite nonnegative positive-mass rows.')
    return a / a.sum(1, keepdims=True)


def distribution_metrics(p, q):
    p, q = probabilities(p), probabilities(q)
    if p.shape != q.shape:
        raise ValueError('Distributions must have the same topic and vocabulary order.')
    midpoint = (p + q) / 2
    js = .5 * (rel_entr(p, midpoint).sum(1) + rel_entr(q, midpoint).sum(1))
    tv = .5 * np.abs(p - q).sum(1)
    # Independent SciPy distance oracle: its square, not its distance, is JS.
    with np.errstate(invalid='ignore'):
        oracle = jensenshannon(p, q, base=None, axis=1) ** 2
    # Nearly identical rows can produce a negative ~machine-epsilon sum in
    # SciPy's square root. Only tolerate NaN there when our unsquared sum is
    # itself within 2e-14 of zero; never modify the affinity/probability arrays.
    finite = np.isfinite(oracle)
    if ((~finite & (np.abs(js) > 2e-14)).any() or
            not np.allclose(js[finite], oracle[finite], atol=2e-14, rtol=1e-9)):
        raise ValueError('Jensen-Shannon divergence failed its independent oracle.')
    if (js < -2e-14).any() or (js > np.log(2) + 2e-14).any() or (tv > 1 + 2e-14).any():
        raise ValueError('Divergence outside mathematical bounds.')
    js[js < 0] = 0.  # Roundoff only, after the strict lower-bound check above.
    return {'js': js, 'tv': tv}


def decoded_profiles(a, gain, offset):
    z = np.asarray(a, dtype=np.float64) * gain + offset
    return np.exp(z - logsumexp(z, axis=1, keepdims=True))


def target_recall(readout, targets):
    actual = descriptor_indices(readout)
    if len(actual) != len(targets) or any(len(x) != 10 or len(set(x)) != 10 for x in targets):
        raise ValueError('Require a complete ten-word target for each topic.')
    return np.asarray([len(set(a) & set(t)) / 10 for a, t in zip(actual, targets)])


def case_inputs(bundle, dataset, seed):
    if dataset == 'Primary':
        f = bundle['primary'] / 'proposed_control_cases' / ('seed' + str(seed))
        z = arrays(f / 'frozen_test_arrays.npz')
        case = read_json(f / 'case_result.json')
        vocabulary = read_json(bundle['preparation'] / 'vocabulary.json')['tokens']
        group_ids = read_json(bundle['primary'] / 'test_population.json')['group_ids']
        a, b, theta = z['native'], z['proposed'], z['theta_test']
        native = next(r for r in case['rows'] if r['role'] == 'native')
        proposed = next(r for r in case['rows'] if r['role'] == 'proposed')
        word_ids = {w:i for i,w in enumerate(vocabulary)}
        reported = [[word_ids[w] for w in t] for t in proposed['topics']]
        saved = read_json(Path(__file__).parent/'data/primary_fitting_targets.json')
        reference = next(r for r in saved['rows'] if r['seed']==seed)
        if (not saved['selected_before_test'] or
                hashlib.sha256(b.astype(np.float64).tobytes()).hexdigest()!=reference['proposed_float64_bytes_sha256']):
            raise ValueError('Primary readout differs from the pre-test acquisition reference.')
        targets = reference['targets']
        if any(set(a)!=set(b) for a,b in zip(targets,reported)):
            raise ValueError('Reported words differ from independent fitting-acquired targets.')
    else:
        f = bundle['external'] / 'cases/proposed_parent' / ('seed' + str(seed))
        z = arrays(f / 'predictive_arrays.npz')
        case = read_json(f / 'case_result.json')
        vocabulary = read_json(bundle['external'] / 'preparation/vocabulary.json')['tokens']
        split = read_json(bundle['external'] / 'preparation/split_group_ids.json')
        masks = read_json(bundle['external'] / 'preparation/completion_masks.json')
        dev = [g for g in split['development'] if masks[g]['eligible']]
        group_ids = [g for g in split['test'] if masks[g]['eligible']]
        a, b = z['readout_native'], z['readout_proposed']
        theta = z['theta_observed'][len(dev):]
        native = next(r for r in case['rows'] if r['method'] == 'NeuroMax')
        proposed = next(r for r in case['rows'] if r['method'] == 'Proposed')
        targets = case['refinement']['targets']
    if len(theta) != len(group_ids) or len(set(group_ids)) != len(group_ids):
        raise ValueError('Every frozen completion group must be retained exactly once.')
    if a.shape != b.shape or a.shape != (10, len(vocabulary)):
        raise ValueError('Matched affinity dimensions disagree.')
    return dict(a=a, b=b, theta=theta, gain=z['decoder_gain'], offset=z['decoder_offset'],
                prior=z['prior'], native=native, proposed=proposed, targets=targets,
                vocabulary=vocabulary, group_ids=group_ids, folder=f)


def calculate(bundle, folder, log):
    seeds, topics, documents, details = [], [], [], {}
    for dataset in ['Primary', 'External']:
        for seed in SEEDS:
            c = case_inputs(bundle, dataset, seed)
            a, b = c['a'], c['b']
            affinity = distribution_metrics(a, b)
            prototype = distribution_metrics(decoded_profiles(a,c['gain'],c['offset']),
                                             decoded_profiles(b,c['gain'],c['offset']))
            before = logits(c['theta'],a,c['gain'],c['offset'])
            after = logits(c['theta'],b,c['gain'],c['offset'])
            raw = distribution_metrics(np.exp(before),np.exp(after))
            final = distribution_metrics(np.exp(mix_logp(before,c['prior'],c['native']['rho'])),
                                         np.exp(mix_logp(after,c['prior'],c['proposed']['rho'])))
            nr, pr = target_recall(a,c['targets']), target_recall(b,c['targets'])
            if not np.all(pr == 1):
                raise ValueError('Proposed fails actual-readout target membership.')
            scale = float(a.max())
            row_error = float(np.max(np.abs(a.sum(1)-b.sum(1))) / scale)
            column_error = float(np.max(np.abs(a.sum(0)-b.sum(0))) / scale)
            if row_error > 1e-6 or column_error > 1e-6 or (b < 0).any():
                raise ValueError('Conservation/nonnegativity failed on saved affinity.')
            r = dict(dataset=dataset,seed=seed,topics=10,vocabulary=a.shape[1],
                     test_completion_groups=len(c['group_ids']),
                     affinity_js_nats=float(affinity['js'].mean()),
                     affinity_tv=float(affinity['tv'].mean()),
                     prototype_js_nats=float(prototype['js'].mean()),
                     prototype_tv=float(prototype['tv'].mean()),
                     raw_document_js_nats=float(raw['js'].mean()),
                     final_document_js_nats=float(final['js'].mean()),
                     final_document_tv=float(final['tv'].mean()),
                     relative_l1_percent=float(100*np.abs(b-a).sum()/a.sum()),
                     native_target_recall=float(nr.mean()),proposed_target_recall=float(pr.mean()),
                     max_row_total_error_native_max_scaled=row_error,
                     max_column_total_error_native_max_scaled=column_error,
                     rho_native=c['native']['rho'],rho_proposed=c['proposed']['rho'])
            seeds.append(r)
            for k in range(10):
                topics.append(dict(dataset=dataset,seed=seed,topic_index=k,
                    affinity_js_nats=float(affinity['js'][k]),affinity_tv=float(affinity['tv'][k]),
                    prototype_js_nats=float(prototype['js'][k]),prototype_tv=float(prototype['tv'][k]),
                    native_target_recall=float(nr[k]),proposed_target_recall=float(pr[k])))
            for i,g in enumerate(c['group_ids']):
                documents.append(dict(dataset=dataset,seed=seed,group_id=g,
                    raw_js_nats=float(raw['js'][i]),final_js_nats=float(final['js'][i]),
                    final_tv=float(final['tv'][i])))
            dest=folder/'numerical_evidence'/dataset.lower()/('seed'+str(seed))
            dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'affinity_before_after.npz',native=a,proposed=b,delta=b-a,
                row_normalized_native=probabilities(a),row_normalized_proposed=probabilities(b),
                unit_topic_decoder_native=decoded_profiles(a,c['gain'],c['offset']),
                unit_topic_decoder_proposed=decoded_profiles(b,c['gain'],c['offset']),
                targets=np.asarray(c['targets'],dtype=np.int64))
            write_json(dest/'vocabulary.json',{'tokens':c['vocabulary'],'topic_rows_matched_within_parent':True})
            details[(dataset,seed)] = c
            log.info('DIVERGENCE | %s seed=%d | affinity JS=%.4g | prototype JS=%.4g | final document JS=%.4g | target recall %.3f -> %.3f',
                     dataset,seed,r['affinity_js_nats'],r['prototype_js_nats'],r['final_document_js_nats'],nr.mean(),pr.mean())
    numerical=folder/'numerical_evidence'
    for name,rows in [('divergence_per_seed',seeds),('divergence_per_topic',topics),('divergence_per_document',documents)]:
        write_csv(numerical/(name+'.csv'),rows,list(rows[0]))
    return seeds,details
