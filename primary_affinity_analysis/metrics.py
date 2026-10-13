"""Independent float64 arithmetic; no model, encoder or optimiser imports."""
from __future__ import annotations
import numpy as np


def normalize(value):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.size == 0 or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Distributions require finite, nonnegative two-dimensional arrays.')
    totals = a.sum(axis=1, keepdims=True)
    if (totals <= 0).any() or not np.isfinite(totals).all():
        raise ValueError('Every distribution must have positive finite mass.')
    return a / totals


def divergence(left, right):
    p, q = normalize(left), normalize(right)
    if p.shape != q.shape:
        raise ValueError('Both distributions must retain the same population and word order.')
    midpoint = (p + q) / 2
    terms = []
    for value in (p, q):
        term = np.zeros_like(value)
        positive = value > 0
        term[positive] = value[positive] * np.log(value[positive] / midpoint[positive])
        terms.append(term.sum(axis=1))
    js = (terms[0] + terms[1]) / 2
    if (not np.isfinite(js).all() or (js < -2e-14).any()
            or (js > np.log(2) + 2e-14).any()):
        raise ValueError('JS divergence is outside its natural-log mathematical bounds.')
    # Only roundoff in the scalar divergence is removed; never alter affinities.
    js = np.where(js < 0, 0.0, js)
    tv = np.abs(p - q).sum(axis=1) / 2
    return {'js': js, 'tv': tv}


def log_softmax(value):
    z = np.asarray(value, dtype=np.float64)
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError('Decoder logits must be finite rows.')
    shifted = z - z.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def predict(theta, affinity, gain, offset, prior, rho):
    theta, affinity = np.asarray(theta, dtype=np.float64), np.asarray(affinity, dtype=np.float64)
    gain, offset, prior = [np.asarray(x, dtype=np.float64) for x in (gain, offset, prior)]
    if (theta.ndim != 2 or affinity.ndim != 2 or theta.shape[1] != affinity.shape[0]
            or any(x.shape != (affinity.shape[1],) for x in (gain, offset, prior))
            or not all(np.isfinite(x).all() for x in (theta, affinity, gain, offset, prior))
            or (theta < 0).any() or (affinity < 0).any() or (prior <= 0).any()
            or not np.isclose(prior.sum(), 1, rtol=0, atol=1e-10) or not 0 <= rho < 1):
        raise ValueError('Invalid frozen latent, decoder, prior or backoff interface.')
    raw = log_softmax((theta @ affinity) * gain + offset)
    return raw if rho == 0 else np.logaddexp(np.log1p(-rho) + raw, np.log(rho) + np.log(prior)[None, :])


def top_words(affinity, count=10):
    a = np.asarray(affinity, dtype=np.float64)
    normalize(a)
    if type(count) is not int or count <= 0 or count > a.shape[1]:
        raise ValueError('Invalid descriptor count.')
    return [np.lexsort((np.arange(a.shape[1]), -row))[:count] for row in a]


def target_inclusion(affinity, targets):
    a = np.asarray(affinity, dtype=np.float64)
    if (len(targets) != a.shape[0] or any(len(t) != 10 or len(set(t)) != 10 for t in targets)
            or any(type(i) not in (int, np.int64) or not 0 <= i < a.shape[1] for t in targets for i in t)):
        raise ValueError('Require ten distinct, in-vocabulary, independently saved targets per topic.')
    return np.array([len(set(t) & set(actual)) / 10 for t, actual in zip(targets, top_words(a))])


def changed_columns(native, proposed, count=24):
    a, b = np.asarray(native), np.asarray(proposed)
    if a.shape != b.shape or not 0 < count <= a.shape[1]:
        raise ValueError('Invalid matched affinity display dimensions.')
    movement = np.abs(b-a).sum(axis=0)
    return np.lexsort((np.arange(a.shape[1]), -movement))[:count]


def losses(logp, group_ids, masks, features):
    if logp.shape[0] != len(group_ids) or len(set(group_ids)) != len(group_ids):
        raise ValueError('Frozen completion groups are missing or duplicated.')
    result = []
    for i, gid in enumerate(group_ids):
        mask = masks[gid]
        ids = np.asarray(mask['target_ids'], dtype=np.int64)
        if (not mask['eligible'] or features[gid]['split'] != 'test' or ids.size == 0
                or ids.min() < 0 or ids.max() >= logp.shape[1]):
            raise ValueError('Invalid held-out target population.')
        selected = logp[i, ids]
        if not np.isfinite(selected).all():
            raise ValueError('A held-out token has undefined loss; do not clip its probability.')
        result.append({'group_id': gid, 'source': features[gid]['source'],
                       'target_tokens': int(ids.size), 'negative_log_probability': float(-selected.sum())})
    return result


def aggregate(rows):
    if not rows or len({r['group_id'] for r in rows}) != len(rows):
        raise ValueError('Expected nonempty independent completion groups.')
    sources = sorted({r['source'] for r in rows})
    by_source = {}
    for source in sources:
        selected = [r for r in rows if r['source'] == source]
        if any(type(r['target_tokens']) is not int or r['target_tokens'] <= 0 or not np.isfinite(r['negative_log_probability']) for r in selected):
            raise ValueError('Invalid target counts or losses.')
        by_source[source] = sum(r['negative_log_probability'] for r in selected) / sum(r['target_tokens'] for r in selected)
    return {'macro_source_nll': float(np.mean(list(by_source.values()))),
            'micro_nll': sum(r['negative_log_probability'] for r in rows) / sum(r['target_tokens'] for r in rows),
            'by_source': by_source}


def paired_interval(cases, reference, seeds):
    """Replay the original shared-group/seed bootstrap with its saved budget."""
    rng = np.random.default_rng(int(reference['seed']))
    n = int(reference['resamples'])
    if n != 10000 or len(seeds) != 5:
        raise ValueError('The locked uncertainty design changed.')
    seed_weights = rng.multinomial(5, np.full(5, .2), size=n)
    draws = np.zeros(n)
    sources = sorted({r['source'] for r in cases[seeds[0]]['proposed']})
    for source in sources:
        first = sorted([r for r in cases[seeds[0]]['proposed'] if r['source'] == source], key=lambda r: r['group_id'])
        groups = [r['group_id'] for r in first]
        tokens = np.array([r['target_tokens'] for r in first], dtype=float)
        group_weights = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=n)
        denominator = group_weights @ tokens
        for j, seed in enumerate(seeds):
            p = {r['group_id']: r for r in cases[seed]['proposed'] if r['source'] == source}
            c = {r['group_id']: r for r in cases[seed]['native'] if r['source'] == source}
            if set(p) != set(groups) or set(c) != set(groups):
                raise ValueError('Bootstrap population pairing changed.')
            if any(p[g]['target_tokens'] != tokens[i] or c[g]['target_tokens'] != tokens[i] for i, g in enumerate(groups)):
                raise ValueError('Bootstrap target counts changed.')
            difference = np.array([c[g]['negative_log_probability']-p[g]['negative_log_probability'] for g in groups])
            draws += (group_weights @ difference) / denominator * seed_weights[:, j] / 5 / len(sources)
    return tuple(map(float, np.quantile(draws, [.025, .975])))

