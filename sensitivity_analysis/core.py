from __future__ import annotations
import copy
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .vendor.acquisition.native_diversity_targets import recovery_targets
from .vendor.acquisition.native_targets import native_indices
from .vendor.projection.minimum_tv import certify
from .vendor.models.neural_prediction import log_predictions, backoff, score_log_predictions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def safe(root, relative):
    if not isinstance(relative, str) or '\\' in relative:
        raise ValueError('Invalid input path.')
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Input path escapes project.')
    result = (Path(root) / p).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError('Input path escapes project.')
    return result


def verified_inputs(root, plan):
    paths, audit = {}, []
    for entry in plan['input_files']:
        # The archive includes authentic saved assets, never newly fitted models.
        packed = safe(root, entry['packed_path'])
        original = safe(root, entry['source_path'])
        if not packed.is_file() or packed.stat().st_size != entry['bytes'] or sha(packed) != entry['sha256']:
            raise ValueError('Bundled input checksum failed: ' + entry['packed_path'])
        if original.is_file() and sha(original) != entry['sha256']:
            raise ValueError('Existing saved input differs from reviewed input: ' + entry['source_path'])
        paths[entry['source_path']] = packed
        audit.append({**entry, 'existing_original_verified': original.is_file()})
    for entry in plan['vendor_source_files']:
        if sha(safe(root, entry['vendored_path'])) != entry['sha256']:
            raise ValueError('Verified numerical implementation changed: ' + entry['vendored_path'])
    return paths, audit


def load_population(paths, plan):
    prefix = plan['canonical'] + '/'
    def lines(name):
        return [json.loads(x) for x in paths[prefix+name].read_text(encoding='utf-8').splitlines() if x]
    records = lines('feature_records.jsonl'); masks = lines('completion_masks.jsonl')
    if len({r['group_id'] for r in records}) != len(records) or len({m['group_id'] for m in masks}) != len(masks):
        raise ValueError('Duplicate independent population identifiers.')
    by_id = {r['group_id']: r for r in records}; mask = {m['group_id']: m for m in masks}
    fit_ids = read(paths[prefix+'model_fit_group_ids.json'])
    dev_ids = read(paths[prefix+'completion_group_ids_by_split.json'])['development']
    if len(set(fit_ids)) != len(fit_ids) or len(set(dev_ids)) != len(dev_ids) or set(fit_ids) & set(dev_ids):
        raise ValueError('Fitting/development firewall failed.')
    if any(by_id[g]['split'] != 'fit' for g in fit_ids) or any(by_id[g]['split'] != 'development' or not mask[g]['eligible'] for g in dev_ids):
        raise ValueError('Foreign groups or ineligible development targets.')
    fit_observed = [g for g in fit_ids if mask[g]['eligible']]
    metadata = read(paths[prefix+'vocabulary.json'])
    return {'fit_ids': fit_ids, 'fit_observed_ids': fit_observed, 'dev_ids': dev_ids,
            'fit_records': [by_id[g] for g in fit_ids], 'dev_records': [by_id[g] for g in dev_ids],
            'masks': {g:mask[g] for g in fit_observed+dev_ids}, 'metadata': metadata,
            'vocabulary': metadata['tokens'], 'sources': sorted({by_id[g]['source'] for g in fit_ids})}


def seed_bundle(paths, spec, data):
    def npz(path):
        with np.load(paths[path], allow_pickle=False) as z:
            return {k:z[k] for k in z.files}
    arrays = npz(spec['arrays_folder']+'/readouts_and_latents.npz')
    evidence = npz(spec['evidence_folder']+'/association_evidence.npz')
    evidence.update(read(paths[spec['evidence_folder']+'/association_words.json']))
    samples = read(paths[spec['evidence_folder']+'/sampling_group_ids.json'])
    contexts = read(paths[spec['evidence_folder']+'/range_contexts.json'])['group_ids']
    fit_set = set(data['fit_ids']); positions = {g:i for i,g in enumerate(data['fit_observed_ids'])}
    if evidence['fit_groups'] != len(fit_set) or len(samples) != 20 or any(len(s)!=len(set(s)) or not set(s)<=fit_set for s in samples):
        raise ValueError('Association stability evidence contains foreign or repeated groups.')
    if len(contexts) != 64 or len(set(contexts)) != 64 or not set(contexts)<=set(positions):
        raise ValueError('Range contexts must be the same 64 fitting-observed groups.')
    shapes = {'native':(10,len(data['vocabulary'])), 'theta_fit':(len(positions),10),
              'theta_development':(len(data['dev_ids']),10), 'decoder_gain':(len(data['vocabulary']),),
              'decoder_offset':(len(data['vocabulary']),), 'prior':(len(data['vocabulary']),)}
    for k,shape in shapes.items():
        if arrays[k].shape != shape or not np.isfinite(arrays[k]).all():
            raise ValueError('Incorrect saved numerical interface: '+k)
    for k in ['theta_fit','theta_development']:
        if (arrays[k]<0).any() or not np.allclose(arrays[k].sum(axis=1),1.,atol=1e-6):
            raise ValueError('Saved topic proportions are not normalized.')
    selected = read(paths[spec['selected_case_folder']+'/case_result.json'])
    readouts = npz(spec['selected_case_folder']+'/readouts.npz')
    if selected['seed'] != spec['seed'] or selected['graph'] != 'signed_stable':
        raise ValueError('Selected recipe source mismatch.')
    return {'arrays':arrays, 'evidence':evidence, 'contexts':arrays['theta_fit'][[positions[g] for g in contexts]],
            'selected':selected, 'selected_readouts':readouts, 'context_ids':contexts}


def predict(bundle, affinity, data, rho):
    a = bundle['arrays']
    lp = log_predictions(a['theta_development'], affinity, a['decoder_gain'], a['decoder_offset'])
    scores, losses = score_log_predictions(backoff(lp, a['prior'], rho), data['dev_records'],data['masks'],data['sources'])
    return scores, losses


def replay_selected(bundle, data, plan):
    maximum = 0.
    for mode in plan['projection_objectives']:
        saved = bundle['selected']['rows'][mode]
        if saved['rho'] != plan['backoff_rho'] or saved['moment_correction']:
            raise ValueError('Fixed final prediction recipe does not match saved development row.')
        check = certify(bundle['arrays']['native'],bundle['selected_readouts'][mode],bundle['selected']['targets'],margin=plan['selected_margin'])
        if not check['certified']:
            raise ValueError('Saved selected readout fails independent certification.')
        scores, losses = predict(bundle,bundle['selected_readouts'][mode],data,plan['backoff_rho'])
        if abs(scores['macro_source_nll'] - saved['prediction']['macro_source_nll']) > 1e-10:
            raise ValueError('Development NLL does not reproduce its reviewed source.')
        old = saved['document_losses']
        if [(r['group_id'],r['source'],r['target_tokens']) for r in losses] != [(r['group_id'],r['source'],r['target_tokens']) for r in old]:
            raise ValueError('Development prediction population/order differs from saved source.')
        maximum=max(maximum,max(abs(x['negative_log_probability']-y['negative_log_probability']) for x,y in zip(losses,old)))
    if maximum > 1e-9:
        raise ValueError('Saved group losses failed replay.')
    return maximum


def acquire(bundle, data, policy, support):
    settings = copy.deepcopy(policy); settings['pair_group_support']=int(support)
    return recovery_targets(bundle['arrays']['native'],data['vocabulary'],data['metadata'],bundle['evidence'],settings,'signed_stable')


def topic_indices(affinity):
    return native_indices(affinity)


def coherence(affinity, data):
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    import warnings
    indices = topic_indices(affinity)
    topics = [[data['vocabulary'][i] for i in row] for row in indices]
    dictionary=Dictionary([data['vocabulary']]); texts=[r['tokens'] for r in data['fit_records']]
    result={'topics':indices, 'td':len({w for t in indices for w in t})/100.,'all_100_slots_retained':True}
    for name,width in [('cv',110),('npmi',10)]:
        with warnings.catch_warnings(record=True) as warns, np.errstate(divide='ignore',invalid='ignore'):
            warnings.simplefilter('always',RuntimeWarning)
            cm=CoherenceModel(topics=topics,texts=texts,dictionary=dictionary,coherence='c_v' if name=='cv' else 'c_npmi',window_size=width,topn=10,processes=1)
            values=np.asarray(cm.get_coherence_per_topic())
        result[name]=float(values.mean()) if np.isfinite(values).all() else None
        result[name+'_per_topic']=[float(v) if np.isfinite(v) else None for v in values]
        result[name+'_warnings']=sorted({str(w.message) for w in warns})
    return result


def summarize(rows):
    # No mean over a silently reduced set of successful seeds.
    groups={}
    for row in rows:
        key=(row['study'],row['support'],row['margin'],row['objective'])
        groups.setdefault(key,[]).append(row)
    metrics=['cv','npmi','td','development_nll','delta_native_nll','relative_l1_percent','target_recall','edited_topics','replacements','fitting_logit_range']
    output=[]
    for key,items in groups.items():
        r=dict(zip(['study','support','margin','objective'],key))
        r.update(planned_seeds=5,completed_slots=len(items),certified_seeds=sum(i['certified'] for i in items),
                 selected_setting=all(i['selected_setting'] for i in items),
                 total_seconds=sum(i['wall_seconds'] for i in items))
        for name in metrics:
            values=[i.get(name) for i in items]
            complete=len(items)==5 and len({i['seed'] for i in items})==5 and all(x is not None and np.isfinite(x) for x in values)
            r[name+'_mean']=float(np.mean(values)) if complete else None
            r[name+'_sd']=float(np.std(values,ddof=1)) if complete else None
        output.append(r)
    return output


def csv_write(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
