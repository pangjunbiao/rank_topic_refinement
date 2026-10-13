"""Load-only neural inference and verified fitting-only classical replay.

This module never imports a neural training entry point. Development replay
must reproduce the frozen interface before the returned model sees test input.
"""
from __future__ import annotations
import hashlib
import io
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from ..runtime.artifacts import atomic_bytes, digest, file_hash, read_json, write_json
from .neural_training import make_model as make_native, export_latents
from .competitor_neural import make_model as make_competitor, decode_model
from .neural_prediction import affine_parameters
from .classical_completion import normalized_reconstruction
from .competitor_classical import gsdmm_predict


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def assert_replay(actual, expected, name, atol=1e-6, rtol=1e-6):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.shape != expected.shape or not np.isfinite(actual).all()
            or not np.isfinite(expected).all()
            or not np.allclose(actual, expected, atol=atol, rtol=rtol)):
        raise ValueError('Frozen development replay failed: ' + name + '. Test scoring was not substituted.')
    return float(np.max(np.abs(actual - expected))) if actual.size else 0.


def assert_top10_replay(actual,expected):
    index=np.arange(actual.shape[1])
    if any(not np.array_equal(np.lexsort((index,-a))[:10],np.lexsort((index,-b))[:10])
           for a,b in zip(actual,expected)):
        raise ValueError('Frozen model TOP10 descriptors changed during inference replay.')


def checked_local(root, relative, prefix):
    root = Path(root).resolve()
    if not isinstance(relative, str) or '\\' in relative:
        raise ValueError('Invalid local frozen asset path.')
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts or rel.parts[:len(prefix)] != tuple(prefix):
        raise ValueError('Frozen asset path is outside its declared cache.')
    path = (root / rel).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise RuntimeError('Saved asset missing: ' + relative + '. Keep your original data/cache; no neural refit is performed.')
    return path


def checkpoint_index(root, details, prefix):
    path = checked_local(root, details['checkpoint_relative_path'], prefix)
    if file_hash(path) != details['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint bytes changed: ' + details['checkpoint_relative_path'])
    index = read_json(path.with_suffix('.json'))
    identity = details.get('identity', details.get('training_identity'))
    if (identity is None or index.get('identity') != identity
            or index.get('checkpoint_sha256') != details['checkpoint_sha256']
            or index.get('epoch') != details['epochs']):
        raise ValueError('Frozen checkpoint index/epoch differs from completed training.')
    return path, index


def restore_neural(root, method, profile, seed, details, folder, vectors, configs, device):
    import torch
    native = method in ['neuromax', 'encot']
    path, index = checkpoint_index(root, details, ('data', 'cache', 'step03_native' if native else 'step04_training'))
    identity = index['identity']
    if identity.get('device') != device or identity.get('torch') != torch.__version__:
        raise ValueError('Use the original frozen model device and PyTorch build; no replacement model is trained.')
    if int(identity['seed']) != int(seed) or identity.get('base', identity.get('method')) != method:
        raise ValueError('Checkpoint method/seed differs from the frozen case.')
    if native:
        if identity['profile'] != profile:
            raise ValueError('Native frozen profile changed.')
        model = make_native(method, folder, len(vectors['words']), vectors['fit_full'].shape[1],
            configs['step03_neural'], vectors['words'] if profile == 'chinese_words' else None, seed)
    else:
        if identity['profile'] != profile:
            raise ValueError('Competitor frozen profile changed.')
        model = make_competitor(method, folder, len(vectors['words']), vectors['fit_full'].shape[1],
            profile, words=vectors['words'] if method == 'glocom' else None, seed=seed)
    # Locally generated checkpoint is hash-verified against the reviewed result.
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('identity') != identity or saved.get('epoch') != index['epoch']:
        raise ValueError('Checkpoint payload differs from its frozen index.')
    if 'topic_word_matrix' in saved['model']:
        if not hasattr(model, 'beta') or not torch.equal(saved['model']['topic_word_matrix'], saved['model']['beta']):
            raise ValueError('Saved CTM topic-word alias differs from beta.')
        model.topic_word_matrix = model.beta
    model.load_state_dict(saved['model'], strict=True)
    model = model.to(device).eval()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite frozen model parameter.')
    return model


def saved_embeddings(root, manifest, data):
    path = checked_local(root, 'data/cache/step03_embeddings/' + manifest['key'][:20] + '/vectors.npz',
                         ('data', 'cache', 'step03_embeddings'))
    live = read_json(path.parent / 'manifest.json')
    if live != manifest or file_hash(path) != manifest['vectors_sha256']:
        raise ValueError('The saved fitting/development encoder cache changed.')
    expected = {'fit_full':len(data['fit_ids']), 'fit_observed':len(data['fit_observed_ids']),
                'development_observed':len(data['dev_ids']), 'words':len(data['vocabulary'])}
    vectors = arrays(path)
    for name, count in expected.items():
        if vectors[name].shape != (count, manifest['dimension']) or not np.isfinite(vectors[name]).all():
            raise ValueError('Frozen encoder views differ from their accepted row contract.')
    if (manifest['fitting_group_ids'] != data['fit_ids']
            or manifest['development_observed_group_ids'] != data['dev_ids']
            or manifest['fitting_observed_group_ids'] != data['fit_observed_ids']
            or manifest['test_or_external_encoded']):
        raise ValueError('Frozen encoder groups or original test firewall changed.')
    return vectors


def frozen_encoder_folder(root, assets):
    """Verify the exact revision folder created by resolve_encoder.

    The policy local_directory is a base directory. The earlier stages store
    files under local_directory/revision, not directly under the base. Reuse
    those exact bytes without downloading or choosing another encoder.
    """
    from .encoder_assets import FILES
    policy = assets['policy']
    revision = policy.get('revision')
    if (not isinstance(revision, str) or len(revision) != 40
            or any(c not in '0123456789abcdef' for c in revision)
            or assets.get('revision') != revision
            or assets.get('repository') != policy.get('repository')):
        raise ValueError('Frozen encoder repository/revision differs from its saved policy.')
    base = policy.get('local_directory')
    if not isinstance(base, str) or '\\' in base:
        raise ValueError('Invalid frozen encoder base directory.')
    relative = Path(base) / revision
    if (relative.is_absolute() or '..' in relative.parts
            or relative.parts[:2] != ('data', 'assets')):
        raise ValueError('Frozen encoder directory is outside data/assets.')
    files = assets.get('files')
    hashes = policy.get('file_sha256')
    if (not isinstance(files, list) or not isinstance(hashes, dict)
            or len(files) != len(FILES) or set(hashes) != set(FILES)
            or any(not isinstance(item, dict) or item.get('path') not in FILES for item in files)
            or len({item['path'] for item in files}) != len(FILES)):
        raise ValueError('Frozen encoder manifest must retain all five distinct pinned files.')
    for item in files:
        name = item['path']
        expected = item.get('sha256')
        size = item.get('bytes')
        if (expected != hashes[name] or not isinstance(expected, str) or len(expected) != 64
                or any(c not in '0123456789abcdef' for c in expected)
                or type(size) is not int or size < 0):
            raise ValueError('Frozen encoder file metadata differs from its pinned policy: ' + name)
        path = checked_local(root, (relative / name).as_posix(), ('data', 'assets'))
        if path.stat().st_size != size or file_hash(path) != expected:
            raise ValueError('Original pinned encoder file changed: ' + name)
    return (Path(root).resolve() / relative).resolve()


def encode_test_observed(root, manifest, group_ids, texts, device, output, log):
    import torch
    from transformers import AutoModel, AutoTokenizer
    policy = manifest['assets']['policy']
    folder = frozen_encoder_folder(root, manifest['assets'])
    identity = {'assets':manifest['assets'], 'group_ids':group_ids, 'texts':texts,
        'view':'primary_test_observed_text_only', 'device':device, 'torch':torch.__version__,
        'pooling':'cls_l2', 'max_length':policy['max_length']}
    path = Path(output) / 'test_observed_embeddings.npz'; lock = path.with_suffix('.json')
    if lock.is_file():
        saved = read_json(lock)
        if saved['identity'] != identity or file_hash(path) != saved['vectors_sha256']:
            raise ValueError('Test embedding cache identity changed.')
        result = arrays(path)['test_observed']
    else:
        model = AutoModel.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False,
                                         use_safetensors=True).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False)
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(texts), policy['batch_size']):
                selected = texts[start:start + policy['batch_size']]
                inputs = tokenizer(selected, return_tensors='pt', padding=True, truncation=True,
                                   max_length=policy['max_length'])
                values = model(**{k:v.to(device) for k,v in inputs.items()}).last_hidden_state[:,0]
                chunks.append(torch.nn.functional.normalize(values, dim=1).cpu().numpy().astype(np.float32))
                log.info('TEST ENCODER | observed text only | %d/%d | %s', min(start+len(selected),len(texts)),len(texts),device)
        result = np.concatenate(chunks)
        stream=io.BytesIO();np.savez_compressed(stream,test_observed=result);atomic_bytes(path,stream.getvalue())
        write_json(lock, {'identity':identity,'vectors_sha256':file_hash(path),'full_test_text_encoded':False})
        del model
        if device == 'cuda':torch.cuda.empty_cache()
    if (result.shape != (len(group_ids),manifest['dimension']) or not np.isfinite(result).all()
            or not np.allclose(np.linalg.norm(result,axis=1),1.,atol=1e-5)):
        raise ValueError('Invalid test observed encoder output.')
    return result


def context_inputs(counts, embeddings, bank, centers):
    # Assignment uses the fixed fitting centers and observed-only embeddings.
    if (counts.ndim!=2 or bank.shape[1]!=counts.shape[1] or centers.shape[0]!=bank.shape[0]
            or embeddings.shape!=(len(counts),centers.shape[1])):
        raise ValueError('Incompatible frozen context bank.')
    # Restore the sufficient fitted state for the same pinned KMeans.predict
    # implementation used originally, including float32/tie handling. No fit
    # occurs, and the original fitting centers are never recomputed.
    from sklearn.cluster import KMeans
    cluster=KMeans(n_clusters=len(centers),n_init=1)
    cluster.cluster_centers_=np.ascontiguousarray(centers)
    cluster.n_features_in_=centers.shape[1]
    cluster._n_threads=1
    labels=cluster.predict(embeddings)
    return np.concatenate([counts,bank[labels]],axis=1).astype(np.float32)


def native_latents(model, dev_inputs, test_inputs, expected, policy, device):
    import torch
    with torch.inference_mode(): native=model.get_beta().cpu().double().numpy()
    gain,offset=affine_parameters(model)
    tol={'atol':policy['native_replay_atol'],'rtol':policy['native_replay_rtol']}
    checks={'readout':assert_replay(native,expected['native'],'native readout',**tol),
        'decoder_gain':assert_replay(gain,expected['decoder_gain'],'decoder gain',**tol),
        'decoder_offset':assert_replay(offset,expected['decoder_offset'],'decoder offset',**tol)}
    assert_top10_replay(native,expected['native'])
    theta_dev=export_latents(model,dev_inputs,device)
    checks['development_theta']=assert_replay(theta_dev,expected['theta_development'],'native development theta',**tol)
    theta_test=export_latents(model,test_inputs,device)
    return theta_test,checks


def competitor_probabilities(method, model, data, vectors, test_counts, test_vectors, expected, seed, policy):
    dev_context=test_context=None
    if method=='glocom':
        bank,centers=expected['global_bank'],expected['global_centers']
        dev_context={'observed':context_inputs(data['dev_observed'],vectors['development_observed'],bank,centers)}
        test_context={'observed':context_inputs(test_counts,test_vectors,bank,centers)}
    weights,dev_prob,_,_=decode_model(method,model,data['dev_observed'],vectors,dev_context,seed,policy['combinedtm_posterior_samples'])
    tol={'atol':policy['native_replay_atol'],'rtol':policy['native_replay_rtol']}
    checks={'descriptor_weights':assert_replay(weights,expected['native_descriptor_weights'],'competitor descriptors',**tol),
            'development_probabilities':assert_replay(dev_prob,expected['native_decoded_probabilities'],'competitor development prediction',**tol)}
    assert_top10_replay(weights,expected['native_descriptor_weights'])
    view={**vectors,'development_observed':test_vectors}
    weights,test_prob,theta,_=decode_model(method,model,test_counts,view,test_context,seed,policy['combinedtm_posterior_samples'])
    return test_prob,theta,checks


def restore_gsdmm(expected, details, profile, vocabulary):
    counts = expected['native_descriptor_weights']
    if counts.shape!=(10,len(vocabulary)) or (counts<0).any() or not np.allclose(counts,np.rint(counts)):
        raise ValueError('Saved GSDMM sufficient statistics are not valid integer counts.')
    if not np.array_equal(counts.sum(1),np.asarray(details['cluster_word_count'])):
        raise ValueError('GSDMM cluster totals differ from frozen sufficient statistics.')
    return SimpleNamespace(K=10, alpha=profile['alpha'], beta=profile['beta'], vocab_size=len(vocabulary),
        cluster_doc_count=details['cluster_doc_count'],cluster_word_count=details['cluster_word_count'],
        cluster_word_distribution=[{vocabulary[int(j)]:int(row[j]) for j in np.flatnonzero(row)} for row in counts])


def classical_probabilities(method, data, test_counts, seed, profile, expected, details, policy, output, log):
    tol={'atol':policy['classical_replay_atol'],'rtol':policy['classical_replay_rtol']}
    if method=='gsdmm':
        model=restore_gsdmm(expected,details,profile,data['vocabulary'])
        weights,dev_prob,_=gsdmm_predict(model,data['dev_observed'],data['vocabulary'])
        checks={'descriptor_weights':assert_replay(weights,expected['native_descriptor_weights'],'GSDMM counts',**tol),
            'development_probabilities':assert_replay(dev_prob,expected['native_decoded_probabilities'],'GSDMM saved prediction',**tol)}
        if test_counts is None:return None,None,checks
        _,test_prob,theta=gsdmm_predict(model,test_counts,data['vocabulary'])
        return test_prob,theta,checks
    fit=sparse.csr_matrix(data['fit'],dtype=np.float64)
    dev=sparse.csr_matrix(data['dev_observed'],dtype=np.float64)
    test=sparse.csr_matrix(test_counts,dtype=np.float64) if test_counts is not None else None
    transformer=None;dev_rng=None
    path=Path(output)/'classical_frozen_model.pkl';index=path.with_suffix('.json')
    cache_identity={'method':method,'seed':seed,'profile':profile,'vocabulary':data['vocabulary'],
        'fitting_ids_sha256':digest(data['fit_ids']),
        'fit_sha256':hashlib.sha256(data['fit'].tobytes()).hexdigest(),
        'development_counts_sha256':hashlib.sha256(data['dev_observed'].tobytes()).hexdigest(),
        'expected_weights_sha256':hashlib.sha256(expected['native_descriptor_weights'].tobytes()).hexdigest(),
        'expected_probabilities_sha256':hashlib.sha256(expected['native_decoded_probabilities'].tobytes()).hexdigest(),
        'inference_policy':policy}
    cached=path.is_file() and index.is_file()
    if cached:
        saved=read_json(index)
        if saved.get('identity')!=cache_identity or file_hash(path)!=saved['sha256']:
            raise ValueError('Frozen classical estimator cache differs from its fitting-only preparation.')
        # Only this pipeline's locally generated, hash-verified estimator is read.
        state=pickle.loads(path.read_bytes())
        if state['method']!=method:raise ValueError('Frozen classical estimator method changed.')
        model,transformer,dev_rng=state['model'],state['transformer'],state['development_rng_state']
    else:
        log.new_classical_fits=getattr(log,'new_classical_fits',0)+1
        log.info('FROZEN CLASSICAL REPLAY | %s seed=%d | original fitting data/profile | no tuning',method,seed)
    if method=='lda':
        from gensim.models import LdaModel
        def corpus(matrix):
            return [[(int(j),float(v)) for j,v in zip(matrix[i].indices,matrix[i].data)] for i in range(matrix.shape[0])]
        if not cached:
            model=LdaModel(corpus=corpus(fit),id2word=dict(enumerate(data['vocabulary'])),num_topics=10,
                random_state=int(seed),alpha=profile['alpha'],eta=profile['eta'],passes=profile['passes'],
                iterations=profile['iterations'],minimum_probability=0.)
            dev_rng=model.random_state.get_state()
        else:model.random_state.set_state(dev_rng)
        weights=model.get_topics().astype(np.float64)
        gamma,_=model.inference(corpus(dev),collect_sstats=False)
        dev_prob=normalized_reconstruction(normalized_reconstruction(gamma)@normalized_reconstruction(weights))
        # Freeze a test inference stream only after original development replay.
        model.random_state=np.random.RandomState(int(seed)+policy['lda_posterior_inference_seed_offset'])
        inference=lambda x:normalized_reconstruction(model.inference(corpus(x),collect_sstats=False)[0])
    elif method=='nmf':
        from sklearn.decomposition import NMF
        from sklearn.feature_extraction.text import TfidfTransformer
        if not cached:
            transformer=TfidfTransformer();transformed=transformer.fit_transform(fit)
            model=NMF(n_components=10,init=profile['init'],solver=profile['solver'],beta_loss=profile['beta_loss'],
                max_iter=profile['max_iter'],tol=profile['tol'],random_state=int(seed),alpha_W=0,alpha_H=0,l1_ratio=0.)
            model.fit_transform(transformed)
        weights=model.components_.astype(np.float64)
        dev_prob=normalized_reconstruction(model.transform(transformer.transform(dev))@weights)
        inference=lambda x:model.transform(transformer.transform(x))
    else:raise ValueError('Unknown classical frozen interface.')
    if method=='lda':tol={'atol':policy['lda_float32_replay_atol'],'rtol':policy['lda_float32_replay_rtol']}
    checks={'descriptor_weights':assert_replay(weights,expected['native_descriptor_weights'],method+' readout',**tol),
        'development_probabilities':assert_replay(dev_prob,expected['native_decoded_probabilities'],method+' development prediction',**tol)}
    from .adapters.pilot_anchors import topics_from_weights
    if topics_from_weights(weights,data['vocabulary'])!=topics_from_weights(expected['native_descriptor_weights'],data['vocabulary']):
        raise ValueError('Classical replay changed the frozen TOP10 descriptors.')
    if method=='lda':
        from .neural_prediction import backoff as mix,score_log_predictions as score
        prior=data['fit'].astype(np.float64).sum(0)+.5;prior/=prior.sum()
        original=expected['native_decoded_probabilities']
        # Fixed rho=.2 is the locked LDA profile's saved development choice.
        # This checks drift only; no backoff or profile is selected here.
        actual_scores,_=score(mix(np.log(dev_prob),prior,.2),data['dev_records'],data['masks'],sorted({r['source'] for r in data['fit_records']}))
        original_scores,_=score(mix(np.log(original),prior,.2),data['dev_records'],data['masks'],sorted({r['source'] for r in data['fit_records']}))
        drift=abs(actual_scores['macro_source_nll']-original_scores['macro_source_nll'])
        checks['development_macro_nll_drift']=drift
        if drift>policy['maximum_lda_development_macro_nll_drift']:
            raise ValueError('LDA reconstruction changed development NLL beyond the declared float32 replay allowance.')
    if not cached:
        atomic_bytes(path,pickle.dumps({'method':method,'model':model,'transformer':transformer,
                                     'development_rng_state':dev_rng},protocol=4))
        write_json(index,{'identity':cache_identity,'method':method,'seed':seed,'profile':profile,'sha256':file_hash(path),
            'fitting_only':True,'development_replay':checks,'test_used_for_fit_or_selection':False})
    if test is None:return None,None,checks
    theta=inference(test)
    test_prob=normalized_reconstruction(theta@(normalized_reconstruction(weights) if method=='lda' else weights))
    return test_prob,theta,checks
