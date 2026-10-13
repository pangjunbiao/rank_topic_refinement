"""Immutable external contract, provenance, and primary-result protection."""
from __future__ import annotations
import copy
import importlib.metadata
from pathlib import Path
from rank_refinement.runtime.artifacts import digest, file_hash, read_json

PROTOCOL = 'STEP07_EXTERNAL_TRANSIT_RECIPE_REFIT_R1_20261010'
DATA_SHA = '3e386a2d6e0ba4023559992415880687326a43c86a7be36d226d960e1a39ec9d'
FREEZE_ID = '88b9b4de3e1f7d877709c0d104f377b26507093b0849fdbfb10d09f1658ba852'
LABELS = {'neuromax':'NeuroMax','encot':'EnCOT','lda':'LDA','nmf':'NMF',
          'gsdmm':'GSDMM','bertopic':'BERTopic','combinedtm':'CombinedTM',
          'fastopic':'FASTopic','glocom':'GloCOM'}
ENCODER = {
 'repository':'BAAI/bge-small-en-v1.5',
 'revision':'5c38ec7c405ec4b44b94cc5a9bb96e735b38267a',
 'weights':'model.safetensors',
 'weights_sha256':'3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad',
 'file_sha256':{
  'config.json':'094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750',
  'model.safetensors':'3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad',
  'vocab.txt':'07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3',
  'tokenizer_config.json':'9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3',
  'special_tokens_map.json':'b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3'},
 'endpoints':['https://huggingface.co','https://hf-mirror.com'],
 'local_directory':'data/assets/bge-small-en-v1.5',
 'batch_size':32,'max_length':512,'pooling':'cls_l2','instruction':'none'}

def recipe(configs):
    """Method and comparator profiles are inherited, never reselected externally."""
    from rank_refinement.stages.step05_ablations import profiles_to_run, profile_settings, LABEL_TO_METHOD
    old = configs['step05_replication']
    panel = [{'method':LABEL_TO_METHOD[label], 'label':label,
              'profile':profile_settings(configs,LABEL_TO_METHOD[label],name)}
             for label,name in profiles_to_run(old)]
    return {'protocol':PROTOCOL,'external_mode':'frozen_recipe_refit',
        'dataset_sha256':DATA_SHA,'primary_freeze_id':FREEZE_ID,
        'language':'English','seeds':[809,810,811,812,813],
        'topics':10,'topn':10,'base':'neuromax','profile':'random_words',
        'graph':'signed_stable','projection':'range','moment_correction':False,
        'native_training':copy.deepcopy(configs['step03_neural']),
        'acquisition':copy.deepcopy(configs['step03_improvement']['acquisition']),
        'calibration':copy.deepcopy(configs['step02_refinement']['acquisition']),
        'projection_settings':copy.deepcopy(configs['step03_improvement']['projection']),
        'secondary_settings':copy.deepcopy(configs['step03_improvement']['controls']),
        'prediction':copy.deepcopy(configs['step03_neural']['projection_pilot']),
        'phrases':copy.deepcopy(configs['data']['primary']['phrases']),
        'vocabulary':copy.deepcopy(configs['data']['primary']['vocabulary']),
        'near_duplicates':copy.deepcopy(configs['data']['primary']['near_duplicates']),
        'completion':{**copy.deepcopy(configs['data']['primary']['completion']),'seed':20261018},
        'split_seed':20261017,'split_ratios':{'fit':.7,'development':.1,'test':.2},
        'tokenizer':'English surface words, lowercase, preserved negation, sklearn stoplist',
        'encoder':copy.deepcopy(ENCODER),'comparators':panel,
        'controls':['primary_tv','display_only','moment_correction'],
        'bootstrap_resamples':10000,'bootstrap_seed':7809,
        'labels_used':False,'test_used_for_selection':False,
        'no_edit_policy':'native abstention when acquisition has no supported edit',
        'projection_failure_policy':'explicit unavailable slot; never replace failed projection with native',
        'numerical_retry':'same_LP_affinity_variables_only_for_tiny_negative_affinity; original_time_and_TV_budgets; no_clipping',
        'development_role':'fixed backoff grid only; no method/profile/epoch search'}

def implementation(root):
    paths = list((root/'step07_external').rglob('*.py')) + [root/'step07_settings.json']
    return {p.relative_to(root).as_posix():file_hash(p) for p in sorted(paths)}

def primary_guard(root,configs):
    from step06_finalize import verify_exposure
    from rank_refinement.stages.step02_completion import check_indexed_tree
    output,identity,ledger_sha = verify_exposure(root,configs)
    check_indexed_tree(output)
    result = read_json(output/'run_result.json')
    if result.get('status') != 'STEP06_FROZEN_PRIMARY_TEST_COMPLETE' or not result.get('engineering_passed'):
        raise RuntimeError('Step 6 must be fully completed before external evaluation.')
    paths = list((root/'state').glob('step0[123456]*.json'))
    paths += [p for p in output.rglob('*') if p.is_file()]
    return {'freeze_id':identity and FREEZE_ID,'exposure_ledger_sha256':ledger_sha,
            'protected':{p.relative_to(root).as_posix():file_hash(p) for p in sorted(paths)}}

def assert_primary_unchanged(root,snapshot):
    for relative,expected in snapshot['protected'].items():
        p = root/relative
        if not p.is_file() or file_hash(p) != expected:
            raise RuntimeError('Primary result changed during external run: '+relative)

def versions(full=False):
    from rank_refinement.runtime.config import check_environment
    from rank_refinement.runtime.step02_contract import dependency_versions
    result = check_environment()
    result.update(dependency_versions({'scikit-learn':'1.5.2'},'requirements-step07.txt'))
    if full:
        from rank_refinement.stages.step04_primary_benchmark import dependencies
        result.update(dependencies())
    result.update({name:importlib.metadata.version(name) for name in ['numpy','scipy','gensim','scikit-learn']})
    return result

def input_file(root,override=None):
    if override:
        p = Path(override)
        p = p if p.is_absolute() else root/p
        found = [p.resolve()]
    else:
        found = [p.resolve() for directory in ['data/raw/external','data/external']
                 for p in (root/directory).glob('*.csv')]
        found = sorted(set(found))
    matching = [p for p in found if p.is_file() and file_hash(p)==DATA_SHA]
    if len(matching) != 1:
        raise RuntimeError('Keep one copy of Transit Tweet Data.csv in data/raw/external. '
                           'Step 7 requires the inspected CSV bytes; do not edit or replace them. '
                           'Use --external-input PATH only to locate that same file.')
    return matching[0]
