"""Verify frozen result bytes and preserve all earlier artifacts."""
from __future__ import annotations
from pathlib import Path
from rank_refinement.runtime.artifacts import digest, file_hash, read_json

PROTOCOL = 'STEP08_VERIFIED_PAPER_ASSEMBLY_R1_20261010'
PRIMARY_ID = '88b9b4de3e1f7d877709c0d104f377b26507093b0849fdbfb10d09f1658ba852'
EXTERNAL_ID = 'e36f6cc7b16b57d24e32888349b7803605a2c42062c1c5a61a88bf2051c74b86'
EXTERNAL_FREEZE = '83e04b9f5223b3aa7b635f15720305535192a0c8590e82230d2b59b5d4c38bae'
SEEDS = [809, 810, 811, 812, 813]

def safe(root, relative):
    if not isinstance(relative, str) or '\\' in relative:
        raise ValueError('Invalid saved path: '+str(relative))
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe saved path: '+relative)
    path = (Path(root)/rel).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Saved path leaves the project: '+relative)
    return path

def verify_tree(folder, optional_checkpoints=False):
    folder=Path(folder).resolve();records=read_json(folder/'artifact_manifest.json')
    names=[r['path']for r in records]
    if len(names)!=len(set(names)):raise ValueError('Duplicate artifact entries: '+str(folder))
    present={p.relative_to(folder).as_posix()for p in folder.rglob('*')
             if p.is_file()and p.name!='artifact_manifest.json'}
    extra=present-set(names)
    if extra:raise ValueError('Unindexed result files: '+str(sorted(extra)[:5]))
    omitted=[]
    for r in records:
        p=safe(folder,r['path'])
        if not p.is_file():
            # Inspection ZIPs intentionally omit unused large training states.
            # Every predictive array, case, report, mask and index is mandatory.
            if optional_checkpoints and r['path'].startswith('workspace/data/cache/')and p.suffix=='.pt':
                omitted.append(r['path']);continue
            raise ValueError('Required result file missing: '+str(p))
        if p.stat().st_size!=r['bytes']or file_hash(p)!=r['sha256']:
            raise ValueError('Saved result bytes changed: '+str(p))
    return {'indexed_files':len(records),'omitted_unused_training_checkpoints':omitted}

def verify_code(root, expected, prefix):
    actual={p.relative_to(root).as_posix():file_hash(p)for p in sorted((root/prefix).rglob('*.py'))}
    # The public release edits only the teaching analogy in one vendored docstring.
    # Accept that exact byte change, while rejecting every other source mismatch.
    verified = dict(actual)
    path = 'src/rank_refinement/vendor/gsdmm_mgp.py'
    if (expected.get(path) == 'a4f704258106c005f563021283382eda2201e4ca61f4c6109829575c54926262'
            and actual.get(path) == '0d8eb606f4c7199b1d61fd2b51a1e3079d1f12dcc1ec83e332a48fecf37be44d'):
        verified[path] = expected[path]
    if verified != expected:
        raise ValueError('Frozen implementation changed: '+prefix)

def protect(root):
    from rank_refinement.runtime.config import load_configs
    root=Path(root).resolve();configs=load_configs(root)
    primary=root/'outputs/steps/step06_frozen_test'/PRIMARY_ID[:20]
    external=root/'outputs/steps/step07_external'/EXTERNAL_ID[:20]
    ledger=read_json(root/'state/step06_test_exposure.json')
    if ledger['freeze_id']!=PRIMARY_ID or digest(ledger['identity'])!=PRIMARY_ID:
        raise ValueError('Require the reviewed completed primary exposure.')
    if digest(configs)!=ledger['identity']['configs_sha256']:
        raise ValueError('Frozen configurations changed.')
    verify_code(root,ledger['identity']['implementation'],'src/rank_refinement')
    if read_json(primary/'protocol_resolved_before_test.json')!=ledger['identity']:
        raise ValueError('Primary result protocol differs from its exposure.')
    a=read_json(primary/'run_result.json');b=read_json(external/'run_result.json')
    if a['status']!='STEP06_FROZEN_PRIMARY_TEST_COMPLETE'or not a['engineering_passed']:
        raise ValueError('Complete Step 6 first; this stage does not repair or refit models.')
    if b['status']!='STEP07_EXTERNAL_REPLICATION_COMPLETE'or not b['engineering_passed']or b['experiment_id']!=EXTERNAL_ID:
        raise ValueError('Require the reviewed complete external replication.')
    identity=read_json(external/'protocol_resolved_before_training.json')
    if digest(identity)!=EXTERNAL_ID or identity['primary_freeze_id']!=PRIMARY_ID:
        raise ValueError('External recipe identity changed.')
    from step07_external.protocol import implementation
    if implementation(root)!=identity['implementation']:
        raise ValueError('Frozen Step 7 implementation changed; retain the supplied Step 7 files.')
    freeze=read_json(root/'state/step07_external_exposure.json')
    if (freeze['freeze_id']!=EXTERNAL_FREEZE or digest(freeze['identity'])!=EXTERNAL_FREEZE
            or freeze['identity']['experiment_id']!=EXTERNAL_ID
            or read_json(external/'external_test_freeze.json')!=freeze):
        raise ValueError('External exposure record changed.')
    audits={'primary':verify_tree(primary),'external':verify_tree(external,True)}
    for relative,sha in freeze['identity']['fit_cases'].items():
        folder=safe(external,relative)
        verify_tree(folder)
        if file_hash(folder/'artifact_manifest.json')!=sha:
            raise ValueError('Externally frozen predictive case changed: '+relative)
    state=read_json(root/'state/step01_accepted.json')
    preparation=safe(root,state['processed_relative_path'])
    verify_tree(preparation)
    if state['preparation_id']!=a['preparation_id']:
        raise ValueError('Primary preparation changed.')
    # Verify the primary snapshot made BEFORE the external experiment.
    previous=read_json(external/'primary_result_protection.json')
    for relative,sha in previous['protected'].items():
        p=safe(root,relative)
        if not p.is_file()or file_hash(p)!=sha:
            raise ValueError('Earlier primary payload changed: '+relative)
    files=list(primary.rglob('*'))+list(external.rglob('*'))+list(preparation.rglob('*'))
    files+=list((root/'state').glob('step0[1234567]*.json'))
    snapshot={p.relative_to(root).as_posix():file_hash(p)for p in sorted(set(files))if p.is_file()}
    return {'primary':primary,'external':external,'preparation':preparation,'configs':configs,
            'snapshot':snapshot,'integrity':audits,'external_identity':identity}

def unchanged(root, snapshot):
    for relative,sha in snapshot.items():
        p=safe(root,relative)
        if not p.is_file()or file_hash(p)!=sha:
            raise ValueError('Read-only assembly changed an earlier artifact: '+relative)

def implementation(root):
    paths=sorted((root/'step08_assembly').glob('*.py'))+[root/'step08_settings.json']
    return {p.relative_to(root).as_posix():file_hash(p)for p in paths}
