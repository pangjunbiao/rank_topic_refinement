#!/usr/bin/env python3
"""Restore verified frozen report inputs into the matching public source release."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

FORMAT = 'frozen-report-packaging-v1.0'
PREFIXES = ('data/processed/', 'data/transportation_audit_v1/', 'state/', 'outputs/steps/')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe(root, relative):
    rel = PurePosixPath(relative)
    if not relative or rel.is_absolute() or '..' in rel.parts or '\\' in relative or ':' in relative:
        raise ValueError('Unsafe saved path: ' + relative)
    p = root.joinpath(*rel.parts)
    if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
        raise ValueError('Saved path leaves its root: ' + relative)
    return p


def check_source(root, archived):
    lineage = json.loads((root / 'docs/source_lineage.json').read_text(encoding='utf-8'))['files']
    required = Counter((r['archived_sha256'], r['archived_bytes']) for r in lineage)
    supplied = Counter((r['sha256'], r['bytes']) for r in archived)
    optional = Counter((r['archived_sha256'], r['archived_bytes']) for r in lineage
                       if r['optional_in_initial_archive'])
    missing = required - supplied
    if supplied - required or missing - optional:
        raise ValueError('Archive does not correspond to the documented source lineage.')
    for r in lineage:
        p = safe(root, r['public_path'])
        if not p.is_file() or p.stat().st_size != r['public_bytes'] or sha(p) != r['public_sha256']:
            raise ValueError('Public source differs from the audited release: ' + r['public_path'])


def restore(bundle, root):
    root = Path(root).resolve()
    with zipfile.ZipFile(bundle) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members.')
        manifest = json.loads(z.read('FROZEN_REPORTING_MANIFEST.json'))
        if manifest.get('format') != FORMAT:
            raise ValueError('Unsupported reporting archive format.')
        check_source(root, manifest['source_files'])
        records = manifest['files']
        paths = [r['path'] for r in records]
        if len(paths) != len(set(paths)):
            raise ValueError('Duplicate payload paths.')
        expected = {'FROZEN_REPORTING_MANIFEST.json', 'README.md'} | {'payload/' + p for p in paths}
        if set(names) != expected:
            raise ValueError('Unregistered ZIP members.')
        for number, r in enumerate(records, 1):
            if not r['path'].startswith(PREFIXES):
                raise ValueError('Unapproved reporting input location: ' + r['path'])
            p = safe(root, r['path'])
            h, size = hashlib.sha256(), 0
            with z.open('payload/' + r['path']) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk); size += len(chunk)
            if size != r['bytes'] or h.hexdigest() != r['sha256']:
                raise ValueError('Archive input checksum failed: ' + r['path'])
            if p.exists() and (not p.is_file() or sha(p) != r['sha256']):
                raise ValueError('Existing different input will not be overwritten: ' + r['path'])
            if number % 500 == 0:
                print('VERIFIED INPUTS |', number, '/', len(records), flush=True)
        count = 0
        for r in records:
            p = safe(root, r['path'])
            if p.exists():
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            with z.open('payload/' + r['path']) as source, p.open('xb') as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            count += 1
        for r in records:
            if sha(safe(root, r['path'])) != r['sha256']:
                raise ValueError('Restored input checksum failed: ' + r['path'])
    print('RESTORATION COMPLETE | new files=', count, '| original saved bytes retained | models fitted=0')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--project-root', default=Path(__file__).resolve().parent, type=Path)
    args = parser.parse_args()
    try:
        restore(args.bundle, args.project_root)
        return 0
    except Exception as error:
        print('FAILED | ' + type(error).__name__ + ': ' + str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
