"""Regression checks for report-only completion of the exposed Step 6 run."""
from __future__ import annotations
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rank_refinement.runtime.artifacts import artifact_manifest, digest, file_hash, read_json, write_json
import step06_finalize as completion


class ReportCacheRepair(unittest.TestCase):
    def fixture(self, directory):
        view = {'coherence_records': [{'group_id': 'g', 'tokens': ['a', 'b']}],
                'vocabulary': ['a', 'b']}
        topics = [['a', 'b']]
        identity = {'topics': topics, 'reference': [('g', ['a', 'b'])],
                    'vocabulary': view['vocabulary'], 'windows': [110, 10], 'retain_all_slots': True}
        path = Path(directory) / (digest(identity) + '.json')
        audit = {'metrics': {'c_v': {'mean': .4, 'complete': True},
                             'c_npmi': {'mean': -.1, 'complete': True}},
                 'reference_groups': 1, 'supported_slots': 2, 'total_slots': 2,
                 'reference_absent_words': []}
        write_json(path, audit)
        write_json(path.with_suffix('.index.json'), {'identity': identity, 'audit_sha256': file_hash(path)})
        return topics, view, path, audit

    def test_tuple_list_cache_reuses_original_bytes_and_negative_score(self):
        with tempfile.TemporaryDirectory() as d:
            topics, view, path, audit = self.fixture(d)
            before = {p.name: file_hash(p) for p in Path(d).iterdir()}
            with patch('step06_finalize.audit_native_reference', side_effect=AssertionError('valid cache must be reused')):
                scores, saved = completion.corrected_descriptor_audit(topics, view, d)
            self.assertEqual(saved, audit)
            self.assertEqual(scores['c_npmi'], -.1)
            self.assertEqual(before, {p.name: file_hash(p) for p in Path(d).iterdir()})

    def test_corrupt_cache_is_rejected_without_recomputation(self):
        with tempfile.TemporaryDirectory() as d:
            topics, view, path, _ = self.fixture(d)
            path.write_bytes(b'{}')
            with patch('step06_finalize.audit_native_reference') as compute:
                with self.assertRaisesRegex(ValueError, 'genuinely changed'):
                    completion.corrected_descriptor_audit(topics, view, d)
                compute.assert_not_called()

    def test_partial_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            topics, view, path, _ = self.fixture(d)
            path.with_suffix('.index.json').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete coherence cache'):
                completion.corrected_descriptor_audit(topics, view, d)

    def test_missing_word_diagnostic_stays_na_and_retains_slots(self):
        with tempfile.TemporaryDirectory() as d:
            topics, view, path, audit = self.fixture(d)
            audit['metrics']['c_v'] = {'mean': None, 'complete': False}
            audit['metrics']['c_npmi'] = {'mean': None, 'complete': False}
            audit['supported_slots'] = 1; audit['reference_absent_words'] = ['b']
            path.unlink(); path.with_suffix('.index.json').unlink()
            with patch('step06_finalize.audit_native_reference', return_value=audit) as compute:
                scores, saved = completion.corrected_descriptor_audit(topics, view, d)
            compute.assert_called_once_with(topics, view['coherence_records'], view['vocabulary'])
            self.assertIsNone(scores['c_v']); self.assertIsNone(scores['c_npmi'])
            self.assertEqual(scores['total_slots'], 2); self.assertTrue(scores['all_descriptor_slots_retained'])
            self.assertEqual(saved['reference_absent_words'], ['b'])


class FrozenReportInputs(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory); key = 'a' * 64
        relative = 'data/processed/primary/' + key[:20]
        folder = root / relative
        cfg = {'default': {'seed': 809}, 'data': {'primary': {}}, 'evaluation': {'window': 10}}
        write_json(folder / 'configuration_snapshot.json', cfg)
        write_json(folder / 'input_manifest.json', [{'source': 's', 'sha256': 'original_raw_hash'}])
        state = {'preparation_id': key, 'processed_relative_path': relative,
                 'engineering_passed': True, 'preparation_gate_passed': True,
                 'artifacts': artifact_manifest(folder)}
        write_json(root / 'state/step01_accepted.json', state)
        return root, folder, {'preparation_id': key}, cfg

    def test_report_only_preparation_verifies_frozen_artifacts_without_raw_csv(self):
        with tempfile.TemporaryDirectory() as d:
            root, folder, identity, cfg = self.fixture(d)
            accepted = completion.frozen_preparation(root, identity, cfg)
            self.assertEqual(accepted['preparation_id'], identity['preparation_id'])
            self.assertFalse((root / 'data/raw').exists())
            self.assertEqual(read_json(folder / 'input_manifest.json')[0]['sha256'], 'original_raw_hash')

    def test_changed_processed_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root, folder, identity, cfg = self.fixture(d)
            write_json(folder / 'input_manifest.json', [])
            with self.assertRaisesRegex(ValueError, 'Accepted artifact changed'):
                completion.frozen_preparation(root, identity, cfg)

    def test_changed_preparation_config_or_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root, folder, identity, cfg = self.fixture(d)
            changed = copy.deepcopy(cfg); changed['default']['seed'] = 810
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                completion.frozen_preparation(root, identity, changed)
            with self.assertRaisesRegex(ValueError, 'differs from the frozen test'):
                completion.frozen_preparation(root, {'preparation_id': 'b' * 64}, cfg)

    def test_reporting_fields_are_excluded_but_prediction_changes_are_detected(self):
        row = {'prediction': {'macro_source_nll': 6.1}, 'topics': [['a']], 'test_coherence': {'c_v': None}}
        a = completion.predictive_digest([row])
        row['test_coherence']['c_v'] = .5
        self.assertEqual(a, completion.predictive_digest([row]))
        row['prediction']['macro_source_nll'] = 6.0
        self.assertNotEqual(a, completion.predictive_digest([row]))


if __name__ == '__main__':
    unittest.main()
