"""Mathematical and provenance checks independent of real training data."""
from __future__ import annotations
import hashlib
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
from . import metrics
from .runner import Inputs, safe


class DivergenceTests(unittest.TestCase):
    def test_identical(self):
        p=np.array([[.2,.3,.5]])
        np.testing.assert_allclose(metrics.divergence(p,p)['js'],[0],atol=1e-15)

    def test_disjoint_js_is_log_two(self):
        self.assertAlmostEqual(metrics.divergence([[1,0]],[[0,1]])['js'][0],math.log(2),places=15)

    def test_disjoint_tv_is_one(self):
        self.assertEqual(metrics.divergence([[1,0]],[[0,1]])['tv'][0],1)

    def test_known_nontrivial_value(self):
        self.assertAlmostEqual(metrics.divergence([[1,0]],[[.5,.5]])['js'][0],.21576155433883565,places=15)

    def test_not_square_root_distance(self):
        js=metrics.divergence([[1,0]],[[0,1]])['js'][0]
        self.assertGreater(abs(js-math.sqrt(math.log(2))),.1)

    def test_scale_invariant_affinity_shares(self):
        a,b=np.array([[1,2,3.]]),np.array([[3,2,1.]])
        np.testing.assert_allclose(metrics.divergence(a,b)['js'],metrics.divergence(a*10,b*.01)['js'],atol=1e-15)

    def test_symmetry(self):
        np.testing.assert_allclose(metrics.divergence([[1,2]],[[4,1]])['js'],metrics.divergence([[4,1]],[[1,2]])['js'])

    def test_zero_mass_rejected(self):
        with self.assertRaises(ValueError):metrics.divergence([[0,0]],[[1,0]])

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):metrics.divergence([[-1,2]],[[1,1]])

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):metrics.divergence([[np.nan,1]],[[1,1]])

    def test_population_shape_rejected(self):
        with self.assertRaises(ValueError):metrics.divergence([[1,1]],[[1,1],[1,1]])

    def test_arrays_never_modified(self):
        a,b=np.array([[1.,2.]]),np.array([[3.,4.]])
        original_a,original_b=a.copy(),b.copy();metrics.divergence(a,b)
        np.testing.assert_array_equal(a,original_a);np.testing.assert_array_equal(b,original_b)


class PredictionTests(unittest.TestCase):
    def test_softmax_normalized(self):
        p=np.exp(metrics.log_softmax([[1000,1001,999]]))
        self.assertAlmostEqual(p.sum(),1,places=14)

    def test_zero_gain_preserves_prediction_under_edit(self):
        a,b=np.array([[1.,2.]]),np.array([[4.,9.]])
        args=([0,0],[0,0],[.5,.5],0)
        np.testing.assert_array_equal(metrics.predict([[1]],a,*args),metrics.predict([[1]],b,*args))

    def test_frozen_backoff(self):
        p=np.exp(metrics.predict([[1]],[[2.,0.]],[1,1],[0,0],[.25,.75],.2))
        raw=np.exp(metrics.log_softmax([[2.,0.]]))
        np.testing.assert_allclose(p,.8*raw+.2*np.array([[.25,.75]]),atol=1e-15)

    def test_invalid_backoff_rejected(self):
        with self.assertRaises(ValueError):metrics.predict([[1]],[[1.,2.]],[1,1],[0,0],[.5,.5],1)

    def test_dimension_mismatch_rejected(self):
        with self.assertRaises(ValueError):metrics.predict([[1,0]],[[1.,2.]],[1,1],[0,0],[.5,.5],0)

    def test_source_macro_differs_from_micro(self):
        rows=[{'group_id':'a','source':'a','target_tokens':1,'negative_log_probability':1.},
              {'group_id':'b','source':'b','target_tokens':9,'negative_log_probability':18.}]
        actual=metrics.aggregate(rows)
        self.assertEqual(actual['macro_source_nll'],1.5);self.assertEqual(actual['micro_nll'],1.9)

    def test_repeated_target_occurrences_retained(self):
        logp=np.log(np.array([[.25,.75]]))
        masks={'x':{'eligible':True,'target_ids':[0,0]}}
        features={'x':{'source':'s','split':'test'}}
        row=metrics.losses(logp,['x'],masks,features)[0]
        self.assertEqual(row['target_tokens'],2)
        self.assertAlmostEqual(row['negative_log_probability'],-2*math.log(.25))

    def test_duplicate_groups_rejected(self):
        row={'group_id':'x','source':'s','target_tokens':1,'negative_log_probability':1.}
        with self.assertRaises(ValueError):metrics.aggregate([row,row])


class SelectionTests(unittest.TestCase):
    def test_target_inclusion_uses_independent_target(self):
        a=np.array([list(range(20,0,-1))],dtype=float)
        np.testing.assert_array_equal(metrics.target_inclusion(a,[list(range(10,20))]),[0])

    def test_duplicate_targets_rejected(self):
        with self.assertRaises(ValueError):metrics.target_inclusion(np.ones((1,20)),[[0]*10])

    def test_out_of_vocabulary_targets_rejected(self):
        with self.assertRaises(ValueError):metrics.target_inclusion(np.ones((1,20)),[list(range(15,25))])

    def test_top_tie_break_word_id(self):
        self.assertEqual(metrics.top_words(np.ones((1,12)))[0].tolist(),list(range(10)))

    def test_changed_columns_stable_tie_break(self):
        a,b=np.ones((2,30)),np.ones((2,30));b[:,8]+=1;b[:,4]+=1
        self.assertEqual(metrics.changed_columns(a,b,2).tolist(),[4,8])

    def test_changed_columns_use_full_topic_column_movement(self):
        a,b=np.ones((2,30)),np.ones((2,30));b[0,3]+=2;b[:,7]+=1.1
        self.assertEqual(metrics.changed_columns(a,b,1).tolist(),[7])


class ProvenanceTests(unittest.TestCase):
    def test_parent_path_rejected(self):
        with self.assertRaises(ValueError):safe(Path.cwd(),'../outside')

    def test_absolute_path_rejected(self):
        with self.assertRaises(ValueError):safe(Path.cwd(),str(Path.cwd()/'absolute'))

    def test_windows_separator_rejected(self):
        with self.assertRaises(ValueError):safe(Path.cwd(),'a\\b')

    def test_source_change_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);p=root/'input';p.write_bytes(b'original')
            reader=Inputs.__new__(Inputs);reader.root=root;reader.used={}
            reader.expected={'input':{'path':'input','bytes':8,'sha256':hashlib.sha256(b'original').hexdigest()}}
            reader.path('input');p.write_bytes(b'changed!')
            with self.assertRaises(ValueError):reader.unchanged()

    def test_unregistered_input_rejected(self):
        reader=Inputs.__new__(Inputs);reader.expected={}
        with self.assertRaises(ValueError):reader.path('not-registered')

    def test_wrong_input_hash_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'input').write_bytes(b'fake')
            reader=Inputs.__new__(Inputs);reader.root=root;reader.used={}
            reader.expected={'input':{'path':'input','bytes':4,'sha256':hashlib.sha256(b'true').hexdigest()}}
            with self.assertRaises(ValueError):reader.path('input')

    def test_packaged_reference_is_primary_only(self):
        from .runner import PACKAGE, read_json
        reference=read_json(PACKAGE/'data/reference_inputs.json')
        records=reference['files']
        self.assertEqual(len(records),len({r['path'] for r in records}))
        self.assertTrue(all('/external/' not in r['path'] and 'step07_external' not in r['path'] for r in records))

    def test_packaged_glosses_have_english_labels(self):
        from .runner import PACKAGE, read_json
        glossary=read_json(PACKAGE/'data/labels_en.json')
        self.assertEqual(len(glossary),24)
        self.assertTrue(all(g['label'].isascii() for g in glossary.values()))

    def test_transport_translations_preserve_six_evidence_ids(self):
        from .runner import PACKAGE, read_json
        translations=read_json(PACKAGE/'data/transport_examples_en.json')
        self.assertEqual(len(translations),6)
        self.assertTrue(all(len(g['original_excerpt_sha256'])==64 and g['english_excerpt'] for g in translations.values()))

