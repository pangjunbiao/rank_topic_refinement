"""Independent mathematical and matched-topic checks; no real model fits."""
import unittest
import numpy as np
from scipy.special import softmax
from .divergence import probabilities,distribution_metrics,decoded_profiles,target_recall


class DivergenceTests(unittest.TestCase):
    def test_identity(self):
        r=distribution_metrics([[.2,.3,.5]],[[.2,.3,.5]])
        self.assertAlmostEqual(r['js'][0],0,places=14)
        self.assertEqual(r['tv'][0],0)
    def test_disjoint_maximum(self):
        r=distribution_metrics([[1,0]],[[0,1]])
        self.assertAlmostEqual(r['js'][0],np.log(2),places=14)
        self.assertEqual(r['tv'][0],1)
    def test_symmetric(self):
        a,b=[[1,2,4]],[[2,3,1]]
        np.testing.assert_allclose(distribution_metrics(a,b)['js'],distribution_metrics(b,a)['js'])
    def test_scale_invariant(self):
        np.testing.assert_allclose(distribution_metrics([[1,2]],[[2,1]])['js'],
                                   distribution_metrics([[10,20]],[[200,100]])['js'])
    def test_reject_negative(self):
        with self.assertRaises(ValueError):probabilities([[-1,2]])
    def test_reject_nan(self):
        with self.assertRaises(ValueError):probabilities([[float('nan'),2]])
    def test_reject_zero_row(self):
        with self.assertRaises(ValueError):probabilities([[0,0]])
    def test_reject_shape(self):
        with self.assertRaises(ValueError):distribution_metrics([[1,2]],[[1,2,3]])
    def test_one_zero_allowed(self):
        r=distribution_metrics([[0,1]],[[.5,.5]])
        self.assertTrue(np.isfinite(r['js']).all())
    def test_near_identity(self):
        p=np.array([[.3,.7],[.123,.877]])
        r=distribution_metrics(p,p+np.array([[1e-15,-1e-15],[-1e-15,1e-15]]))
        self.assertTrue((r['js']>=0).all())
    def test_decoder_affine(self):
        a=np.array([[1.,3.,2.]])
        np.testing.assert_allclose(decoded_profiles(a,np.array([2.,1.,4.]),np.array([.1,.2,.3])),
                                   softmax([[2.1,3.2,8.3]],axis=1))
    def test_affinity_not_decoder(self):
        a=np.array([[1.,3.,2.]])
        self.assertFalse(np.allclose(probabilities(a),decoded_profiles(a,np.ones(3),np.zeros(3))))
    def test_decoder_mixture_not_profile_mixture(self):
        a=np.array([[1.,3.,2.],[4.,0.,2.]])
        t=np.array([[.4,.6]])
        self.assertFalse(np.allclose(softmax(t@a,axis=1),t@softmax(a,axis=1)))
    def test_target_recall(self):
        a=np.arange(15.,dtype=float)[None,:]
        self.assertEqual(target_recall(a,[list(range(5,15))])[0],1.)
        self.assertEqual(target_recall(a,[list(range(10))])[0],.5)
    def test_topic_rows_matched(self):
        a=np.array([[.8,.2],[.2,.8]])
        self.assertGreater(distribution_metrics(a,a[::-1])['js'].mean(),0)
    def test_duplicate_target_rejected(self):
        with self.assertRaises(ValueError):target_recall(np.arange(15.)[None,:],[[1]*10])
