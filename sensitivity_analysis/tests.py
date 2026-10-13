from __future__ import annotations
import copy
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
from .core import (safe,load_population,acquire,replay_selected,predict,summarize,write,read,sha)
from .runner import cache_load
from .vendor.projection.replay_solver import primary_tv,range_secondary
from .vendor.projection.minimum_tv import certify
from .vendor.models.neural_prediction import backoff,log_predictions


def run_checks(root,plan,data,bundles,cfg):
    class Checks(unittest.TestCase):
        def test_grid_budget_and_markers(self):
            self.assertEqual(5*len(plan['support_values'])+5*len(plan['margin_values'])*len(plan['projection_objectives']),65)
            self.assertIn(plan['selected_support'],plan['support_values']);self.assertIn(plan['selected_margin'],plan['margin_values'])
        def test_ablation_display_vs_decoder(self):
            from .report import ablation_readout_recall
            scores=ablation_readout_recall(root/'sensitivity_analysis')
            self.assertEqual(scores['Same-core display only'],'81.00 ± 2.92')
            for name in ['Proposed (prediction-aware range secondary)','Primary minimum-TV only','Index secondary','L2 projection']:
                self.assertEqual(scores[name],'100.00 ± 0.00')
        def test_firewall(self):
            self.assertFalse(set(data['fit_ids'])&set(data['dev_ids']))
            self.assertEqual(len(data['fit_ids']),1436);self.assertEqual(len(data['dev_ids']),203)
        def test_path_absolute_rejected(self):
            with self.assertRaises(ValueError):safe(root,'/tmp/foreign')
        def test_path_parent_rejected(self):
            with self.assertRaises(ValueError):safe(root,'../foreign')
        def test_missing_seed_is_not_averaged(self):
            rows=[{'study':'x','support':5,'margin':1e-6,'objective':'range','seed':809+i,'certified':True,'selected_setting':True,'wall_seconds':0.,'development_nll':1.} for i in range(4)]
            self.assertIsNone(summarize(rows)[0]['development_nll_mean'])
        def test_failed_seed_is_not_averaged(self):
            rows=[{'study':'x','support':5,'margin':1e-6,'objective':'range','seed':809+i,'certified':i<4,'selected_setting':True,'wall_seconds':0.,'development_nll':1. if i<4 else None} for i in range(5)]
            self.assertIsNone(summarize(rows)[0]['development_nll_mean']);self.assertEqual(summarize(rows)[0]['certified_seeds'],4)
        def test_backoff_zero_exact(self):
            lp=np.log(np.array([[.2,.8]]));self.assertTrue(np.array_equal(backoff(lp,np.array([.4,.6]),0.),lp))
        def test_log_decoder_independent(self):
            a=np.array([[1.,2.,3.],[3.,1.,2.]]);theta=np.array([[.3,.7]]);gain=np.array([1.,2.,-1.]);offset=np.array([.1,-.2,.3])
            logits=(theta@a)*gain+offset;expected=np.exp(logits-logits.max());expected/=expected.sum()
            self.assertTrue(np.allclose(np.exp(log_predictions(theta,a,gain,offset)),expected,atol=1e-14))
        def test_projection_conservation_and_secondary(self):
            a=np.array([[4.,3.,2.,1.],[1.,2.,3.,4.]])
            target=[[0,2],[1,3]];phi,pc=primary_tv(a,target,1e-5,5.)
            self.assertIsNotNone(phi);self.assertTrue(pc['certified'])
            contexts=np.array([[.2,.8],[.8,.2]])
            result,sc=range_secondary(a,target,pc['solver_tv_scaled'],contexts,np.ones(4),phi,1e-5,1e-5,5.,50,1e-8)
            self.assertIsNotNone(result);self.assertTrue(sc['certified'])
            self.assertTrue(np.allclose(a.sum(axis=0),result.sum(axis=0),atol=1e-9));self.assertTrue(np.allclose(a.sum(axis=1),result.sum(axis=1),atol=1e-9))
            self.assertLessEqual(sc['tv_scaled'],pc['solver_tv_scaled']+1e-5+1e-7)
        def test_infeasible_retained(self):
            a=np.ones((2,4));phi,pc=primary_tv(a,[[0,1],[0,1]],.1,5.)
            self.assertIsNone(phi);self.assertFalse(pc['certified'])
        def test_cache_mutation_detected(self):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d);write(p/'evidence.json',{'x':1});write(p/'result.json',{'key':'x','row':{},'output_hashes':{'evidence.json':sha(p/'evidence.json')}})
                write(p/'evidence.json',{'x':2})
                with self.assertRaises(ValueError):cache_load(p,'x')
        def test_cache_identity_detected(self):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d);write(p/'result.json',{'key':'a','row':{},'output_hashes':{}})
                with self.assertRaises(ValueError):cache_load(p,'b')
        def test_json_nonfinite_rejected(self):
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError):write(Path(d)/'bad.json',{'x':float('nan')})
    def make_replay(seed):
        def test(self):self.assertLess(replay_selected(bundles[seed],data,plan),1e-9)
        return test
    def make_population(seed):
        def test(self):
            score,losses=predict(bundles[seed],bundles[seed]['arrays']['native'],data,0.)
            self.assertEqual([r['group_id'] for r in losses],data['dev_ids']);self.assertEqual(set(score['by_source']),set(data['sources']))
        return test
    def make_targets(seed):
        def test(self):
            target,selection=acquire(bundles[seed],data,cfg['acquisition'],plan['selected_support'])
            self.assertEqual([sorted(t) for t in target],[sorted(t) for t in bundles[seed]['selected']['targets']])
            self.assertTrue(selection['required_diversity_achieved']);self.assertTrue(selection['only_fitting_groups_used'])
        return test
    for seed in plan['seeds']:
        setattr(Checks,f'test_selected_prediction_replay_{seed}',make_replay(seed))
        setattr(Checks,f'test_completion_population_{seed}',make_population(seed))
        setattr(Checks,f'test_original_acquisition_replay_{seed}',make_targets(seed))
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    return {'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'passed':r.wasSuccessful(),'details':stream.getvalue()}
