"""Numerical equivalence, complete separation and honest scientific gates."""
import copy
import unittest
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from rank_refinement.projection.predictive_controls import constraint_system, secondary_lp
from rank_refinement.projection.replay_solver import (primary_tv, index_secondary, range_secondary,
                                                     displacement_system)
from rank_refinement.runtime.config import load_configs
from rank_refinement.stages.step03_saved_replay import (validate_repair, source_folder,
    native_replay_check, summarize_replay)

ROOT = Path(__file__).resolve().parents[1]


class ReplayNumerics(unittest.TestCase):
    def setUp(self):
        self.native=np.array([[.8,.3,.4,.1,.2,.1],[.2,.7,.6,.9,.8,.9]])
        self.targets=[[0,2],[3,4]]
        self.theta=np.array([[.8,.2],[.2,.8],[.4,.6]])
        self.gain=np.arange(1,7,dtype=float)

    def test_displacement_and_affinity_primary_objectives_agree(self):
        a,s,au,bu,ae,be=constraint_system(self.native,self.targets,1e-6)
        n=self.native.size;k=2
        dense=linprog(np.r_[np.zeros(n),np.full(n,.5),np.zeros(k)],A_ub=au,b_ub=bu,
            A_eq=ae,b_eq=be,bounds=(0,None),method='highs-ipm')
        result,cert=primary_tv(self.native,self.targets,seconds=10)
        self.assertTrue(cert['success'])
        self.assertAlmostEqual(cert['solver_tv_scaled'],dense.fun,places=7)
        np.testing.assert_allclose(result.sum(0),self.native.sum(0),atol=1e-9)
        np.testing.assert_allclose(result.sum(1),self.native.sum(1),atol=1e-9)

    def test_index_equivalence_on_original_capped_feasible_set(self):
        primary,p=primary_tv(self.native,self.targets,seconds=10)
        first,_=secondary_lp(self.native,self.targets,p['solver_tv_scaled'],1e-6,1e-5,10,mode='index')
        result,cert=index_secondary(self.native,self.targets,p['solver_tv_scaled'],seconds=10)
        self.assertTrue(cert['success'])
        w=np.arange(1,self.native.size+1)/self.native.size
        self.assertAlmostEqual((first/ self.native.max()).ravel()@w,(result/self.native.max()).ravel()@w,places=7)

    def test_index_weights_are_constant_when_both_margins_are_conserved(self):
        primary,p=primary_tv(self.native,self.targets,seconds=10)
        indexed,cert=index_secondary(self.native,self.targets,p['solver_tv_scaled'],seconds=10)
        w=np.arange(1,self.native.size+1,dtype=float)/self.native.size
        for matrix in [primary,indexed]:
            self.assertAlmostEqual(matrix.ravel()@w,self.native.ravel()@w,places=9)

    def test_exact_separation_matches_dense_full_range_lp(self):
        primary,p=primary_tv(self.native,self.targets,seconds=10)
        dense,dc=secondary_lp(self.native,self.targets,p['solver_tv_scaled'],1e-6,1e-5,10,
            mode='range',contexts=self.theta,decoder_gain=self.gain)
        result,cert=range_secondary(self.native,self.targets,p['solver_tv_scaled'],self.theta,self.gain,primary,seconds=10)
        self.assertTrue(cert['success'])
        objective=lambda x:np.ptp((self.theta@(x-self.native))*self.gain,axis=1).mean()
        self.assertAlmostEqual(objective(result),objective(dense),places=7)
        self.assertLessEqual(cert['full_constraint_max_violation'],1e-8)
        self.assertLessEqual(cert['optimality_gap'],1e-7)

    def test_separation_certifies_all_words_not_just_active_constraints(self):
        rng=np.random.default_rng(10)
        theta=rng.dirichlet(np.ones(2),size=20)
        primary,p=primary_tv(self.native,self.targets,seconds=10)
        result,cert=range_secondary(self.native,self.targets,p['solver_tv_scaled'],theta,self.gain,primary,seconds=10)
        self.assertTrue(cert['success'])
        self.assertEqual(cert['full_range_constraints_count'],240)
        self.assertAlmostEqual(cert['secondary_objective'],np.ptp((theta@(result-self.native))*self.gain,axis=1).mean(),places=10)

    def test_zero_gain_contexts_are_finite_and_keep_constraints(self):
        primary,p=primary_tv(self.native,self.targets,seconds=10)
        result,cert=range_secondary(self.native,self.targets,p['solver_tv_scaled'],self.theta,np.zeros(6),primary,seconds=10)
        self.assertTrue(cert['success'])
        self.assertAlmostEqual(cert['secondary_objective'],0.)

    def test_solver_does_not_claim_success_after_iteration_budget(self):
        primary,p=primary_tv(self.native,self.targets,seconds=10)
        result,cert=range_secondary(self.native,self.targets,p['solver_tv_scaled'],self.theta,self.gain,primary,
            seconds=10,maximum_iterations=0)
        self.assertIsNone(result)
        self.assertFalse(cert['success'])


class ReplayScope(unittest.TestCase):
    def test_repair_cannot_silently_change_protocol_or_reference_gate(self):
        configs=load_configs(ROOT);validate_repair(configs)
        configs['step03_repair']['allow_step4_with_missing_development_coherence']=True
        with self.assertRaises(ValueError):validate_repair(configs)

    def test_repair_does_not_require_or_import_torch(self):
        import inspect
        import rank_refinement.stages.step03_saved_replay as module
        source=inspect.getsource(module)
        self.assertNotIn('import torch',source)
        self.assertNotIn('train_native(',source)
        self.assertNotIn('encode_views(',source)

    def test_standard_missing_reference_keeps_all_slots_and_na(self):
        from rank_refinement.evaluation.reference_audit import audit_reference
        words=['w'+str(i) for i in range(100)]
        topics=[words[k*10:(k+1)*10] for k in range(10)]
        records=[{'tokens':words[:-1]}]*3
        audit=audit_reference(topics,records,words)
        self.assertEqual(audit['total_slots'],100)
        self.assertEqual(audit['supported_slots'],99)
        self.assertEqual(audit['reference_absent_words'],['w99'])
        self.assertTrue(audit['all_descriptor_slots_retained'])
        self.assertEqual(len(audit['metrics']['c_npmi']['per_topic']),10)
        self.assertIsNone(audit['metrics']['c_npmi']['mean'])

    def test_source_path_cannot_escape_project(self):
        from rank_refinement.runtime.artifacts import write_json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);p=r/'outputs/steps/step03_neural_pilot/LATEST.json'
            write_json(p,{'relative_path':'../../outside'})
            with self.assertRaises(ValueError):source_folder(r)

    def test_display_control_in_the_selected_assessment_keeps_native_predictions(self):
        # Exercise the complete assessment on one frozen synthetic cache, with
        # the same real prediction/backoff functions as the saved-array replay.
        from rank_refinement.stages.step03_saved_replay import assess_selected_candidates
        from rank_refinement.stages.step03_neural_pilot import prediction_row
        from rank_refinement.stages.step02_native_refinement import save_arrays
        import tempfile
        policy=load_configs(ROOT)['step03_neural']
        native=np.arange(1,101,dtype=float).reshape(10,10)/100
        candidate=native.copy();candidate[0,0]+=.001;candidate[0,1]-=.001
        theta=np.ones((3,10))/10
        records=[{'group_id':'g'+str(i),'source':'city','tokens':['a','b']} for i in range(3)]
        # Match the production mask contract, without replacing its scorer.
        from rank_refinement.data.completion_masks import make_completion
        words=['a','b']+[str(i) for i in range(8)]
        mask_policy={'seed':809,'observed_fraction':.5,'minimum_in_vocabulary_tokens':2}
        masks={r['group_id']:make_completion(r['group_id'],r['tokens'],dict(zip(words,range(10))),mask_policy) for r in records}
        data={'fit_records':records,'dev_records':records,'masks':masks}
        common=dict(theta_fit=theta,theta_dev=theta,native=native,gain=np.ones(10),offset=np.zeros(10),
            prior=np.ones(10)/10,data=data,policy=policy['projection_pilot'],base='neuromax',profile='random_words',seed=809)
        metric={'scored':False,'c_v':.5,'c_npmi':-.1}
        topics=[[str(j) for j in range(10)] for _ in range(10)]
        n=prediction_row('native','NeuroMax',native,topics,metric,metric,1.,correction=False,deployed=False,**common)
        i=prediction_row('index','Index',candidate,topics,metric,metric,1.,**common)
        case={'base':'neuromax','profile':'random_words','seed':809,'rows':[n,i,{'role':'range','status':'failed'}],
            'acceptances':{'index':{'accepted':True}},'certificates':{}}
        class Log:
            def info(self,*args):pass
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); folder=root/'cases/neuromax/random_words/seed809'
            save_arrays(folder/'readouts_and_latents.npz',native=native,candidate_index=candidate,
                theta_fit=theta,theta_development=theta,decoder_gain=np.ones(10),decoder_offset=np.zeros(10),prior=np.ones(10)/10)
            reports=assess_selected_candidates(root,root,[case],data,policy,Log())
            rows={r['role']:r for r in reports[0]['rows']}
            self.assertEqual(rows['display_only']['prediction'],n['prediction'])
            self.assertEqual(rows['display_only']['document_losses'],n['document_losses'])
            self.assertEqual(rows['display_only']['topics'],rows['proposed']['topics'])
            self.assertFalse(rows['display_only']['projection_deployed'])
            self.assertTrue(rows['proposed']['projection_deployed'])

    def test_secondary_is_optional_and_missing_reference_still_blocks_step4(self):
        policy=load_configs(ROOT)['step03_neural'];cases=[]
        for seed in policy['seeds']:
            rows=[]
            for role in ['native','primary','index','range','l2']:
                score={'scored':False,'c_v':None,'c_npmi':None}
                rows.append({'role':role,'status':'scored','development':score,
                    'fitting':{'c_v':.4 if role=='native' else .5,'c_npmi':-.1 if role=='native' else 0.},
                    'prediction':{'macro_source_nll':6. if role!='range' else 6.0001}})
            cases.append({'base':'neuromax','profile':'random_words','seed':seed,'rows':rows,
                'certificates':{role:{'success':True} for role in ['primary','index','range','l2']},
                'acceptances':{role:{'accepted':True} for role in ['primary','index','range','l2']}})
        summary,choices=summarize_replay(cases,policy)
        self.assertEqual(choices[('neuromax','random_words')],'index')
        self.assertTrue(summary['profiles'][0]['descriptive_mechanism_and_predictive_budget_supported'])
        self.assertFalse(summary['profiles'][0]['complete_development_coherence_reference'])
        self.assertFalse(summary['ready_for_step4'])
        self.assertFalse(summary['original_step03_gate_passed'])


if __name__=='__main__':unittest.main()
