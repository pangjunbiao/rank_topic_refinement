"""Independent mathematical and real-author-interface checks; synthetic data only."""
from __future__ import annotations
import copy
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = all(importlib.util.find_spec(name) for name in ["torch", "geomloss", "torch_kmeans", "osqp", "sentence_transformers"])
if os.environ.get("RANK_STEP03_REQUIRED_CHECKS") and not AVAILABLE:
    raise ImportError("Step 3's required interface checks cannot be skipped; install requirements-step03.txt.")


class PredictionMath(unittest.TestCase):
    def test_moment_identity_is_exact(self):
        from rank_refinement.models.neural_prediction import moment_parameters
        phi = np.ones((3, 5))
        a, b = moment_parameters(np.ones((4, 3)), phi, phi)
        np.testing.assert_array_equal(a, np.ones(5))
        np.testing.assert_array_equal(b, np.zeros(5))

    def test_moments_match_fitting_population(self):
        from rank_refinement.models.neural_prediction import moment_parameters
        theta = np.array([[.8,.2],[.2,.8],[.5,.5]])
        native = np.array([[.9,.2,.7],[.1,.8,.3]])
        refined = np.array([[.6,.3,.8],[.4,.7,.2]])
        a, b = moment_parameters(theta, native, refined)
        corrected = (theta @ refined) * a + b
        np.testing.assert_allclose(corrected.mean(0), (theta @ native).mean(0), atol=1e-12)
        np.testing.assert_allclose(corrected.std(0), (theta @ native).std(0), atol=1e-12)

    def test_constant_moments_stay_finite(self):
        from rank_refinement.models.neural_prediction import moment_parameters
        a, b = moment_parameters(np.ones((3,2)), np.ones((2,4)), np.full((2,4),2.))
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())

    def test_log_space_extremes_no_clipping(self):
        from rank_refinement.models.neural_prediction import log_predictions
        p = log_predictions(np.array([[1.]]), np.array([[0.,1000.]]), np.ones(2), np.zeros(2))
        self.assertAlmostEqual(p[0,0], -1000.)
        self.assertAlmostEqual(p[0,1], 0.)

    def test_backoff_matches_probability_mixture(self):
        from rank_refinement.models.neural_prediction import backoff
        p = np.array([[.2,.8],[.7,.3]])
        prior = np.array([.4,.6])
        np.testing.assert_allclose(np.exp(backoff(np.log(p), prior, .1)), .9*p+.1*prior)
        np.testing.assert_array_equal(backoff(np.log(p),prior,0),np.log(p))

    def test_macro_nll_is_source_balanced(self):
        from rank_refinement.models.neural_prediction import score_log_predictions
        records = [{"group_id":"a","source":"s1"},{"group_id":"b","source":"s2"}]
        masks = {"a":{"eligible":True,"target_ids":[0]},"b":{"eligible":True,"target_ids":[1,1,1]}}
        scores, rows = score_log_predictions(np.log([[.25,.75],[.9,.1]]),records,masks,["s1","s2"])
        self.assertAlmostEqual(scores["macro_source_nll"],(-np.log(.25)-np.log(.1))/2)
        self.assertEqual(len(rows),2)
        self.assertNotAlmostEqual(scores["micro_nll"],scores["macro_source_nll"])

    def test_source_missing_is_not_silently_ignored(self):
        from rank_refinement.models.neural_prediction import score_log_predictions
        with self.assertRaises(ValueError):
            score_log_predictions(np.log([[.25,.75]]),[{"group_id":"a","source":"s1"}],
                {"a":{"eligible":True,"target_ids":[0]}},["s1","s2"])

    def test_logit_range_bounds_actual_cross_entropy_change(self):
        from scipy.special import logsumexp
        rng=np.random.default_rng(11)
        for _ in range(20):
            logits=rng.normal(size=7); delta=rng.normal(size=7); y=rng.dirichlet(np.ones(7))
            change=logsumexp(logits+delta)-logsumexp(logits)-y@delta
            self.assertLessEqual(abs(change),np.ptp(delta)+1e-12)

    def test_latents_are_not_row_normalized(self):
        from rank_refinement.models.neural_prediction import log_predictions
        theta=np.array([[.1,.2]])
        phi=np.array([[.3,.8],[.7,.2]])
        actual=log_predictions(theta,phi,np.ones(2),np.zeros(2))
        normalized=log_predictions(theta/theta.sum(1,keepdims=True),phi,np.ones(2),np.zeros(2))
        self.assertFalse(np.allclose(actual,normalized))


@unittest.skipUnless(AVAILABLE, "Step 3 dependencies are optional for Steps 1/2 only")
class PredictiveOptimization(unittest.TestCase):
    def setUp(self):
        from rank_refinement.projection.minimum_tv import minimum_tv
        self.native=np.array([[.8,.3,.4,.1,.2,.1],[.2,.7,.6,.9,.8,.9]])
        self.targets=[[0,2],[3,4]]
        self.primary,self.cert=minimum_tv(self.native,self.targets,seconds=10)
        self.assertIsNotNone(self.primary)
        self.theta=np.array([[.8,.2],[.2,.8],[.4,.6]])
        self.gain=np.arange(1,7,dtype=float)

    def test_secondary_preserves_all_totals_and_primary_cap(self):
        from rank_refinement.projection.predictive_controls import secondary_lp
        for mode in ["index","range"]:
            result,cert=secondary_lp(self.native,self.targets,self.cert["tv_scaled"],1e-6,1e-5,10,
                mode,self.theta,self.gain)
            self.assertTrue(cert["certified"] and cert["success"])
            np.testing.assert_allclose(result.sum(0),self.native.sum(0),atol=1e-8)
            np.testing.assert_allclose(result.sum(1),self.native.sum(1),atol=1e-8)
            self.assertLessEqual(cert["tv_scaled"],self.cert["tv_scaled"]+1.01e-5)

    def test_range_objective_matches_independent_logit_changes(self):
        from rank_refinement.projection.predictive_controls import secondary_lp
        result,cert=secondary_lp(self.native,self.targets,self.cert["tv_scaled"],1e-6,1e-5,10,
            "range",self.theta,self.gain)
        actual=np.ptp((self.theta@(result-self.native))*self.gain,axis=1).mean()
        self.assertAlmostEqual(actual,cert["secondary_objective"],places=7)
        self.assertAlmostEqual(actual,cert["measured_fitting_logit_range_mean"],places=10)

    def test_range_cannot_be_worse_than_index_on_its_fitting_bound(self):
        from rank_refinement.projection.predictive_controls import secondary_lp
        args=(self.native,self.targets,self.cert["tv_scaled"],1e-6,1e-5,10)
        index,_=secondary_lp(*args,mode="index")
        ranged,_=secondary_lp(*args,mode="range",contexts=self.theta,decoder_gain=self.gain)
        objective=lambda p:np.ptp((self.theta@(p-self.native))*self.gain,axis=1).mean()
        self.assertLessEqual(objective(ranged),objective(index)+1e-7)

    def test_l2_uses_original_feasible_set(self):
        from rank_refinement.projection.predictive_controls import l2_projection
        result,cert=l2_projection(self.native,self.targets,1e-6,10,1e-9)
        self.assertTrue(cert["certified"])
        self.assertFalse(cert["tv_cap_applied"])
        self.assertLessEqual(((result-self.native)**2).sum(),((self.primary-self.native)**2).sum()+1e-7)

    def test_singular_contexts_do_not_break_lp(self):
        from rank_refinement.projection.predictive_controls import secondary_lp
        result,cert=secondary_lp(self.native,self.targets,self.cert["tv_scaled"],1e-6,1e-5,10,
            "range",np.ones((2,2)),np.zeros(6))
        self.assertIsNotNone(result)
        self.assertTrue(cert["certified"])


class DataAndProtocol(unittest.TestCase):
    def test_singleton_batch_is_merged_without_losing_any_document(self):
        from rank_refinement.models.neural_training import batches
        groups=batches(401,200,8)
        self.assertEqual(sorted(np.concatenate(groups)),list(range(401)))
        self.assertTrue(all(len(g)>=2 for g in groups))

    def test_fitting_context_selection_is_unique_and_source_balanced(self):
        from rank_refinement.stages.step03_neural_pilot import fitting_context_ids
        records=[{"source":str(i%3),"group_id":str(i)} for i in range(100)]
        chosen=fitting_context_ids(records,64,809)
        self.assertEqual(len(chosen),64)
        self.assertEqual(len(set(chosen)),64)
        counts=[sum(records[i]["source"]==str(s) for i in chosen) for s in range(3)]
        self.assertLessEqual(max(counts)-min(counts),1)
        np.testing.assert_array_equal(chosen,fitting_context_ids(records,64,809))

    def test_protocol_changes_are_detected(self):
        from rank_refinement.stages.step03_neural_pilot import validate_protocol
        policy=yaml.safe_load((ROOT/'configs/step03_neural.yaml').read_text())
        validate_protocol({"step03_neural":policy})
        policy['projection_pilot']['backoff_grid'].append(.3)
        with self.assertRaises(ValueError):validate_protocol({"step03_neural":policy})

    def test_test_split_is_excluded_from_canonical_model_views(self):
        from rank_refinement.models.neural_training import load_canonical
        candidates=list((ROOT/'data/processed/primary').glob('*/vocabulary.json'))
        if not candidates:self.skipTest('Canonical accepted data not needed for synthetic self-check')
        data=load_canonical(candidates[0].parent)
        self.assertTrue(all(r['split']=='fit' for r in data['fit_records']))
        self.assertTrue(all(r['split']=='development' for r in data['dev_records']))
        self.assertEqual(data['fit'].shape[1],len(data['vocabulary']))

    def test_incomplete_references_do_not_qualify_a_profile(self):
        from rank_refinement.stages.step03_neural_pilot import summarize
        policy=yaml.safe_load((ROOT/'configs/step03_neural.yaml').read_text())
        self.assertFalse(summarize([],policy)['pilot_investment_gate_passed'])
        self.assertIsNone(summarize([],policy)['selected_base'])


@unittest.skipUnless(AVAILABLE, "Step 3 dependencies are optional for Steps 1/2 only")
class AuthorInterfaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        from rank_refinement.models.neural_sources import resolve_sources
        cls.folders,cls.specs=resolve_sources(ROOT)
        cls.policy=yaml.safe_load((ROOT/'configs/step03_neural.yaml').read_text())
        cls.policy['sinkhorn_max_iter']=100
        cls.policy['training']['encot']['num_clusters']=4
        for base in ['encot','neuromax']:
            cls.policy['training'][base]['en_units']=16

    def model(self,base):
        from rank_refinement.models.neural_training import make_model
        return make_model(base,self.folders[base],30,16,self.policy,seed=809)

    def test_author_model_eval_decoders_replay_without_row_normalization(self):
        import torch
        from rank_refinement.models.neural_prediction import verify_native_interface
        for base in ['encot','neuromax']:
            m=self.model(base).eval();x=torch.rand(5,30)
            if base=='encot':x=torch.cat([x,x*3],1)
            cert=verify_native_interface(m,x)
            self.assertTrue(cert['verified'])
            np.testing.assert_allclose(m.get_beta().detach().numpy().sum(0),np.ones(30),atol=1e-6)
            if base=='encot':self.assertTrue(torch.all(m.get_theta(x).sum(1)<1))

    def test_both_author_objectives_have_finite_gradients(self):
        import torch
        for base in ['encot','neuromax']:
            m=self.model(base).train();x=torch.rand(8,30)*2
            inputs=torch.cat([x,x*3],1) if base=='encot' else {'data':x,'contextual_embed':torch.rand(8,16)}
            output=m(inputs) if base=='encot' else m(inputs,epoch_id=10)
            output['loss'].backward()
            self.assertTrue(torch.isfinite(output['loss']))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_neuromax_group_regularizer_runs_on_cpu_after_epoch_ten(self):
        import torch
        m=self.model('neuromax').train()
        x={'data':torch.rand(8,30),'contextual_embed':torch.rand(8,16)}
        m(x,epoch_id=10);output=m(x,epoch_id=11)
        self.assertGreater(float(output['loss_GR'].detach()),0)
        self.assertEqual(m.group_connection_regularizer.device.type,'cpu')

    def test_encot_word_and_cluster_dimensions_match_chinese_width(self):
        from rank_refinement.models.neural_training import make_model
        words=np.random.default_rng(809).normal(size=(30,16)).astype(np.float32)
        model=make_model('encot',self.folders['encot'],30,16,self.policy,words=words)
        self.assertEqual(model.cluster_embeddings.shape[1],16)
        self.assertEqual(model.word_embeddings.shape[1],16)

    def test_neuromax_contrastive_projection_preserves_encoder_width(self):
        model=self.model('neuromax')
        self.assertEqual(model.prj_rep[0].out_features,16)
        self.assertEqual(model.weight_loss_InfoNCE,50.)

    def test_author_sources_are_seven_pinned_modules_only(self):
        self.assertEqual(len(self.specs),7)
        self.assertTrue(all(len(s['commit'])==40 for s in self.specs))
        self.assertTrue(all(Path(s['path']).name!='main.py' for s in self.specs))

    def test_corrupted_native_checkpoint_is_rejected(self):
        import torch
        from rank_refinement.models.neural_training import save_checkpoint
        from rank_refinement.runtime.artifacts import read_json,file_hash
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'checkpoint.pt';m=self.model('neuromax')
            save_checkpoint(p,m,torch.optim.Adam(m.parameters()),None,1,{'a':1},[])
            self.assertEqual(read_json(p.with_suffix('.json'))['checkpoint_sha256'],file_hash(p))
            p.write_bytes(p.read_bytes()+b'corrupt')
            self.assertNotEqual(read_json(p.with_suffix('.json'))['checkpoint_sha256'],file_hash(p))

    def test_checkpoint_resume_reproduces_uninterrupted_training(self):
        from rank_refinement.models.neural_training import train_native
        import torch
        policy=copy.deepcopy(self.policy)
        policy['training']['neuromax']['epochs']=12
        policy['checkpoint_every_epochs']=1
        policy['batch_size']=6
        data={'fit_ids':[str(i) for i in range(12)],'fit':np.random.default_rng(1).integers(0,4,size=(12,30)).astype(np.float32)}
        vectors={name:np.random.default_rng(2).normal(size=shape).astype(np.float32) for name,shape in [('fit_full',(12,16)),('words',(30,16))]}
        class Logger:
            interrupt=False
            def info(self,msg,*args):
                if self.interrupt and msg.startswith('TRAIN') and args[3]==10:
                    self.interrupt=False
                    raise KeyboardInterrupt('synthetic checkpoint interruption')
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            logger=Logger()
            first,_=train_native(Path(a),'neuromax','random_words',809,policy,self.folders,self.specs,data,vectors,{},'cpu',logger)
            logger.interrupt=True
            with self.assertRaises(KeyboardInterrupt):
                train_native(Path(b),'neuromax','random_words',809,policy,self.folders,self.specs,data,vectors,{},'cpu',logger)
            second,details=train_native(Path(b),'neuromax','random_words',809,policy,self.folders,self.specs,data,vectors,{},'cpu',logger)
            self.assertEqual(details['completed_before_run'],10)
            for key,value in first.state_dict().items():
                torch.testing.assert_close(value,second.state_dict()[key],rtol=0,atol=0)
            torch.testing.assert_close(first.group_connection_regularizer,second.group_connection_regularizer,rtol=0,atol=0)

    def test_complete_case_pipeline_and_cached_replay_for_both_author_bases(self):
        from rank_refinement.stages.step03_neural_pilot import run_case
        from rank_refinement.models.neural_training import global_context
        from rank_refinement.runtime.config import load_configs
        from rank_refinement.runtime.artifacts import file_hash
        rng=np.random.default_rng(809)
        v=100;vocabulary=['word'+str(i) for i in range(v)]
        fit=rng.integers(0,4,size=(60,v)).astype(np.float32)
        observed=rng.integers(0,3,size=(30,v)).astype(np.float32)
        def records(count,split,counts):
            result=[]
            for i in range(count):
                ids=np.repeat(np.arange(v),counts[i].astype(int));rng.shuffle(ids)
                result.append({'group_id':split+str(i),'split':split,'source':str(i%3),
                    'tokens':[vocabulary[int(j)] for j in ids]})
            return result
        fr=records(60,'fit',fit);dr=records(30,'development',observed)
        masks={r['group_id']:{'eligible':True,'target_ids':[i%v,(i+1)%v]} for i,r in enumerate(dr)}
        data={'fit':fit,'fit_observed':fit/2,'dev_observed':observed,'fit_ids':[r['group_id'] for r in fr],
            'fit_observed_ids':[r['group_id'] for r in fr],'dev_ids':[r['group_id'] for r in dr],
            'fit_records':fr,'fit_observed_records':fr,'dev_records':dr,'dev_coherence_records':dr,
            'masks':masks,'vocabulary':vocabulary,
            'metadata':{'tokens':vocabulary,'word_to_id':{w:i for i,w in enumerate(vocabulary)},
                        'document_frequency':(fit>0).sum(axis=0).tolist()}}
        vectors={name:rng.normal(size=(count,16)).astype(np.float32) for name,count in
            [('fit_full',60),('fit_observed',60),('development_observed',30),('words',100)]}
        for name in vectors:vectors[name]/=np.linalg.norm(vectors[name],axis=1,keepdims=True)
        policy=copy.deepcopy(self.policy);policy['global_context']['clusters']=4
        policy['checkpoint_every_epochs']=1;policy['batch_size']=20
        for base in ['encot','neuromax']:policy['training'][base]['epochs']=2
        contexts=global_context(data,vectors,policy['global_context'])
        r3=load_configs(ROOT)['step02_refinement']
        class Logger:
            def info(self,*args):pass
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for base in ['encot','neuromax']:
                folder=root/'cases'/base
                result=run_case(root,base,'random_words',809,data,vectors,contexts,self.folders,self.specs,
                    policy,r3,'cpu',Logger(),folder,root/'coherence')
                self.assertTrue(result['interface']['verified'])
                self.assertEqual(len(result['rows']),8)
                self.assertTrue(result['prediction_evaluated'])
                self.assertFalse(result['final_model_result'])
                by_role={r['role']:r for r in result['rows']}
                self.assertEqual(by_role['native']['prediction'],by_role['display_only']['prediction'])
                before=file_hash(folder/'artifact_manifest.json')
                again=run_case(root,base,'random_words',809,data,vectors,contexts,self.folders,self.specs,
                    policy,r3,'cpu',Logger(),folder,root/'coherence')
                self.assertEqual(again,result)
                self.assertEqual(before,file_hash(folder/'artifact_manifest.json'))

    def test_projected_case_has_actual_target_readout_and_native_display_predictor(self):
        # A controlled readout with two mixed descriptors exercises the full
        # projected branch. It is a mathematical fixture, not a model result.
        import torch
        from unittest.mock import patch
        from rank_refinement.stages.step03_neural_pilot import run_case
        from rank_refinement.runtime.config import load_configs
        from rank_refinement.runtime.artifacts import read_json
        vocabulary=['word'+str(i) for i in range(100)]
        counts=np.zeros((210,100),dtype=np.float32)
        records=[]
        for i in range(210):
            topic=i%10;counts[i,topic*10:(topic+1)*10]=2
            records.append({'group_id':'f'+str(i),'source':str(i%3),'split':'fit',
                'tokens':vocabulary[topic*10:(topic+1)*10]*2})
        dev=[dict(r,group_id='d'+str(i),split='development') for i,r in enumerate(records[:30])]
        data={'fit':counts,'fit_records':records,'fit_observed_ids':[r['group_id'] for r in records],
            'fit_observed_records':records,'fit_observed':counts/2,'dev_observed':counts[:30]/2,
            'dev_records':dev,'dev_coherence_records':dev,'vocabulary':vocabulary,'metadata':{},
            'masks':{r['group_id']:{'eligible':True,'target_ids':list(range((i%10)*10,(i%10+1)*10))} for i,r in enumerate(dev)}}
        native=np.full((10,100),.06,dtype=np.float32)
        for topic in range(10):native[topic,topic*10:(topic+1)*10]=.46
        native[0,9],native[0,19]=.06,.46
        native[1,9],native[1,19]=.46,.06
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.decoder_bn=torch.nn.BatchNorm1d(100).eval()
            def get_theta(self,x):return torch.softmax(x.reshape(len(x),10,10).sum(2)/10,dim=1)
            def get_beta(self):return torch.from_numpy(native)
        model=Model().eval()
        targets=[list(range(t*10,(t+1)*10)) for t in range(10)]
        selection={'selected':[{'fixture':True}]}
        bundle={'sample_group_ids':[],'pairs':np.zeros((0,2),dtype=int),'stable':np.zeros(0),
            'full':np.zeros(0),'pair_support':np.zeros(0),'word_support':np.zeros(100),'words':vocabulary,'fit_groups':210}
        policy=copy.deepcopy(self.policy)
        r3=load_configs(ROOT)['step02_refinement']
        class Logger:
            def info(self,*args):pass
        with tempfile.TemporaryDirectory() as tmp,patch('rank_refinement.stages.step03_neural_pilot.train_native',return_value=(model,{'synthetic_fixture':True})),patch('rank_refinement.stages.step03_neural_pilot.choose_targets',return_value=(targets,selection,bundle)):
            folder=Path(tmp)/'case'
            result=run_case(Path(tmp),'neuromax','random_words',809,data,{}, {},self.folders,self.specs,
                policy,r3,'cpu',Logger(),folder,Path(tmp)/'coherence')
            self.assertTrue(result['proposed_fitting_accepted'])
            self.assertTrue(result['all_controls_scored'])
            self.assertTrue(all(c['certified'] for c in result['certificates'].values()))
            rows={r['role']:r for r in result['rows']}
            self.assertEqual(rows['proposed']['method'],'Proposed')
            self.assertTrue(rows['proposed']['projection_deployed'])
            self.assertFalse(rows['display_only']['projection_deployed'])
            self.assertEqual(rows['display_only']['prediction'],rows['native']['prediction'])
            with np.load(folder/'readouts_and_latents.npz',allow_pickle=False) as saved:
                self.assertFalse(np.array_equal(saved['native'],saved['proposed']))
                for t in range(10):self.assertEqual(set(np.argsort(-saved['proposed'][t])[:10]),set(targets[t]))


if __name__=='__main__':unittest.main()
