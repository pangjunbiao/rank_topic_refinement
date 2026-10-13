"""Freeze/firewall checks and real-author load-only inference on synthetic data."""
from __future__ import annotations
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy import sparse
from rank_refinement.runtime.artifacts import atomic_bytes, digest, file_hash, read_json, write_json
from rank_refinement.runtime.config import load_configs
from rank_refinement.models.frozen_inference import (assert_replay,checkpoint_index,restore_neural,
    native_latents,context_inputs,competitor_probabilities,classical_probabilities,
    frozen_encoder_folder,encode_test_observed)
from rank_refinement.evaluation.frozen_test import load_test_view,frozen_probability_scores,summarize_test
from rank_refinement.stages.step06_frozen_test import validate_protocol,freeze_test_exposure,test_row

ROOT=Path(__file__).resolve().parents[1]


class FreezeAndIsolation(unittest.TestCase):
    def test_protocol_is_frozen_and_has_no_neural_refit(self):
        policy,_=validate_protocol(load_configs(ROOT),ROOT)
        self.assertFalse(policy['freeze']['refit_neural_models'])
        self.assertFalse(policy['freeze']['reselect_graph_projection_profiles_seeds_or_rho'])
        self.assertFalse(policy['freeze']['external_evaluation'])

    def test_modified_policy_is_rejected(self):
        cfg=load_configs(ROOT);cfg['step06_frozen_test']['freeze']['correction_in_proposed']=True
        with self.assertRaises(ValueError):validate_protocol(cfg,ROOT)

    def test_exposed_test_allows_identical_restart_but_blocks_reselection(self):
        with tempfile.TemporaryDirectory() as d:
            key,replay=freeze_test_exposure(d,{'method':'frozen','rho':0.})
            self.assertFalse(replay)
            self.assertEqual(freeze_test_exposure(d,{'method':'frozen','rho':0.}),(key,True))
            with self.assertRaises(ValueError):freeze_test_exposure(d,{'method':'frozen','rho':.1})
            with self.assertRaises(ValueError):freeze_test_exposure(d,{'method':'changed','rho':0.})

    def synthetic_preparation(self,folder):
        vocabulary=['a','b','c'];features=[{'group_id':'test0','source':'s','split':'test','tokens':['a','b','c'],'clean_text':'TARGET FULL TEXT'}]
        mask={'group_id':'test0','eligible':True,'observed_ids':[0,0],'target_ids':[1],
              'observed_positions':[0,1],'target_positions':[2],'observed_text':'OBSERVED ONLY'}
        for name,rows in [('feature_records.jsonl',features),('completion_masks.jsonl',[mask])]:
            (folder/name).write_text('\n'.join(__import__('json').dumps(x) for x in rows))
        write_json(folder/'vocabulary.json',{'tokens':vocabulary});write_json(folder/'matrix_rows.json',['test0'])
        write_json(folder/'completion_group_ids_by_split.json',{'test':['test0']})
        sparse.save_npz(folder/'observed_counts.npz',sparse.csr_matrix([[2,0,0]]))
        return mask

    def test_test_view_supplies_observed_text_and_counts_only(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);self.synthetic_preparation(folder)
            view=load_test_view(folder,['fit0'],['development0'])
            self.assertEqual(view['observed_texts'],['OBSERVED ONLY'])
            self.assertNotIn('clean_text',view['records'][0])
            self.assertNotIn('tokens',view['records'][0])
            np.testing.assert_array_equal(view['observed_counts'],[[2,0,0]])

    def test_test_group_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);self.synthetic_preparation(folder)
            with self.assertRaises(ValueError):load_test_view(folder,['test0'],[])

    def test_counts_different_from_observed_masks_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);self.synthetic_preparation(folder)
            sparse.save_npz(folder/'observed_counts.npz',sparse.csr_matrix([[2,1,0]]))
            with self.assertRaises(ValueError):load_test_view(folder,[],[])

    def test_checkpoint_missing_or_changed_fails_without_training(self):
        with tempfile.TemporaryDirectory() as d:
            details={'checkpoint_relative_path':'data/cache/step03_native/x/checkpoint.pt','checkpoint_sha256':'bad','epochs':1,'identity':{}}
            with self.assertRaises(RuntimeError):checkpoint_index(d,details,('data','cache','step03_native'))
            path=Path(d)/details['checkpoint_relative_path'];atomic_bytes(path,b'changed')
            with self.assertRaises(ValueError):checkpoint_index(d,details,('data','cache','step03_native'))

    def test_fixed_rho_is_applied_without_search(self):
        view={'ids':['g'],'records':[{'group_id':'g','source':'s'}],'masks':{'g':{'eligible':True,'target_ids':[1]}}}
        score,_=frozen_probability_scores(np.array([[.8,.2]]),.1,np.array([.5,.5]),view,['s'])
        self.assertAlmostEqual(score['macro_source_nll'],-np.log(.23))

    def test_zero_target_probability_is_not_clipped_or_retuned(self):
        view={'ids':['g'],'records':[{'group_id':'g','source':'s'}],'masks':{'g':{'eligible':True,'target_ids':[1]}}}
        with self.assertRaises(ValueError):frozen_probability_scores(np.array([[1.,0.]]),0.,np.array([.5,.5]),view,['s'])

    def test_zero_non_target_probability_is_valid(self):
        view={'ids':['g'],'records':[{'group_id':'g','source':'s'}],'masks':{'g':{'eligible':True,'target_ids':[0]}}}
        score,_=frozen_probability_scores(np.array([[1.,0.]]),0.,np.array([.5,.5]),view,['s'])
        self.assertEqual(score['macro_source_nll'],0.)

    def test_partial_seed_mean_is_rejected_and_na_is_not_a_win(self):
        base={'method':'M','profile':'p','status':'scored','fitting':{'c_v':.5,'c_npmi':.1},
              'test_coherence':{'c_v':None,'c_npmi':None},'prediction':{'macro_source_nll':1.,'micro_nll':1.},'topic_diversity':1.}
        rows=[{**copy.deepcopy(base),'seed':s} for s in [1,2,3,4,5]]
        with self.assertRaises(ValueError):summarize_test(rows[:-1],[1,2,3,4,5])
        rows[-1]['prediction']['macro_source_nll']=None
        result=summarize_test(rows,[1,2,3,4,5])[0]
        self.assertIsNone(result['test_macro_nll_mean']);self.assertEqual(result['test_macro_nll_available_seeds'],4)
        self.assertIsNone(result['test_c_v_mean'])

    def test_replay_shape_and_numeric_differences_fail(self):
        with self.assertRaises(ValueError):assert_replay(np.ones((2,3)),np.ones((3,2)),'shape')
        with self.assertRaises(ValueError):assert_replay(np.ones((2,3))*2,np.ones((2,3)),'values')

    def test_development_replay_failure_stops_before_test_exposure(self):
        from rank_refinement.stages.step06_frozen_test import run
        from test_step04_competitors import QuietLog
        cfg=load_configs(ROOT);policy=cfg['step06_frozen_test'];old=cfg['step05_replication']
        module='rank_refinement.stages.step06_frozen_test.'
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'outputs/steps/step05_recovery/synthetic';source.mkdir(parents=True)
            write_json(source/'artifact_manifest.json',[])
            for name in ['completed_step05_reference.json','completed_step04_reference.json','source_neural_pilot_reference.json',
                         'repaired_saved_source_reference.json','completed_pilot_reference.json','frozen_proposed_reference.json']:
                write_json(source/name,{'relative_path':'synthetic_fixture'})
            log=QuietLog();log.folder=root/'outputs/runs/synthetic';log.folder.mkdir(parents=True)
            with patch(module+'validate_protocol',return_value=(policy,old)),\
                 patch(module+'require_preparation',return_value={'preparation_id':policy['preparation_id'],'processed_relative_path':'synthetic_data'}),\
                 patch(module+'accepted_source',return_value=(source,[],[],{'synthetic':True})),\
                 patch(module+'source_folders',return_value={}),patch(module+'load_canonical',return_value={}),\
                 patch(module+'preflight',return_value=({}, {}, [])),\
                 patch(module+'native_sources',return_value=({},[])),patch(module+'competitor_sources',return_value=({},[])),\
                 patch(module+'prepare_development_interfaces',side_effect=RuntimeError('development replay failed')),\
                 patch(module+'freeze_test_exposure') as expose,patch(module+'load_test_view') as test_view:
                with self.assertRaisesRegex(RuntimeError,'development replay failed'):
                    run(root,cfg,{}, {'future_model_device':'cpu'},log)
                expose.assert_not_called();test_view.assert_not_called()
            self.assertFalse((root/'state/step06_test_exposure.json').exists())

    def test_context_assignment_matches_fixed_centers(self):
        counts=np.array([[1.,0.],[0.,2.]],dtype=np.float32)
        emb=np.array([[0.,0.],[9.,9.]],dtype=np.float32)
        bank=np.array([[3.,4.],[5.,6.]],dtype=np.float32);centers=np.array([[0.,0.],[10.,10.]],dtype=np.float32)
        np.testing.assert_array_equal(context_inputs(counts,emb,bank,centers),np.concatenate([counts,bank],axis=1))

    def test_restored_context_predictions_equal_original_kmeans_without_fitting(self):
        from sklearn.cluster import KMeans
        rng=np.random.default_rng(809);fit=rng.normal(size=(80,6)).astype(np.float32)
        original=KMeans(n_clusters=7,n_init=3,random_state=809).fit(fit)
        queries=rng.normal(size=(30,6)).astype(np.float32);counts=np.ones((30,10),dtype=np.float32)
        bank=rng.normal(size=(7,10)).astype(np.float32)
        expected=bank[original.predict(queries)]
        with patch('sklearn.cluster.KMeans.fit',side_effect=AssertionError('No fitting of test context bank')):
            actual=context_inputs(counts,queries,bank,original.cluster_centers_)
        np.testing.assert_array_equal(actual[:,10:],expected)


class FrozenEncoderAssets(unittest.TestCase):
    def fixture(self, root):
        from rank_refinement.models.encoder_assets import FILES,resolve_encoder
        from test_step04_competitors import QuietLog
        policy=copy.deepcopy(load_configs(ROOT)['acquisition_anchors']['encoder'])
        folder=Path(root)/policy['local_directory']/policy['revision']
        folder.mkdir(parents=True)
        for name in FILES:(folder/name).write_bytes(('synthetic pinned asset '+name).encode('utf-8'))
        policy['file_sha256']={name:file_hash(folder/name) for name in FILES}
        policy['weights_sha256']=policy['file_sha256']['model.safetensors']
        with patch('urllib.request.urlopen',side_effect=AssertionError('Existing assets must not be downloaded')):
            old_folder,assets=resolve_encoder(root,policy,QuietLog())
        return old_folder,assets

    def test_frozen_encoder_reuses_original_revision_folder_without_download(self):
        with tempfile.TemporaryDirectory() as d:
            original,assets=self.fixture(d)
            (original.parent/'config.json').write_bytes(b'wrong unversioned decoy')
            before={p.name:file_hash(p) for p in original.iterdir() if p.is_file()}
            with patch('urllib.request.urlopen',side_effect=AssertionError('No download during frozen inference')):
                actual=frozen_encoder_folder(d,assets)
            self.assertEqual(actual,original.resolve())
            self.assertEqual(before,{p.name:file_hash(p) for p in original.iterdir() if p.is_file()})

    def test_missing_revision_does_not_fall_back_to_unversioned_encoder(self):
        with tempfile.TemporaryDirectory() as d:
            original,assets=self.fixture(d)
            for item in assets['files']:
                path=original/item['path']
                (original.parent/item['path']).write_bytes(path.read_bytes())
                path.unlink()
            with self.assertRaisesRegex(RuntimeError,'Saved asset missing'):
                frozen_encoder_folder(d,assets)

    def test_changed_pinned_encoder_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            original,assets=self.fixture(d)
            path=original/'model.safetensors';payload=path.read_bytes()
            path.write_bytes(bytes([payload[0]^1])+payload[1:])
            with self.assertRaisesRegex(ValueError,'Original pinned encoder file changed'):
                frozen_encoder_folder(d,assets)

    def test_encoder_identity_escape_duplicate_and_missing_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            _,assets=self.fixture(d)
            changes=[]
            value=copy.deepcopy(assets);value['revision']='0'*40;changes.append(value)
            value=copy.deepcopy(assets);value['repository']='replacement/model';changes.append(value)
            value=copy.deepcopy(assets);value['policy']['local_directory']='data/assets/../../outside';changes.append(value)
            value=copy.deepcopy(assets);value['files'][-1]=copy.deepcopy(value['files'][0]);changes.append(value)
            value=copy.deepcopy(assets);value['files'].pop();changes.append(value)
            for i,value in enumerate(changes):
                with self.subTest(change=i),self.assertRaises(ValueError):frozen_encoder_folder(d,value)

    def test_preflight_checks_original_revision_layout(self):
        from rank_refinement.stages.step06_frozen_test import preflight
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,assets=self.fixture(root)
            source=root/'outputs/source';source.mkdir(parents=True)
            write_json(source/'protocol_resolved_before_scoring.json',{'device':'cpu','versions':{}})
            manifest={'assets':assets};write_json(source/'shared_encoder_manifest.json',manifest)
            sentinel={'synthetic_saved_vectors':True}
            with patch('rank_refinement.stages.step06_frozen_test.saved_embeddings',return_value=sentinel):
                vectors,saved,checkpoints=preflight(root,source,{},[],{}, {},{},'cpu',{})
            self.assertIs(vectors,sentinel);self.assertEqual(saved,manifest);self.assertEqual(checkpoints,[])
            self.assertFalse((root/'state/step06_test_exposure.json').exists())

    def test_observed_encoder_loads_revision_folder_and_reuses_verified_cache(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import Mock
        from test_step04_competitors import QuietLog
        class TinyEncoder(torch.nn.Module):
            def forward(self,input_ids,attention_mask):
                values=torch.ones((*input_ids.shape,3),device=input_ids.device)
                return SimpleNamespace(last_hidden_state=values)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder,assets=self.fixture(root)
            output=root/'outputs/synthetic';output.mkdir(parents=True)
            manifest={'assets':assets,'dimension':3}
            tokenizer=Mock(return_value={'input_ids':torch.ones((1,2),dtype=torch.long),
                                         'attention_mask':torch.ones((1,2),dtype=torch.long)})
            with patch('transformers.AutoModel.from_pretrained',return_value=TinyEncoder()) as loader,\
                 patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer) as token_loader:
                result=encode_test_observed(root,manifest,['test0'],['OBSERVED ONLY'],'cpu',output,QuietLog())
                cached=encode_test_observed(root,manifest,['test0'],['OBSERVED ONLY'],'cpu',output,QuietLog())
            loader.assert_called_once_with(str(folder.resolve()),local_files_only=True,
                                           trust_remote_code=False,use_safetensors=True)
            token_loader.assert_called_once_with(str(folder.resolve()),local_files_only=True,trust_remote_code=False)
            self.assertEqual(tokenizer.call_args.args[0],['OBSERVED ONLY'])
            np.testing.assert_array_equal(result,cached)
            np.testing.assert_allclose(np.linalg.norm(result,axis=1),1.,atol=1e-5)


class RealAuthorFrozenInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        from rank_refinement.models.neural_sources import resolve_sources as native
        from rank_refinement.models.competitor_sources import resolve_sources as competitor
        cls.nf,_=native(ROOT);cls.cf,_=competitor(ROOT)
        cls.configs=load_configs(ROOT)

    def fixture(self):
        from test_step04_competitors import fixture
        return fixture()

    def checkpoint(self,root,model,identity,native):
        import torch
        path=Path(root)/'data/cache'/('step03_native' if native else 'step04_training')/'synthetic/checkpoint.pt'
        saved={'identity':identity,'epoch':1,'model':model.state_dict()}
        stream=io.BytesIO();torch.save(saved,stream);atomic_bytes(path,stream.getvalue())
        write_json(path.with_suffix('.json'),{'identity':identity,'epoch':1,'checkpoint_sha256':file_hash(path)})
        return {'checkpoint_relative_path':path.relative_to(root).as_posix(),'checkpoint_sha256':file_hash(path),
                'epochs':1,'identity' if native else 'training_identity':identity}

    def test_native_neuromax_and_encot_restore_without_optimizer_or_fit(self):
        import torch
        from rank_refinement.models.neural_training import make_model,export_latents
        from rank_refinement.models.neural_prediction import affine_parameters
        data,vectors=self.fixture();cfg=copy.deepcopy(self.configs);p=cfg['step03_neural']
        p['sinkhorn_max_iter']=100
        for method in ['neuromax','encot']:
            with self.subTest(method=method),tempfile.TemporaryDirectory() as d:
                p['training'][method]['en_units']=16
                if method=='encot':p['training'][method]['num_clusters']=4
                profile='random_words'
                model=make_model(method,self.nf[method],100,12,p,seed=809).eval()
                inputs=data['dev_observed'] if method=='neuromax' else np.concatenate([data['dev_observed'],data['dev_observed']],axis=1)
                gain,offset=affine_parameters(model)
                expected={'native':model.get_beta().detach().double().numpy(),'decoder_gain':gain,'decoder_offset':offset,
                          'theta_development':export_latents(model,inputs,'cpu')}
                identity={'base':method,'profile':profile,'seed':809,'device':'cpu','torch':torch.__version__}
                details=self.checkpoint(Path(d),model,identity,True)
                with patch('torch.optim.Adam',side_effect=AssertionError('No optimizer during frozen inference')):
                    restored=restore_neural(d,method,profile,809,details,self.nf[method],vectors,cfg,'cpu')
                    theta,checks=native_latents(restored,inputs,inputs[::-1].copy(),expected,cfg['step06_frozen_test']['inference'],'cpu')
                self.assertEqual(theta.shape,(20,10));self.assertLess(max(checks.values()),1e-6)

    def test_ctm_fastopic_and_glocom_restore_and_predict_with_frozen_reference(self):
        import torch
        from rank_refinement.models.competitor_neural import make_model,decode_model,cluster_contexts
        data,vectors=self.fixture();cfg=self.configs
        devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for method in ['combinedtm','fastopic','glocom']:
                with self.subTest(method=method,device=device),tempfile.TemporaryDirectory() as d:
                    profile=copy.deepcopy(cfg['step04_competitors']['profiles'][method][0])
                    if method=='combinedtm':profile['hidden_sizes']=[16,16]
                    if method=='glocom':profile.update(clusters=4,en_units=16)
                    # The pinned CTM constructor moves beta/priors to CUDA when
                    # available, while its encoder initially remains on CPU.
                    # Place the entire fixture explicitly, as production does.
                    model=make_model(method,self.cf[method],100,12,profile,
                        words=vectors['words'] if method=='glocom' else None,seed=809).to(device).eval()
                    if method=='combinedtm':model.topic_word_matrix=model.beta
                    for tensor in list(model.parameters())+list(model.buffers()):
                        self.assertEqual(tensor.device.type,device)
                    contexts=cluster_contexts(data['fit'],vectors['fit_full'],data['dev_observed'],vectors['development_observed'],profile,809) if method=='glocom' else None
                    w,p,_,_=decode_model(method,model,data['dev_observed'],vectors,contexts,809)
                    expected={'native_descriptor_weights':w,'native_decoded_probabilities':p}
                    if contexts:expected.update(global_bank=contexts['bank'],global_centers=contexts['centers'])
                    identity={'method':method,'profile':profile,'seed':809,'device':device,'torch':torch.__version__}
                    details=self.checkpoint(Path(d),model,identity,False)
                    with patch('torch.optim.Adam',side_effect=AssertionError('No optimizer during frozen inference')):
                        restored=restore_neural(d,method,profile,809,details,self.cf[method],vectors,cfg,device)
                        for tensor in list(restored.parameters())+list(restored.buffers()):
                            self.assertEqual(tensor.device.type,device)
                        result,theta,checks=competitor_probabilities(method,restored,data,vectors,data['dev_observed'][::-1].copy(),
                            vectors['development_observed'][::-1].copy(),expected,809,cfg['step06_frozen_test']['inference'])
                    np.testing.assert_allclose(result.sum(1),1.,atol=1e-10);self.assertEqual(theta.shape,(20,10))
                    self.assertLess(max(checks.values()),1e-6)

    def test_classical_replay_matches_saved_development_before_test(self):
        from rank_refinement.models.classical_completion import fit_and_predict
        from rank_refinement.stages.step05_ablations import profile_settings
        from test_step04_competitors import QuietLog
        data,_=self.fixture()
        for method in ['lda','nmf']:
            with self.subTest(method=method),tempfile.TemporaryDirectory() as d:
                profile=copy.deepcopy(profile_settings(self.configs,method,'fitting_strong_k10'))
                if method=='lda':profile.update(passes=2,iterations=20)
                else:profile.update(max_iter=80)
                _,w,p,details=fit_and_predict(method,data['fit'],data['dev_observed'],data['vocabulary'],809,profile)
                test=data['dev_observed'][::-1].copy()
                expected={'native_descriptor_weights':w,'native_decoded_probabilities':p};log=QuietLog()
                prepared,_,_=classical_probabilities(method,data,None,809,profile,expected,details,self.configs['step06_frozen_test']['inference'],d,log)
                self.assertIsNone(prepared);self.assertEqual(log.new_classical_fits,1)
                block='gensim.models.LdaModel.__init__' if method=='lda' else 'sklearn.decomposition.NMF.fit_transform'
                with patch(block,side_effect=AssertionError('No refit after pretest preparation')):
                    result,_,checks=classical_probabilities(method,data,test,809,profile,expected,details,self.configs['step06_frozen_test']['inference'],d,log)
                self.assertEqual(log.new_classical_fits,1)
                self.assertLess(max(checks.values()),1e-10);np.testing.assert_allclose(result.sum(1),1.,atol=1e-10)
                self.assertTrue((Path(d)/'classical_frozen_model.pkl').is_file())

    def test_gsdmm_restores_sufficient_statistics_without_refitting(self):
        from rank_refinement.models.competitor_classical import fit_gsdmm
        from test_step04_competitors import QuietLog
        data,_=self.fixture();profile=copy.deepcopy(self.configs['step04_competitors']['profiles']['gsdmm'][0]);profile['iterations']=3
        _,w,p,_,details=fit_gsdmm(data['fit'],data['dev_observed'],data['vocabulary'],809,profile,QuietLog())
        with tempfile.TemporaryDirectory() as d:
            with patch('rank_refinement.vendor.gsdmm_mgp.MovieGroupProcess.fit',side_effect=AssertionError('No GSDMM refit')):
                result,_,checks=classical_probabilities('gsdmm',data,data['dev_observed'][::-1].copy(),809,profile,
                    {'native_descriptor_weights':w,'native_decoded_probabilities':p},details,self.configs['step06_frozen_test']['inference'],d,QuietLog())
            self.assertLess(max(checks.values()),1e-12);np.testing.assert_allclose(result.sum(1),1.,atol=1e-10)


class ReportIntegration(unittest.TestCase):
    def test_all_profiles_controls_tables_figures_and_uncertainty_export(self):
        from rank_refinement.stages.step06_frozen_test import report,export_progress
        from rank_refinement.stages.step05_ablations import profiles_to_run
        from rank_refinement.models.neural_prediction import score_log_predictions
        from test_step04_competitors import fixture
        cfg=load_configs(ROOT);policy=copy.deepcopy(cfg['step06_frozen_test']);old=cfg['step05_replication']
        policy['reporting']['bootstrap_resamples']=20
        data,_=fixture()
        for r in data['fit_records']:r['clean_text']=' '.join(r['tokens'])
        lp=np.full(data['dev_observed'].shape,-np.log(len(data['vocabulary'])))
        scores,losses=score_log_predictions(lp,data['dev_records'],data['masks'],['source0','source1','source2'])
        topics=[data['vocabulary'][k*10:(k+1)*10] for k in range(10)]
        baseline={'profile':'random_words','role':'native','status':'scored','topics':topics,
            'fitting':{'c_v':.5,'c_npmi':.1},'test_coherence':{'c_v':None,'c_npmi':None},
            'topic_diversity':1.,'prediction':scores,'document_losses':losses,'rho':0.,
            'mechanism':{'relative_l1_affinity_movement':.01,'target_top10_membership_recall':1.,
                'maximum_topic_total_error_scaled':0.,'maximum_word_total_error_scaled':0.}}
        rows=[];controls=[]
        for method,profile in [('Proposed','random_words')]+profiles_to_run(old):
            for seed in policy['seeds']:
                row={**copy.deepcopy(baseline),'method':method,'profile':profile,'seed':seed}
                if method=='BERTopic':row.update(prediction={'macro_source_nll':None,'micro_nll':None},document_losses=[])
                rows.append(row)
        for role in cfg['step05_recovery']['controls']:
            for seed in policy['seeds']:
                row={**copy.deepcopy(baseline),'method':role,'role':role,'seed':seed}
                if role=='strict_word_disjointness' and seed==812:
                    row.update(status='unavailable',topics=None,fitting={'c_v':None,'c_npmi':None},topic_diversity=None,
                        prediction={'macro_source_nll':None,'micro_nll':None},document_losses=[])
                controls.append(row)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);output=root/'outputs/steps/step06_frozen_test/synthetic';output.mkdir(parents=True)
            export_progress(output,rows,controls,policy,True)
            gates=report(root,output,rows,controls,data,policy,old,{'synthetic':True})
            self.assertEqual(gates['comparison_rows'],60);self.assertEqual(gates['control_slots'],60)
            self.assertEqual(len(gates['unavailable_control_slots']),1)
            self.assertTrue((output/'figures/F6_source_prediction_heterogeneity.pdf').is_file())
            self.assertTrue((output/'figures/S6_actual_projection_tradeoff.png').is_file())
            self.assertTrue((output/'automatic_fitting_cases/topic_examples.csv').is_file())
            intervals=read_json(output/'paired_prediction_intervals.json')
            self.assertEqual(sum(r['primary_contrast'] for r in intervals),1)
            self.assertTrue(all(not r['multiplicity_adjusted'] for r in intervals))
            summary=(output/'SUMMARY.md').read_text()
            self.assertIn('original Step 5 all-metric-win gate remains false',summary)
            self.assertIn('BERTopic word NLL is NA',summary)
