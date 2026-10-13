"""Seed/group uncertainty, strict accounting and live CPU stage integration.

Synthetic fixtures test execution, never claim primary-corpus performance.
"""
from __future__ import annotations
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from rank_refinement.runtime.config import load_configs
from rank_refinement.runtime.artifacts import read_json, write_json, file_hash, artifact_manifest
from rank_refinement.evaluation.replication import ablated_evidence, assert_grid, assert_ablation_grid, hierarchical_gain, readout_mechanism
from rank_refinement.stages.step05_ablations import (validate_protocol, profiles_to_run, accepted_panel, native_case,
    classical_case, build_refinement_source, acquisition_ablation, final_reports, export_progress, profile_settings,
    replicate_new_seeds, parent_replication_gate)
from rank_refinement.stages.step03_improvement import run_case as fullset_case
from rank_refinement.models.neural_sources import resolve_sources
from rank_refinement.models.neural_training import global_context
from rank_refinement.reporting.export import make_inspection_zip
from test_step04_competitors import fixture, QuietLog

ROOT=Path(__file__).resolve().parents[1]


def loss_rows(seed, gain=0., method='Proposed', profile='random_words', role='proposed'):
    losses=[{'group_id':f'{source}_g{g}', 'source':source,'target_tokens':g+2,
             'negative_log_probability':(g+2)*(4.+gain)} for source in ['a','b','c'] for g in range(3)]
    return {'method':method,'profile':profile,'role':role,'seed':seed,'status':'scored','topics':None,
        'fitting':{'c_v':.7 if method=='Proposed' else .5,'c_npmi':.1 if method=='Proposed' else .05},
        'development':{'c_v':None,'c_npmi':None},'topic_diversity':1.,'fixed_k10':True,
        'native_topic_count':10,'prediction_interface':'synthetic_fixture',
        'prediction':{'macro_source_nll':4.+gain,'micro_nll':4.+gain},'rho':0.,'document_losses':losses,
        'test_or_external_used':False,'final_model_result':False}


def panel_rows(policy):
    seeds=policy['seeds']; profiles=profiles_to_run(policy)
    rows=[loss_rows(s) for s in seeds]
    for method,profile in profiles:
        for s in seeds:
            row=loss_rows(s,.02,method,profile,'competitor')
            if method=='BERTopic':row['prediction']={'macro_source_nll':None,'micro_nll':None};row['document_losses']=[]
            rows.append(row)
    ab=[]
    names={'native':'NeuroMax','proposed':'Proposed','pair_acquisition':'Previous pair','moment_control':'Moment',
        'no_backoff':'Without backoff','index_control':'Index','range_control':'Range','l2_control':'L2','display_only':'Display',
        'without_subsample_stability':'Without stability','positive_only_association':'Positive associations'}
    for role in policy['retained_controls']+policy['new_acquisition_ablations']:
        for s in seeds:ab.append(loss_rows(s,0. if role=='proposed' else .02,names[role],role=role))
    certs=[{'seed':s,'certified':True,'within_declared_movement_budget':True,'relative_l1_movement':.01} for s in seeds]
    return rows,ab,certs


class FrozenReplication(unittest.TestCase):
    def setUp(self):self.configs=load_configs(ROOT);self.policy=self.configs['step05_replication']
    def test_frozen_policy_and_exact_budget(self):
        validate_protocol(self.configs,ROOT)
        self.assertEqual(self.policy['seeds'],[809,810,811,812,813])
        self.assertEqual(len(profiles_to_run(self.policy))*len(self.policy['new_seeds']),33)
        self.assertFalse(self.policy['test_evaluation'] or self.policy['external_evaluation'] or self.policy['moment_correction'])
    def test_policy_tampering_rejected(self):
        self.configs['step05_replication']['primary_profiles']['GSDMM']='bounded_low_beta_k10'
        with self.assertRaises(ValueError):validate_protocol(self.configs)
    def test_both_stronger_coherence_challengers_retained(self):
        self.assertIn(('GSDMM','bounded_low_beta_k10'),profiles_to_run(self.policy))
        self.assertIn(('BERTopic','library_default_hdbscan_common_k10'),profiles_to_run(self.policy))
    def test_unknown_profile_rejected(self):
        with self.assertRaises(ValueError):profile_settings(self.configs,'nmf','lucky_unregistered')
    def test_missing_foreign_and_duplicate_seed_cases_rejected(self):
        rows=[{'method':'A','profile':'p','seed':s} for s in [809,810]]
        assert_grid(rows,[('A','p')],[809,810])
        for bad in [rows[:1],rows+rows[:1],[*rows,{'method':'A','profile':'p','seed':811}]]:
            with self.assertRaises(ValueError):assert_grid(bad,[('A','p')],[809,810])
    def test_partial_grid_accepted_without_partial_mean(self):
        rows=[loss_rows(809)];assert_grid(rows,[('Proposed','random_words')],[809,810],allow_partial=True)
        with tempfile.TemporaryDirectory() as tmp:
            means,_=export_progress(Path(tmp),rows,[],[809,810])
            self.assertIsNone(means[0]['fitting_c_v_mean'])
    def test_ablation_grid_missing_and_duplicate_role_rejected(self):
        rows=[{'role':'native','seed':809},{'role':'proposed','seed':809}]
        assert_ablation_grid(rows,['native','proposed'],[809])
        for bad in [rows[:1],rows+rows[:1]]:
            with self.assertRaises(ValueError):assert_ablation_grid(bad,['native','proposed'],[809])
    def test_inspected_pilot_scores_are_reused_unchanged(self):
        state=ROOT/'state/step01_accepted.json'
        if not state.is_file() or not (ROOT/'state/step04_competitor_result.json').is_file():
            self.skipTest('Existing user pilot outputs are kept by installation, not distributed as synthetic data.')
        folder,reused=accepted_panel(ROOT,read_json(state)['preparation_id'],self.policy)
        original=read_json(folder/'all_rows.json')
        self.assertEqual(len(reused),24)
        for row in reused:
            other=next(r for r in original if (r['method'],r['profile'],r['seed'])==(row['method'],row['profile'],row['seed']))
            self.assertEqual(row,other)
    def evidence(self):
        return {'pairs':np.array([[0,1],[0,2],[1,2]]),'full':np.array([.3,-.6,.2]),
            'stable':np.array([.1,-.7,.05]),'pair_support':np.array([5,1,7]),'word_support':np.array([8,9,7]),
            'words':['a','b','c'],'fit_groups':20}
    def test_no_stability_changes_only_stability(self):
        e=self.evidence();a=ablated_evidence(e,'without_subsample_stability')
        np.testing.assert_array_equal(a['stable'],e['full']);self.assertLess(a['stable'][1],0)
        for key in ['full','pairs','pair_support','word_support']:np.testing.assert_array_equal(a[key],e[key])
        a['stable'][0]=99;self.assertEqual(e['stable'][0],.1)
    def test_positive_only_changes_sign_but_preserves_support_units(self):
        e=self.evidence();a=ablated_evidence(e,'positive_only_association')
        np.testing.assert_array_equal(a['stable'],[.1,0,.05]);np.testing.assert_array_equal(a['full'],[.3,0,.2])
        for key in ['pairs','pair_support','word_support']:np.testing.assert_array_equal(a[key],e[key])
        self.assertEqual(a['fit_groups'],20);self.assertEqual(e['stable'][1],-.7)
    def test_unregistered_ablation_rejected(self):
        with self.assertRaises(ValueError):ablated_evidence(self.evidence(),'better_words')
    def test_constant_paired_gain_matches_exact_value(self):
        seeds=[809,810,811,812,813]
        result=hierarchical_gain([loss_rows(s) for s in seeds],[loss_rows(s,.02) for s in seeds],seeds,200)
        for k in ['mean_competitor_minus_proposed_macro_nll','percentile_95_low','percentile_95_high']:
            self.assertAlmostEqual(result[k],.02,places=13)
    def test_seed_uncertainty_and_deterministic_bootstrap(self):
        seeds=[809,810,811,812,813];p=[loss_rows(s) for s in seeds]
        c=[loss_rows(s,x) for s,x in zip(seeds,[-.2,-.1,0,.1,.2])]
        a=hierarchical_gain(p,c,seeds,2000);self.assertEqual(a,hierarchical_gain(p,c,seeds,2000))
        self.assertLess(a['percentile_95_low'],0);self.assertGreater(a['percentile_95_high'],0)
    def test_bootstrap_loss_or_missing_groups_not_hidden(self):
        p=[loss_rows(809)];c=[loss_rows(809,-.1)]
        a=hierarchical_gain(p,c,[809],100);self.assertLess(a['percentile_95_high'],0)
        c[0]['document_losses'].pop()
        with self.assertRaises(ValueError):hierarchical_gain(p,c,[809],100)
    def test_target_counts_cannot_change_across_seeds(self):
        p=[loss_rows(809),loss_rows(810)];c=[loss_rows(809,.1),loss_rows(810,.1)]
        for r in [p[1],c[1]]:r['document_losses'][0]['target_tokens']=5
        with self.assertRaises(ValueError):hierarchical_gain(p,c,[809,810],100)
    def test_display_only_actual_membership_is_not_projected(self):
        native=np.ones((10,101))
        for i in range(10):native[i,i*10:(i+1)*10]=3
        changed=native.copy();changed[0,9]=1;changed[0,100]=3
        targets=[list(range(i*10,(i+1)*10)) for i in range(10)];targets[0]=list(range(9))+[100]
        words=['w'+str(i) for i in range(101)]
        display=readout_mechanism(native,native,targets,words,'display_only',809)
        deployed=readout_mechanism(native,changed,targets,words,'proposed',809)
        self.assertAlmostEqual(display['target_top10_membership_recall'],.99)
        self.assertEqual(deployed['target_top10_membership_recall'],1.)
        self.assertEqual(display['relative_l1_affinity_movement'],0.)
    def test_invalid_or_zero_readout_rejected(self):
        for a in [np.zeros((1,11)),np.full((1,11),np.nan)]:
            with self.assertRaises(ValueError):readout_mechanism(a,a,[list(range(10))],list(range(11)),'native',809)
    def test_complete_reports_generate_table_figure_and_true_gate(self):
        rows,ab,certs=panel_rows(self.policy)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);gates=final_reports(folder,rows,ab,self.policy,QuietLog(),certs)
            self.assertTrue(gates['ready_for_untouched_primary_test_review'])
            self.assertTrue((folder/'figures/seed_tradeoffs.pdf').is_file())
            self.assertEqual(len(read_json(folder/'all_profile_comparisons.json')),11)
            self.assertEqual(len(read_json(folder/'hierarchical_prediction_bootstrap.json')),9)
            self.assertEqual(len(read_json(folder/'all_rows.json')),60)
            self.assertEqual(len(read_json(folder/'ablation_rows.json')),55)
            self.assertIn('Display',(folder/'ABLATION_SUMMARY.md').read_text())
    def test_bad_prediction_does_not_pass_on_engineering_success(self):
        rows,ab,certs=panel_rows(self.policy)
        for r in rows:
            if r['method']=='Proposed':r['prediction']['macro_source_nll']=5.
        with tempfile.TemporaryDirectory() as tmp:
            gates=final_reports(Path(tmp),rows,ab,self.policy,QuietLog(),certs)
            self.assertTrue(gates['engineering_passed']);self.assertFalse(gates['ready_for_untouched_primary_test_review'])
            self.assertFalse(gates['all_available_profile_means_win_or_tie'])
    def test_abstained_acquisition_is_explicit_and_blocks_readiness(self):
        rows,ab,certs=panel_rows(self.policy)
        ab[-1]['status']='abstained';ab[-1]['prediction']={'macro_source_nll':None,'micro_nll':None}
        with tempfile.TemporaryDirectory() as tmp:
            gates=final_reports(Path(tmp),rows,ab,self.policy,QuietLog(),certs)
            self.assertFalse(gates['ready_for_untouched_primary_test_review'])
            self.assertEqual(len(gates['unavailable_ablation_cases']),1)
    def test_uncertified_or_excessive_movement_blocks_readiness(self):
        rows,ab,certs=panel_rows(self.policy);certs[-1]['within_declared_movement_budget']=False
        with tempfile.TemporaryDirectory() as tmp:
            gates=final_reports(Path(tmp),rows,ab,self.policy,QuietLog(),certs)
            self.assertFalse(gates['primary_projection_gate_passed']);self.assertFalse(gates['ready_for_untouched_primary_test_review'])
    def test_parent_gate_requires_actual_descriptor_and_prediction_gains(self):
        rows,_,_=panel_rows(self.policy)
        self.assertTrue(parent_replication_gate(rows,self.policy)['passed'])
        for r in rows:
            if r['method']=='Proposed':r['fitting']['c_npmi']=-.1
        self.assertFalse(parent_replication_gate(rows,self.policy)['passed'])
    def execution_fixture(self,root,good=True):
        calls=[];rows,ab,_=panel_rows(self.policy)
        rows=[r for r in rows if r['seed'] in self.policy['reuse_seeds']]
        ab=[r for r in ab if r['seed'] in self.policy['reuse_seeds'] and r['role'] in self.policy['retained_controls']]
        def native(*args):
            _,base,name,seed=args[:4];folder=args[12];folder.mkdir(parents=True,exist_ok=True)
            write_json(folder/'artifact_manifest.json',[]);calls.append((base,name,seed))
            return {'row':loss_rows(seed,.02,{'neuromax':'NeuroMax','encot':'EnCOT'}[base],name,'native')}
        def classical(*args):
            _,base,profile,seed=args[:4];calls.append((base,profile['name'],seed))
            return {'row':loss_rows(seed,.02,{'lda':'LDA','nmf':'NMF'}[base],profile['name'],'native')}
        def competitor(*args):
            _,base,profile,seed=args[:4];calls.append((base,profile['name'],seed))
            name={'bertopic':'BERTopic','gsdmm':'GSDMM','combinedtm':'CombinedTM','fastopic':'FASTopic','glocom':'GloCOM'}[base]
            return {'row':loss_rows(seed,.02,name,profile['name'],'competitor')}
        def refine(*args):
            seed=args[3]['seed'];folder=args[7];new=[copy.deepcopy(r) for r in panel_rows(self.policy)[1] if r['seed']==seed and r['role'] in self.policy['retained_controls']]
            if not good:next(r for r in new if r['role']=='proposed')['prediction']['macro_source_nll']=5.
            value={'rows':new,'certificates':{'primary':{'success':True,'certified':True,'relative_l1_movement':.01}}}
            write_json(folder/'case_result.json',value);return value
        with patch('rank_refinement.stages.step05_ablations.native_case',side_effect=native), \
             patch('rank_refinement.stages.step05_ablations.classical_case',side_effect=classical), \
             patch('rank_refinement.stages.step05_ablations.competitor_case',side_effect=competitor), \
             patch('rank_refinement.stages.step05_ablations.build_refinement_source',return_value=root/'source'), \
             patch('rank_refinement.stages.step05_ablations.fullset_case',side_effect=refine):
            result=replicate_new_seeds(root,root/'out','synthetic',rows,ab,{}, {}, {}, {}, [], {}, [],self.configs,self.policy,'cpu',np.ones(2)/2,{'vectors_sha256':'synthetic'},QuietLog())
        return calls,result
    def test_execution_order_and_budget_make_all_three_parents_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls,(rows,ab)=self.execution_fixture(Path(tmp))
            self.assertEqual(calls[:3],[('neuromax','random_words',s) for s in [811,812,813]])
            self.assertEqual(len(calls),33);self.assertEqual(len(set(calls)),33)
            self.assertEqual(len(rows),60);self.assertEqual(len(ab),45)
            assert_grid(rows,[('Proposed','random_words')]+profiles_to_run(self.policy),self.policy['seeds'])
    def test_parent_failure_saves_results_and_avoids_thirty_other_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls,(rows,ab)=self.execution_fixture(Path(tmp),False)
            self.assertEqual(len(calls),3);self.assertEqual(len(rows),30);self.assertEqual(len(ab),45)
            self.assertFalse(read_json(Path(tmp)/'out/parent_replication_gate.json')['passed'])
    def test_completed_stage_replay_does_not_download_or_refit(self):
        from rank_refinement.stages.step05_ablations import run
        from rank_refinement.runtime.artifacts import digest
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'configs').mkdir()
            shutil.copy2(ROOT/'configs/upstream_competitor_sources.json',root/'configs/upstream_competitor_sources.json')
            panel=root/'panel';panel.mkdir();write_json(panel/'artifact_manifest.json',[])
            write_json(panel/'protocol_resolved_before_scoring.json',{'device':'cpu','versions':{}})
            prepared={'preparation_id':'synthetic','processed_relative_path':'unused'};versions={'fixture':'1'}
            identity={'policy':self.policy,'preparation_id':'synthetic','step04_index_sha256':file_hash(panel/'artifact_manifest.json'),
                'versions':versions,'device':'cpu','implementation':{}}
            key=digest(identity);out=root/'outputs/steps/step05_replication'/key[:20]
            write_json(out/'protocol_resolved_before_scoring.json',identity)
            result={'status':'STEP05_REPLICATION_COMPLETE','step05_id':key,'new_neural_fits':15,'new_classical_fits':18}
            write_json(out/'run_result.json',result);write_json(out/'artifact_manifest.json',artifact_manifest(out));before=file_hash(out/'artifact_manifest.json')
            log=QuietLog();log.folder=root/'current_run'
            with patch('rank_refinement.stages.step05_ablations.require_preparation',return_value=prepared), \
                 patch('rank_refinement.stages.step05_ablations.accepted_panel',return_value=(panel,[])), \
                 patch('rank_refinement.stages.step05_ablations.load_canonical',return_value={}), \
                 patch('rank_refinement.stages.step05_ablations.proposed_source',return_value=(root,root,[],[],[])), \
                 patch('rank_refinement.stages.step05_ablations.resolve_native_sources',side_effect=AssertionError('No download on replay')):
                replay,folder=run(root,self.configs,versions,{'future_model_device':'cpu'},log)
            self.assertTrue(replay['cache_replay']);self.assertEqual(replay['new_neural_fits'],0);self.assertEqual(folder,out)
            self.assertEqual(before,file_hash(out/'artifact_manifest.json'));self.assertEqual(read_json(out/'run_result.json'),result)
    def test_environment_mismatch_blocks_mixed_old_new_seed_panel(self):
        from rank_refinement.stages.step05_ablations import run
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'configs').mkdir()
            shutil.copy2(ROOT/'configs/upstream_competitor_sources.json',root/'configs/upstream_competitor_sources.json')
            panel=root/'panel';write_json(panel/'protocol_resolved_before_scoring.json',{'device':'cuda','versions':{'torch':'2.7.0+cu128'}})
            prepared={'preparation_id':'synthetic','processed_relative_path':'unused'}
            with patch('rank_refinement.stages.step05_ablations.require_preparation',return_value=prepared), \
                 patch('rank_refinement.stages.step05_ablations.accepted_panel',return_value=(panel,[])):
                with self.assertRaises(ValueError):run(root,self.configs,{'torch':'2.6.0'}, {'future_model_device':'cuda'},QuietLog())


class LiveStageIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.configs=load_configs(ROOT);cls.folders,cls.specs=resolve_sources(ROOT)
    def data(self):
        data,vectors=fixture()
        data.update(fit_observed=data['fit'].copy(),fit_observed_records=data['fit_records'],fit_observed_ids=data['fit_ids'])
        vectors['fit_observed']=vectors['fit_full'].copy()
        data['metadata']={'tokens':data['vocabulary'],'document_frequency':(data['fit']>0).sum(0).tolist()}
        return data,vectors
    def test_live_native_stage_replays_both_author_interfaces_without_refits(self):
        data,vectors=self.data();policy=copy.deepcopy(self.configs['step03_neural'])
        policy['global_context']['clusters']=4;policy['batch_size']=40;policy['checkpoint_every_epochs']=1
        for base in ['neuromax','encot']:policy['training'][base]['epochs']=2
        contexts=global_context(data,vectors,policy['global_context']);prior=data['fit'].astype(np.float64).sum(0)+.5;prior/=prior.sum()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);log=QuietLog()
            for base in ['neuromax','encot']:
                folder=root/base
                a=native_case(root,base,'random_words',811,data,vectors,contexts,self.folders,self.specs,policy,'cpu',prior,folder,log,{'seed':811,'base':base})
                self.assertEqual(a['row']['status'],'scored');self.assertEqual(a['training_details']['new_training_epochs'],2)
                self.assertTrue(read_json(folder/'native_interface_certificate.json')['verified'])
                count=log.new_neural_fits;before=file_hash(folder/'artifact_manifest.json')
                b=native_case(root,base,'random_words',811,data,vectors,contexts,self.folders,self.specs,policy,'cpu',prior,folder,log,{'seed':811,'base':base})
                self.assertEqual(a,b);self.assertEqual(before,file_hash(folder/'artifact_manifest.json'));self.assertEqual(count,log.new_neural_fits)
                with self.assertRaises(ValueError):native_case(root,base,'random_words',812,data,vectors,contexts,self.folders,self.specs,policy,'cpu',prior,folder,log,{'seed':812,'base':base})
            source=build_refinement_source(root,root/'stage05',811,root/'neuromax',data,self.configs,log)
            case={'base':'neuromax','profile':'random_words','seed':811};folder=source/'cases/neuromax/random_words/seed811'
            original=file_hash(folder/'artifact_manifest.json')
            self.assertEqual(source,build_refinement_source(root,root/'stage05',811,root/'neuromax',data,self.configs,log))
            self.assertEqual(original,file_hash(folder/'artifact_manifest.json'))
            result=fullset_case(root,source,source,case,data,policy,self.configs['step03_improvement'],root/'refined',log,{'seed':811,'fixture':'CPU'})
            self.assertEqual(len(result['rows']),9)
            for variant in self.configs['step05_replication']['new_acquisition_ablations']:
                a=acquisition_ablation(root,811,variant,folder,folder,data,self.configs,root/variant,log,{'seed':811,'variant':variant})
                self.assertEqual(a['row']['status'],'abstained')
                self.assertIsNone(a['row']['prediction']['macro_source_nll'])
                self.assertEqual(a['row']['failure_kind'],'readout_unavailable_no_native_substitution')
    def test_live_classical_stage_preserves_predictions_and_cache(self):
        data,_=self.data();prior=data['fit'].astype(np.float64).sum(0)+.5;prior/=prior.sum()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);log=QuietLog()
            for method in ['lda','nmf']:
                profile=profile_settings(self.configs,method,'fitting_strong_k10')
                if method=='lda':profile.update(max_iter=2)
                a=classical_case(root,method,profile,811,data,prior,root/method,log,{'seed':811,'method':method},[0,.1])
                self.assertEqual(a['row']['status'],'scored')
                count=log.new_classical_fits
                b=classical_case(root,method,profile,811,data,prior,root/method,log,{'seed':811,'method':method},[0,.1])
                self.assertEqual(a,b);self.assertEqual(count,log.new_classical_fits)
    def test_inspection_contains_step04_source_and_step05_state(self):
        from zipfile import ZipFile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);panel=root/'outputs/steps/step04_competitor_pilot/example';panel.mkdir(parents=True)
            write_json(panel/'all_rows.json',[{'fixture':'preserved'}]);write_json(panel/'artifact_manifest.json',artifact_manifest(panel))
            out=root/'outputs/steps/step05_replication/example';out.mkdir(parents=True)
            write_json(out/'completed_step04_reference.json',{'relative_path':panel.relative_to(root).as_posix(),'artifact_manifest_sha256':file_hash(panel/'artifact_manifest.json')})
            write_json(root/'state/step05_replication_result.json',{'fixture':True})
            class Run:
                step=5;run_id='synthetic';folder=root/'outputs/runs/synthetic/step05_replication';log_path=root/'logs/synthetic.log'
            Run.folder.mkdir(parents=True);Run.log_path.parent.mkdir();Run.log_path.write_text('test')
            archive=make_inspection_zip(root,Run(),out,'step05')
            with ZipFile(archive) as z:
                self.assertIn('outputs/steps/step04_competitor_pilot/example/all_rows.json',z.namelist())
                self.assertIn('state/step05_replication_result.json',z.namelist())
            write_json(panel/'artifact_manifest.json',{})
            with self.assertRaises(ValueError):make_inspection_zip(root,Run(),out,'step05')
