"""Recovery gates, immutable evidence, global selection and completion budget.

Synthetic fixtures exercise orchestration; they are never model-performance
evidence. Existing saved-user-data checks are separate and clearly optional.
"""
from __future__ import annotations
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from rank_refinement.runtime.config import load_configs
from rank_refinement.runtime.artifacts import read_json, write_json, artifact_manifest, file_hash
from rank_refinement.acquisition.native_diversity_targets import recovery_targets
from rank_refinement.stages import step05_recovery as stage
from rank_refinement.evaluation.replication import assert_ablation_grid
from rank_refinement.reporting.export import make_inspection_zip
from test_step05_replication import loss_rows
from test_step04_competitors import QuietLog

ROOT=Path(__file__).resolve().parents[1]


def grid(policy):
    parents=[loss_rows(s,.01,'NeuroMax') for s in policy['seeds']]
    rows=[]
    for g in policy['recovery_graphs']:
        for p in policy['projection_candidates']:
            for seed in policy['seeds']:
                r=loss_rows(seed,0. if p=='range' else .002,g+'/'+p,role='candidate')
                r['fitting']={'c_v':.7,'c_npmi':.1}
                r.update(relative_l1_movement=.001,projection_certificate_passed=True)
                rows.append(r)
    return rows,parents


class RecoveryContracts(unittest.TestCase):
    def setUp(self):self.configs=load_configs(ROOT);self.policy,self.old=stage.validate_protocol(self.configs,ROOT)
    def test_named_revision_and_zero_parent_training_budget(self):
        self.assertFalse(self.policy['frozen_model_retraining'])
        self.assertEqual(self.policy['recovery_new_neural_fits'],0)
        self.assertEqual(self.policy['recovery_new_encoder_calls'],0)
        self.assertEqual(self.policy['maximum_new_completion_baseline_fits'],30)
    def test_policy_tampering_rejected(self):
        for key,value in [('per_seed_method_selection',True),('test_evaluation',True),('moment_correction',True),('seeds',[809,810])]:
            c=copy.deepcopy(self.configs);c['step05_recovery'][key]=value
            with self.assertRaises(ValueError):stage.validate_protocol(c)
    def test_parent_floor_replaces_absolute_union_without_mutating_inputs(self):
        a=np.ones((10,15));a[:,:10]=2.;e={'stable':np.array([-.4,.2]),'full':np.array([-.3,.5])}
        original=copy.deepcopy(e);p={'required_unique_words':100}
        with patch('rank_refinement.acquisition.native_diversity_targets.choose_complete_targets',return_value=([],{})) as choose:
            _,report=recovery_targets(a,list(map(str,range(15))),{},e,p,'signed_stable')
            self.assertEqual(choose.call_args.args[-1]['required_unique_words'],10)
            self.assertEqual(report['native_diversity_floor_words'],10)
            self.assertTrue(report['promoted_cores_disjoint'])
        self.assertEqual(p['required_unique_words'],100)
        np.testing.assert_array_equal(e['stable'],original['stable'])
    def test_full_graph_control_does_not_change_support(self):
        a=np.ones((10,15));e={'stable':np.array([-.4,.2]),'full':np.array([-.3,.5]),'pair_support':np.array([1,9])}
        with patch('rank_refinement.acquisition.native_diversity_targets.choose_complete_targets',return_value=([],{})) as choose:
            recovery_targets(a,list(map(str,range(15))),{},e,{},'signed_full')
            changed=choose.call_args.args[-2]
            np.testing.assert_equal(changed['stable'],e['full']);np.testing.assert_equal(changed['pair_support'],e['pair_support'])
    def test_positive_only_control_and_strict_100_floor_are_explicit(self):
        a=np.ones((10,15));e={'stable':np.array([-.4,.2]),'full':np.array([-.3,.5])}
        with patch('rank_refinement.acquisition.native_diversity_targets.choose_complete_targets',return_value=([],{})) as choose:
            _,r=recovery_targets(a,list(map(str,range(15))),{},e,{},'positive_only_stable',strict=True)
            np.testing.assert_array_equal(choose.call_args.args[-2]['stable'],[0.,.2])
            self.assertEqual(choose.call_args.args[-1]['required_unique_words'],100)
            self.assertEqual(r['diversity_contract'],'absolute_100_word_control')
    def test_unknown_graph_rejected(self):
        with self.assertRaises(ValueError):recovery_targets(None,None,None,None,None,'chosen_after_test')
    def test_one_global_choice_can_retain_a_losing_seed(self):
        rows,parents=grid(self.policy)
        part=[r for r in rows if r['method']=='signed_stable/range']
        part[2]['prediction']['macro_source_nll']=4.011 # tiny loss retained, not excluded
        decision=stage.select_global(rows,parents,self.policy)
        self.assertEqual(decision['selected_configuration'],'signed_full/range')
        self.assertFalse(decision['per_seed_method_selection'])
        self.assertEqual(len(decision['decisions']),4)
    def test_incomplete_configuration_is_ineligible_without_partial_mean(self):
        rows,parents=grid(self.policy);rows=[r for r in rows if not (r['method']=='signed_full/range' and r['seed']==812)]
        d=stage.select_global(rows,parents,self.policy)
        p=next(x for x in d['decisions'] if x['configuration']=='signed_full/range')
        self.assertFalse(p['passed']);self.assertIsNone(p['mean_native_minus_candidate_nll'])
        self.assertIsNone(next(m for m in d['means'] if m['method']=='signed_full/range')['development_macro_nll_mean'])
    def test_failed_seed_not_replaced_by_parent(self):
        rows,parents=grid(self.policy)
        for r in rows:
            if r['seed']==812:r.update(status='failed');r['prediction']['macro_source_nll']=None
        d=stage.select_global(rows,parents,self.policy)
        self.assertFalse(d['passed']);self.assertIsNone(d['selected_configuration'])
    def test_diversity_regression_disqualifies_configuration(self):
        rows,parents=grid(self.policy)
        for r in rows:r['topic_diversity']=.99
        self.assertFalse(stage.select_global(rows,parents,self.policy)['passed'])
    def test_uncertified_or_excessive_movement_disqualifies(self):
        for key,value in [('projection_certificate_passed',False),('relative_l1_movement',.051)]:
            rows,parents=grid(self.policy)
            for r in rows:r[key]=value
            self.assertFalse(stage.select_global(rows,parents,self.policy)['passed'])
    def test_descriptor_regression_disqualifies(self):
        rows,parents=grid(self.policy)
        for r in rows:r['fitting']['c_npmi']=-.5
        self.assertFalse(stage.select_global(rows,parents,self.policy)['passed'])
    def test_duplicate_or_foreign_seed_rejected(self):
        rows,parents=grid(self.policy)
        for bad in [rows+rows[:1],rows+[dict(rows[0],seed=999)]]:
            with self.assertRaises(ValueError):stage.select_global(bad,parents,self.policy)
    def test_thirty_remaining_jobs_exclude_all_neuromax_parent_fits(self):
        rows=[loss_rows(s,method=m,profile=p) for m,p in stage.profiles_to_run(self.old) for s in self.old['reuse_seeds']]
        rows += [loss_rows(s,method='NeuroMax') for s in self.old['new_seeds']]
        rows += [loss_rows(s) for s in self.old['seeds']]
        jobs=stage.completion_jobs(self.old,rows)
        self.assertEqual(len(jobs),30);self.assertTrue(all(m!='NeuroMax' for s,m,p in jobs))
        self.assertEqual(len({(s,m,p) for s,m,p in jobs}),30)
        self.assertIn((811,'GSDMM','bounded_low_beta_k10'),jobs)
        self.assertIn((811,'BERTopic','library_default_hdbscan_common_k10'),jobs)
    def test_completed_panel_does_not_resolve_encoder_or_author_sources(self):
        rows=[loss_rows(s,method=m,profile=p) for m,p in stage.profiles_to_run(self.old) for s in self.old['seeds']]
        rows += [loss_rows(s) for s in self.old['seeds']]
        with patch.object(stage,'encode_views',side_effect=AssertionError('unneeded encoder')),patch.object(stage,'native_sources',side_effect=AssertionError('unneeded author lookup')):
            result=stage.complete_baselines(ROOT,ROOT,'fixture',rows,[],{},self.configs,self.old,'cpu',QuietLog())
        self.assertIs(result,rows)
    def test_completion_dispatches_locked_thirty_cases_and_saves_sixty_slots(self):
        rows=[loss_rows(s,method=m,profile=p) for m,p in stage.profiles_to_run(self.old) for s in self.old['reuse_seeds']]
        rows += [loss_rows(s,method='NeuroMax') for s in self.old['new_seeds']]
        rows += [loss_rows(s) for s in self.old['seeds']]
        def fake_case(*args):
            method=args[1];profile=args[2];seed=args[3]
            name=profile if isinstance(profile,str) else profile['name']
            label=next(label for label,m in stage.LABEL_TO_METHOD.items() if m==method)
            return {'row':loss_rows(seed,.02,label,name,'native')}
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            with patch.object(stage,'native_sources',return_value=({},{})),patch.object(stage,'competitor_sources',return_value=({},{})), \
                 patch.object(stage,'encode_views',return_value=({}, {'vectors_sha256':'synthetic_fixture'})) as encoder, \
                 patch.object(stage,'global_context',return_value={}),patch.object(stage,'native_case',side_effect=fake_case) as native, \
                 patch.object(stage,'classical_case',side_effect=fake_case) as classical,patch.object(stage,'competitor_case',side_effect=fake_case) as competitors:
                completed=stage.complete_baselines(output,output,'fixture',rows,[],{'fit':np.ones((5,12))},self.configs,self.old,'cpu',QuietLog())
            self.assertEqual(len(completed),60);self.assertEqual(native.call_count,3)
            self.assertEqual(classical.call_count,6);self.assertEqual(competitors.call_count,21)
            self.assertEqual(encoder.call_count,1)
            self.assertTrue(all(c.args[1]=='encot' for c in native.call_args_list))
            self.assertEqual(len(read_json(output/'all_rows.json')),60)
    def test_failed_control_keeps_explicit_na(self):
        row=stage.unavailable(812,'strict_word_disjointness','infeasible',self.configs)
        self.assertEqual(row['status'],'failed');self.assertIsNone(row['prediction']['macro_source_nll'])
        self.assertIsNone(row['fitting']['c_npmi'])
    def test_saved_contexts_reject_foreign_or_duplicate_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);a={'theta_fit':np.ones((64,10))};data={'fit_observed_ids':list(map(str,range(64)))}
            for bad in [list(map(str,range(63)))+['foreign'],['0']*64]:
                write_json(folder/'range_contexts.json',{'group_ids':bad})
                with self.assertRaises(ValueError):stage.saved_contexts(folder,a,data)
            write_json(folder/'range_contexts.json',{'group_ids':list(map(str,range(64)))})
            np.testing.assert_equal(stage.saved_contexts(folder,a,data),a['theta_fit'])
    def test_real_original_parent_sources_verify_and_remain_unchanged(self):
        state=ROOT/'state/step05_replication_result.json'
        if not state.is_file():self.skipTest('User saved outputs are preserved by installation, not included in the code ZIP.')
        prep=read_json(ROOT/'state/step01_accepted.json')['preparation_id']
        folder,rows,snapshot=stage.accepted_source(ROOT,prep,self.policy)
        before=file_hash(folder/'artifact_manifest.json')
        for seed in self.policy['seeds']:
            e,a,z=stage.seed_sources(ROOT,folder,seed,self.old)
            self.assertEqual(z['native'].shape[0],10)
        self.assertEqual(file_hash(folder/'artifact_manifest.json'),before)
        self.assertEqual(len([r for r in rows if r['method']=='NeuroMax']),5)


class RecoveryOrchestration(unittest.TestCase):
    def test_failed_global_gate_prevents_every_extra_fit_and_encoder_call(self):
        configs=load_configs(ROOT);p,old=stage.validate_protocol(configs,ROOT);rr,parents=grid(p)
        for r in rr:r['prediction']['macro_source_nll']=10.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'outputs/steps/step05_replication/fixture';source.mkdir(parents=True)
            write_json(source/'artifact_manifest.json',[]);write_json(source/'ablation_rows.json',[])
            log=QuietLog();log.folder=root/'run';log.folder.mkdir()
            def case(*args,**kwargs):
                seed,graph=args[2:4];return {'rows':{mode:next(r for r in rr if r['seed']==seed and r['method']==graph+'/'+mode) for mode in p['projection_candidates']}}
            with patch.object(stage,'require_preparation',return_value={'preparation_id':'fixture','processed_relative_path':'data/fixture'}), \
                 patch.object(stage,'validate_protocol',return_value=(p,old)), \
                 patch.object(stage,'accepted_source',return_value=(source,parents,{'device':'cpu','versions':{}})), \
                 patch.object(stage,'load_canonical',return_value={'fit_ids':['f'],'fit_observed_ids':['f'],'dev_ids':['d']}),patch.object(stage,'projected_case',side_effect=case), \
                 patch.object(stage,'check_indexed_tree'),patch.object(stage,'complete_baselines',side_effect=AssertionError('no extra fits')):
                result,folder=stage.run(root,configs,{}, {'future_model_device':'cpu'},log)
            self.assertFalse(result['complete_five_seed_grid']);self.assertEqual(result['new_neural_fits'],0)
            self.assertFalse(result['test_or_external_used']);self.assertEqual(result['status'],'STEP05_RECOVERY_REVIEW_REQUIRED')
            self.assertTrue((folder/'recovery_candidate_rows.json').is_file())
            self.assertEqual(len(read_json(folder/'recovery_candidate_rows.json')),20)
            self.assertFalse((root/'state/step05_replication_result.json').exists())
    def test_actual_control_report_keeps_infeasible_seed_and_diversity_losses(self):
        cfg=load_configs(ROOT);p,old=stage.validate_protocol(cfg,ROOT)
        rows=[loss_rows(s) for s in p['seeds']]
        for method,profile in stage.profiles_to_run(old):
            for seed in p['seeds']:
                r=loss_rows(seed,.02,method,profile,'native')
                if method=='BERTopic':r['prediction']={'macro_source_nll':None,'micro_nll':None};r['document_losses']=[]
                rows.append(r)
        ab=[];mechanism=[]
        for role in p['controls']:
            for seed in p['seeds']:
                r=loss_rows(seed,0. if role=='proposed' else .02,stage.LABELS[role],role=role)
                if role=='strict_word_disjointness' and seed==812:r=stage.unavailable(seed,role,'infeasible',cfg)
                ab.append(r)
        for r in rows:
            if r['method']=='Proposed':r['topic_diversity']=.994
        for seed in p['seeds']:mechanism.append({'role':'proposed','seed':seed,'target_top10_membership_recall':1.,
            'maximum_topic_total_error_scaled':0.,'maximum_word_total_error_scaled':0.})
        assert_ablation_grid(ab,p['controls'],p['seeds'])
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);g=stage.report_tables(folder,rows,ab,p,old,{'passed':True,'selected_configuration':'signed_stable/range'},mechanism)
            self.assertTrue(g['engineering_passed']);self.assertFalse(g['all_available_profile_means_win_or_tie'])
            self.assertFalse(g['ready_for_untouched_primary_test_review'])
            self.assertEqual(g['unavailable_ablation_cases'][0]['seed'],812)
            self.assertTrue((folder/'figures/seed_tradeoffs.pdf').is_file())
            self.assertIn('NA',(folder/'ABLATION_SUMMARY.md').read_text())


if __name__=='__main__':unittest.main()
