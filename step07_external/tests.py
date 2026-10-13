"""Independent synthetic checks. No primary/external real-data model fitting."""
from __future__ import annotations
import copy
import csv
import io
import logging
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
from rank_refinement.runtime.artifacts import read_json,write_json,digest,file_hash
from rank_refinement.runtime.config import load_configs
from rank_refinement.models.neural_sources import resolve_sources
from rank_refinement.models.competitor_sources import resolve_sources as resolve_competitor_sources
from rank_refinement.models.neural_prediction import log_predictions,backoff
from rank_refinement.projection.replay_solver import primary_tv,range_secondary
from rank_refinement.projection.minimum_tv import certify
from .data import clean,tokenize,read_records,prepare
from .protocol import recipe,DATA_SHA,implementation,primary_guard,assert_primary_unchanged
from .models import frozen_scores,save_arrays,fit_native_case,fit_baseline_case,cached,combined_views,refine
from .report import audit_topics,summary_rows,paired_intervals,export,score_cases
from .runner import freeze_exposure

PROJECT_ROOT=Path(__file__).resolve().parents[1]

class QuietLog:
    def info(self,*args):pass

def fixture(path,rows=300):
    groups=[]
    for topic in range(10):
        words=['word'+chr(97+topic)+chr(97+i)for i in range(16)]
        for j in range(rows//10):
            selected=[words[(j+i)%16]for i in range(10)]
            groups.append({'Created_at':f'2020-01-{j%28+1:02d} 01:00:00+00:00','Year':'2020','Month':'January',
                'Text':' '.join(selected)+' train bus '+chr(97+j%26)*7+'_'+str(j),
                'latitude':'0','longitude':'0'})
    # Unique numerical suffixes are provenance only; tokenization drops numbers.
    s=io.StringIO(newline='');writer=csv.DictWriter(s,fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
    path.write_text(s.getvalue(),encoding='utf-8')

class DataContractTests(unittest.TestCase):
    def test_anonymization_and_negation(self):
        text=clean("@[agency name] can't stop at [location name] station https://bad.example/a")
        words=tokenize(text)
        self.assertIn('cannot',words);self.assertNotIn('agency',words);self.assertNotIn('location',words);self.assertIn('station',words)
    def test_english_phrase_spacing(self):
        self.assertEqual('bus_stop'.replace('_',' '),'bus stop')
        self.assertIn('not',tokenize(clean("isn't safe")))
    def test_multiline_csv_is_one_record(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'x.csv';p.write_text('Created_at,Year,Month,Text,latitude,longitude\n2020,2020,Jan,"bus\ntrain",0,0\n,,,,,\n',encoding='utf-8')
            records,audit=read_records(p)
            self.assertEqual(len(records),1);self.assertEqual(audit['raw_rows'],2);self.assertEqual(audit['multiline_records'],1)
    def test_strict_schema_rejects_labels_as_text(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'x.csv';p.write_text('sentiment,label\nnegative,1\n')
            with self.assertRaises(ValueError):read_records(p)
    def test_empty_text_cannot_become_a_model_document(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'x.csv';p.write_text('Created_at,Year,Month,Text,latitude,longitude\n,,,,,\n')
            with self.assertRaises(ValueError):read_records(p)
    def test_split_and_vocabulary_firewall(self):
        with tempfile.TemporaryDirectory()as d:
            d=Path(d);p=d/'x.csv';fixture(p);policy=recipe(load_configs(PROJECT_ROOT));data=prepare(p,policy,d/'prepared')
            sets=[set(x)for x in data['all_split_ids'].values()]
            self.assertFalse(sets[0]&sets[1]);self.assertFalse(sets[0]&sets[2]);self.assertFalse(sets[1]&sets[2])
            counts={w:sum(w in set(r['tokens'])for r in data['fit_records'])for w in data['vocabulary']}
            self.assertEqual([counts[w]for w in data['vocabulary']],data['metadata']['document_frequency'])
            self.assertTrue(all(r['split']=='fit'for r in data['fit_records']))
    def test_occurrence_masks_partition_tokens(self):
        from rank_refinement.data.completion_masks import make_completion
        tokens=['bus','bus','train','driver','bus'];mask=make_completion('g',tokens,{'bus':0,'train':1,'driver':2},
            {'seed':12,'observed_fraction':.5,'minimum_in_vocabulary_tokens':2})
        a=set(mask['observed_positions']);b=set(mask['target_positions'])
        self.assertFalse(a&b);self.assertEqual(a|b,set(range(5)));self.assertEqual(len(mask['target_ids'])+len(mask['observed_ids']),5)
    def test_duplicate_groups_stay_together(self):
        from rank_refinement.data.duplicate_groups import group_records
        row={'source':'x','source_file':'f','source_id':'1','normalized_body':'bus complaint','comparison_text':'bus complaint'}
        groups,_=group_records([{**row,'record_id':'a'},{**row,'source_id':'2','record_id':'b'}],{'enabled':False})
        self.assertEqual(len(groups),1);self.assertEqual(len(groups[0]['members']),2)
    def test_combined_adapter_never_injects_test_full_text(self):
        data={'dev_observed':np.ones((2,4)), 'test_observed':np.zeros((3,4))}
        vectors={'development_observed':np.ones((2,384)), 'test_observed':np.zeros((3,384))}
        a,v=combined_views(data,vectors);self.assertEqual(a['dev_observed'].shape,(5,4));self.assertEqual(v['development_observed'].shape,(5,384))
    def test_fixed_recipe_and_primary_source(self):
        p=recipe(load_configs(PROJECT_ROOT))
        self.assertEqual(p['seeds'],[809,810,811,812,813]);self.assertFalse(p['moment_correction'])
        self.assertEqual(p['acquisition']['word_group_support'],20);self.assertEqual(p['acquisition']['maximum_replacements'],4)
        self.assertEqual(len(p['comparators']),11);self.assertEqual(p['graph'],'signed_stable')
        before=primary_guard(PROJECT_ROOT,load_configs(PROJECT_ROOT));assert_primary_unchanged(PROJECT_ROOT,before)

class PredictionProjectionTests(unittest.TestCase):
    def test_completion_loss_independent_oracle(self):
        p=np.array([[.1,.2,.7],[.5,.25,.25]])
        data={'test_records':[{'group_id':'a','source':'external_transit'},{'group_id':'b','source':'external_transit'}],
              'masks':{'a':{'target_ids':[2,2]},'b':{'target_ids':[0]}}}
        scores,losses=frozen_scores(np.log(p),0,np.ones(3)/3,data)
        expected=(-2*math.log(.7)-math.log(.5))/3
        self.assertAlmostEqual(scores['macro_source_nll'],expected,12);self.assertEqual(losses[0]['target_tokens'],2)
    def test_true_zero_targets_are_not_clipped(self):
        data={'test_records':[{'group_id':'a','source':'external_transit'}],'masks':{'a':{'target_ids':[1]}}}
        logp=np.array([[0.,-np.inf]])
        value,losses=frozen_scores(logp,0,np.ones(2)/2,data)
        self.assertIsNone(value['macro_source_nll']);self.assertIsNone(losses[0]['negative_log_probability'])
        value,_=frozen_scores(logp,.1,np.ones(2)/2,data);self.assertAlmostEqual(value['macro_source_nll'],-math.log(.05))
    def test_actual_projection_totals_and_rank(self):
        native=np.array([[.8,.7,.6,.5,.4],[.4,.5,.6,.7,.8]])
        target=[[3,4],[0,1]]
        phi,certificate=primary_tv(native,target,1e-6,20)
        self.assertTrue(certificate['certified']);self.assertIsNotNone(phi)
        np.testing.assert_allclose(phi.sum(0),native.sum(0),atol=1e-8);np.testing.assert_allclose(phi.sum(1),native.sum(1),atol=1e-8)
        for row,t in zip(phi,target):self.assertEqual(set(np.argsort(row)[-2:]),set(t))
    def test_range_bound_on_actual_cross_entropy(self):
        native=np.array([[.8,.7,.6,.5,.4],[.4,.5,.6,.7,.8]]);target=[[3,4],[0,1]]
        phi,cert=primary_tv(native,target,1e-6,20);theta=np.array([[.8,.2],[.2,.8]])
        refined,report=range_secondary(native,target,cert['solver_tv_scaled'],theta,np.ones(5),phi,seconds=20)
        self.assertTrue(report['certified']);self.assertTrue(report['success'])
        before=log_predictions(theta,native,np.ones(5),np.zeros(5));after=log_predictions(theta,refined,np.ones(5),np.zeros(5))
        delta=theta@(refined-native);bound=np.ptp(delta,axis=1)
        self.assertTrue((np.abs(after-before)<=bound[:,None]+1e-7).all())
    def test_supported_edit_acquisition_and_actual_readout_deployment(self):
        with tempfile.TemporaryDirectory()as directory:
            d=Path(directory);fixture(d/'fixture.csv',600)
            policy=recipe(load_configs(PROJECT_ROOT));policy['phrases']['enabled']=False
            data=prepare(d/'fixture.csv',policy,d/'prepared')
            native=np.full((10,len(data['vocabulary'])),.004)
            theme=[]
            for t in range(10):
                ids=[j for j,w in enumerate(data['vocabulary'])if w.startswith('word'+chr(97+t))]
                self.assertEqual(len(ids),16);theme.append(ids)
                native[t,ids[:9]]=.30-np.arange(9)*.001
                native[t,ids[9:]]=.28-np.arange(7)*.001
            for t in range(10):native[t,theme[(t+1)%10][0]]=.285
            theta=np.random.default_rng(33).dirichlet(np.ones(10),size=len(data['fit_observed_ids']))
            readouts,report,evidence=refine(native,theta,np.ones(native.shape[1]),data,policy,809,QuietLog())
            self.assertTrue(report['selection']['selected']);self.assertEqual(report['deployment'],'certified_projection_deployed')
            self.assertEqual(report['mechanism']['target_top10_membership_recall'],1.)
            np.testing.assert_allclose(readouts['proposed'].sum(0),native.sum(0),atol=1e-7)
            np.testing.assert_allclose(readouts['proposed'].sum(1),native.sum(1),atol=1e-7)
    def test_persistent_case_detects_modified_arrays(self):
        from rank_refinement.runtime.artifacts import artifact_manifest
        with tempfile.TemporaryDirectory()as d:
            p=Path(d);identity={'a':1};write_json(p/'case_result.json',{'identity':identity})
            save_arrays(p/'predictive_arrays.npz',x=np.ones(2));write_json(p/'artifact_manifest.json',artifact_manifest(p))
            self.assertIsNotNone(cached(p,identity));save_arrays(p/'predictive_arrays.npz',x=np.zeros(2))
            with self.assertRaises(ValueError):cached(p,identity)
    def test_exposure_blocks_reselection(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);one=freeze_exposure(root,{'recipe':'a'});two=freeze_exposure(root,{'recipe':'a'})
            self.assertEqual(one,two)
            with self.assertRaises(RuntimeError):freeze_exposure(root,{'recipe':'b'})
    def test_missing_seeds_cannot_receive_a_mean(self):
        with self.assertRaises(ValueError):summary_rows([{'method':'x','profile':'p','seed':809}], [809,810])
    def test_absent_reference_words_remain_all_slots(self):
        topics=[['word'+chr(97+t)+chr(97+i)for i in range(10)]for t in range(10)]
        reference=[{'tokens':topics[0],'group_id':'a'}]
        audit=audit_topics(topics,reference,[w for t in topics for w in t])
        self.assertEqual(audit['total_slots'],100);self.assertEqual(audit['supported_slots'],10)
        self.assertEqual(len(audit['audit']['reference_absent_words']),90)

class EndToEndAuthorFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.output=Path(cls.temp.name)
        cls.policy=recipe(load_configs(PROJECT_ROOT));cls.log=QuietLog()
        cls.policy['native_training']['training']['neuromax']['epochs']=2
        cls.policy['native_training']['training']['encot']['epochs']=2
        cls.policy['native_training']['sinkhorn_max_iter']=100
        cls.policy['native_training']['global_context']['clusters']=10
        cls.policy['native_training']['training']['encot']['num_clusters']=10
        # Author adapters are tested with their real objectives on synthetic input.
        cls.native_folders,cls.native_specs=resolve_sources(PROJECT_ROOT,cls.log)
        cls.comp_folders,cls.comp_specs=resolve_competitor_sources(PROJECT_ROOT,cls.log)
        p=cls.output/'fixture.csv';fixture(p);cls.data=prepare(p,cls.policy,cls.output/'prepared')
        rng=np.random.default_rng(507)
        sizes={'fit_full':len(cls.data['fit_ids']),'fit_observed':len(cls.data['fit_observed_ids']),
            'development_observed':len(cls.data['dev_ids']),'test_observed':len(cls.data['test_ids']),
            'words':len(cls.data['vocabulary'])}
        cls.vectors={}
        for name,size in sizes.items():
            a=rng.normal(size=(size,384)).astype(np.float32);cls.vectors[name]=a/np.linalg.norm(a,axis=1,keepdims=True)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_real_author_neuromax_fit_readout_and_restart(self):
        identity={'synthetic_only':True,'seed':809}
        value=fit_native_case(PROJECT_ROOT,self.output,809,self.data,self.vectors,self.policy,'cpu',self.native_folders,self.native_specs,self.log,identity)
        again=fit_native_case(PROJECT_ROOT,self.output,809,self.data,self.vectors,self.policy,'cpu',self.native_folders,self.native_specs,self.log,identity)
        self.assertEqual(value,again);self.assertEqual(len(value['rows']),5)
        parent=next(r for r in value['rows']if r['method']=='NeuroMax');display=next(r for r in value['rows']if r['method']=='Same-core display only')
        if display['status']=='scored':self.assertEqual(parent['development_prediction'],display['development_prediction'])
    def test_all_ten_remaining_frozen_comparator_interfaces(self):
        for item in self.policy['comparators']:
            if item['method']=='neuromax':continue
            item=copy.deepcopy(item);profile=item['profile']
            if 'epochs'in profile:profile['epochs']=2
            if item['method']=='glocom':profile['clusters']=10;profile['sinkhorn_max_iter']=100
            if item['method']=='lda':profile['passes']=1;profile['iterations']=20
            if item['method']=='nmf':profile['max_iter']=40
            if item['method']=='gsdmm':profile['iterations']=5
            with self.subTest(method=item['method'],profile=profile['name']):
                value=fit_baseline_case(PROJECT_ROOT,self.output,item,809,self.data,self.vectors,self.policy,'cpu',
                    self.native_folders,self.native_specs,self.comp_folders,self.comp_specs,self.log,
                    {'synthetic_only':True,'method':item['method'],'name':profile['name']})
                self.assertEqual(len(value['rows']),1);self.assertFalse(value['rows'][0]['test_used_for_training_or_selection'])
                if item['method']=='bertopic':self.assertIsNone(value['rows'][0]['rho'])
                else:self.assertIsNotNone(value['rows'][0]['development_prediction']['macro_source_nll'])
    def test_complete_test_report_from_saved_synthetic_slots(self):
        # Ensure both previous integration checks have executed, regardless of unittest ordering.
        self.test_real_author_neuromax_fit_readout_and_restart();self.test_all_ten_remaining_frozen_comparator_interfaces()
        paths=sorted((self.output/'cases').glob('**/case_result.json'));paths=[p.parent for p in paths]
        rows=score_cases(self.output,paths,self.data,self.policy,self.log)
        self.assertEqual(len(rows),15)
        # Five-seed report arithmetic fixture copies are not independent model runs.
        # They remain inside this temporary synthetic test and are never published as results.
        grid=[]
        for seed in self.policy['seeds']:
            for r in rows:
                row=copy.deepcopy(r);row['seed']=seed;grid.append(row)
        source=read_json(self.output/'cases/proposed_parent/seed809/refinement.json')
        for seed in self.policy['seeds'][1:]:write_json(self.output/f'cases/proposed_parent/seed{seed}/refinement.json',source)
        result=export(self.output,grid,self.data,self.policy,True,self.log)
        self.assertEqual(result['comparison_slots'],60);self.assertEqual(result['control_slots'],15)
        self.assertTrue((self.output/'tables/T7_external_comparison.csv').is_file())
        self.assertTrue((self.output/'figures/S7_external_reference_support.pdf').is_file())

if __name__=='__main__':unittest.main()
