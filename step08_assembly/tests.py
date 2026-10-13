"""Independent reporting checks; real-data validation is performed by the full command."""
from __future__ import annotations
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
from rank_refinement.runtime.artifacts import artifact_manifest,read_json,write_json,write_text
from .integrity import safe,verify_tree,unchanged,SEEDS
from .metrics import aggregate,mix_logp,replay_losses,summaries,grid,normalize,point_winners,logits
from .tables import fmt,escape,comparison_table
from .audit import mechanism,check_masks
from .runner import zip_files

def row(seed=809,value=2.):
    return {'dataset':'External','method':'Proposed','profile':'fixed','seed':seed,'control':False,
        'role':'proposed','status':'scored','fixed_k10':True,'native_topic_count':10,
        'rho':0.,'fit_c_v':.4,'fit_npmi':-.1,'td':1.,'test_nll':value,'test_micro_nll':value,
        'test_c_v':None,'test_npmi':None}

class ReportingChecks(unittest.TestCase):
    def test_token_weighted_loss_differs_from_document_average(self):
        r=aggregate([{'group_id':'a','source':'s','target_tokens':1,'negative_log_probability':2.},
                     {'group_id':'b','source':'s','target_tokens':3,'negative_log_probability':3.}])
        self.assertEqual(r['micro_nll'],1.25)

    def test_macro_sources_are_equal_weighted(self):
        r=aggregate([{'group_id':'a','source':'a','target_tokens':1,'negative_log_probability':2.},
                     {'group_id':'b','source':'b','target_tokens':3,'negative_log_probability':3.}])
        self.assertEqual(r['macro_source_nll'],1.5);self.assertEqual(r['micro_nll'],1.25)

    def test_duplicate_loss_group_rejected(self):
        r={'group_id':'a','source':'s','target_tokens':1,'negative_log_probability':2.}
        with self.assertRaises(ValueError):aggregate([r,r])

    def test_nonfinite_loss_rejected(self):
        with self.assertRaises(ValueError):aggregate([{'group_id':'a','source':'s','target_tokens':1,'negative_log_probability':None}])

    def test_backoff_matches_direct_mixture(self):
        actual=np.exp(mix_logp(np.log([[.25,.75]]),np.array([.5,.5]),.2))
        np.testing.assert_allclose(actual,[[.3,.7]],atol=1e-15)

    def test_zero_non_target_probability_remains_zero(self):
        value=mix_logp(np.array([[-np.inf,0.]]),np.array([.5,.5]),0.)
        self.assertTrue(np.isneginf(value[0,0]))

    def test_invalid_probability_mass_rejected(self):
        with self.assertRaises(ValueError):mix_logp(np.log([[.2,.2]]),np.array([.5,.5]),0.)

    def test_repeated_target_occurrences_counted(self):
        masks={'g':{'eligible':True,'target_ids':[1,1],'target_positions':[1,2],'observed_positions':[0]}}
        expected=-2*math.log(.75)
        r={'rho':0.,'document_losses':[{'group_id':'g','source':'s','target_tokens':2,'negative_log_probability':expected}],
           'prediction':{'macro_source_nll':expected/2,'micro_nll':expected/2,'by_source':{'s':expected/2}}}
        checked=replay_losses(np.log([[.25,.75]]),np.array([.5,.5]),r,['g'],masks,{'g':{'source':'s'}})
        self.assertLess(checked['max_loss_error'],1e-15);self.assertEqual(checked['target_tokens'],2)

    def test_zero_target_probability_is_not_clipped(self):
        masks={'g':{'eligible':True,'target_ids':[0],'target_positions':[1],'observed_positions':[0]}}
        with self.assertRaises(ValueError):
            replay_losses(np.array([[-np.inf,0.]]),np.array([.5,.5]),{'rho':0.},['g'],masks,{'g':{'source':'s'}})

    def test_missing_seed_prevents_mean(self):
        with self.assertRaises(ValueError):summaries([row(s)for s in SEEDS[:-1]])

    def test_nonzero_sample_sd(self):
        result=summaries([row(s,float(i+1))for i,s in enumerate(SEEDS)])[0]
        self.assertEqual(result['test_nll_mean'],3.);self.assertAlmostEqual(result['test_nll_sd'],math.sqrt(2.5))

    def test_unavailable_seed_is_not_removed(self):
        rows=[row(s)for s in SEEDS];rows[0]['fit_npmi']=None
        result=summaries(rows)[0]
        self.assertIsNone(result['fit_npmi_mean']);self.assertEqual(result['fit_npmi_available_seeds'],4)

    def test_duplicate_seed_rejected(self):
        rows=[row(s)for s in SEEDS];rows[-1]['seed']=809
        with self.assertRaises(ValueError):grid(rows,5)

    def test_real_zero_sd_is_retained(self):
        result=summaries([row(s)for s in SEEDS])[0]
        self.assertEqual(result['fit_c_v_sd'],0.);self.assertIn('0.0000',fmt(.4,0.))

    def test_negative_npmi_can_be_best(self):
        items=[]
        for dataset in ['Primary','External']:
            items.extend([{'dataset':dataset,'method':m,'profile':'x','control':False,
                'fit_c_v_mean':.4,'fit_npmi_mean':v,'td_mean':1.,'test_nll_mean':2.}
                for m,v in [('A',-.1),('B',-.3)]])
        winning=[r for r in point_winners(items)if r['metric']=='fit_npmi']
        self.assertTrue(all(r['method']=='A'for r in winning))

    def test_variable_topic_metrics_preserved_as_unavailable(self):
        r=normalize({'method':'GSDMM','profile':'x','seed':809,'status':'scored','topics':None,'fixed_k10':False,
                     'native_topic_count':1,'fitting_coherence':{'c_v':.4,'c_npmi':.1},'prediction':{'macro_source_nll':2.}},'External')
        self.assertIsNone(r['fit_c_v']);self.assertEqual(r['test_nll'],2.)

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            with self.assertRaises(ValueError):safe(Path(tmp),'../outside.json')

    def test_renamed_or_modified_payload_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);write_text(root/'report.txt','original');write_json(root/'artifact_manifest.json',artifact_manifest(root))
            write_text(root/'report.txt','modified')
            with self.assertRaises(ValueError):verify_tree(root)

    def test_unindexed_extra_file_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);write_text(root/'report.txt','original');write_json(root/'artifact_manifest.json',artifact_manifest(root))
            write_text(root/'extra.txt','extra')
            with self.assertRaises(ValueError):verify_tree(root)

    def test_only_unused_training_checkpoint_can_be_omitted(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);p=root/'workspace/data/cache/m/checkpoint.pt';p.parent.mkdir(parents=True);p.write_bytes(b'checkpoint')
            write_text(root/'scores.json','scores');write_json(root/'artifact_manifest.json',artifact_manifest(root));p.unlink()
            self.assertEqual(len(verify_tree(root,True)['omitted_unused_training_checkpoints']),1)
            (root/'scores.json').unlink()
            with self.assertRaises(ValueError):verify_tree(root,True)

    def test_affine_readout_softmax_oracle(self):
        actual=np.exp(logits(np.array([[1.,0.]]),np.array([[0.,math.log(3)],[2.,3.]]),np.ones(2),np.zeros(2)))
        np.testing.assert_allclose(actual,[[.25,.75]],atol=1e-15)

    def test_projection_totals_and_topwords_verified(self):
        native=np.tile(np.linspace(2,1,12),(10,1));words=[str(i)for i in range(12)]
        actual=[words[:10]for _ in range(10)]
        result=mechanism(native,native.copy(),words,actual,809,'fixture')
        self.assertEqual(result['membership_recall'],1.);self.assertEqual(result['relative_l1_movement'],0.)

    def test_changed_reported_words_rejected(self):
        native=np.tile(np.linspace(2,1,12),(10,1));words=[str(i)for i in range(12)]
        actual=[words[:10]for _ in range(10)];actual[0]=words[1:11]
        with self.assertRaises(ValueError):mechanism(native,native.copy(),words,actual,809,'fixture')

    def test_all_iv_occurrences_must_be_partitioned(self):
        f={'g':{'split':'test','tokens':['a','b','a']}}
        m={'g':{'eligible':True,'observed_positions':[0],'target_positions':[1],
                'observed_ids':[0],'target_ids':[1]}}
        with self.assertRaises(ValueError):check_masks(f,m,['g'],'test',['a','b'])

    def test_latex_special_characters_escaped(self):
        self.assertEqual(escape('a_b & 50%'),r'a\_b \& 50\%')

    def test_exact_csv_and_na_presentation(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);s=summaries([row(seed)for seed in SEEDS]);s[0]['test_nll_mean']=None;s[0]['test_nll_sd']=None
            comparison_table(root,'table',s,'Fixture')
            import csv
            with(root/'table.csv').open(encoding='utf-8-sig',newline='')as f:saved=list(csv.DictReader(f))
            self.assertEqual(float(saved[0]['fit_c_v_mean']),.4)
            self.assertEqual(saved[0]['test_nll_mean'],'')
            self.assertIn('NA',(root/'table.md').read_text())
            self.assertIn(r'\begin{table*}',(root/'table.tex').read_text())

    def test_review_zip_keeps_flat_relative_paths(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);write_text(root/'SUMMARY.md','summary')
            p=zip_files(root/'paper.zip',root,[root/'SUMMARY.md'])
            from zipfile import ZipFile
            with ZipFile(p)as z:self.assertIn('SUMMARY.md',z.namelist());self.assertIsNone(z.testzip())
