"""Full-set search, exact diversity accounting and revision gate checks."""
from pathlib import Path
from copy import deepcopy
from itertools import combinations
import tempfile
import unittest
import numpy as np
from rank_refinement.acquisition.native_targets import allocate
from rank_refinement.acquisition.complete_targets import complete_quality, choose_complete_targets, evidence_matrices
from rank_refinement.runtime.config import load_configs
from rank_refinement.runtime.artifacts import write_json
from rank_refinement.stages.step03_improvement import (validate_improvement, completion_source_folder,
    load_evidence, quality_gates, failed_row)
from rank_refinement.stages.step03_completion import aggregate_rows


class FullsetChecks(unittest.TestCase):
    def setUp(self):
        self.configs = load_configs(Path(__file__).resolve().parents[1])
        _, self.policy = validate_improvement(self.configs)

    def test_frozen_scope_has_no_new_fits_or_test_evaluation(self):
        for field in ["new_neural_fits", "new_classical_fits", "new_encoder_calls"]:
            self.assertEqual(self.policy[field], 0)
        self.assertFalse(self.policy["moment_correction"])
        self.assertFalse(self.policy["test_or_external_used"])

    def test_post_scoring_configuration_change_is_rejected(self):
        self.configs["step03_improvement"]["acquisition"]["beam_width"] = 13
        with self.assertRaises(ValueError):
            validate_improvement(self.configs)

    def test_all_45_signed_pairs_count(self):
        matrix = np.full((10, 10), -.5); np.fill_diagonal(matrix, 0)
        matrix[0, 1] = matrix[1, 0] = 1.
        expected = (1 - 44*.5)/45
        self.assertAlmostEqual(complete_quality(list(range(10)), dict(enumerate(range(10))), matrix), expected)

    def test_duplicate_descriptor_word_is_rejected(self):
        with self.assertRaises(ValueError):
            complete_quality([0]*10, {0:0}, np.zeros((1,1)))

    def test_legacy_allocator_still_requires_pairs(self):
        c = {"topic":0, "target":[2,3], "core":[2], "expected_gain":1.}
        with self.assertRaises(ValueError):
            allocate([[0,1],[2,3]], [c], 6)

    def test_exact_union_constraint_prevents_fake_diversity(self):
        native = [[0,1],[1,2]]
        candidates = [{"topic":0,"target":[0,3],"core":[3],"expected_gain":.3},
                      {"topic":1,"target":[1,3],"core":[3],"expected_gain":.7}]
        targets, report = allocate(native,candidates,6,core_sizes=(1,),required_unique=4)
        self.assertEqual(len({w for row in targets for w in row}),4)
        self.assertTrue(report["required_diversity_achieved"])
        self.assertEqual(targets, [[0,3],[1,2]])

    def test_infeasible_disjoint_pool_is_saved_as_abstention(self):
        targets,report=allocate([[0,1],[1,2]],[],6,core_sizes=(1,),required_unique=4)
        self.assertEqual(targets,[[0,1],[1,2]])
        self.assertFalse(report["required_diversity_achieved"])

    def fixture(self):
        vocabulary=["w"+str(i) for i in range(120)]
        native=np.ones((10,120))
        for t in range(10): native[t,t*10:(t+1)*10]=np.arange(100,90,-1)
        native[0,100:104]=np.arange(90,86,-1)
        pairs=np.array(list(combinations(range(120),2)),dtype=np.int64)
        stable=np.full(len(pairs),-.2); full=stable.copy(); support=np.ones(len(pairs),dtype=np.int64)
        clique=set(list(range(6))+list(range(100,104)))
        for j,(a,b) in enumerate(pairs):
            if a in clique and b in clique: stable[j]=.6; full[j]=.7; support[j]=10
        bundle={"words":vocabulary,"pairs":pairs,"stable":stable,"full":full,
                "pair_support":support,"word_support":np.full(120,30,dtype=np.int64)}
        metadata={"tokens":vocabulary,"document_frequency":[30]*120}
        return native,vocabulary,metadata,bundle

    def test_more_than_two_words_can_change_with_final_support(self):
        native,vocabulary,metadata,bundle=self.fixture()
        target,report=choose_complete_targets(native,vocabulary,metadata,bundle,self.policy["acquisition"])
        self.assertTrue(report["fully_disjoint_targets"])
        self.assertEqual(len(set(target[0])-set(range(10))),4)
        self.assertEqual(len(set(target[0])&set(range(10))),6)
        _,_,mapping,(stable,_,support)=evidence_matrices(bundle,vocabulary,metadata,native,self.policy["acquisition"])
        for item in report["selected"]:
            for a in item["core"]:
                degree=sum(support[mapping[a],mapping[b]]>=5 and stable[mapping[a],mapping[b]]>=.1
                           for b in item["target"] if b!=a)
                self.assertGreaterEqual(degree,1)
        self.assertFalse(report["development_inputs_used"])

    def test_missing_pair_evidence_is_rejected(self):
        native,vocabulary,metadata,bundle=self.fixture()
        bundle["pairs"]=bundle["pairs"][:-1]
        with self.assertRaises(ValueError):
            evidence_matrices(bundle,vocabulary,metadata,native,self.policy["acquisition"])

    def test_wrong_fitting_frequency_is_rejected(self):
        native,vocabulary,metadata,bundle=self.fixture();bundle["word_support"][0]=31
        with self.assertRaises(ValueError):
            evidence_matrices(bundle,vocabulary,metadata,native,self.policy["acquisition"])

    def test_pair_support_cannot_exceed_group_word_support(self):
        native,vocabulary,metadata,bundle=self.fixture();bundle["pair_support"][0]=31
        with self.assertRaises(ValueError):
            evidence_matrices(bundle,vocabulary,metadata,native,self.policy["acquisition"])

    def test_nonfinite_association_is_rejected(self):
        native,vocabulary,metadata,bundle=self.fixture();bundle["stable"][0]=np.nan
        with self.assertRaises(ValueError):
            evidence_matrices(bundle,vocabulary,metadata,native,self.policy["acquisition"])

    def test_source_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            write_json(Path(d)/"state/step03_completion_result.json",{"processed_relative_path":"../foreign"})
            with self.assertRaises(ValueError): completion_source_folder(Path(d))

    def test_missing_source_has_clear_run_instruction(self):
        with tempfile.TemporaryDirectory() as d,self.assertRaisesRegex(RuntimeError,"--complete"):
            completion_source_folder(Path(d))

    def test_foreign_group_in_calibration_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);np.savez(folder/"association_evidence.npz",stable=np.array([.1]))
            write_json(folder/"association_words.json",{"words":["a","b"],"fit_groups":1})
            write_json(folder/"sampling_group_ids.json",[["development-group"]]*20)
            with self.assertRaises(ValueError): load_evidence(folder,[{"group_id":"f","split":"fit"}])

    def row(self, role, seed, nll, cv=.6, npmi=.1, td=1.):
        return {"method":role,"role":role,"underlying_base":"neuromax","profile":"random_words","seed":seed,
            "status":"scored","fitting":{"c_v":cv,"c_npmi":npmi},"development":{"c_v":None,"c_npmi":None},
            "topic_diversity":td,"prediction":{"macro_source_nll":nll,"micro_nll":nll},"rho":0.,
            "moment_correction":False,"projection_deployed":role!="native"}

    def cases(self):
        return [{"seed":s,"rows":[self.row("native",s,6.,.5,0.),self.row("pair_acquisition",s,5.997,.55,.02),
            self.row("proposed",s,5.99)],"certificates":{"primary":{"success":True,"certified":True,"relative_l1_movement":.001}}}
            for s in [809,810]]

    def gates(self, cases):
        means=aggregate_rows([r for c in cases for r in c["rows"]],[809,810])
        baseline=deepcopy(next(m for m in means if m["role"]=="native"));baseline["method"]="competitor"
        return quality_gates(cases,means,[baseline],{},self.policy)

    def test_complete_mechanism_and_prediction_allow_bounded_followup(self):
        gate=self.gates(self.cases());self.assertTrue(gate["bounded_followup_allowed"])
        self.assertFalse(gate["full_final_experiment_allowed"])
        self.assertFalse(gate["heldout_coherence_superiority_established"])

    def test_one_harmed_native_seed_blocks_revision(self):
        cases=self.cases();cases[1]["rows"][2]["prediction"]["macro_source_nll"]=6.001
        self.assertFalse(self.gates(cases)["revision_gate_passed"])

    def test_missing_seed_blocks_revision(self):
        self.assertFalse(self.gates(self.cases()[:1])["revision_gate_passed"])

    def test_duplicate_seed_blocks_revision(self):
        c=self.cases()[0];self.assertFalse(self.gates([c,c])["revision_gate_passed"])

    def test_readout_diversity_loss_blocks_revision(self):
        cases=self.cases();cases[1]["rows"][2]["topic_diversity"]=.99
        self.assertFalse(self.gates(cases)["revision_gate_passed"])

    def test_excess_affinity_movement_blocks_revision(self):
        cases=self.cases();cases[1]["certificates"]["primary"]["relative_l1_movement"]=.051
        self.assertFalse(self.gates(cases)["revision_gate_passed"])

    def test_mean_prediction_regression_to_r2_blocks_revision(self):
        cases=self.cases()
        for c in cases: c["rows"][2]["prediction"]["macro_source_nll"]=5.998
        self.assertFalse(self.gates(cases)["revision_gate_passed"])

    def test_failed_candidate_retains_na_without_becoming_native_result(self):
        cases=self.cases();cases[1]["rows"][2]=failed_row("proposed","Proposed",810,"solver_failed",self.policy)
        gate=self.gates(cases);self.assertFalse(gate["revision_gate_passed"])
        self.assertFalse(gate["all_tested_baseline_profiles_win_or_tie"])


if __name__=="__main__": unittest.main()
