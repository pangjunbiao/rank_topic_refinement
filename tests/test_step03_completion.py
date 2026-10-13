"""Independent R2 scoring, selection, leakage and policy checks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from rank_refinement.models.classical_completion import normalized_reconstruction, select_probability_backoff, fit_and_predict
from rank_refinement.runtime.artifacts import write_json
from rank_refinement.runtime.config import load_configs
from rank_refinement.stages.step03_completion import validate_completion, profile_decisions, aggregate_rows, repaired_source_folder, reference_scores


class CompletionChecks(unittest.TestCase):
    def setUp(self):
        self.configs = load_configs(Path(__file__).resolve().parents[1])
        self.native, self.policy = validate_completion(self.configs)

    def test_declared_revision_has_no_moments(self):
        self.assertIs(self.policy["moment_correction"], False)
        self.assertEqual(self.policy["canonical_projection"], "primary_minimum_tv")

    def test_policy_change_rejected(self):
        self.configs["step03_completion"]["moment_correction"] = True
        with self.assertRaises(ValueError):
            validate_completion(self.configs)

    def test_backoff_grid_change_rejected(self):
        self.configs["step03_completion"]["prediction"]["shared_backoff_grid"] = [0, .9]
        with self.assertRaises(ValueError):
            validate_completion(self.configs)

    def test_normalization_known_ratio(self):
        np.testing.assert_allclose(normalized_reconstruction([[2, 3], [10, 90]]), [[.4, .6], [.1, .9]])

    def test_zero_prediction_row_rejected(self):
        with self.assertRaises(ValueError):
            normalized_reconstruction([[0., 0.]])

    def test_negative_prediction_rejected(self):
        with self.assertRaises(ValueError):
            normalized_reconstruction([[-1., 2.]])

    def test_nonfinite_prediction_rejected(self):
        with self.assertRaises(ValueError):
            normalized_reconstruction([[float("nan"), 1.]])

    def fixtures(self):
        records = [{"group_id": "a", "source": "x"}, {"group_id": "b", "source": "y"}]
        masks = {"a": {"eligible": True, "target_ids": [1, 1]}, "b": {"eligible": True, "target_ids": [1]}}
        return records, masks

    def test_zero_probability_retained_then_explicitly_mixed(self):
        records, masks = self.fixtures()
        chosen, curve = select_probability_backoff(np.array([[1., 0.], [1., 0.]]), np.array([.5, .5]), [0., .2], records, masks, ["x", "y"])
        self.assertEqual(curve[0]["status"], "zero_probability_not_clipped")
        self.assertIsNone(curve[0]["macro_source_nll"])
        self.assertEqual(chosen["rho"], .2)
        self.assertAlmostEqual(chosen["scores"]["macro_source_nll"], -np.log(.1))

    def test_no_explicit_finite_path_rejected(self):
        records, masks = self.fixtures()
        with self.assertRaises(ValueError):
            select_probability_backoff(np.array([[1., 0.], [1., 0.]]), np.array([.5, .5]), [0.], records, masks, ["x", "y"])

    def test_zero_nontarget_probability_does_not_force_backoff(self):
        records, masks = self.fixtures()
        chosen, curve = select_probability_backoff(np.array([[0., 1.], [0., 1.]]), np.array([.5, .5]), [0., .2], records, masks, ["x", "y"])
        self.assertEqual(chosen["rho"], 0.)
        self.assertEqual(curve[0]["status"], "scored")
        self.assertEqual(chosen["scores"]["macro_source_nll"], 0.)

    def test_probability_rows_must_sum_to_one(self):
        records, masks = self.fixtures()
        with self.assertRaises(ValueError):
            select_probability_backoff(np.ones((2, 2)), np.array([.5, .5]), [0.], records, masks, ["x", "y"])

    def test_source_macro_not_document_macro(self):
        records, masks = self.fixtures()
        p = np.array([[.8, .2], [.4, .6]])
        chosen, _ = select_probability_backoff(p, np.array([.5, .5]), [0.], records, masks, ["x", "y"])
        self.assertAlmostEqual(chosen["scores"]["macro_source_nll"], (-np.log(.2) - np.log(.6))/2)
        self.assertAlmostEqual(chosen["scores"]["micro_nll"], (-2*np.log(.2) - np.log(.6))/3)

    def case(self, base, profile, seed, native=6., proposed=5.99, accepted=True):
        return {"base": base, "profile": profile, "seed": seed, "fitting_accepted": accepted,
            "rows": [{"role": "native", "prediction": {"macro_source_nll": native}},
                     {"role": "proposed", "prediction": {"macro_source_nll": proposed}}]}

    def test_two_seed_positive_candidate_selected(self):
        cases = [self.case("neuromax", "random_words", s) for s in [809, 810]]
        _, chosen = profile_decisions(cases, self.native, self.policy)
        self.assertEqual((chosen["base"], chosen["profile"]), ("neuromax", "random_words"))

    def test_missing_seed_blocks_selection(self):
        _, chosen = profile_decisions([self.case("neuromax", "random_words", 809)], self.native, self.policy)
        self.assertIsNone(chosen)

    def test_duplicate_seed_blocks_selection(self):
        _, chosen = profile_decisions([self.case("neuromax", "random_words", 809)]*2, self.native, self.policy)
        self.assertIsNone(chosen)

    def test_one_harmed_seed_blocks_selection(self):
        cases = [self.case("neuromax", "random_words", 809, proposed=5.9), self.case("neuromax", "random_words", 810, proposed=6.001)]
        _, chosen = profile_decisions(cases, self.native, self.policy)
        self.assertIsNone(chosen)

    def test_tiny_mean_gain_blocks_selection(self):
        cases = [self.case("neuromax", "random_words", s, proposed=5.9999) for s in [809, 810]]
        _, chosen = profile_decisions(cases, self.native, self.policy)
        self.assertIsNone(chosen)

    def test_failed_fitting_gate_blocks_selection(self):
        cases = [self.case("neuromax", "random_words", s, accepted=False) for s in [809, 810]]
        _, chosen = profile_decisions(cases, self.native, self.policy)
        self.assertIsNone(chosen)

    def test_missing_repair_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(RuntimeError, "--repair"):
            repaired_source_folder(Path(d))

    def test_source_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            write_json(Path(d)/"state/step03_repair_result.json", {"processed_relative_path": "../foreign"})
            with self.assertRaises(ValueError):
                repaired_source_folder(Path(d))

    def test_repaired_pointer_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rel = "outputs/steps/step03_saved_replay/abc"
            (root/rel).mkdir(parents=True)
            write_json(root/"state/step03_repair_result.json", {"processed_relative_path": rel})
            self.assertEqual(repaired_source_folder(root), (root/rel).resolve())

    def test_common_reference_keeps_missing_word_slots(self):
        vocabulary = ["w"+str(i) for i in range(100)]
        topics = [vocabulary[i:i+10] for i in range(0,100,10)]
        data = {"vocabulary": vocabulary,
            "fit_records": [{"group_id": "f"+str(i), "tokens": t} for i,t in enumerate(topics)],
            "dev_coherence_records": [{"group_id": "d", "tokens": topics[0]}]}
        with tempfile.TemporaryDirectory() as d:
            fit, dev, audit = reference_scores(topics, data, Path(d))
            self.assertTrue(fit["scored"])
            self.assertFalse(dev["scored"])
            self.assertIsNone(dev["c_v"])
            self.assertEqual(audit["development"]["total_slots"], 100)
            self.assertEqual(audit["development"]["supported_slots"], 10)
            with patch("rank_refinement.stages.step03_completion.audit_reference", side_effect=AssertionError("should reuse verified cache")):
                self.assertEqual(reference_scores(topics, data, Path(d)), (fit, dev, audit))

    def test_nmf_idf_is_fitting_only_and_predictions_are_normalized(self):
        rng = np.random.default_rng(109)
        fit = rng.integers(0, 3, (30, 120)).astype(float)
        observed = rng.integers(0, 3, (4, 120)).astype(float)
        profile = self.policy["classical_panel"]["profiles"]["nmf"][0]
        profile = {**profile, "max_iter": 20}
        _, weights, pred, detail = fit_and_predict("nmf", fit, observed, ["w"+str(i) for i in range(120)], 809, profile)
        expected_idf = np.log((1+len(fit))/(1+(fit>0).sum(0)))+1
        np.testing.assert_allclose(detail["fitting_only_idf"], expected_idf)
        np.testing.assert_allclose(pred.sum(1), 1.)
        _, weights2, _, detail2 = fit_and_predict("nmf", fit, observed[:,::-1], ["w"+str(i) for i in range(120)], 809, profile)
        np.testing.assert_array_equal(weights, weights2)
        self.assertEqual(detail["fitting_only_idf"], detail2["fitting_only_idf"])

    def test_lda_prediction_is_observed_only_and_normalized(self):
        rng = np.random.default_rng(91)
        profile = self.policy["classical_panel"]["profiles"]["lda"][0]
        fit = rng.integers(0, 3, (30,120)); observed = rng.integers(0,3,(4,120))
        topics, _, pred, details = fit_and_predict("lda", fit, observed, ["w"+str(i) for i in range(120)], 809, profile)
        self.assertEqual((len(topics),len(topics[0])),(10,10))
        self.assertFalse(details["development_full_text_used_for_inference"])
        np.testing.assert_allclose(pred.sum(1),1.)

    def test_empty_observed_context_rejected(self):
        with self.assertRaises(ValueError):
            fit_and_predict("lda", np.ones((20,120)), np.zeros((1,120)), [str(i) for i in range(120)],809,self.policy["classical_panel"]["profiles"]["lda"][0])

    def row(self, seed):
        return {"method": "Proposed", "role": "proposed", "underlying_base": "neuromax", "profile": "random_words", "seed": seed,
            "fitting": {"c_v": .5,"c_npmi": .1}, "development": {"c_v": None,"c_npmi": None},
            "topic_diversity": 1., "prediction": {"macro_source_nll":6.,"micro_nll":6.1},
            "rho":0.,"moment_correction":False,"projection_deployed":True}

    def test_incomplete_seed_mean_not_reported(self):
        summary=aggregate_rows([self.row(809)],[809,810])[0]
        self.assertFalse(summary["complete_seed_grid"])
        self.assertIsNone(summary["development_macro_nll_mean"])
        self.assertIsNone(summary["fitting_c_v_mean"])

    def test_absent_heldout_metrics_remain_na(self):
        summary=aggregate_rows([self.row(809),self.row(810)],[809,810])[0]
        self.assertTrue(summary["complete_seed_grid"])
        self.assertIsNone(summary["development_c_v_mean"])
        self.assertIsNone(summary["development_c_npmi_mean"])
        self.assertEqual(summary["fitting_c_v_mean"],.5)
        self.assertEqual(summary["fitting_c_v_sd"],0.)


if __name__ == "__main__":
    unittest.main()
