"""Independent numerical examples and acceptance/firewall regression checks."""
import copy
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from rank_refinement.runtime.config import load_configs
from rank_refinement.acquisition.native_targets import allocate, calibrate, native_indices
from rank_refinement.projection.minimum_tv import minimum_tv, certify
from rank_refinement.stages.step02_native_refinement import validate_protocol, fitting_acceptance, metric_score, run_case
from rank_refinement.runtime.artifacts import artifact_manifest, write_json


class NativeRefinementTests(unittest.TestCase):
    def policy(self):
        return validate_protocol(load_configs(Path(__file__).resolve().parents[1]))

    def test_protocol_and_preparation_configuration_are_separate(self):
        root = Path(__file__).resolve().parents[1]
        config = load_configs(root)
        before = copy.deepcopy({k: config[k] for k in ["default", "data", "evaluation"]})
        self.assertEqual(validate_protocol(config)["protocol"], "R3_NATIVE_TOP10_DIVERSITY_20261009")
        self.assertEqual(before, {k: config[k] for k in before})

    def test_support_cannot_be_lowered_silently(self):
        config = load_configs(Path(__file__).resolve().parents[1])
        config["step02_refinement"]["acquisition"]["word_group_support"] = 5
        with self.assertRaises(ValueError):
            validate_protocol(config)

    def test_conserved_two_by_two_optimum_has_known_tv(self):
        # Every feasible conserved matrix is [[a,1-a],[1-a,a]].
        # Reversing both leading words requires a<= (1-margin)/2.
        # The minimum TV is 1+margin, with a=(1-margin)/2.
        native = np.eye(2)
        result, report = minimum_tv(native, [[1], [0]], margin=1e-4)
        self.assertIsNotNone(result)
        self.assertTrue(report["certified"])
        np.testing.assert_allclose(result, [[.49995, .50005], [.50005, .49995]], atol=1e-9)
        self.assertAlmostEqual(report["tv_scaled"], 1.0001, places=8)
        self.assertTrue(report["primary_optimal"])
        self.assertFalse(report["post_solve_clipping"])

    def test_infeasible_zero_column_target_is_reported(self):
        native = np.array([[1., 0.], [1., 0.]])
        result, report = minimum_tv(native, [[1], [1]])
        self.assertIsNone(result)
        self.assertFalse(report["success"])
        self.assertFalse(report["certified"])

    def test_exhausted_solver_budget_returns_a_saved_failure(self):
        result, report = minimum_tv(np.eye(2), [[1], [0]], seconds=1e-20)
        self.assertIsNone(result)
        self.assertFalse(report["success"])
        self.assertFalse(report["solver_attempts"])

    def test_certificate_detects_column_error_even_with_correct_rows(self):
        native = np.array([[.7, .3], [.3, .7]])
        changed = np.array([[.71, .29], [.31, .69]])
        self.assertFalse(certify(native, changed, [[0], [1]])["certified"])

    def test_certificate_detects_wrong_rank_set(self):
        native = np.array([[.7, .3], [.3, .7]])
        report = certify(native, native, [[1], [0]])
        self.assertFalse(report["actual_top_sets_realized"])
        self.assertFalse(report["certified"])

    def test_scale_invariance_preserves_actual_affinity_totals(self):
        native = np.array([[.7, .3], [.3, .7]])
        a, ca = minimum_tv(native, [[1], [0]])
        b, cb = minimum_tv(native * 1000, [[1], [0]])
        np.testing.assert_allclose(a * 1000, b, atol=1e-8)
        self.assertAlmostEqual(ca["relative_l1_movement"], cb["relative_l1_movement"], places=10)

    def test_allocator_does_not_invent_unique_words(self):
        native = [[0, 1], [2, 3]]
        candidate = {"topic": 0, "target": [2, 3], "core": [2, 3], "expected_gain": 10.0}
        targets, report = allocate(native, [candidate], 4)
        self.assertEqual(targets, native)
        self.assertFalse(report["selected"])
        self.assertEqual(report["target_unique_words"], 4)

    def test_allocator_can_replace_diversity_preservingly(self):
        native = [[0, 1], [2, 3]]
        candidate = {"topic": 0, "target": [4, 5], "core": [4, 5], "expected_gain": 0.1}
        targets, report = allocate(native, [candidate], 6)
        self.assertEqual(targets, [[4, 5], [2, 3]])
        self.assertEqual(len(report["selected"]), 1)

    def test_allocator_respects_pair_disjointness(self):
        native = [[0, 1], [2, 3]]
        a = {"topic": 0, "target": [4, 5], "core": [4, 5], "expected_gain": 0.2}
        b = {"topic": 1, "target": [5, 6], "core": [5, 6], "expected_gain": 0.1}
        targets, report = allocate(native, [a, b], 7)
        self.assertEqual(len(report["selected"]), 1)
        self.assertEqual(report["selected"][0]["topic"], 0)

    def test_no_candidate_preserves_native_sets(self):
        native = [[0, 1], [2, 3]]
        targets, report = allocate(native, [], 4)
        self.assertEqual(targets, native)
        self.assertTrue(report["allocation_verified"])

    def test_calibration_rejects_validation_and_duplicate_groups(self):
        rows = [{"group_id": "a", "source": "x", "tokens": ["a", "b"], "split": "validation"}]
        with self.assertRaises(ValueError):
            calibrate(rows, ["a", "b"], 809, self.policy()["acquisition"])
        rows[0]["split"] = "fit"
        with self.assertRaises(ValueError):
            calibrate(rows * 2, ["a", "b"], 809, self.policy()["acquisition"])

    def test_calibration_samples_are_reproducible_and_source_stratified(self):
        rows = [{"group_id": source + str(i), "source": source, "tokens": ["a", "b"], "split": "fit"}
                for source in ["x", "y"] for i in range(4)]
        p = self.policy()["acquisition"]
        a = calibrate(rows, ["a", "b"], 809, p)
        b = calibrate(rows, ["a", "b"], 809, p)
        self.assertEqual(a["sample_group_ids"], b["sample_group_ids"])
        self.assertEqual(a["pair_support"].tolist(), [8])
        for chosen in a["sample_group_ids"]:
            self.assertEqual(sum(x.startswith("x") for x in chosen), 2)
            self.assertEqual(sum(x.startswith("y") for x in chosen), 2)

    def test_acceptance_rejects_a_diversity_loss(self):
        n = {"scored": True, "c_v": .4, "c_npmi": -.1}
        c = {"scored": True, "c_v": .5, "c_npmi": -.05}
        cert = {"success": True, "certified": True, "relative_l1_movement": .01}
        accepted, reasons = fitting_acceptance(n, c, .8, .79, cert, self.policy())
        self.assertFalse(accepted)
        self.assertIn("actual_top10_diversity_decreases", reasons)

    def test_negative_npmi_can_improve_without_being_replaced(self):
        n = {"scored": True, "c_v": .4, "c_npmi": -.1}
        c = {"scored": True, "c_v": .5, "c_npmi": -.05}
        cert = {"success": True, "certified": True, "relative_l1_movement": .01}
        self.assertTrue(fitting_acceptance(n, c, .8, .8, cert, self.policy())[0])

    def test_positive_npmi_does_not_override_a_cv_loss(self):
        n = {"scored": True, "c_v": .6, "c_npmi": -.1}
        c = {"scored": True, "c_v": .5, "c_npmi": .1}
        cert = {"success": True, "certified": True, "relative_l1_movement": .01}
        self.assertFalse(fitting_acceptance(n, c, .8, .8, cert, self.policy())[0])

    def test_unscorable_fitting_cannot_be_accepted(self):
        n = {"scored": False, "c_v": None, "c_npmi": None}
        c = {"scored": True, "c_v": .5, "c_npmi": .1}
        cert = {"success": True, "certified": True, "relative_l1_movement": .01}
        self.assertFalse(fitting_acceptance(n, c, .8, .8, cert, self.policy())[0])

    def test_unavailable_candidate_does_not_invent_td_or_movement_losses(self):
        n = {"scored": True, "c_v": .4, "c_npmi": -.1}
        cert = {"success": False, "certified": False}
        accepted, reasons = fitting_acceptance(n, None, .8, None, cert, self.policy())
        self.assertFalse(accepted)
        self.assertIn("no_scored_projection_candidate", reasons)
        self.assertNotIn("actual_top10_diversity_decreases", reasons)
        self.assertNotIn("movement_exceeds_five_percent", reasons)

    def test_reference_absence_keeps_all_topic_slots(self):
        words = ["word" + str(i) for i in range(10)]
        with tempfile.TemporaryDirectory() as d:
            result = metric_score([words], [{"tokens": words[:-1]}] * 4, words, Path(d))
        self.assertEqual(result["absent_words"], ["word9"])
        self.assertTrue(result["all_ten_topics_and_words_retained"])
        if not result["scored"]:
            self.assertIsNone(result["c_npmi"])

    def test_native_indices_resolve_ties_by_original_column_order(self):
        matrix = np.ones((2, 12))
        self.assertEqual(native_indices(matrix), [list(range(10)), list(range(10))])

    def check_cached_readout_closed(self, changed):
        original_load = np.load
        handles = []
        def observed_load(*args, **kwargs):
            handle = original_load(*args, **kwargs)
            handles.append(handle)
            return handle
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "case"
            folder.mkdir()
            native = np.ones((10, 100))
            final = native + 1.0 if changed else native.copy()
            np.savez_compressed(folder / "readout.npz", native=native, final=final)
            write_json(folder / "result.json", {"fitting_accepted": False})
            write_json(folder / "artifact_manifest.json", artifact_manifest(folder))
            args = (None, None, None, None, "accepted_phrases", 1, 809, "lda",
                    self.policy(), folder, folder / "cache", SimpleNamespace(info=lambda *x: None))
            with patch("rank_refinement.stages.step02_native_refinement.np.load", side_effect=observed_load):
                if changed:
                    with self.assertRaisesRegex(ValueError, "rejected case changed"):
                        run_case(*args)
                else:
                    self.assertFalse(run_case(*args)["fitting_accepted"])
            self.assertEqual(len(handles), 1)
            self.assertIsNone(handles[0].zip)
            self.assertIsNone(handles[0].fid)
            (folder / "readout.npz").unlink()

    def test_cached_readout_handle_is_closed_before_return(self):
        self.check_cached_readout_closed(False)

    def test_cached_readout_handle_is_closed_before_validation_error(self):
        self.check_cached_readout_closed(True)
