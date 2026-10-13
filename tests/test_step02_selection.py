import copy
import unittest
from rank_refinement.acquisition.pilot import choose_setting


class SelectionOracles(unittest.TestCase):
    def setUp(self):
        self.grid = [{"index": 1, "name": "recipe_a", "word_min": 10, "pair_min": 3, "gamma": 0}]
        self.rows = [{"fold": f, "seed": s, "setting_name": "recipe_a", "method": "proposed_stable",
                      "scored": True, "structural_passed": True, "c_v": .7, "c_npmi": .2,
                      "core_reference_token_fraction": .3} for f in [1, 2, 3] for s in [809, 810]]
        self.anchors = [{"fold": f, "seed": s, "method": m, "scored": True,
                         "c_v": .6 if m == "lda" else .5, "c_npmi": .1 if m == "nmf" else .05}
                        for m in ["lda", "nmf", "gsdmm", "bertopic"] for f in [1, 2, 3] for s in [809, 810]]

    def select(self):
        return choose_setting(self.rows, self.anchors, self.grid)[0]

    def test_both_endpoint_leadership_and_three_fold_advantages_qualify(self):
        self.assertIsNotNone(self.select())

    def test_a_single_failed_support_run_blocks_selection(self):
        self.rows[0]["structural_passed"] = False
        self.assertIsNone(self.select())

    def test_no_positive_overall_npmi_blocks_selection(self):
        for r in self.rows:
            r["c_npmi"] = -.01
        self.assertIsNone(self.select())

    def test_npmi_win_without_cv_win_does_not_qualify(self):
        for r in self.rows:
            r["c_v"] = .59
        self.assertIsNone(self.select())

    def test_missing_seed_cannot_disappear_from_mean(self):
        self.rows.pop()
        self.assertIsNone(self.select())

    def test_duplicate_fold_seed_observation_is_rejected(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        self.assertIsNone(self.select())

    def test_missing_anchor_blocks_comparative_gate(self):
        self.anchors = [r for r in self.anchors if r["method"] != "bertopic"]
        self.assertIsNone(self.select())

    def test_unscorable_anchor_is_not_dropped(self):
        self.anchors[0]["scored"] = False
        self.assertIsNone(self.select())

    def test_aggregate_gain_with_only_one_winning_fold_fails(self):
        for r in self.rows:
            r["c_v"] = 1 if r["fold"] == 1 else .59
        self.assertIsNone(self.select())

    def test_two_winning_folds_with_aggregate_lead_qualify(self):
        for r in self.rows:
            r["c_v"] = .8 if r["fold"] in [1, 2] else .5
        self.assertIsNotNone(self.select())

    def test_exact_ties_do_not_count_as_superiority(self):
        for r in self.rows:
            r["c_v"] = .6
        self.assertIsNone(self.select())

    def test_fixed_configuration_index_breaks_complete_recipe_tie(self):
        self.grid.append({**self.grid[0], "index": 2, "name": "recipe_b"})
        self.rows.extend({**r, "setting_name": "recipe_b"} for r in list(self.rows))
        self.assertEqual(self.select()["index"], 1)
