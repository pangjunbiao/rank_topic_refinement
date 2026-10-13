import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from rank_refinement.acquisition.global_cores import (
    pair_means, pack_candidates, verify_packing, generate_candidates, select_global_cores)
from rank_refinement.acquisition.pilot import acquire
from rank_refinement.runtime.config import load_configs
from rank_refinement.runtime.step02_contract import validate_contract


class GlobalCoreSearchChecks(unittest.TestCase):
    def policy(self):
        return {"eta": .25, "max_accepted_swaps_per_core": 10,
                "global_core_search": {"diversity_draws": 4, "word_subset_fractions": [.5, .8],
                "repair_drop_counts": [1, 2], "solver_time_limit_seconds": 2,
                "diagnostic_time_limit_seconds": 2, "solver_node_limit": 1000}}

    def blocks(self):
        r = np.full((100, 100), -.7); a = np.zeros((100, 100))
        for k in range(10):
            r[10*k:10*(k+1), 10*k:10*(k+1)] = .6
            a[10*k:10*(k+1), 10*k:10*(k+1)] = 1
        np.fill_diagonal(r, 0); np.fill_diagonal(a, 0)
        return a, r, (a * 8).astype(int)

    def test_negative_pairs_remain_in_core_mean(self):
        r = np.array([[0, .8, -.7], [.8, 0, -.7], [-.7, -.7, 0]])
        q, _ = pair_means([0, 1, 2], r, np.maximum(r, 0))
        self.assertAlmostEqual(q, -.2)

    def test_exact_candidate_packing_matches_exhaustive_lexicographic_optimum(self):
        c = [(0, 1), (1, 2), (2, 3), (4, 5), (0, 5)]
        q = np.array([.2, .8, .3, .4, .7])
        feasible = [s for k in range(3) for s in itertools.combinations(range(len(c)), k)
                    if len({i for j in s for i in c[j]}) == 2*k]
        expected = max((len(s), sum(q[j] for j in s)) for s in feasible)
        ids, d = pack_candidates(c, q, 6, topics=2, depth=2)
        self.assertEqual(len(ids), expected[0]); self.assertAlmostEqual(sum(q[ids]), expected[1])
        self.assertTrue(d["optimal_within_candidate_pool"])

    def test_cardinality_dominates_negative_diagnostic_quality(self):
        ids, _ = pack_candidates([(0, 1), (2, 3), (1, 2)], [-.8, -.8, 1.2], 4, 2, 2)
        self.assertEqual(ids, [0, 1])

    def test_overlapping_core_words_rejected(self):
        with self.assertRaises(ValueError): verify_packing([[0, 1], [1, 2]], 4, 2, 2)

    def test_padded_core_rejected(self):
        with self.assertRaises(ValueError): verify_packing([[0, 0]], 4, 2, 2)

    def test_duplicate_candidates_rejected(self):
        with self.assertRaises(ValueError): pack_candidates([(0, 1), (0, 1)], [1, 1], 4, 2, 2)

    def test_empty_pool_is_explicit(self):
        ids, d = pack_candidates([], [], 100)
        self.assertEqual(ids, []); self.assertEqual(d["status"], "empty_pool")

    def test_timed_feasible_incumbent_is_not_called_optimal(self):
        fake = SimpleNamespace(x=np.array([1., 0.]), status=1, message="budget", fun=-1,
                               mip_gap=.1, mip_node_count=1)
        with patch("rank_refinement.acquisition.global_cores.milp", return_value=fake):
            ids, d = pack_candidates([(0, 1), (2, 3)], [.3, .5], 4, 2, 2)
        self.assertEqual(ids, [0]); self.assertFalse(d["optimal_within_candidate_pool"])

    def test_fractional_solver_incumbent_rejected(self):
        fake = SimpleNamespace(x=np.array([.5]), status=1, message="budget", fun=-1)
        with patch("rank_refinement.acquisition.global_cores.milp", return_value=fake):
            with self.assertRaises(ValueError): pack_candidates([(0, 1)], [.3], 4, 2, 2)

    def test_hundred_word_blocks_return_ten_cores_and_all450_pairs(self):
        a, r, s = self.blocks()
        cores, reports, pairs, cert = select_global_cores(a, r, s, np.arange(100),
                                      [f"w{i}" for i in range(100)], 809, self.policy())
        self.assertEqual(cores, [list(range(10*k,10*(k+1))) for k in range(10)])
        self.assertEqual(len(pairs), 450); self.assertTrue(all(x["signed_cohesion"] > 0 for x in reports))
        self.assertFalse(cert["validation_used_in_search"])
        json.dumps(cert, allow_nan=False); json.dumps(reports, allow_nan=False)

    def test_masks_seeded_and_saved_pool_replays_selected_evidence(self):
        a, r, s = self.blocks()
        with tempfile.TemporaryDirectory() as d:
            result = select_global_cores(a, r, s, np.arange(100), [f"w{i}" for i in range(100)], 809, self.policy(), destination=d)
            with np.load(Path(d)/"candidate_pool.npz", allow_pickle=False) as z:
                for core in result[0]:
                    j = np.flatnonzero((z["cores"] == core).all(1))[0]
                    self.assertAlmostEqual(z["signed_cohesion"][j], pair_means(core, r, a)[0])
                self.assertEqual(z["fitting_word_masks"].shape, (5, 100))
            # Windows cannot remove an NPZ while its ZipFile is still open.
            self.assertIsNone(z.zip)
            self.assertIsNone(z.fid)

    def test_global_search_does_not_mutate_fitting_matrices(self):
        a, r, s = self.blocks(); snapshots = [x.copy() for x in (a, r, s)]
        select_global_cores(a, r, s, np.arange(100), [f"w{i}" for i in range(100)], 809, self.policy())
        for before, after in zip(snapshots, [a, r, s]): np.testing.assert_array_equal(before, after)

    def test_asymmetric_evidence_rejected(self):
        a, r, s = self.blocks(); r[0, 1] = .2
        with self.assertRaises(ValueError):
            select_global_cores(a, r, s, np.arange(100), [f"w{i}" for i in range(100)], 809, self.policy())

    def test_generated_candidates_use_only_available_fitting_words(self):
        rng = np.random.default_rng(2); r = rng.uniform(-.8,.8,(15,15)); r=(r+r.T)/2; np.fill_diagonal(r,0)
        a=np.maximum(r,0); pool={}; available=[1,3,5,7,9,11]
        generate_candidates(available,r+.25*a,r,a,3,4,pool)
        self.assertTrue(pool); self.assertTrue(all(set(c).issubset(available) for c in pool))
        for c,(q,g) in pool.items(): self.assertEqual((q,g),pair_means(c,r,a))

    def test_changing_fitting_budget_after_scoring_is_blocked(self):
        c = load_configs(Path(__file__).resolve().parents[1]);c["acquisition"]["pilot"]["global_core_search"]["diversity_draws"] = 999
        with self.assertRaises(ValueError): validate_contract(c)

    def test_turning_off_positive_core_gate_is_blocked(self):
        c = load_configs(Path(__file__).resolve().parents[1]);c["acquisition"]["pilot"]["every_core_positive_signed_cohesion"] = False
        with self.assertRaises(ValueError): validate_contract(c)
