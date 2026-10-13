import itertools
import unittest
import numpy as np
from rank_refinement.acquisition.partition import minimum_size_assignment, spectral_partition
from rank_refinement.acquisition.graph import UnsupportedAcquisition
from rank_refinement.acquisition.cores import select_cores
from rank_refinement.acquisition.alignment import align_cores


class PartitionAndCoreOracles(unittest.TestCase):
    def test_minimum_size_cost_is_global_optimum_by_exhaustive_enumeration(self):
        cost = np.array([[0, 9], [1, 8], [2, 1], [0, 3], [0, 4]], float)
        actual = minimum_size_assignment(cost, 2)
        expected = min(sum(cost[i, z] for i, z in enumerate(labels)) for labels in itertools.product(range(2), repeat=5) if min(labels.count(0), labels.count(1)) >= 2)
        self.assertAlmostEqual(sum(cost[i, z] for i, z in enumerate(actual)), expected)

    def test_three_cluster_quota_cost_matches_exhaustive_search(self):
        cost = np.random.default_rng(18).random((6, 3))
        labels = minimum_size_assignment(cost, 2)
        expected = min(sum(cost[i, z] for i, z in enumerate(row)) for row in itertools.product(range(3), repeat=6) if all(row.count(k) >= 2 for k in range(3)))
        self.assertAlmostEqual(float(cost[np.arange(6), labels].sum()), expected)

    def test_undersized_graph_fails_instead_of_padding(self):
        with self.assertRaises(UnsupportedAcquisition):
            minimum_size_assignment(np.ones((9, 2)), 5)

    def test_nonfinite_assignment_cost_rejected(self):
        with self.assertRaises(ValueError):
            minimum_size_assignment([[float("nan"), 0]], 1)

    def test_spectral_partition_recovers_two_separate_supported_blocks(self):
        a = np.zeros((12, 12))
        a[:6, :6] = 1; a[6:, 6:] = 1; np.fill_diagonal(a, 0)
        groups, d = spectral_partition(a, np.arange(12), 809, topics=2, minimum=3)
        self.assertEqual(groups, [list(range(6)), list(range(6, 12))])
        self.assertTrue(d["assignment_converged"])

    def test_exact_ten_core_sets_have_450_audited_pairs(self):
        a = np.zeros((100, 100)); r = np.full((100, 100), -.5)
        communities = [list(range(10 * k, 10 * (k + 1))) for k in range(10)]
        for c in communities:
            a[np.ix_(c, c)] = 1; r[np.ix_(c, c)] = .8
        np.fill_diagonal(a, 0); np.fill_diagonal(r, 0)
        cores, reports, pairs = select_cores(communities, a, r, (a * 10).astype(int), [f"w{i}" for i in range(100)])
        self.assertEqual(len(pairs), 450)
        self.assertEqual(len(set(i for core in cores for i in core)), 100)
        self.assertTrue(all(abs(q["signed_cohesion"] - .8) < 1e-12 for q in reports))

    def test_each_accepted_swap_improves_declared_objective(self):
        rng = np.random.default_rng(23)
        r = rng.uniform(-.5, .9, (16, 16)); r = (r + r.T) / 2; np.fill_diagonal(r, 0)
        a = np.maximum(r, 0)
        _, reports, _ = select_cores([list(range(16))], a, r, (a > 0).astype(int), [f"w{i}" for i in range(16)])
        self.assertLessEqual(reports[0]["accepted_swaps"], 10)
        self.assertTrue(all(s["after"] > s["before"] for s in reports[0]["swap_history"]))

    def test_affinity_alignment_selects_maximum_row_normalized_score(self):
        cores = [[0, 1], [2, 3]]
        b = [[1, 1, 8, 8], [9, 9, 1, 1]]
        aligned, scores = align_cores(cores, b)
        self.assertEqual(aligned, [cores[1], cores[0]])
        self.assertGreater(scores[0, 1] + scores[1, 0], scores[0, 0] + scores[1, 1])

    def test_negative_affinity_is_not_a_compatible_interface(self):
        with self.assertRaises(ValueError):
            align_cores([[0], [1]], [[1, -1], [0, 1]])
