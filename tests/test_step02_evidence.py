from __future__ import annotations
import itertools
import unittest
import numpy as np
from gensim.corpora import Dictionary
from gensim.topic_coherence.text_analysis import WordOccurrenceAccumulator
from gensim.topic_coherence.direct_confirmation_measure import log_ratio_measure
from rank_refinement.acquisition.windows import count_group_evidence, virtual_windows
from rank_refinement.acquisition.associations import signed_npmi, association_bundle
from rank_refinement.acquisition.subsampling import group_subsamples
from rank_refinement.acquisition.graph import build_graph


class EvidenceOracles(unittest.TestCase):
    def setUp(self):
        self.words = ["a", "b", "c", "d"]
        self.texts = [["a", "b", "a", "unknown", "c", "a", "b", "d"], ["c", "d"], [], ["a", "b", "a", "b"]]
        self.e = count_group_evidence(self.texts, self.words, 3)

    def test_window_counts_match_independent_whole_corpus_gensim_call(self):
        dictionary = Dictionary([self.words])
        acc = WordOccurrenceAccumulator(set(dictionary.token2id.values()), dictionary).accumulate(self.texts, 3)
        for i, word in enumerate(self.words):
            self.assertEqual(int(self.e["word_windows"][:, i].sum()), int(acc[dictionary.token2id[word]]))
        totals = self.e["pair_windows"].sum(axis=0).A1
        for (i, j), value in zip(self.e["pairs"], totals):
            self.assertEqual(int(value), int(acc[dictionary.token2id[self.words[i]], dictionary.token2id[self.words[j]]]))
        self.assertEqual(int(self.e["windows_per_group"].sum()), acc.num_docs)

    def test_true_pair_support_counts_distinct_groups_not_windows(self):
        pair = np.flatnonzero((self.e["pairs"] == [0, 1]).all(axis=1))[0]
        self.assertEqual(int(self.e["pair_groups"][:, pair].sum()), 2)
        self.assertGreater(int(self.e["pair_windows"][:, pair].sum()), 2)

    def test_group_word_support_not_repeated_token_counts(self):
        self.assertEqual(int(self.e["word_groups"][:, 0].sum()), 2)

    def test_oov_positions_do_not_compress_local_support(self):
        e = count_group_evidence([["a", "outside", "outside", "b"]], ["a", "b"], 3)
        self.assertEqual(int(e["pair_groups"].sum()), 0)

    def test_document_boundaries_do_not_create_pairs(self):
        e = count_group_evidence([["a"], ["b"]], ["a", "b"], 10)
        self.assertEqual(int(e["pair_windows"].sum()), 0)
        self.assertEqual(int(e["pair_groups"].sum()), 0)

    def test_short_and_empty_documents_match_library_denominator(self):
        self.assertEqual(self.e["windows_per_group"].tolist(), [6, 1, 1, 2])

    def test_virtual_windows_keep_full_lexical_positions(self):
        self.assertEqual(list(virtual_windows([1, 2, 3, 4], 3)), [[1, 2, 3], [2, 3, 4]])

    def test_all_six_pairs_retained_including_never_observed(self):
        self.assertEqual(self.e["pairs"].tolist(), list(map(list, itertools.combinations(range(4), 2))))

    def test_signed_npmi_matches_standard_direct_confirmation_measure(self):
        texts = [["a", "b"], ["a", "b"], ["c", "d"], ["c", "d"]]
        dictionary = Dictionary([self.words])
        acc = WordOccurrenceAccumulator(set(dictionary.token2id.values()), dictionary).accumulate(texts, 10)
        pairs = np.array(list(itertools.combinations(range(4), 2)))
        word_counts = [acc[dictionary.token2id[w]] for w in self.words]
        pair_counts = [acc[dictionary.token2id[self.words[i]], dictionary.token2id[self.words[j]]] for i, j in pairs]
        expected = [log_ratio_measure([[(dictionary.token2id[self.words[i]], dictionary.token2id[self.words[j]])]], acc, normalize=True)[0] for i, j in pairs]
        actual = signed_npmi(word_counts, pair_counts, pairs, acc.num_docs)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_absent_marginal_is_declared_pessimistic_not_division_by_zero(self):
        self.assertEqual(signed_npmi([0, 1], [0], [[0, 1]], 3).tolist(), [-1])

    def test_ubiquitous_joint_is_declared_uninformative(self):
        self.assertEqual(signed_npmi([3, 3], [3], [[0, 1]], 3).tolist(), [0])

    def test_subsamples_are_reproducible_unique_groups(self):
        ids = [f"g{i:03d}" for i in range(30)]
        a = group_subsamples(ids, 809)
        self.assertEqual(a, group_subsamples(ids, 809))
        self.assertNotEqual(a, group_subsamples(ids, 810))
        self.assertEqual(len(a[0]), 20)
        self.assertTrue(all(len(set(row)) == 24 for row in a[1]))

    def test_duplicate_evidence_units_rejected(self):
        with self.assertRaises(ValueError):
            group_subsamples(["x", "x"], 809)

    def test_unsorted_evidence_rows_rejected_before_subsampling(self):
        with self.assertRaises(ValueError):
            association_bundle(self.e, ["z", "b", "c", "d"], 809, {"subsamples": 20, "group_fraction": .8, "empirical_quantile": .1})

    def test_pair_threshold_filters_graph_without_erasing_signed_penalty(self):
        bundle = {"words": ["a", "b", "c"], "pairs": np.array([[0, 1], [0, 2], [1, 2]]),
                  "stable": np.array([.8, -.5, .2]), "full": np.array([.9, -.3, .3]),
                  "word_support": np.array([20, 20, 20]), "pair_support": np.array([5, 2, 2]), "groups": 100}
        a, r, support, active, d = build_graph(bundle, 10, 3, 0)
        self.assertEqual(a[0, 1], 1)
        self.assertEqual(a[0, 2], 0)
        self.assertEqual(r[0, 2], -.5)
        self.assertEqual(r[1, 2], .2)
        self.assertEqual(active.tolist(), [0, 1])

    def test_quantile_is_empirical_linear_tenth_percentile(self):
        p = {"subsamples": 20, "group_fraction": .8, "empirical_quantile": .1}
        b = association_bundle(self.e, ["a", "b", "c", "d"], 809, p)
        ordered = np.sort(b["subsample_scores"], axis=0)
        expected = .1 * ordered[1] + .9 * ordered[2]  # (20-1)*.1 = 1.9
        np.testing.assert_allclose(b["stable"], expected, atol=1e-14)
