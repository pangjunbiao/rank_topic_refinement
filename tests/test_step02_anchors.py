import importlib.util
import unittest
import numpy as np
from scipy import sparse
from rank_refinement.models.adapters.pilot_anchors import topics_from_weights, fit_anchor, UnusableAnchor
from rank_refinement.runtime.config import load_configs
from pathlib import Path


class NativeAnchorChecks(unittest.TestCase):
    def setUp(self):
        self.words = [f"term{i:03d}" for i in range(100)]
        rows = []
        rng = np.random.default_rng(809)
        for k in range(10):
            for _ in range(12):
                row = np.zeros(100); row[k * 10:(k + 1) * 10] = rng.integers(1, 4, 10); rows.append(row)
        self.x = sparse.csr_matrix(rows)
        self.profiles = load_configs(Path(__file__).resolve().parents[1])["acquisition_anchors"]["profiles"]

    def test_native_ranking_uses_vocabulary_index_for_weight_ties(self):
        weights = np.ones((10, 100))
        topics = topics_from_weights(weights, self.words)
        self.assertTrue(all(t == self.words[:10] for t in topics))

    def test_zero_descriptor_weight_padding_is_rejected(self):
        with self.assertRaises(UnusableAnchor):
            topics_from_weights(np.zeros((10, 100)), self.words)

    def test_lda_returns_complete_native_topics_on_known_blocks(self):
        topics, weights, details = fit_anchor("lda", self.x, self.words, 809, self.profiles["lda"][0])
        self.assertEqual(weights.shape, (10, 100)); self.assertTrue(all(len(t) == 10 for t in topics))
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-6)

    def test_nmf_recovers_disjoint_observed_word_blocks(self):
        topics, weights, details = fit_anchor("nmf", self.x, self.words, 809, self.profiles["nmf"][0])
        recovered = {frozenset(t) for t in topics}
        expected = {frozenset(self.words[k * 10:(k + 1) * 10]) for k in range(10)}
        self.assertEqual(recovered, expected)

    def test_gsdmm_pinned_source_recovers_populated_known_blocks(self):
        topics, weights, details = fit_anchor("gsdmm", self.x, self.words, 809, self.profiles["gsdmm"][0])
        self.assertEqual(details["occupied_clusters"], 10)
        self.assertTrue(all(len(t) == 10 for t in topics))
        self.assertTrue(np.isfinite(weights).all())

    def test_gsdmm_does_not_change_callers_numpy_random_stream(self):
        np.random.seed(555); state = np.random.get_state()
        fit_anchor("gsdmm", self.x, self.words, 809, self.profiles["gsdmm"][0])
        actual = np.random.random(10); np.random.set_state(state); expected = np.random.random(10)
        np.testing.assert_array_equal(actual, expected)

    @unittest.skipUnless(importlib.util.find_spec("bertopic") and importlib.util.find_spec("torch"), "Conditional BERTopic dependencies not installed; no anchor panel may run until installed.")
    def test_real_bertopic_package_preserves_ten_observed_word_blocks(self):
        rng = np.random.default_rng(31)
        vectors = np.zeros((120, 12), dtype=np.float32)
        for k in range(10):
            vectors[k * 12:(k + 1) * 12, k] = 10
        vectors += rng.normal(0, .01, vectors.shape).astype(np.float32)
        profiles = [p for p in self.profiles["bertopic"] if p["cluster"] == "kmeans"]
        topics, weights, details = fit_anchor("bertopic", self.x, self.words, 809, profiles[0], embeddings=vectors)
        self.assertEqual(weights.shape, (10, 100))
        self.assertEqual({frozenset(t) for t in topics}, {frozenset(self.words[k * 10:(k + 1) * 10]) for k in range(10)})

    @unittest.skipUnless(importlib.util.find_spec("bertopic") and importlib.util.find_spec("torch"), "Conditional BERTopic dependencies not installed.")
    def test_real_bertopic_preserves_chinese_words_and_phrase_underscores(self):
        words = [chr(0x4e00+i) + "_公交" for i in range(100)]
        rng = np.random.default_rng(31)
        vectors = np.zeros((120,12),dtype=np.float32)
        for k in range(10): vectors[k*12:(k+1)*12,k] = 10
        vectors += rng.normal(0,.01,vectors.shape).astype(np.float32)
        profile = [p for p in self.profiles["bertopic"] if p["cluster"] == "kmeans"][0]
        topics, weights, details = fit_anchor("bertopic",self.x,words,809,profile,embeddings=vectors)
        self.assertTrue(np.isfinite(weights).all())
        self.assertEqual({frozenset(t) for t in topics},{frozenset(words[k*10:(k+1)*10]) for k in range(10)})
        self.assertEqual(details["lexical_preprocessing"],"multilingual_preserves_chinese_and_phrase_underscores")
