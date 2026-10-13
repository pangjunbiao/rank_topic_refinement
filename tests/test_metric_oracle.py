import math
import tempfile
import unittest
import warnings
from rank_refinement.evaluation.coherence import score_topics,run_metric_oracles,UnscorableCoherence
from rank_refinement.evaluation.diversity import topic_diversity
from rank_refinement.evaluation.likelihood import document_loss,aggregate_nll


class MetricTests(unittest.TestCase):
    def test_direct_library_known_event_and_cache_oracles(self):
        with tempfile.TemporaryDirectory() as d:self.assertTrue(run_metric_oracles(d)["passed"])

    def test_no_descriptor_padding(self):
        with self.assertRaises(ValueError):topic_diversity([["a"]],expected_words=10)

    def test_duplicate_within_topic_rejected(self):
        with self.assertRaises(ValueError):topic_diversity([["a","a"]],expected_words=2)

    def test_diversity_structural_ties(self):
        topics=[["a","b"],["c","d"]]
        self.assertEqual(topic_diversity(topics,2),topic_diversity(list(reversed(topics)),2))

    def test_out_of_vocabulary_descriptor_rejected(self):
        with self.assertRaises(ValueError):score_topics([["a","unknown"]],[["a","b"]],["a","b"],topn=2)

    def test_empty_reference_not_scored_as_zero(self):
        with self.assertRaises(UnscorableCoherence):score_topics([["a","b"]],[[]],["a","b"],topn=2)

    def test_target_occurrences_not_unique_types(self):
        loss=document_loss([.25,.75],[0,0,1])
        self.assertAlmostEqual(loss["negative_log_probability"],-2*math.log(.25)-math.log(.75),places=12)
        self.assertEqual(loss["target_tokens"],3)

    def test_macro_and_micro_are_distinct(self):
        rows=[{**document_loss([.8,.2],[0]*100),"source":"a"},{**document_loss([.8,.2],[1]),"source":"b"}]
        result=aggregate_nll(rows,["a","b"])
        self.assertAlmostEqual(result["macro_source_nll"],(-math.log(.8)-math.log(.2))/2,places=12)
        self.assertAlmostEqual(result["micro_nll"],(-100*math.log(.8)-math.log(.2))/101,places=12)
