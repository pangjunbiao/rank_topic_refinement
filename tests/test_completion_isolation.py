import unittest
from rank_refinement.data.completion_masks import make_completion
from rank_refinement.evaluation.likelihood import aggregate_nll,document_loss

POLICY={"seed":20261011,"observed_fraction":.5,"minimum_in_vocabulary_tokens":2}


class CompletionTests(unittest.TestCase):
    def test_partition_covers_each_occurrence_once(self):
        tokens=["扫码","扫码","支付","空调","未知"]
        mask=make_completion("g",tokens,{"扫码":0,"支付":1,"空调":2},POLICY)
        self.assertEqual(set(mask["observed_positions"])|set(mask["target_positions"]),{0,1,2,3})
        self.assertFalse(set(mask["observed_positions"])&set(mask["target_positions"]))
        self.assertEqual(len(mask["observed_ids"])+len(mask["target_ids"]),4)

    def test_replay_identical(self):
        self.assertEqual(make_completion("g",["a","b","a"],{"a":0,"b":1},POLICY),make_completion("g",["a","b","a"],{"a":0,"b":1},POLICY))

    def test_unique_targets_absent_from_encoder_view(self):
        tokens=["unique"+str(i) for i in range(12)]
        mask=make_completion("g",tokens,{t:i for i,t in enumerate(tokens)},POLICY)
        for i in mask["target_positions"]:self.assertNotIn(tokens[i],mask["observed_text"].split())

    def test_two_tokens_leave_both_sides_nonempty(self):
        mask=make_completion("g",["a","b"],{"a":0,"b":1},POLICY)
        self.assertEqual(len(mask["observed_ids"]),1)
        self.assertEqual(len(mask["target_ids"]),1)

    def test_one_token_is_not_fabricated_completion(self):
        self.assertFalse(make_completion("g",["a"],{"a":0},POLICY)["eligible"])

    def test_oov_does_not_shift_into_target_ids(self):
        mask=make_completion("g",["a","unknown","b"],{"a":0,"b":1},POLICY)
        self.assertEqual(set(mask["observed_positions"]+mask["target_positions"]),{0,2})

    def test_zero_target_probability_not_clipped(self):
        self.assertEqual(document_loss([1.,0.],[1])["status"],"infinite_nll_zero_target_probability")

    def test_unnormalized_predictor_rejected(self):
        with self.assertRaises(ValueError):document_loss([.5,.6],[0])

    def test_missing_source_rejected(self):
        with self.assertRaises(ValueError):aggregate_nll([{**document_loss([.5,.5],[0]),"source":"a"}],["a","b"])

    def test_numpy_target_indices_supported(self):
        import numpy as np
        self.assertEqual(document_loss([.5,.5],np.array([0,1],dtype=np.int64))["target_tokens"],2)

    def test_fractional_target_ids_rejected(self):
        with self.assertRaises(ValueError):document_loss([.5,.5],[0.5])

    def test_duplicate_sources_rejected(self):
        with self.assertRaises(ValueError):aggregate_nll([], ["a","a"])
