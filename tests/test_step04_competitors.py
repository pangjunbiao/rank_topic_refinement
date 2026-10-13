"""Independent native-interface, leakage, checkpoint and reporting checks.

Tiny synthetic contexts and counts only; no encoder download or primary model
training in --self-check. Live author interfaces are mandatory in Step 4.
"""
from __future__ import annotations
import copy
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.special import logsumexp
from rank_refinement.runtime.config import load_configs
from rank_refinement.runtime.artifacts import digest, read_json
from rank_refinement.models.competitor_classical import native_topics, stable_gsdmm_score, gsdmm_predict, lexical_docs
from rank_refinement.evaluation.competitor_evidence import summarize, paired_groups, bootstrap_gain, availability_gates, audit_native_reference
from rank_refinement.stages.step04_primary_benchmark import validate_protocol, compare_profiles, selected_profiles, failed_case

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = all(importlib.util.find_spec(n) for n in ["torch", "bertopic", "umap", "hdbscan"])
if os.environ.get("RANK_STEP04_REQUIRED_CHECKS") and not AVAILABLE:
    raise ImportError("Step 4 native interface checks cannot be skipped. Install requirements-step04.txt.")


class QuietLog:
    def __init__(self):
        import logging
        self.logger = logging.getLogger("step04-test")
        self.logger.propagate = False
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())
    def info(self, *args):
        pass


def fixture():
    rng = np.random.default_rng(809)
    vocabulary = [chr(0x4e00 + i) + "_公交" for i in range(100)]
    fit = np.zeros((120, 100), dtype=np.float32)
    vectors = np.zeros((120, 12), dtype=np.float32)
    for k in range(10):
        fit[k * 12:(k + 1) * 12, k * 10:(k + 1) * 10] = rng.integers(1, 4, (12, 10))
        vectors[k * 12:(k + 1) * 12, k] = 10
    vectors += rng.normal(0, .01, vectors.shape).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    observed = fit[::6].copy()
    for row in observed:
        row[np.flatnonzero(row)[5:]] = 0
    context = vectors[::6].copy()
    words = rng.normal(size=(100, 12)).astype(np.float32)
    words /= np.linalg.norm(words, axis=1, keepdims=True)
    fit_ids = ["fit" + str(i) for i in range(len(fit))]
    dev_ids = ["development" + str(i) for i in range(len(observed))]
    fit_records = [{"group_id": g, "source": "source" + str(i % 3), "split": "fit",
                    "tokens": lexical_docs(fit[i:i + 1], vocabulary)[0].split()} for i, g in enumerate(fit_ids)]
    dev_records = [{"group_id": g, "source": "source" + str(i % 3), "split": "development"} for i, g in enumerate(dev_ids)]
    dev_coherence = [{**r, "tokens": lexical_docs(fit[::6][i:i + 1], vocabulary)[0].split()} for i, r in enumerate(dev_records)]
    masks = {g: {"eligible": True, "target_ids": np.flatnonzero(fit[::6][i] - observed[i]).tolist()} for i, g in enumerate(dev_ids)}
    return {"fit": fit, "dev_observed": observed, "fit_ids": fit_ids, "dev_ids": dev_ids,
            "fit_records": fit_records, "dev_records": dev_records, "dev_coherence_records": dev_coherence,
            "masks": masks, "vocabulary": vocabulary}, {"fit_full": vectors, "development_observed": context, "words": words}


def fastopic_theta_oracle(observed, fitting, topics, temperature):
    """Independent float64 oracle for the author's fixed-reference inference.

    Direct squared differences avoid sharing the Torch matrix-multiplication
    code under test. The fitting column normalizer stays fixed for every query.
    """
    observed, fitting, topics = [np.asarray(a, dtype=np.float64)
                                for a in (observed, fitting, topics)]
    query_log = -np.square(observed[:, None, :] - topics[None, :, :]).sum(2) / temperature
    fitting_log = -np.square(fitting[:, None, :] - topics[None, :, :]).sum(2) / temperature
    logits = query_log - logsumexp(fitting_log, axis=0, keepdims=True)
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


class ProtocolAndEvidence(unittest.TestCase):
    def test_frozen_twenty_case_policy_validates(self):
        policy = validate_protocol(load_configs(ROOT), ROOT)
        self.assertEqual(sum(len(p) for p in policy["profiles"].values()) * len(policy["seeds"]), 20)
        self.assertFalse(policy["test_evaluation"] or policy["external_evaluation"] or policy["retrain_proposed"])

    def test_policy_change_requires_explicit_revision(self):
        configs = load_configs(ROOT)
        configs["step04_competitors"]["profiles"]["combinedtm"][0]["epochs"] = 1
        with self.assertRaises(ValueError):
            validate_protocol(configs)

    def test_frozen_source_keeps_proposed_and_every_baseline_metric(self):
        from rank_refinement.stages.step04_primary_benchmark import proposed_source
        state = ROOT / "state/step01_accepted.json"
        if not state.is_file() or not (ROOT / "state/step03_improvement_result.json").is_file():
            self.skipTest("Existing accepted Step 3 results are preserved by installation; not included as synthetic fixtures.")
        _, completed, proposed, baselines, _ = proposed_source(ROOT, read_json(state)["preparation_id"])
        historical = read_json(completed / "case_results.json")
        original = [r for c in historical["neural"] for r in c["rows"] if r["role"] == "native"] + [c["row"] for c in historical["classical"]]
        self.assertEqual(len(baselines), 16)
        self.assertTrue(all(r["method"] == "Proposed" and r["projection_deployed"] for r in proposed))
        for a, b in zip(baselines, original):
            for field in ["fitting", "development", "prediction", "document_losses", "topics", "topic_diversity"]:
                self.assertEqual(a[field], b[field])
            self.assertEqual(a["source_method_name"], b["method"])

    def test_empty_topics_are_not_prior_padded(self):
        matrix = np.zeros((10, 100)); matrix[:3, :10] = 1
        topics, info = native_topics(matrix, [str(i) for i in range(100)])
        self.assertIsNone(topics)
        self.assertEqual(info["topics_without_ten_supported_words"], list(range(3, 10)))

    def test_occupied_native_topics_keep_actual_k(self):
        matrix = np.zeros((10, 100)); matrix[:3, :10] = 1
        topics, info = native_topics(matrix, [str(i) for i in range(100)], [0, 1, 2])
        self.assertEqual(len(topics), 3)
        self.assertFalse(info["fixed_k10"])

    def test_ties_use_canonical_vocabulary_order(self):
        topics, _ = native_topics(np.ones((10, 100)), [str(i) for i in range(100)])
        self.assertEqual(topics[0], [str(i) for i in range(10)])

    def test_lexical_document_conversion_preserves_chinese_and_phrases(self):
        self.assertEqual(lexical_docs(np.array([[2, 1]]), ["地铁_服务", "公交"]), ["地铁_服务 地铁_服务 公交"])

    def test_formula3_log_repair_matches_pinned_finite_probabilities(self):
        from rank_refinement.vendor.gsdmm_mgp import MovieGroupProcess
        model = MovieGroupProcess.from_data(2, .1, .1, 10, 5, [4, 6], [7, 8], [{"a": 4, "b": 3}, {"b": 5, "c": 3}])
        np.testing.assert_allclose(stable_gsdmm_score(model, ["a", "b"]), model.score(["a", "b"]), rtol=1e-12)

    def test_formula3_long_document_underflow_is_stable(self):
        from rank_refinement.vendor.gsdmm_mgp import MovieGroupProcess
        model = MovieGroupProcess.from_data(2, .1, .1, 10, 2000, [4, 6], [7, 8], [{"a": 7}, {"b": 8}])
        probability = np.array(stable_gsdmm_score(model, [str(i) for i in range(1000)]))
        self.assertTrue(np.isfinite(probability).all())
        self.assertAlmostEqual(probability.sum(), 1)

    def test_gsdmm_conditional_predictive_matches_manual_posterior(self):
        from rank_refinement.vendor.gsdmm_mgp import MovieGroupProcess
        model = MovieGroupProcess.from_data(2, .1, .2, 10, 3, [4, 6], [7, 8], [{"a": 4, "b": 3}, {"b": 5, "c": 3}])
        _, actual, theta = gsdmm_predict(model, np.array([[2, 0, 0]]), ["a", "b", "c"])
        p = np.array([[5.2, 3.2, .2], [1.2, 5.2, 3.2]])
        p /= p.sum(1, keepdims=True)
        np.testing.assert_allclose(actual, theta @ p, atol=1e-14)

    def test_complete_metric_slots_reproduce_standard_gensim(self):
        from rank_refinement.evaluation.reference_audit import audit_reference
        data, _ = fixture()
        topics = [data["vocabulary"][k * 10:(k + 1) * 10] for k in range(10)]
        a = audit_native_reference(topics, data["fit_records"], data["vocabulary"])
        b = audit_reference(topics, data["fit_records"], data["vocabulary"])
        for m in ["c_v", "c_npmi"]:
            self.assertAlmostEqual(a["metrics"][m]["mean"], b["metrics"][m]["mean"], places=14)

    def test_failed_case_stays_unavailable(self):
        row = failed_case("glocom", {"name": "bad"}, 809, {}, "test failure")["row"]
        self.assertEqual(row["status"], "failed")
        self.assertIsNone(row["prediction"]["macro_source_nll"])
        self.assertIsNone(row["topic_diversity"])

    def test_incomplete_seed_grid_does_not_report_single_seed_mean(self):
        row = failed_case("glocom", {"name": "bad"}, 809, {}, "test failure")["row"]
        row["prediction"]["macro_source_nll"] = 5.
        entry = summarize([row], [809, 810])[0]
        self.assertFalse(entry["complete_seed_grid"])
        self.assertIsNone(entry["development_macro_nll_mean"])

    def paired(self):
        proposed, competitors = [], []
        for seed in [809, 810]:
            losses = [{"group_id": str(i), "source": "s" + str(i % 2), "target_tokens": 1 + i % 3,
                       "negative_log_probability": 2. * (1 + i % 3)} for i in range(8)]
            larger = [{**r, "negative_log_probability": r["negative_log_probability"] + .5 * r["target_tokens"]} for r in losses]
            proposed.append({"seed": seed, "document_losses": losses})
            competitors.append({"seed": seed, "document_losses": larger})
        return proposed, competitors

    def test_bootstrap_constant_gain_is_exact(self):
        a, b = self.paired()
        result = bootstrap_gain(paired_groups(a, b, [809, 810]), [809, 810], 200)
        for n in ["mean_competitor_minus_proposed_macro_nll", "percentile_95_low", "percentile_95_high"]:
            self.assertAlmostEqual(result[n], .5, places=14)

    def test_paired_groups_reject_missing_or_changed_targets(self):
        a, b = self.paired(); b[0]["document_losses"][0]["target_tokens"] += 1
        with self.assertRaises(ValueError): paired_groups(a, b, [809, 810])

    def test_negative_gain_is_retained_as_a_loss(self):
        row = {"method": "Proposed", "profile": "p", "complete_seed_grid": True, "fixed_k10_all_seeds": True,
               "fitting_c_v_mean": .5, "fitting_c_npmi_mean": .1, "topic_diversity_mean": 1., "development_macro_nll_mean": 6.}
        better = {**row, "method": "Other", "profile": "o", "development_macro_nll_mean": 5.}
        result = compare_profiles([row, better])[0]
        self.assertEqual(result["development_macro_nll_proposed_gain"], -1.)
        self.assertFalse(result["development_macro_nll_win_or_tie"])


@unittest.skipUnless(AVAILABLE, "Step 4 neural/BERTopic libraries are optional for earlier stages")
class LiveNativeInterfaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        from rank_refinement.models.competitor_sources import resolve_sources
        cls.folders, cls.specs = resolve_sources(ROOT)
        cls.policy = load_configs(ROOT)["step04_competitors"]

    def profiles(self, method):
        p = copy.deepcopy(self.policy["profiles"][method][0])
        if method not in ["gsdmm", "bertopic"]:
            p["epochs"] = 3
        if method == "glocom":
            p.update(clusters=4, en_units=16)
        if method == "combinedtm":
            p["hidden_sizes"] = [16, 16]
        return p

    def test_three_real_author_objectives_and_native_predictions(self):
        import torch
        from rank_refinement.models.competitor_neural import train_model, decode_model, cluster_contexts
        data, vectors = fixture()
        for method in ["combinedtm", "fastopic", "glocom"]:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                p = self.profiles(method)
                contexts = cluster_contexts(data["fit"], vectors["fit_full"], data["dev_observed"], vectors["development_observed"], p, 809) if method == "glocom" else None
                model, detail = train_model(Path(tmp), method, p, 809, self.folders[method], data["fit"], vectors,
                    data["fit_ids"], self.specs, "cpu", 1, QuietLog(), contexts)
                weights, probabilities, theta, _ = decode_model(method, model, data["dev_observed"], vectors, contexts, 809)
                self.assertEqual(weights.shape, (10, 100)); self.assertEqual(theta.shape, (20, 10))
                np.testing.assert_allclose(probabilities.sum(1), 1., atol=1e-14)
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertEqual(len(detail["history"]), 3)
                self.assertTrue(all(r["fit_documents_seen"] == 120 for r in detail["history"]))

    def test_ctm_sampling_matches_actual_author_forward(self):
        import torch
        from rank_refinement.models.competitor_neural import make_model, decode_ctm
        data, vectors = fixture()
        model = make_model("combinedtm", self.folders["combinedtm"], 100, 12, self.profiles("combinedtm"), seed=809).cpu().eval()
        _, probabilities, _ = decode_ctm(model, data["dev_observed"], vectors["development_observed"], 809, 20)
        torch.manual_seed(809 + 470004)
        with torch.no_grad():
            native = np.mean([model(torch.from_numpy(data["dev_observed"]), torch.from_numpy(vectors["development_observed"]))[5].numpy().astype(np.float64) for _ in range(20)], axis=0)
        native /= native.sum(1, keepdims=True)
        np.testing.assert_allclose(probabilities, native, atol=2e-15)

    def test_glocom_probability_is_native_bn_decode_not_row_normalized_affinity(self):
        import torch
        from rank_refinement.models.competitor_neural import make_model, decode_model, cluster_contexts
        data, vectors = fixture(); p = self.profiles("glocom")
        context = cluster_contexts(data["fit"], vectors["fit_full"], data["dev_observed"], vectors["development_observed"], p, 809)
        model = make_model("glocom", self.folders["glocom"], 100, 12, p, words=vectors["words"]).cpu().eval()
        weights, actual, _, _ = decode_model("glocom", model, data["dev_observed"], vectors, context, 809)
        with torch.no_grad():
            native = torch.softmax(model.decoder_bn(model.get_theta(torch.from_numpy(context["observed"])) @ model.get_beta()), dim=1).numpy().astype(np.float64)
        native /= native.sum(1, keepdims=True)
        np.testing.assert_allclose(actual, native, atol=1e-14)
        np.testing.assert_allclose(weights.sum(0), 1., atol=2e-7)
        self.assertFalse(np.allclose(weights.sum(1), 1))

    def test_fastopic_theta_uses_fixed_fitting_reference(self):
        import torch
        from rank_refinement.models.competitor_neural import make_model
        data, vectors = fixture()
        model = make_model("fastopic", self.folders["fastopic"], 100, 12, self.profiles("fastopic")).cpu().eval()
        with torch.no_grad():
            for dtype, tolerance in [(torch.float32, 2 * np.finfo(np.float32).eps),
                                     (torch.float64, 2e-13)]:
                with self.subTest(dtype=dtype):
                    model.to(dtype=dtype)
                    observed = torch.from_numpy(vectors["development_observed"]).to(dtype=dtype)
                    fit = torch.from_numpy(vectors["fit_full"]).to(dtype=dtype)
                    together = model.get_theta(observed, fit).numpy()
                    expected = fastopic_theta_oracle(observed.numpy(), fit.numpy(),
                        model.topic_embeddings.detach().numpy(), model.theta_temp)
                    # Float32 matrix products can differ by a few ULPs across
                    # batch sizes/backends. Use explicit dtype-aware tolerances;
                    # float64 and the independent oracle still test the formula.
                    np.testing.assert_allclose(together, expected, rtol=tolerance, atol=tolerance)
                    np.testing.assert_allclose(together.sum(1), 1., rtol=0, atol=tolerance)
                    for chunk in [1, 3, 7]:
                        separate = np.concatenate([model.get_theta(observed[i:i+chunk], fit).numpy()
                                                   for i in range(0, len(observed), chunk)])
                        np.testing.assert_allclose(together, separate, rtol=tolerance, atol=tolerance)
                    reverse = model.get_theta(observed.flip(0), fit).numpy()[::-1]
                    np.testing.assert_allclose(together, reverse, rtol=tolerance, atol=tolerance)

    def test_fastopic_decoder_rejects_query_batch_as_fitting_reference(self):
        import torch
        from rank_refinement.models.competitor_neural import make_model, decode_model
        data, vectors = fixture()
        model = make_model("fastopic", self.folders["fastopic"], 100, 12, self.profiles("fastopic")).cpu().eval()
        _, _, actual, _ = decode_model("fastopic", model, data["dev_observed"], vectors, None, 809)
        expected = fastopic_theta_oracle(vectors["development_observed"], vectors["fit_full"],
            model.topic_embeddings.detach().numpy(), model.theta_temp)
        tolerance = 2 * np.finfo(np.float32).eps
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        # A genuinely batch-dependent denominator must fail the same assertion,
        # even when it still returns finite, normalized probability vectors.
        with torch.no_grad():
            observed = torch.from_numpy(vectors["development_observed"])
            wrong = model.get_theta(observed, observed[:1]).numpy()
        self.assertGreater(float(np.max(np.abs(wrong - expected))), .01)
        np.testing.assert_allclose(wrong.sum(1), 1., rtol=0, atol=tolerance)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(wrong, expected, rtol=tolerance, atol=tolerance)

    def test_glo_context_bank_contains_only_fit_counts(self):
        from rank_refinement.models.competitor_neural import cluster_contexts
        data, vectors = fixture(); p = self.profiles("glocom")
        before = cluster_contexts(data["fit"], vectors["fit_full"], data["dev_observed"], vectors["development_observed"], p, 809)
        after = cluster_contexts(data["fit"], vectors["fit_full"], data["dev_observed"] * 10000, vectors["development_observed"], p, 809)
        np.testing.assert_array_equal(before["bank"], after["bank"])
        np.testing.assert_array_equal(before["fit"], after["fit"])
        np.testing.assert_allclose(before["bank"].sum(0), data["fit"].sum(0), atol=0)

    def test_interrupted_training_resume_reproduces_uninterrupted_model(self):
        from rank_refinement.models.competitor_neural import train_model
        data, vectors = fixture(); p = self.profiles("combinedtm")
        class StopAfterCheckpoint(QuietLog):
            def info(self, message, *args):
                if message.startswith("COMPETITOR EPOCH"):
                    raise KeyboardInterrupt("synthetic interruption after verified checkpoint")
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            args = ("combinedtm", p, 809, self.folders["combinedtm"], data["fit"], vectors, data["fit_ids"], self.specs, "cpu", 1)
            uninterrupted, _ = train_model(Path(first), *args, QuietLog())
            with self.assertRaises(KeyboardInterrupt): train_model(Path(second), *args, StopAfterCheckpoint())
            resumed, details = train_model(Path(second), *args, QuietLog())
            for name, value in uninterrupted.state_dict().items():
                np.testing.assert_array_equal(value.numpy(), resumed.state_dict()[name].numpy())
            self.assertEqual(len(details["history"]), 3)

    def test_real_bertopic_preserves_chinese_and_has_no_invented_likelihood(self):
        from rank_refinement.models.competitor_classical import fit_bertopic
        data, vectors = fixture()
        topics, weights, probability, theta, details = fit_bertopic(data["fit"], data["dev_observed"], data["vocabulary"], 809,
            self.profiles("bertopic"), vectors, QuietLog())
        self.assertEqual(len(topics), 10); self.assertEqual(weights.shape, (10, 100))
        self.assertIsNone(probability); self.assertIsNone(theta)
        self.assertTrue(all("_公交" in w for topic in topics for w in topic))
        self.assertEqual(len(details["observed_development_labels"]), 20)

    def test_real_gsdmm_fit_does_not_change_callers_numpy_stream(self):
        from rank_refinement.models.competitor_classical import fit_gsdmm
        data, _ = fixture(); np.random.seed(311); state = np.random.get_state()
        _, _, p, _, details = fit_gsdmm(data["fit"], data["dev_observed"], data["vocabulary"], 809, self.profiles("gsdmm"), QuietLog())
        actual = np.random.random(5); np.random.set_state(state); expected = np.random.random(5)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(p.sum(1), 1., atol=1e-14)
        self.assertEqual(details["native_topic_count"], 10)

    def test_case_export_cache_and_native_result_are_complete(self):
        from rank_refinement.stages.step04_primary_benchmark import competitor_case
        from rank_refinement.stages.step02_completion import check_indexed_tree
        data, vectors = fixture(); p = self.profiles("combinedtm")
        prior = data["fit"].astype(np.float64).sum(0) + .5; prior /= prior.sum()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root / "cases/combinedtm/test/seed809"
            args = (root, "combinedtm", p, 809, data, vectors, self.folders, self.specs, self.policy,
                    "cpu", prior, folder, QuietLog(), {"synthetic_test": True})
            result = competitor_case(*args)
            self.assertEqual(result["row"]["status"], "scored")
            self.assertEqual(len(result["row"]["document_losses"]), 20)
            check_indexed_tree(folder)
            cached = competitor_case(*args)
            self.assertEqual(result, cached)
            with np.load(folder / "native_readout_and_predictions.npz", allow_pickle=False) as z:
                self.assertEqual(z["native_descriptor_weights"].shape, (10, 100))
                self.assertEqual(z["native_decoded_probabilities"].shape, (20, 100))

    def test_case_failure_is_saved_and_cannot_be_ranked_as_a_win(self):
        from rank_refinement.stages.step04_primary_benchmark import competitor_case
        data, vectors = fixture(); p = self.profiles("combinedtm")
        prior = np.full(100, .01)
        with tempfile.TemporaryDirectory() as tmp, patch("rank_refinement.stages.step04_primary_benchmark.train_model", side_effect=RuntimeError("synthetic deliberate failure")):
            root = Path(tmp); folder = root / "case"
            result = competitor_case(root, "combinedtm", p, 809, data, vectors, self.folders, self.specs,
                self.policy, "cpu", prior, folder, QuietLog(), {"synthetic_test": True})
            self.assertEqual(result["row"]["status"], "failed")
            self.assertIsNone(result["row"]["prediction"]["macro_source_nll"])
            self.assertTrue((folder / "failure_traceback.txt").is_file())
