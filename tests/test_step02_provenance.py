import json
import tempfile
import unittest
from pathlib import Path
from rank_refinement.runtime.artifacts import digest, write_json, write_jsonl
from rank_refinement.stages.step02_acquisition_pilot import fit_and_validation, cached_result
from rank_refinement.runtime.step02_contract import validate_contract
from rank_refinement.runtime.config import load_configs


class PilotProvenanceChecks(unittest.TestCase):
    def fixture(self, folder):
        rows = [{"group_id": "g1", "split": "fit", "tokens": ["a"]},
                {"group_id": "g2", "split": "validation", "tokens": ["b"]}]
        write_jsonl(folder / "feature_records.jsonl", rows)
        write_json(folder / "vocabulary.json", {"fit_group_ids_sha256": digest(["g1"])})
        return rows

    def test_outer_heldout_group_cannot_enter_internal_pilot(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d); self.fixture(folder)
            with self.assertRaises(AssertionError):
                fit_and_validation(folder, {"g1", "g3"})

    def test_fold_vocabulary_fitted_on_validation_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d); self.fixture(folder)
            write_json(folder / "vocabulary.json", {"fit_group_ids_sha256": digest(["g1", "g2"])})
            with self.assertRaises(AssertionError):
                fit_and_validation(folder, {"g1", "g2"})

    def test_duplicate_group_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d); rows = self.fixture(folder)
            write_jsonl(folder / "feature_records.jsonl", rows + [rows[0]])
            with self.assertRaises(AssertionError):
                fit_and_validation(folder, {"g1", "g2"})

    def test_valid_fold_keeps_fitting_and_validation_roles(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d); self.fixture(folder)
            fit, val, _ = fit_and_validation(folder, {"g1", "g2"})
            self.assertEqual([r["group_id"] for r in fit], ["g1"])
            self.assertEqual([r["group_id"] for r in val], ["g2"])

    def test_cached_complete_failure_is_reused_without_retrying_fit(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            def producer(folder):
                calls.append(1)
                return {"structural_passed": False, "failure": "no supported core"}
            a, reused = cached_result(Path(d), producer)
            b, reused2 = cached_result(Path(d), producer)
            self.assertEqual(a, b); self.assertFalse(reused); self.assertTrue(reused2); self.assertEqual(len(calls), 1)

    def test_tampered_cached_result_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            cached_result(folder, lambda p: {"scored": False})
            write_json(folder / "result.json", {"scored": True})
            with self.assertRaises(ValueError):
                cached_result(folder, lambda p: {"scored": False})

    def test_existing_contract_passes_without_modifying_data_rules(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue(validate_contract(load_configs(root)))

    def test_lowering_support_after_failure_is_blocked(self):
        root = Path(__file__).resolve().parents[1]
        config = load_configs(root); config["acquisition"]["pilot"]["word_group_support"] = [5, 10]
        with self.assertRaises(ValueError):
            validate_contract(config)
