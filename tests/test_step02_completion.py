"""Guard the new limited handoff against result suppression and false approval."""
import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rank_refinement.reporting.result_labels import result_identity
from rank_refinement.runtime.artifacts import artifact_manifest, read_json, write_json, write_text
from rank_refinement.runtime.config import load_configs
from rank_refinement.stages.step02_native_refinement import validate_protocol, summarize
from rank_refinement.stages.step02_completion import (check_case_grid, check_indexed_tree,
    check_feature_groups, compare_saved_summary, long_results, require_pilot_handoff, safe_source_path, validate_completion_config)

ROOT = Path(__file__).resolve().parents[1]


def fixture_cases():
    cases = []
    for rep in ["accepted_phrases", "unigrams"]:
        for base in ["lda", "nmf"]:
            for fold in [1,2,3]:
                for seed in [809,810]:
                    score = {"scored": fold != 1, "c_v": .4 if fold != 1 else None,
                             "c_npmi": -.1 if fold != 1 else None,
                             "absent_words": [] if fold != 1 else ["missing"],
                             "failure": None if fold != 1 else "standard nonfinite"}
                    final = copy.deepcopy(score)
                    if final["scored"]:
                        final["c_v"], final["c_npmi"] = .5, -.05
                    topics = [["w"+str(i*10+j) for j in range(10)] for i in range(10)]
                    cases.append({"representation": rep, "base": base, "fold": fold, "seed": seed,
                        "native_validation": score, "final_validation": final,
                        "native_td": 1., "final_td": 1., "relative_l1_final": .01,
                        "fitting_accepted": True, "native_topics": topics, "final_topics": topics})
    return cases


class CompletionTests(unittest.TestCase):
    def policy(self):
        return validate_protocol(load_configs(ROOT))

    def test_native_and_proposed_names_preserve_underlying_neural_base(self):
        for base, native in [("encot","EnCOT"),("neuromax","NeuroMax")]:
            self.assertEqual(result_identity("native",base,"pilot")["model"],native)
            proposed=result_identity("proposed",base,"pilot")
            self.assertEqual(proposed["model"],"Proposed")
            self.assertEqual(proposed["underlying_base"],native)

    def test_unknown_roles_and_omitted_scope_are_rejected(self):
        with self.assertRaises(ValueError):result_identity("winner","encot","pilot")
        with self.assertRaises(ValueError):result_identity("proposed","unknown","pilot")
        with self.assertRaises(ValueError):result_identity("proposed","encot","")

    def test_completed_grid_cannot_substitute_a_duplicate_for_a_missing_case(self):
        cases=fixture_cases();check_case_grid(cases,self.policy())
        cases[-1]=copy.deepcopy(cases[0])
        with self.assertRaisesRegex(ValueError,"24 unique"):check_case_grid(cases,self.policy())

    def test_partial_run_is_not_finalized(self):
        with self.assertRaisesRegex(ValueError,"24 unique"):check_case_grid(fixture_cases()[:-1],self.policy())

    def test_primary_development_or_test_group_cannot_enter_saved_feature_view(self):
        rows=[{"group_id":"fit1","split":"fit"},{"group_id":"val1","split":"validation"}]
        check_feature_groups(rows,["fit1"],["val1"])
        rows[1]["group_id"]="primary_test1"
        with self.assertRaisesRegex(ValueError,"fitting-only"):check_feature_groups(rows,["fit1"],["val1"])

    def test_duplicate_feature_groups_cannot_be_collapsed_silently(self):
        rows=[{"group_id":"fit1","split":"fit"},{"group_id":"val1","split":"validation"}]
        with self.assertRaisesRegex(ValueError,"duplicate"):check_feature_groups(rows+[rows[0]],["fit1"],["val1"])

    def test_missing_cases_and_negative_npmi_are_retained_in_both_method_rows(self):
        rows,topics=long_results(fixture_cases())
        self.assertEqual(len(rows),48);self.assertEqual(len(topics),480)
        self.assertEqual(sum(not r["scored"] for r in rows),16)
        proposed=[r for r in rows if r["method_role"]=="proposed"]
        self.assertEqual(len(proposed),24)
        self.assertEqual(sum(r["npmi"]==-.05 for r in proposed),16)
        self.assertTrue(all(r["model"]=="Proposed" and not r["final_proposed_model_result"] for r in proposed))

    def test_summary_replay_preserves_original_failed_reference_gate(self):
        cases=fixture_cases();saved=summarize(cases,self.policy())
        result=compare_saved_summary(cases,self.policy(),saved)
        self.assertTrue(result["mechanism_promising"])
        self.assertFalse(result["complete_coherence_reference_gate_passed"])
        self.assertFalse(result["acquisition_gate_passed"])
        self.assertFalse(result["full_experiment_gate_passed"])

    def test_false_scientific_gate_cannot_be_changed_to_a_pass(self):
        cases=fixture_cases();saved=summarize(cases,self.policy());saved["acquisition_gate_passed"]=True
        with self.assertRaisesRegex(ValueError,"acquisition_gate"):compare_saved_summary(cases,self.policy(),saved)

    def test_indexed_payload_changes_and_extra_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);write_text(folder/"a.txt","original")
            write_json(folder/"artifact_manifest.json",artifact_manifest(folder))
            self.assertEqual(len(check_indexed_tree(folder)),1)
            write_text(folder/"extra.txt","extra")
            with self.assertRaisesRegex(ValueError,"extra"):check_indexed_tree(folder)
            (folder/"extra.txt").unlink();write_text(folder/"a.txt","changed")
            with self.assertRaisesRegex(ValueError,"changed"):check_indexed_tree(folder)

    def test_size_metadata_is_checked_even_when_hash_is_correct(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);write_text(folder/"a.txt","a")
            manifest=artifact_manifest(folder);manifest[0]["bytes"]=9
            write_json(folder/"artifact_manifest.json",manifest)
            with self.assertRaisesRegex(ValueError,"size"):check_indexed_tree(folder)

    def test_source_pointer_cannot_escape_or_point_at_final_paper_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for unsafe in ["../elsewhere","/tmp/elsewhere","outputs/paper/main/example","outputs\\steps\\step02_native_refinement\\id"]:
                with self.subTest(unsafe=unsafe):
                    with self.assertRaises(ValueError):safe_source_path(root,unsafe)
            good=root/"outputs/steps/step02_native_refinement/id";good.mkdir(parents=True)
            self.assertEqual(safe_source_path(root,"outputs/steps/step02_native_refinement/id"),good.resolve())

    def test_completion_config_cannot_authorize_a_full_benchmark(self):
        configs=load_configs(ROOT);validate_completion_config(configs)
        configs["step02_completion"]["allow_partial_reference_for_benchmark"]=True
        with self.assertRaises(ValueError):validate_completion_config(configs)

    def test_missing_handoff_cannot_start_followup(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,"Finalize Step 2"):require_pilot_handoff(Path(d))

    def test_handoff_cannot_expand_its_approved_scope(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_json(root/"state/step02_pilot_accepted.json",{
                "completion_protocol":"STEP02_R3_COMPLETION_R1_20261009","step02_pilot_gate_passed":True,
                "bounded_neural_pilot_allowed":True,"handoff_scope":"full_benchmark",
                "full_experiment_gate_passed":True,"final_paper_benchmark_ready":True})
            with self.assertRaisesRegex(ValueError,"scope"):require_pilot_handoff(root)

    def test_finalize_flag_is_rejected_for_wrong_step_and_other_modes(self):
        for args in [["--step","1","--finalize"],["--step","2","--finalize","--self-check"],
                     ["--step","2","--finalize","--doctor-only"]]:
            result=subprocess.run([sys.executable,str(ROOT/"main.py"),*args],capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertIn("--finalize",result.stderr)
