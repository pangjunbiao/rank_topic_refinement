import tempfile
import unittest
from pathlib import Path
from rank_refinement.runtime.state import require_preparation,accepted_preparation
from rank_refinement.runtime.state import ProjectLock
from rank_refinement.runtime.artifacts import write_json,artifact_manifest


class StageGuardTests(unittest.TestCase):
    def test_no_accepted_preparation_no_advance(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):require_preparation(d)

    def test_failed_quality_gate_no_advance(self):
        with tempfile.TemporaryDirectory() as d:
            write_json(Path(d)/"state/step01_accepted.json",{"engineering_passed":True,"preparation_gate_passed":False})
            with self.assertRaises(RuntimeError):require_preparation(d)

    def test_changed_protocol_after_downstream_acceptance_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            write_json(Path(d)/"state/step01_accepted.json",{"preparation_id":"old"})
            write_json(Path(d)/"state/step02_accepted.json",{"passed":True})
            with self.assertRaises(RuntimeError):accepted_preparation(d,"new")

    def test_matching_preparation_artifacts_verified(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/"data/processed/primary/example"
            write_json(folder/"vocabulary.json",{"tokens":["a"]})
            write_json(Path(d)/"state/step01_accepted.json",{"preparation_id":"same","processed_relative_path":"data/processed/primary/example",
                "engineering_passed":True,"preparation_gate_passed":True,"artifacts":artifact_manifest(folder)})
            self.assertEqual(require_preparation(d)["preparation_id"],"same")

    def test_concurrent_project_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with ProjectLock(d):
                with self.assertRaises(RuntimeError):
                    with ProjectLock(d):pass
            with ProjectLock(d):pass  # OS lock releases; a persistent lock file is harmless.
