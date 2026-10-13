import tempfile
import unittest
from pathlib import Path
from rank_refinement.runtime.artifacts import artifact_manifest,verify_artifacts,write_json,read_json,digest


class ArtifactTests(unittest.TestCase):
    def test_changed_artifact_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"words.json"
            write_json(p,{"words":["设备"]})
            manifest=artifact_manifest(d)
            self.assertTrue(verify_artifacts(d,manifest))
            write_json(p,{"words":["different"]})
            with self.assertRaises(ValueError):verify_artifacts(d,manifest)

    def test_nonfinite_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):write_json(Path(d)/"bad.json",{"score":float("nan")})

    def test_dictionary_order_does_not_change_identity(self):
        self.assertEqual(digest({"b":2,"a":1}),digest({"a":1,"b":2}))

    def test_missing_artifact_detected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):verify_artifacts(d,[{"path":"missing.json","sha256":"x"}])

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):verify_artifacts(d,[{"path":"../escape","sha256":"x"}])

    def test_saved_csr_matrices_and_occurrences_agree(self):
        from rank_refinement.data.vocabulary import fit_features
        from rank_refinement.stages.step01_prepare import persist_features,verify_saved_matrices
        docs={"a":{"base_tokens":["a","a","b"],"record_id":"x:1","source":"x","clean_text":"a a b"},
              "b":{"base_tokens":["a","c"],"record_id":"x:2","source":"x","clean_text":"a c"}}
        with tempfile.TemporaryDirectory() as d:
            views,meta=fit_features(docs,list(docs),{"enabled":False},{"min_document_frequency":1,"max_document_fraction":1},d)
            persist_features(d,docs,views,meta,{"a":"fit","b":"development"},{"seed":1,"observed_fraction":.5,"minimum_in_vocabulary_tokens":2})
            self.assertTrue(verify_saved_matrices(d)["passed"])
