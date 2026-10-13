import csv
import tempfile
import unittest
from pathlib import Path
from rank_refinement.data.cleaning import clean_body
from rank_refinement.data.io import locate_primary, read_source
from rank_refinement.data.duplicate_groups import group_records
from rank_refinement.data.splits import allocate_counts, assign_splits, make_internal_folds, recover_split
from rank_refinement.data.vocabulary import fit_features

NEAR = {"enabled": True, "threshold": 0.9, "minimum_chars": 50, "minimum_length_ratio": 0.9, "shingle_size": 5}


def record(index, text, source="beijing"):
    return {"record_id": source + ":" + str(index), "source": source, "source_id": str(index),
            "source_file": source + ".csv", "normalized_body": text, "comparison_text": text,
            "clean_text": text}


class DataFirewallTests(unittest.TestCase):
    def test_negation_and_content_preserved(self):
        self.assertEqual(clean_body("[公交不能扫码，建议改进 网页链接 https://t.cn/a]") , "公交不能扫码,建议改进")

    def test_real_list_wrapper_is_unwrapped(self):
        self.assertEqual(clean_body("['没有空调']"), "没有空调")

    def test_empty_brackets_remain_empty(self):
        self.assertEqual(clean_body("[ ]"), "")

    def test_announcements_are_not_relevance_filtered(self):
        self.assertIn("集团公告", clean_body("[集团公告：开通新线路]"))

    def test_gb18030_strict_decode_and_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "beijing.csv"
            p.write_bytes("序号,正文,发布时间\n7,[公交不能扫码],11月3日\n".encode("gb18030"))
            rows, meta = read_source(p, "beijing", {"encodings": ["utf-8-sig", "gb18030"], "body_column": "正文", "id_column": "序号", "timestamp_column": "发布时间"})
            self.assertEqual(rows[0]["record_id"], "beijing:7")
            self.assertEqual(meta["encoding"], "gb18030")
            self.assertEqual(meta["raw_records"], 1)

    def test_bad_bytes_never_use_replacement_decode(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"beijing.csv"
            p.write_bytes(b'\xff')
            with self.assertRaises(ValueError):
                read_source(p, "beijing", {"encodings": ["utf-8-sig", "gb18030"], "body_column": "正文", "id_column": "序号", "timestamp_column": "发布时间"})

    def test_repeated_source_ids_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"beijing.csv"
            p.write_text("序号,正文\n1,公交\n1,空调\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_source(p,"beijing", {"encodings":["utf-8-sig"],"body_column":"正文","id_column":"序号","timestamp_column":"发布时间"})

    def test_fallback_folder_supported_without_ambiguity(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/"data/primary"
            folder.mkdir(parents=True)
            for source in ["beijing", "shanghai", "xiamen"]:
                (folder/(source+"(5).csv")).write_text("placeholder", encoding="utf-8")
            policy={"input_dir":"data/raw/primary","fallback_input_dir":"data/primary","sources":{s:s+"*.csv" for s in ["beijing","shanghai","xiamen"]}}
            self.assertEqual(len(locate_primary(d,policy)),3)
            other=Path(d)/"data/raw/primary"
            other.mkdir(parents=True)
            (other/"beijing.csv").write_text("other",encoding="utf-8")
            with self.assertRaises(ValueError): locate_primary(d,policy)

    def test_cross_source_exact_duplicate_has_one_group(self):
        groups,_=group_records([record(1,"相同正文"),record(2,"相同正文","shanghai")],NEAR)
        self.assertEqual(len(groups),1)
        self.assertEqual(len(groups[0]["members"]),2)
        self.assertEqual(groups[0]["member_sources"],["beijing","shanghai"])

    def test_grouping_is_input_order_invariant(self):
        rows=[record(1,"相同正文"),record(2,"相同正文"),record(3,"独立正文")]
        self.assertEqual(group_records(rows,NEAR),group_records(list(reversed(rows)),NEAR))

    def test_longest_representative_rule(self):
        a=record(1,"同一正文")
        b=record(2,"同一正文")
        b["normalized_body"]="同一 正文"
        groups,_=group_records([a,b],NEAR)
        self.assertEqual(groups[0]["representative"]["record_id"],"beijing:2")

    def test_near_duplicate_groups_cannot_split(self):
        text="乘客建议优化公交扫码支付设备并改善空调服务和车辆停站流程"*4
        second=text[:-1]+"。"
        groups,edges=group_records([record(1,text),record(2,second)],NEAR)
        self.assertEqual(len(groups),1)
        self.assertTrue(any(e["kind"]=="near" for e in edges))

    def test_split_rounding_preserves_population(self):
        for n in range(3,100):
            counts=allocate_counts(n,{"fit":.7,"development":.1,"test":.2})
            self.assertEqual(sum(counts.values()),n)
            self.assertTrue(all(v>=1 for v in counts.values()))

    def test_split_and_fold_order_invariance(self):
        rows=[record(i,"独立正文"+str(i),source) for source in ["beijing","shanghai","xiamen"] for i in range(1,16)]
        groups,_=group_records(rows,{**NEAR,"enabled":False})
        policy={"split_seed":20261009,"split_ratios":{"fit":.7,"development":.1,"test":.2}}
        a,_=assign_splits(groups,policy)
        b,_=assign_splits(list(reversed(groups)),policy)
        self.assertEqual(a,b)
        folds=make_internal_folds(groups,a,3,20261010)
        validation=[g for f in folds for g in f["validation_group_ids"]]
        fit={g for g,v in a.items() if v=="fit"}
        self.assertEqual(set(validation),fit)
        self.assertEqual(len(validation),len(fit))
        for fold in folds:self.assertFalse(set(fold["fit_group_ids"]) & set(fold["validation_group_ids"]))

    def test_historical_group_conflict_rejected(self):
        groups,_=group_records([record(1,"相同"),record(2,"相同","shanghai")],NEAR)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"previous.csv"
            p.write_text("record_id,split\nbeijing:1,fit\nshanghai:2,test\n",encoding="utf-8")
            with self.assertRaises(ValueError):recover_split(groups,p)

    def test_historical_split_not_replaced_by_new_seed(self):
        groups,_=group_records([record(1,"独立")],NEAR)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"previous.csv"
            p.write_text("record_id,split\nbeijing:1,test\n",encoding="utf-8")
            assignment=recover_split(groups,p)
            self.assertEqual(list(assignment.values()),["test"])

    def test_heldout_words_and_phrases_do_not_fit(self):
        docs={"fit1":{"base_tokens":["扫码","支付","正常"]},"fit2":{"base_tokens":["扫码","支付","设备"]},
              "heldout":{"base_tokens":["秘密目标","泄漏词"]*50}}
        with tempfile.TemporaryDirectory() as d:
            views,meta=fit_features(docs,["fit1","fit2"],{"enabled":True,"min_count":1,"threshold":.5,"scoring":"npmi","delimiter":"_"},
                                    {"min_document_frequency":1,"max_document_fraction":1},d)
            self.assertNotIn("秘密目标",meta["word_to_id"])
            self.assertNotIn("秘密目标_泄漏词",meta["word_to_id"])
            self.assertEqual(meta["fit_groups"],2)

    def test_group_df_is_presence_not_occurrences(self):
        docs={"a":{"base_tokens":["设备"]*100},"b":{"base_tokens":["设备","空调"]}}
        with tempfile.TemporaryDirectory() as d:
            _,meta=fit_features(docs,["a","b"],{"enabled":False}, {"min_document_frequency":1,"max_document_fraction":1},d)
            df=dict(zip(meta["tokens"],meta["document_frequency"]))
            self.assertEqual(df["设备"],2)
            self.assertEqual(df["空调"],1)
