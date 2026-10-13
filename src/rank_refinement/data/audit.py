from collections import Counter
import re
import statistics


def raw_audit(records, groups, edges):
    rows = []
    for source in sorted({r["source"] for r in records}):
        subset = [r for r in records if r["source"] == source]
        rows.append({"source": source, "raw_records": len(subset),
                     "empty_cleaned_bodies": sum(not r["clean_text"] for r in subset),
                     "yearless_timestamps": sum(not re.search(r"(?:19|20)\d{2}", r["published_at"]) for r in subset),
                     "median_normalized_characters": statistics.median(len(r["normalized_body"]) for r in subset)})
    return {"sources": rows, "raw_records": len(records), "independent_groups_before_lexical_filter": len(groups),
            "duplicate_groups": sum(len(g["members"]) > 1 for g in groups),
            "extra_duplicate_records": sum(len(g["members"]) - 1 for g in groups),
            "cross_source_groups": sum(len(g["member_sources"]) > 1 for g in groups),
            "exact_edges": sum(e["kind"] == "exact" for e in edges), "near_edges": sum(e["kind"] == "near" for e in edges),
            "source_means_collection_not_verified_geography": True, "historical_exposure": True}


def feature_summary(documents, transformed, metadata, assignments, masks):
    vocab = set(metadata["tokens"])
    result = []
    for split in sorted(set(assignments.values())):
        for source in ["ALL"] + sorted({d["source"] for d in documents.values()}):
            selected = [g for g in sorted(documents) if assignments[g] == split and (source == "ALL" or documents[g]["source"] == source)]
            total = sum(len(transformed[g]) for g in selected)
            retained = sum(sum(w in vocab for w in transformed[g]) for g in selected)
            result.append({"split": split, "source": source, "representatives": len(selected),
                           "lexical_tokens": total, "in_vocabulary_tokens": retained,
                           "coverage": retained / total if total else 0.0,
                           "model_nonempty_documents": sum(any(w in vocab for w in transformed[g]) for g in selected),
                           "completion_eligible_documents": sum(masks[g]["eligible"] for g in selected),
                           "completion_eligible_fraction": sum(masks[g]["eligible"] for g in selected) / len(selected) if selected else 0.0,
                           "median_lexical_tokens": statistics.median([len(transformed[g]) for g in selected]) if selected else 0,
                           "oov_token_fraction": 1 - retained / total if total else 1.0})
    return result
