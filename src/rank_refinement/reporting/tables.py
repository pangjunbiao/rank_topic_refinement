from pathlib import Path
from ..runtime.artifacts import write_csv, write_text


def dataset_table(raw_report, documents, assignments, transformed, masks, word_to_id):
    result = []
    raw_counts = {r["source"]: r["raw_records"] for r in raw_report["sources"]}
    import statistics
    vocabulary = set(word_to_id)
    for source in ["ALL"] + sorted(raw_counts):
        ids = [g for g, d in documents.items() if source == "ALL" or d["source"] == source]
        def heldout_coverage(split):
            subset = [g for g in ids if assignments[g] == split]
            total = sum(len(transformed[g]) for g in subset)
            kept = sum(sum(token in vocabulary for token in transformed[g]) for g in subset)
            return kept / total if total else 0.0
        result.append({"dataset": "primary", "collection_source": source,
                       "raw_records": raw_report["raw_records"] if source == "ALL" else raw_counts[source],
                       "eligible_representatives": len(ids),
                       "fit": sum(assignments[g] == "fit" for g in ids),
                       "development": sum(assignments[g] == "development" for g in ids),
                       "test": sum(assignments[g] == "test" for g in ids), "vocabulary": len(word_to_id),
                       "median_lexical_tokens": statistics.median([len(transformed[g]) for g in ids]) if ids else 0,
                       "development_token_coverage": heldout_coverage("development"),
                       "test_token_coverage": heldout_coverage("test"),
                       "completion_eligible_dev": sum(assignments[g] == "development" and masks[g]["eligible"] for g in ids),
                       "completion_eligible_test": sum(assignments[g] == "test" and masks[g]["eligible"] for g in ids)})
    return result


def export_table(folder, name, rows, columns, note=""):
    folder = Path(folder)
    write_csv(folder / (name + ".csv"), rows, columns)
    markdown = "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |\n"
    markdown += "".join("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in columns) + " |\n" for row in rows)
    if note:
        markdown += "\n" + note + "\n"
    write_text(folder / (name + ".md"), markdown)
    def escape(value):
        return str(value).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")
    latex = "\\begin{tabular}{" + "l" * len(columns) + "}\n\\hline\n"
    latex += " & ".join(escape(c) for c in columns) + " \\\\\n\\hline\n"
    latex += "".join(" & ".join(escape(row[c]) for c in columns) + " \\\\\n" for row in rows)
    latex += "\\hline\n\\end{tabular}\n"
    write_text(folder / (name + ".tex"), latex)
