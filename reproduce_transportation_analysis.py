#!/usr/bin/env python3
"""Read-only, reproducible transportation case audit for saved primary results.

Python 3.10+. Bundled CSV mode uses the standard library only. Live compact
mode also uses the project's existing NumPy installation to read saved arrays.
This program does not train models, select settings, or infer service labels.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION = "transportation-case-audit-v1.0"
SEEDS = (809, 810, 811, 812, 813)
K = 10
M = 10
EXAMPLE_SEED = 809


class EvidenceError(ValueError):
    """Evidence cannot support a consistent matched case audit."""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise EvidenceError(f"Cannot export an empty table: {path.name}")
    # UTF-8 BOM helps Excel on Windows display original Chinese tokens.
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tex(text: Any) -> str:
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
               "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
               "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(c, c) for c in str(text))


def strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise EvidenceError(f"Boolean value is not a valid {field}.")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise EvidenceError(f"Invalid integer for {field}: {value!r}") from error
    if str(result) != str(value):
        raise EvidenceError(f"Non-integer representation for {field}: {value!r}")
    return result


def validate_inventory(rows: list[dict[str, Any]]) -> dict:
    groups: dict[tuple[int, str, int], list[tuple[int, int, str]]] = {}
    vocabulary: dict[int, str] = {}
    for row in rows:
        seed = strict_int(row.get("seed"), "seed")
        topic = strict_int(row.get("topic"), "topic")
        rank = strict_int(row.get("rank"), "rank")
        word_id = strict_int(row.get("word_id"), "word_id")
        method = str(row.get("method", "")).strip().lower()
        token = str(row.get("token", ""))
        if seed not in SEEDS or topic not in range(1, K + 1):
            raise EvidenceError(f"Unexpected seed/topic: {seed}/{topic}.")
        if method not in {"native", "proposed"} or rank not in range(1, M + 1):
            raise EvidenceError(f"Unexpected method/rank: {method}/{rank}.")
        if not token or word_id < 0:
            raise EvidenceError("Every word must have a nonempty token and nonnegative ID.")
        if word_id in vocabulary and vocabulary[word_id] != token:
            raise EvidenceError(f"Vocabulary ID {word_id} maps to different tokens.")
        vocabulary[word_id] = token
        groups.setdefault((seed, method, topic), []).append((rank, word_id, token))
    expected = {(s, m, t) for s in SEEDS for m in ("native", "proposed")
                for t in range(1, K + 1)}
    if set(groups) != expected:
        missing = sorted(expected - set(groups))
        raise EvidenceError(f"Incomplete matched seed/topic inventory. Missing: {missing[:5]}")
    output = {}
    for key, values in groups.items():
        if len(values) != M or {v[0] for v in values} != set(range(1, M + 1)):
            raise EvidenceError(f"Duplicate or missing ranks for {key}.")
        if len({v[1] for v in values}) != M or len({v[2] for v in values}) != M:
            raise EvidenceError(f"Duplicate descriptor words for {key}.")
        output[key] = [v[2] for v in sorted(values)]
    return output


def load_bundle(data: Path, notes_path: Path | None) -> tuple[dict, dict, dict, dict]:
    manifest_path = data / "input_manifest.json"
    if not manifest_path.is_file():
        raise EvidenceError(f"Input manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if (manifest.get("schema") != "transportation-audit-inputs-v1"
            or manifest.get("seeds") != list(SEEDS)
            or manifest.get("topics") != K or manifest.get("descriptor_words") != M):
        raise EvidenceError("Unexpected bundled protocol.")
    hashes = {}
    for name, expected in manifest["files"].items():
        path = data / name
        if not path.is_file():
            raise EvidenceError(f"Required input missing: {path}")
        actual = sha(path.read_bytes())
        if actual != expected:
            raise EvidenceError(f"Bundled input changed: {name}. Use live compact mode for new outputs.")
        hashes[str(path.resolve())] = actual
    hashes[str(manifest_path.resolve())] = sha(manifest_path.read_bytes())
    with (data / "topic_words_all_primary_seeds.csv").open(encoding="utf-8-sig", newline="") as handle:
        inventory = validate_inventory(list(csv.DictReader(handle)))
    notes_file = notes_path or data / "presentation_notes.json"
    notes = read_json(notes_file)
    hashes[str(notes_file.resolve())] = sha(notes_file.read_bytes())
    identity = {"mode": "bundled_saved_primary_evidence", "snapshot_source": manifest["snapshot_source"],
                "snapshot_sha256": manifest["snapshot_sha256"], "scope": manifest["scope"],
                "input_hashes": hashes}
    return inventory, read_json(data / "case_selection.json"), notes, identity


def load_compact(folder: Path, notes_file: Path) -> tuple[dict, dict, dict, dict]:
    try:
        import numpy as np
    except ImportError as error:
        raise EvidenceError("Live array mode needs NumPy from your existing project environment.") from error
    evidence = folder / "numerical_evidence"
    application = evidence / "application_selection.json"
    if not application.is_file():
        raise EvidenceError(f"Saved application selection not found: {application}")
    rows = []
    hashes = {str(application.resolve()): sha(application.read_bytes()),
              str(notes_file.resolve()): sha(notes_file.read_bytes())}
    expected_vocabulary = None
    for seed in SEEDS:
        arrays_file = evidence / "primary" / f"seed{seed}" / "affinity_before_after.npz"
        vocabulary_file = arrays_file.with_name("vocabulary.json")
        if not arrays_file.is_file() or not vocabulary_file.is_file():
            raise EvidenceError(f"Saved primary arrays or vocabulary missing for seed {seed}.")
        vocabulary_json = read_json(vocabulary_file)
        vocabulary = vocabulary_json["tokens"] if isinstance(vocabulary_json, dict) else vocabulary_json
        if not isinstance(vocabulary, list) or not all(isinstance(x, str) and x for x in vocabulary):
            raise EvidenceError("Invalid saved vocabulary.")
        if len(set(vocabulary)) != len(vocabulary) or len(vocabulary) <= M:
            raise EvidenceError("Vocabulary must contain unique tokens and exceed descriptor depth.")
        if expected_vocabulary is None:
            expected_vocabulary = vocabulary
        elif vocabulary != expected_vocabulary:
            raise EvidenceError("Primary seeds must share the same fitting vocabulary.")
        with np.load(arrays_file, allow_pickle=False) as saved:
            if not {"native", "proposed", "targets"}.issubset(saved.files):
                raise EvidenceError(f"Required array keys missing for seed {seed}.")
            native = np.asarray(saved["native"], dtype=np.float64)
            proposed = np.asarray(saved["proposed"], dtype=np.float64)
            targets = np.asarray(saved["targets"])
        shape = (K, len(vocabulary))
        if native.shape != shape or proposed.shape != shape:
            raise EvidenceError(f"Unexpected primary affinity shape for seed {seed}.")
        if not np.isfinite(native).all() or not np.isfinite(proposed).all():
            raise EvidenceError("Affinities must be finite.")
        if (native < 0).any() or (proposed < 0).any() or (native.sum(1) <= 0).any():
            raise EvidenceError("Affinities must be nonnegative with positive native row totals.")
        scale = float(native.max())
        for axis in (0, 1):
            if float(np.max(np.abs(proposed.sum(axis) - native.sum(axis)))) / scale > 1e-7:
                raise EvidenceError(f"Conserved affinity totals failed for seed {seed}.")
        if targets.shape != (K, M) or not np.issubdtype(targets.dtype, np.integer):
            raise EvidenceError("Saved targets must be an integer K-by-M array.")
        if (targets < 0).any() or (targets >= len(vocabulary)).any():
            raise EvidenceError("Saved target IDs are outside the vocabulary.")
        indices = np.arange(len(vocabulary))
        for method, array in (("native", native), ("proposed", proposed)):
            for t in range(K):
                selected = np.lexsort((indices, -array[t]))[:M]
                if method == "proposed" and set(selected) != set(targets[t]):
                    raise EvidenceError(f"Actual target membership failed: seed={seed}, topic={t + 1}.")
                for rank, word_id in enumerate(selected, 1):
                    rows.append({"seed": seed, "method": method, "topic": t + 1,
                                 "rank": rank, "word_id": int(word_id), "token": vocabulary[word_id]})
        hashes[str(arrays_file.resolve())] = sha(arrays_file.read_bytes())
        hashes[str(vocabulary_file.resolve())] = sha(vocabulary_file.read_bytes())
    identity = {"mode": "live_saved_compact_arrays", "compact_directory": str(folder.resolve()),
                "scope": "Saved primary affinities and previously selected fitting posts. No retraining or test selection.",
                "input_hashes": hashes}
    return validate_inventory(rows), read_json(application), read_json(notes_file), identity


def edit_rows(inventory: dict) -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        for topic in range(1, K + 1):
            native = inventory[seed, "native", topic]
            proposed = inventory[seed, "proposed", topic]
            retained = set(native) & set(proposed)
            added = [w for w in proposed if w not in set(native)]
            removed = [w for w in native if w not in set(proposed)]
            if len(added) != len(removed):
                raise EvidenceError("Matched descriptors must have equal depth.")
            rows.append({"seed": seed, "topic": topic,
                         "native_words": " | ".join(native), "proposed_words": " | ".join(proposed),
                         "removed_words": " | ".join(removed), "added_words": " | ".join(added),
                         "retained_words": len(retained), "replacements": len(added),
                         "native_membership_retained_percent": len(retained) / M * 100})
    return rows


def build_cases(inventory: dict, selection: dict, notes: dict) -> list[dict[str, Any]]:
    if selection.get("seed") != EXAMPLE_SEED or selection.get("human_labels_or_accuracy") is not False:
        raise EvidenceError("Expected the archived seed-809 fitting illustrations without accuracy labels.")
    note_list = notes.get("cases", [])
    note_by_id = {n["group_id"]: n for n in note_list}
    if len(note_by_id) != len(note_list):
        raise EvidenceError("Duplicate presentation-note group IDs.")
    glossary = notes.get("glosses", {})
    originals = selection.get("selected_original_rows", [])
    if len(originals) != 6:
        raise EvidenceError("Expected all six original examples; do not omit unfavorable cases.")
    seen_topics, seen_groups = set(), set()
    edits = {(r["seed"], r["topic"]): r for r in edit_rows(inventory)}
    output = []
    for original in originals:
        topic = strict_int(original.get("topic_index"), "saved zero-based topic index") + 1
        group_id = original.get("group_id")
        if topic not in range(1, K + 1) or topic in seen_topics or group_id in seen_groups:
            raise EvidenceError("Invalid or repeated case topic/group.")
        seen_topics.add(topic)
        seen_groups.add(group_id)
        if original.get("dataset") != "Primary" or str(original.get("seed")) != str(EXAMPLE_SEED):
            raise EvidenceError("Case must belong to the primary seed-809 experiment.")
        if "fitting-only" not in original.get("scope", ""):
            raise EvidenceError("Case split/provenance must explicitly identify fitting-only evidence.")
        words = original.get("descriptor_words", "").split(" | ")
        if len(words) != M or set(words) != set(inventory[EXAMPLE_SEED, "proposed", topic]):
            raise EvidenceError(f"Archived case descriptor does not match actual Proposed membership for T{topic}.")
        note = note_by_id.get(group_id)
        if note is None:
            raise EvidenceError(f"No verified English presentation note for group {group_id}. Supply --notes for changed cases.")
        text = original.get("fitting_text", "")
        if sha(text.encode("utf-8")) != note.get("expected_text_sha256"):
            raise EvidenceError(f"Original post changed for {group_id}; its English translation needs review.")
        edit = edits[EXAMPLE_SEED, topic]
        added = edit["added_words"].split(" | ") if edit["added_words"] else []
        removed = edit["removed_words"].split(" | ") if edit["removed_words"] else []
        missing_glosses = [w for w in added + removed if w not in glossary]
        if missing_glosses:
            raise EvidenceError(f"English glosses need review for: {missing_glosses}. Update --notes for new descriptors.")
        overlap = strict_int(original.get("descriptor_words_present"), "archived overlap count")
        if not 0 <= overlap <= M:
            raise EvidenceError("Archived overlap count is outside descriptor depth.")
        output.append({"seed": EXAMPLE_SEED, "topic": topic, "source": original["source"],
                       "group_id": group_id, "split": "fitting", "post_type": note["post_type"],
                       "service_context": note["service_context"],
                       "native_words": edit["native_words"], "proposed_words": edit["proposed_words"],
                       "removed_words": edit["removed_words"], "added_words": edit["added_words"],
                       "removed_words_en": "; ".join(glossary[w] for w in removed) or "None",
                       "added_words_en": "; ".join(glossary[w] for w in added) or "None",
                       "retained_words": edit["retained_words"], "replacements": edit["replacements"],
                       "archived_proposed_overlap_count_not_recomputed": overlap,
                       "original_post": text, "excerpt_en": note["excerpt_en"],
                       "potential_analyst_followup": note["potential_analyst_followup"]})
    return sorted(output, key=lambda r: r["topic"])


def seed_summaries(inventory: dict, edits: list[dict]) -> list[dict]:
    rows = []
    for seed in SEEDS:
        current = [r for r in edits if r["seed"] == seed]
        if any(r["replacements"] > 4 for r in current):
            raise EvidenceError(f"Frozen R=4 replacement limit failed in seed {seed}.")
        promotions = [set(inventory[seed, "proposed", t]) - set(inventory[seed, "native", t])
                      for t in range(1, K + 1)]
        if sum(map(len, promotions)) != len(set().union(*promotions)):
            raise EvidenceError(f"A word was promoted for more than one topic in seed {seed}.")
        native_unique = len(set().union(*(set(inventory[seed, "native", t]) for t in range(1, K + 1))))
        proposed_unique = len(set().union(*(set(inventory[seed, "proposed", t]) for t in range(1, K + 1))))
        if proposed_unique < native_unique:
            raise EvidenceError(f"Native descriptor diversity was reduced in seed {seed}.")
        rows.append({"seed": seed, "topics": K, "descriptor_positions": K * M,
                     "replacement_positions": sum(r["replacements"] for r in current),
                     "retained_positions": sum(r["retained_words"] for r in current),
                     "native_unique_words": native_unique, "proposed_unique_words": proposed_unique,
                     "native_td": native_unique / (K * M), "proposed_td": proposed_unique / (K * M)})
    return rows


def main_table(cases: list[dict]) -> str:
    lines = [r"\begin{table*}[!t]",
             r"\caption{Matched descriptor changes and supporting transit posts (primary corpus, seed 809).}",
             r"\label{tab:transportation_examples}", r"\centering", r"\begingroup", r"\footnotesize",
             r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.08}",
             r"\begin{tabularx}{\textwidth}{@{}",
             r">{\raggedright\arraybackslash}p{0.04\textwidth}",
             r">{\raggedright\arraybackslash}p{0.15\textwidth}",
             r">{\raggedright\arraybackslash}p{0.18\textwidth}",
             r">{\raggedright\arraybackslash}p{0.17\textwidth}",
             r">{\raggedright\arraybackslash}X@{}}", r"\toprule",
             r"Topic & Service context & Removed words & Added words & Supporting post (condensed English)\\",
             r"\midrule"]
    for c in cases:
        evidence = c["post_type"] + ": " + c["excerpt_en"]
        lines.append(" & ".join(tex(x) for x in (f"T{c['topic']}", c["service_context"],
                                                  c["removed_words_en"], c["added_words_en"], evidence)) + r"\\")
        lines.append(r"\addlinespace[2pt]")
    lines.pop()
    lines += [r"\bottomrule", r"\end{tabularx}", r"\endgroup", r"\end{table*}"]
    return "\n".join(lines) + "\n"


SUBSECTION = r"""\subsection{Transportation Service Interpretation and Descriptor Changes}
\label{subsec:transportation_examples}

The transportation application is to support examination of recurring
service concerns through topic descriptors and their source posts.
Table~\ref{tab:transportation_examples} compares native and refined
descriptors for the same six previously selected fitting posts from
seed 809. The original service-word screening and post-selection rules
are retained. Each row lists words removed from and added to the
actual ten-word affinity descriptor. Entries are English glosses of
distinct original tokens, alongside a condensed English
translation of the supporting post. Original texts, complete word
lists, and group identifiers are retained in the accompanying evidence.

The cases distinguish passenger complaints from service announcements
and incident reports. The waiting post requests help with departure
availability and boarding queues, while the ventilation post requests
continuous air conditioning or permission to open windows. These posts
identify operating conditions that a transport agency could examine.
Other cases concern fleet allocation, crowding information, payment,
route changes, and driver safety.

"""


def subsection_text(cases: list[dict]) -> str:
    # Specific interpretations are emitted only when the corresponding
    # word changes are present in the actual saved evidence.
    by_context = {c["service_context"]: c for c in cases}
    statements = []
    for context, required, phrase in [
        ("Express-route fleet allocation", {"路"}, "the fleet example adds a route term"),
        ("Fare-card conflict and driver safety", {"上车", "下车", "乘客"},
         "the incident example adds boarding, alighting, and passenger terms")]:
        c = by_context.get(context)
        if c and required.issubset(set(c["added_words"].split(" | "))):
            statements.append(phrase)
    waiting = by_context.get("Waiting and boarding difficulties")
    if waiting and {"不", "没", "没有"} & set(waiting["added_words"].split(" | ")):
        statements.append("the waiting example adds negation terms")
    paragraph = "Refinement changes the emphasis of the descriptors. "
    if statements:
        paragraph += "; ".join(statements).capitalize() + ".\n"
    ventilation = by_context.get("Cabin ventilation")
    if ventilation and "开_空调" in ventilation["removed_words"].split(" | "):
        paragraph += ("Specific service details can also be lost: the ventilation descriptor\n"
                      "removes the air-conditioning expression, although the concern remains\n"
                      "explicit in its supporting post. ")
    paragraph += ("Service interpretation therefore\n"
                  "uses the full post together with the descriptor. This matched analysis\n"
                  "provides a traceable illustration of service-theme exploration;\n"
                  "the examples do not measure requirement-detection accuracy or\n"
                  "operational improvement.\n")
    return SUBSECTION + paragraph


def export(out: Path, inventory: dict, cases: list[dict], identity: dict) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    edits = edit_rows(inventory)
    summaries = seed_summaries(inventory, edits)
    write_csv(out / "transportation_case_audit.csv", cases)
    write_csv(out / "all_primary_descriptor_edits.csv", edits)
    write_csv(out / "descriptor_edits_by_seed.csv", summaries)
    counts = [{"seed": c["seed"], "topic": f"T{c['topic']}", "source": c["source"],
               "post_type": c["post_type"], "native_words_retained": c["retained_words"],
               "replacement_words": c["replacements"], "descriptor_depth": M} for c in cases]
    write_csv(out / "matched_case_edit_counts.csv", counts)
    table = main_table(cases)
    subsection = subsection_text(cases)
    (out / "transportation_table.tex").write_text(table, encoding="utf-8")
    (out / "transportation_subsection.tex").write_text(subsection + "\n" + table, encoding="utf-8")
    (out / "transportation_subsection_latex.txt").write_text(subsection + "\n" + table, encoding="utf-8")
    supplementary = ("\\paragraph{Matched transportation case audit.}\n"
                     "The six original fitting examples are retained. Comparisons use the same\n"
                     "topic, seed, and source post. Added and removed words are obtained from\n"
                     "set differences between the saved native and refined top-ten affinities.\n"
                     "English translations and post-type descriptions serve presentation only.\n"
                     "Previously archived lexical-overlap counts are retained without being\n"
                     "recomputed; no tokenization replay or accuracy assessment is performed.\n"
                     "All fifty matched primary topic cases are exported separately. Topic\n"
                     "indices are matched within a seed and are not semantic alignments across seeds.\n")
    (out / "supplementary_transportation_note.tex").write_text(supplementary, encoding="utf-8")
    replacements = sum(c["replacements"] for c in cases)
    retained = sum(c["retained_words"] for c in cases)
    overview = {"six_case_seed": EXAMPLE_SEED, "six_cases": len(cases),
                "retained_descriptor_positions": retained, "replacement_positions": replacements,
                "position_denominator": len(cases) * M,
                "all_primary_topic_cases": len(edits), "neural_fits": 0, "encoder_calls": 0,
                "service_accuracy_established": False, "operational_improvement_established": False,
                "edit_count_interpretation": "Exact membership changes; not service quality or accuracy scores."}
    write_json(out / "case_edit_summary.json", overview)
    metadata = {"software": VERSION, "python": platform.python_version(), "identity": identity,
                "protocol": {"seeds": list(SEEDS), "K": K, "M": M, "case_seed": EXAMPLE_SEED,
                             "case_selection": "All six previously archived service-screened fitting cases.",
                             "rank_ties": "Ascending vocabulary index.",
                             "new_training_or_model_selection": False}, "results": overview}
    write_json(out / "provenance.json", metadata)
    markdown = ["# Matched transportation case analysis", "", "Replace the existing transportation subsection; do not add a second repeated subsection.",
                "", "This is a reproducible qualitative case audit, not an independent task-performance benchmark.",
                "", f"Input mode: `{identity['mode']}`. Five seeds and all {len(edits)} primary topic cases are retained.",
                f"In the six seed-809 examples, {retained}/{len(cases)*M} native descriptor positions remain and {replacements} are replaced.",
                "These counts describe edits and do not measure service benefit.", "", "| Topic | Retained / 10 | Removed words | Added words |", "|---|---:|---|---|"]
    markdown += [f"| T{c['topic']} | {c['retained_words']} | {c['removed_words_en']} | {c['added_words_en']} |" for c in cases]
    markdown += ["", "English text is a condensed, reviewed presentation translation; original Chinese remains in the CSV.",
                 "The six post-type descriptions are presentation notes, not service labels predicted by the model.",
                 "", "Read `transportation_subsection.tex` for the complete replacement text and table.",
                 "Existing supplementary overlap counts remain compatible with the retained cases.",
                 "Independent transportation-usefulness validation still requires suitable external labels or reader judgments."]
    (out / "SUMMARY.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return overview


def run_self_checks(data: Path) -> bool:
    inventory, selection, notes, _ = load_bundle(data, None)
    with (data / "topic_words_all_primary_seeds.csv").open(encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))

    class Checks(unittest.TestCase):
        def test_archived_case_membership(self):
            cases = build_cases(inventory, selection, notes)
            self.assertEqual([c["topic"] for c in cases], [3, 4, 6, 7, 8, 10])
            self.assertEqual([c["replacements"] for c in cases], [1, 4, 2, 1, 3, 3])
            self.assertEqual(sum(c["retained_words"] for c in cases), 46)

        def test_preserve_unfavorable_case(self):
            cases = build_cases(inventory, selection, notes)
            ventilation = next(c for c in cases if c["topic"] == 6)
            self.assertIn("开_空调", ventilation["removed_words"].split(" | "))

        def test_changed_edits_do_not_reuse_old_claims(self):
            cases = build_cases(inventory, selection, notes)
            for case in cases:
                case["added_words"] = ""
                case["removed_words"] = ""
            paragraph = subsection_text(cases)
            self.assertNotIn("adds a route term", paragraph)
            self.assertNotIn("removes the air-conditioning expression", paragraph)

        def test_incomplete_inventory_fails(self):
            with self.assertRaises(EvidenceError):
                validate_inventory(raw_rows[:-1])

        def test_duplicate_rank_fails(self):
            with self.assertRaises(EvidenceError):
                validate_inventory(raw_rows + [copy.deepcopy(raw_rows[0])])

        def test_vocabulary_mismatch_fails(self):
            altered = copy.deepcopy(raw_rows)
            altered[0]["token"] = "incorrect_vocabulary_token"
            with self.assertRaises(EvidenceError):
                validate_inventory(altered)

        def test_changed_case_descriptor_fails(self):
            changed = copy.deepcopy(selection)
            changed["selected_original_rows"][0]["descriptor_words"] = "wrong word"
            with self.assertRaises(EvidenceError):
                build_cases(inventory, changed, notes)

        def test_translation_identity_fails(self):
            changed = copy.deepcopy(selection)
            changed["selected_original_rows"][0]["fitting_text"] += " changed"
            with self.assertRaises(EvidenceError):
                build_cases(inventory, changed, notes)

        def test_case_omission_fails(self):
            changed = copy.deepcopy(selection)
            changed["selected_original_rows"].pop()
            with self.assertRaises(EvidenceError):
                build_cases(inventory, changed, notes)

        def test_all_topic_cases_and_diversity(self):
            edits = edit_rows(inventory)
            self.assertEqual(len(edits), 50)
            summary = seed_summaries(inventory, edits)
            self.assertEqual([x["native_unique_words"] for x in summary], [100, 99, 100, 98, 100])
            self.assertEqual([x["proposed_unique_words"] for x in summary], [100, 99, 100, 98, 100])

        def test_export_preserves_input_bytes(self):
            before = {p.name: sha(p.read_bytes()) for p in data.iterdir() if p.is_file()}
            with tempfile.TemporaryDirectory(prefix="transport_audit_check_") as temp:
                out = Path(temp) / "results"
                cases = build_cases(inventory, selection, notes)
                report = export(out, inventory, cases, {"mode": "self_check_fixture"})
                self.assertEqual(report["replacement_positions"], 14)
                self.assertIn("turn on air conditioning", (out / "transportation_table.tex").read_text())
            self.assertEqual(before, {p.name: sha(p.read_bytes()) for p in data.iterdir() if p.is_file()})

    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(f"SELF CHECK | tests={result.testsRun} | failures={len(result.failures)} | errors={len(result.errors)}")
    return result.wasSuccessful()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data-dir", type=Path, help="Bundled evidence folder (default: data/transportation_audit_v1).")
    parser.add_argument("--compact-dir", type=Path, help="Read saved Step 8 compact primary arrays and application selection directly.")
    parser.add_argument("--notes", type=Path, help="Reviewed presentation notes for the saved cases.")
    parser.add_argument("--output-root", type=Path, help="Default: outputs/steps/step08_transportation.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Compatibility flag. All saved-file analysis runs on CPU; no encoder is called.")
    parser.add_argument("--self-check", action="store_true", help="Check the shipped historical fixture; do not create a production report.")
    args = parser.parse_args()
    started = time.perf_counter()
    root = args.project_root.resolve()
    data = (args.data_dir or root / "data" / "transportation_audit_v1").resolve()
    try:
        if args.self_check:
            if args.compact_dir:
                raise EvidenceError("Run --self-check separately; --compact-dir is for an actual saved-artifact audit.")
            return 0 if run_self_checks(data) else 1
        print(f"START | {VERSION} | requested_device={args.device} | saved-file computation=CPU | neural fits=0")
        if args.compact_dir:
            notes_file = (args.notes or data / "presentation_notes.json").resolve()
            inventory, selection, notes, identity = load_compact(args.compact_dir.resolve(), notes_file)
        else:
            inventory, selection, notes, identity = load_bundle(data, args.notes.resolve() if args.notes else None)
        print(f"SOURCE | {identity['mode']}")
        if identity["mode"] == "bundled_saved_primary_evidence":
            print("SNAPSHOT | archived primary results; does not verify later revised benchmark tables")
        cases = build_cases(inventory, selection, notes)
        edits = edit_rows(inventory)
        seed_summaries(inventory, edits)
        print(f"VERIFIED | matched topic cases={len(edits)} | fixed fitting examples={len(cases)} | one-based topic numbering")
        input_id = sha(json.dumps({"version": VERSION, "hashes": identity["input_hashes"]}, sort_keys=True).encode())[:12]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_root = (args.output_root or root / "outputs" / "steps" / "step08_transportation").resolve()
        if output_root == data or data in output_root.parents:
            raise EvidenceError("Output directory must be outside the evidence-input folder.")
        out = output_root / f"{stamp}_{input_id}"
        result = export(out, inventory, cases, identity)
        for path, old_hash in identity["input_hashes"].items():
            if sha(Path(path).read_bytes()) != old_hash:
                raise EvidenceError(f"Input changed during report generation: {path}")
        report = {"status": "MATCHED_TRANSPORTATION_CASE_AUDIT_COMPLETE", "software": VERSION,
                  "input_identity": input_id, "wall_seconds": time.perf_counter() - started,
                  "requested_device": args.device, "actual_compute": "CPU", "model_calls": 0,
                  "results": result, "output_directory": str(out)}
        write_json(out / "run_result.json", report)
        manifest = {p.name: sha(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()}
        write_json(out / "output_manifest.json", manifest)
        print(f"CASE EDITS | retained={result['retained_descriptor_positions']}/60 | replacements={result['replacement_positions']}/60")
        print("SCOPE | qualitative case audit | requirement-detection accuracy and operational improvement not established")
        print(f"READ FIRST | {out / 'SUMMARY.md'}")
        print(f"MANUSCRIPT | {out / 'transportation_subsection.tex'}")
        print(f"AUDIT CSV | {out / 'transportation_case_audit.csv'}")
        print(f"REPORT | {out / 'run_result.json'}")
        print(f"COMPLETE | wall={report['wall_seconds']:.3f} sec | input bytes unchanged | new neural fits=0")
        return 0
    except (EvidenceError, OSError, KeyError, json.JSONDecodeError, csv.Error, ValueError) as error:
        print(f"FAILED | {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
