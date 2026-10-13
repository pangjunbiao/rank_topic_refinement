from __future__ import annotations
import csv
import io
from pathlib import Path
from ..runtime.artifacts import file_hash
from .cleaning import clean_body, comparison_text, unwrap_body


def locate_primary(root, policy):
    candidates = [Path(root) / policy["input_dir"], Path(root) / policy["fallback_input_dir"]]
    active = []
    for directory in dict.fromkeys(candidates):
        found = {source: sorted(directory.glob(pattern)) for source, pattern in policy["sources"].items()}
        if any(found.values()):
            active.append((directory, found))
    if len(active) != 1:
        raise ValueError("Put exactly one copy of the three primary CSVs in data/raw/primary (or data/primary). Found active directories: " + str([str(d) for d, _ in active]))
    directory, found = active[0]
    result = {}
    for source, files in found.items():
        if len(files) != 1:
            raise ValueError(f"Expected one {source} CSV; found {len(files)} in {directory}.")
        result[source] = files[0]
    return result


def read_source(path, source, policy):
    path = Path(path)
    raw = path.read_bytes()
    text, encoding = None, None
    for candidate in policy["encodings"]:
        try:
            text = raw.decode(candidate, errors="strict")
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("Cannot strictly decode " + str(path) + "; no replacement decoding is allowed.")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not reader.fieldnames or policy["body_column"] not in reader.fieldnames or policy["id_column"] not in reader.fieldnames:
        raise ValueError("Missing required CSV headers in " + str(path))
    if len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("Duplicate CSV headers in " + str(path))
    rows, seen = [], set()
    for ordinal, row in enumerate(reader, 1):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f"Malformed field count in {path.name}, data row {ordinal}.")
        source_id = row[policy["id_column"]].strip()
        if not source_id or source_id in seen:
            raise ValueError(f"Empty or repeated source ID in {path.name}, data row {ordinal}.")
        seen.add(source_id)
        body = row[policy["body_column"]]
        rows.append({"record_id": source + ":" + source_id, "source": source, "source_id": source_id,
                     "source_file": path.name, "source_row": ordinal, "raw_body": body,
                     "normalized_body": unwrap_body(body), "clean_text": clean_body(body),
                     "comparison_text": comparison_text(body),
                     "published_at": row.get(policy["timestamp_column"], "")})
    if not rows:
        raise ValueError("Empty primary CSV: " + str(path))
    return rows, {"source": source, "file_name": path.name, "sha256": file_hash(path),
                  "bytes": len(raw), "encoding": encoding, "raw_records": len(rows), "columns": reader.fieldnames}
