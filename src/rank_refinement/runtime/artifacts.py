"""Portable, atomic artifacts. No non-finite JSON and no silent partial files."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import os
import uuid
from pathlib import Path


def stable_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp_" + uuid.uuid4().hex)
    try:
        with tmp.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_text(path, text):
    atomic_bytes(path, text.encode("utf-8"))


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path, records):
    write_text(path, "".join(stable_json(r) + "\n" for r in records))


def write_csv(path, rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, ("\ufeff" + stream.getvalue()).encode("utf-8"))


def artifact_manifest(folder):
    folder = Path(folder)
    return [{"path": p.relative_to(folder).as_posix(), "bytes": p.stat().st_size, "sha256": file_hash(p)}
            for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"]


def verify_artifacts(folder, records):
    folder = Path(folder).resolve()
    for item in records:
        path = (folder / item["path"]).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise ValueError("Missing or unsafe accepted artifact: " + item["path"])
        if file_hash(path) != item["sha256"]:
            raise ValueError("Accepted artifact changed: " + item["path"])
    return True
