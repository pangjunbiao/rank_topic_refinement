from __future__ import annotations
import json
import os
import uuid
from pathlib import Path
from .artifacts import read_json, verify_artifacts, write_json


class ProjectLock:
    def __init__(self, root):
        self.path = Path(root) / "state" / "step01.lock"
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError("Another experiment stage holds the project lock. Wait for that run to finish.") from exc
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            try:
                if os.name == "nt":
                    import msvcrt
                    self.handle.seek(0)
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            finally:
                self.handle.close()
                self.handle = None


def accepted_preparation(root, signature):
    path = Path(root) / "state" / "step01_accepted.json"
    if not path.exists():
        return None
    accepted = read_json(path)
    if accepted["preparation_id"] != signature:
        downstream = [p for p in (Path(root) / "state").glob("step*_accepted.json") if p.name != "step01_accepted.json"]
        if downstream:
            raise RuntimeError("Preparation changed after a later stage was accepted. Preserve the frozen project and use a separate project for a revised protocol.")
        return None
    folder = (Path(root) / accepted["processed_relative_path"]).resolve()
    verify_artifacts(folder, accepted["artifacts"])
    return accepted


def require_preparation(root):
    path = Path(root) / "state" / "step01_accepted.json"
    if not path.exists():
        raise RuntimeError("Step 1 has no accepted preparation gate.")
    state = read_json(path)
    if not state.get("engineering_passed") or not state.get("preparation_gate_passed"):
        raise RuntimeError("Step 1 preparation gate did not pass.")
    verify_artifacts(Path(root) / state["processed_relative_path"], state["artifacts"])
    folder = Path(root) / state["processed_relative_path"]
    from .config import load_configs
    from ..data.io import locate_primary
    from .artifacts import file_hash
    configs_path = Path(root) / "configs" / "data.yaml"
    if configs_path.is_file():
        configs = load_configs(root)
        sources = locate_primary(root, configs["data"]["primary"])
        expected = read_json(folder / "input_manifest.json")
        if {source: file_hash(path) for source, path in sources.items()} != {r["source"]: r["sha256"] for r in expected}:
            raise RuntimeError("Raw primary inputs changed after Step 1. Rerun and inspect preparation before a later stage.")
        snapshot = read_json(folder / "configuration_snapshot.json")
        if any(configs[key] != snapshot[key] for key in ["default", "data", "evaluation"]):
            raise RuntimeError("The accepted data/evaluation contract changed. Rerun preparation before proceeding.")
        previous = Path(root) / configs["data"]["primary"]["previous_split_path"]
        current_previous_hash = file_hash(previous) if previous.is_file() else None
        if current_previous_hash != state.get("previous_split_sha256"):
            raise RuntimeError("Historical split input changed after acceptance; rerun preparation.")
    return state


def save_acceptance(root, state):
    write_json(Path(root) / "state" / "step01_accepted.json", state)
