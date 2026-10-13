"""Fetch only hash-pinned author model modules; no author scripts or telemetry."""
from __future__ import annotations
import importlib.util
import sys
import types
import urllib.request
from pathlib import Path
from ..runtime.artifacts import atomic_bytes, file_hash, read_json


def resolve_sources(root, log=None):
    root = Path(root)
    specs = read_json(root / "configs/upstream_neural_sources.json")
    required = {("encot", "NewMethod.py"), ("encot", "ECR.py"), ("encot", "_ETP.py"),
                ("encot", "_model_utils.py"), ("neuromax", "NeuroMax.py"),
                ("neuromax", "ECR.py"), ("neuromax", "GR.py")}
    if {(r["base"], Path(r["path"]).name) for r in specs} != required or len(specs) != 7:
        raise ValueError("Author source manifest must contain exactly the seven required model modules.")
    folders = {}
    for item in specs:
        if item["base"] not in ["encot", "neuromax"] or len(item["commit"]) != 40:
            raise ValueError("Invalid pinned author source identity.")
        folder = root / "data/cache/neural_sources" / item["base"] / item["commit"]
        destination = folder / Path(item["path"]).name
        folders[item["base"]] = folder
        if not destination.is_file():
            if log:
                log.info("AUTHOR SOURCE | %s %s | pinned commit", item["base"], destination.name)
            url = "https://raw.githubusercontent.com/" + item["repository"] + "/" + item["commit"] + "/" + item["path"]
            last_error = None
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=30) as response:
                        payload = response.read()
                    import hashlib
                    if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
                        raise ValueError("Author source checksum mismatch.")
                    atomic_bytes(destination, payload)
                    break
                except Exception as exc:
                    last_error = exc
            else:
                raise RuntimeError("Cannot fetch pinned author module " + destination.name + ": " + str(last_error))
        if destination.stat().st_size != item["bytes"] or file_hash(destination) != item["sha256"]:
            raise ValueError("Cached author source changed: " + str(destination))
    return folders, specs


def author_class(base, folder):
    # Isolated package names permit the authors' relative imports without
    # importing their experiment launchers, datasets, wandb, or CLI code.
    name = "rank_author_" + base + "_" + Path(folder).name
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(folder).resolve())]
        sys.modules[name] = package
    module_name = "NewMethod" if base == "encot" else "NeuroMax"
    qualified = name + "." + module_name
    if qualified not in sys.modules:
        spec = importlib.util.spec_from_file_location(qualified, Path(folder) / (module_name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    return getattr(sys.modules[qualified], module_name)
