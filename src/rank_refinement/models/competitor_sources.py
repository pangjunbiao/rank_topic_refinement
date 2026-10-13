"""Seven pinned author modules, loaded in isolated packages.

Only two absolute import paths are relocated. Author objectives, priors,
normalizers and model parameters remain unchanged. No author launcher runs.
"""
from __future__ import annotations
import hashlib
import importlib.util
import sys
import types
import urllib.request
from pathlib import Path
from ..runtime.artifacts import atomic_bytes, file_hash, read_json

REQUIRED = {("glocom", "GloCOM.py"), ("glocom", "ECR.py"),
            ("fastopic", "_fastopic.py"), ("fastopic", "_ETP.py"),
            ("fastopic", "_model_utils.py"), ("combinedtm", "decoding_network.py"),
            ("combinedtm", "inference_network.py")}


def resolve_sources(root, log=None):
    root = Path(root)
    specs = read_json(root / "configs/upstream_competitor_sources.json")
    if len(specs) != 7 or {(s["method"], Path(s["path"]).name) for s in specs} != REQUIRED:
        raise ValueError("Require exactly the seven pinned competitor model modules.")
    folders = {}
    for spec in specs:
        if (spec["method"] not in {m for m, _ in REQUIRED}
                or len(spec["commit"]) != 40 or any(c not in "0123456789abcdef" for c in spec["commit"])
                or Path(spec["path"]).is_absolute() or ".." in Path(spec["path"]).parts):
            raise ValueError("Unsafe pinned competitor source.")
        folder = root / "data/cache/step04_sources" / spec["method"] / spec["commit"]
        folders[spec["method"]] = folder
        destination = folder / Path(spec["path"]).name
        if not destination.is_file():
            if log:
                log.info("AUTHOR SOURCE | %s | %s | pinned commit", spec["method"], destination.name)
            url = f"https://raw.githubusercontent.com/{spec['repository']}/{spec['commit']}/{spec['path']}"
            error = None
            for _ in range(3):
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "RankRefinement-ReproducibleResearch"})
                    with urllib.request.urlopen(request, timeout=30) as response:
                        payload = response.read()
                    if len(payload) != spec["bytes"] or hashlib.sha256(payload).hexdigest() != spec["sha256"]:
                        raise ValueError("Pinned competitor source checksum mismatch.")
                    atomic_bytes(destination, payload)
                    break
                except Exception as exc:
                    error = exc
            else:
                raise RuntimeError(f"Cannot fetch {spec['method']}/{destination.name}: {error}. Rerun to reuse verified downloads.")
        if destination.stat().st_size != spec["bytes"] or file_hash(destination) != spec["sha256"]:
            raise ValueError("Cached author source changed: " + str(destination))
    return folders, specs


def author_class(method, folder):
    folder = Path(folder).resolve()
    name = "rank_competitor_" + method + "_" + folder.name
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(folder)]
        sys.modules[name] = package
    modules = {"glocom": ("GloCOM", "GloCOM"), "fastopic": ("_fastopic", "fastopic"),
               "combinedtm": ("decoding_network", "DecoderNetwork")}
    module_name, class_name = modules[method]
    qualified = name + "." + module_name
    if qualified not in sys.modules:
        path = folder / (module_name + ".py")
        source = path.read_text(encoding="utf-8")
        if method == "glocom":
            old, new = "from models.GloCOM.ECR import ECR", "from .ECR import ECR"
        elif method == "combinedtm":
            old = "from contextualized_topic_models.networks.inference_network import ("
            new = "from .inference_network import ("
        else:
            old = new = None
        if old:
            if source.count(old) != 1:
                raise ValueError("Pinned import relocation is no longer unique.")
            source = source.replace(old, new, 1)
        module = types.ModuleType(qualified)
        module.__file__ = str(path)
        module.__package__ = name
        sys.modules[qualified] = module
        try:
            exec(compile(source, str(path), "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(qualified, None)
            raise
    return getattr(sys.modules[qualified], class_name)
