"""Explicit configuration loading; later-model settings do not fit anything."""
from __future__ import annotations
import importlib.metadata
import sys
from pathlib import Path
from .artifacts import digest

PINS = {"numpy": "1.26.4", "scipy": "1.13.1", "PyYAML": "6.0.3", "gensim": "4.4.0",
        "jieba": "0.42.1", "stopwordsiso": "0.6.1", "psutil": "7.0.0",
        "smart_open": "8.0.2", "wrapt": "2.5.0", "setuptools": "80.9.0"}


def check_environment():
    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        raise RuntimeError("Use Python 3.10, 3.11 or 3.12; Python 3.10 is recommended for your project.")
    found, problems = {}, []
    for name, expected in PINS.items():
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
        if found[name] != expected:
            problems.append(f"{name}: installed={found[name]}, required={expected}")
    if problems:
        raise RuntimeError("Step 1 dependency pins not satisfied. Run python -m pip install -r requirements.txt\n" + "\n".join(problems))
    import unicodedata
    found["python_major_minor"] = ".".join(str(x) for x in sys.version_info[:2])
    found["unicode_data_version"] = unicodedata.unidata_version
    return found


def load_configs(root):
    import yaml
    configs = {}
    for path in sorted((Path(root) / "configs").glob("*.yaml")):
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Empty/invalid configuration: " + str(path))
        configs[path.stem] = value
    needed = {"default", "data", "evaluation", "acquisition", "models", "projection", "prediction", "resources", "ablations", "sensitivity"}
    if not needed.issubset(configs):
        raise ValueError("Missing configuration files: " + str(sorted(needed - configs.keys())))
    project, data, metrics = configs["default"]["project"], configs["data"]["primary"], configs["evaluation"]["coherence"]
    if project["topics"] != 10 or project["top_words"] != 10 or metrics["topn"] != 10:
        raise ValueError("The accepted R1 contract requires K=M=10.")
    if metrics["cv_window"] != 110 or metrics["npmi_window"] != 10 or metrics["processes"] != 1:
        raise ValueError("Coherence windows/processes differ from the accepted contract.")
    ratios = data["split_ratios"]
    if set(ratios) != {"fit", "development", "test"} or any(float(x) <= 0 for x in ratios.values()) or abs(sum(ratios.values()) - 1) > 1e-12:
        raise ValueError("Split ratios must be positive fit/development/test fractions summing to one.")
    if data["internal_folds"] != 3 or data["completion"]["minimum_in_vocabulary_tokens"] < 2:
        raise ValueError("Three folds and at least two completion tokens are required.")
    if not 0 < data["completion"]["observed_fraction"] < 1:
        raise ValueError("Completion observed fraction must lie strictly between zero and one.")
    vocab, phrase, near = data["vocabulary"], data["phrases"], data["near_duplicates"]
    if vocab["min_document_frequency"] < 1 or not 0 < vocab["max_document_fraction"] <= 1:
        raise ValueError("Invalid vocabulary frequency policy.")
    if phrase["scoring"] != "npmi" or not 0 < phrase["threshold"] < 1 or phrase["min_count"] < 1:
        raise ValueError("Invalid fitting-only phrase policy.")
    if not 0 < near["threshold"] <= 1 or not 0 < near["minimum_length_ratio"] <= 1 or near["shingle_size"] < 1:
        raise ValueError("Invalid near-duplicate policy.")
    acq = configs["acquisition"]["pilot"]
    if len(acq["word_group_support"]) * len(acq["pair_group_support"]) * len(acq["gamma"]) != acq["settings"] or acq["settings"] != 12:
        raise ValueError("The bounded acquisition grid must contain exactly twelve settings.")
    if configs["models"]["predictive_pilot"]["final_base_count"] != 1:
        raise ValueError("Keep one selected final proposed base after the two-base pilot.")
    return configs


def preparation_signature(configs, inputs, assets, versions, code_hash):
    """Signature excludes GPU/absolute paths and later model implementation files."""
    return digest({"contract": configs["default"], "data": configs["data"],
                   "evaluation": configs["evaluation"], "inputs": inputs,
                   "assets": assets, "versions": versions, "preparation_code_hash": code_hash})
