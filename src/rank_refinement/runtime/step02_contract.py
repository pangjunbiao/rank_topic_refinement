"""Step 2 additions leave the accepted Step 1 preparation contract unchanged."""
from __future__ import annotations
import importlib.metadata
from itertools import product

CORE_PINS = {"scikit-learn": "1.5.2", "joblib": "1.4.2", "threadpoolctl": "3.6.0"}
ANCHOR_PINS = {"bertopic": "0.17.3", "sentence-transformers": "3.4.1", "transformers": "4.51.3",
               "tokenizers": "0.21.1", "huggingface-hub": "0.30.2", "safetensors": "0.5.3",
               "pandas": "2.2.3", "umap-learn": "0.5.7", "hdbscan": "0.8.40", "numba": "0.61.2",
               "llvmlite": "0.44.0", "plotly": "5.24.1"}


def dependency_versions(pins, requirement):
    actual, errors = {}, []
    for name, version in pins.items():
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual[name] = None
        if actual[name] != version:
            errors.append(f"{name}: installed={actual[name]}, required={version}")
    if errors:
        raise RuntimeError(f"Install this stage's pins: python -m pip install -r {requirement}\n" + "\n".join(errors))
    return actual


def validate_contract(configs):
    p = configs["acquisition"]["pilot"]
    for key, required in {"word_group_support": [10, 20], "pair_group_support": [3, 5, 10], "gamma": [0.0, 0.5],
                          "eta": 0.25, "subsamples": 20, "group_fraction": 0.8, "empirical_quantile": 0.1,
                          "window": 10, "max_accepted_swaps_per_core": 10, "minimum_supported_nodes": 100,
                          "quantile_method": "linear", "association_epsilon": 1e-12,
                          "zero_marginal_signed_score": -1.0, "ubiquitous_joint_signed_score": 0.0,
                          "partition_restarts": 5, "partition_max_iterations": 100,
                          "every_core_positive_signed_cohesion": True,
                          "core_search": "global_signed_candidate_packing",
                          "core_objective": "signed_all_pairs_plus_internal_graph_strength"}.items():
        if p.get(key) != required:
            raise ValueError(f"Frozen Step 2 R2 contract mismatch: {key}. Revise explicitly in a separate protocol.")
    expected_search = {"diversity_draws": 128, "word_subset_fractions": [0.35, 0.5, 0.7, 0.9],
                       "repair_drop_counts": [1, 2], "solver_time_limit_seconds": 10.0,
                       "diagnostic_time_limit_seconds": 5.0, "solver_node_limit": 10000}
    if p.get("global_core_search") != expected_search:
        raise ValueError("Frozen fitting-only R2 search budget changed; declare a new protocol before scoring.")
    if configs["default"]["project"]["pilot_seeds"] != [809, 810]:
        raise ValueError("The bounded pilot requires seeds 809 and 810.")
    a = configs.get("acquisition_anchors")
    if not a or a["panel"] != ["lda", "nmf", "gsdmm", "bertopic"]:
        raise ValueError("All four prespecified screening anchors are required if a setting survives support.")
    for name in a["panel"]:
        if not 1 <= len(a["profiles"].get(name, [])) <= 2:
            raise ValueError("Anchor profiles must be frozen and bounded before scoring.")
    if a["encoder"]["repository"] != "BAAI/bge-small-zh-v1.5" or a["encoder"]["revision"] != "7999e1d3359715c523056ef9478215996d62a620":
        raise ValueError("The Chinese encoder asset changed; do not substitute an encoder after scoring.")
    from pathlib import Path
    from .artifacts import file_hash, read_json
    vendor = Path(__file__).resolve().parents[1] / "vendor"
    provenance = read_json(vendor / "gsdmm_provenance.json")
    if provenance["commit"] != "4ad1b6b6976743681ee4976b4573463d359214ee" or file_hash(vendor / "gsdmm_mgp.py") != provenance["upstream_file_sha256"]:
        raise ValueError("The pinned GSDMM implementation changed.")
    return True
