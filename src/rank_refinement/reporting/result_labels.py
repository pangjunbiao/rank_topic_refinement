"""Separate the reported method name from the underlying trained base."""
from __future__ import annotations

BASE_NAMES = {"lda": "LDA", "nmf": "NMF", "encot": "EnCOT", "neuromax": "NeuroMax"}
ROLE_NAMES = {"proposed": "Proposed", "display_only": "Display-only",
              "original_secondary": "Proposed (original secondary)", "l2": "Proposed (L2 control)"}


def result_identity(role, base, scope):
    if base not in BASE_NAMES:
        raise ValueError("Unknown underlying base: " + str(base))
    if role != "native" and role not in ROLE_NAMES:
        raise ValueError("Unknown result role: " + str(role))
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("A result requires an explicit experimental scope.")
    return {"model": BASE_NAMES[base] if role == "native" else ROLE_NAMES[role],
            "method_role": role, "underlying_base": BASE_NAMES[base], "base_key": base,
            "result_scope": scope}
