"""Globally conserved minimum-TV readout projection, with independent checks.

No claim of a unique primary-optimal solution is made. Input affinities are
converted to float64 before sums; the LP uses global-max scaling. No readout
clipping or row normalization follows optimization.
"""
from __future__ import annotations
import time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def certify(native, refined, targets, margin=1e-6, tolerance=1e-7,
            solver_tv_scaled=None, objective_tolerance=1e-5):
    a, b = np.asarray(native, dtype=np.float64), np.asarray(refined, dtype=np.float64)
    if a.ndim != 2 or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid native/projected matrix in certificate.")
    scale = float(a.max())
    if scale <= 0 or (a < 0).any():
        raise ValueError("Invalid native affinity scale.")
    scaled_a, scaled_b = a / scale, b / scale
    row_error = float(np.max(np.abs(scaled_b.sum(axis=1) - scaled_a.sum(axis=1))))
    column_error = float(np.max(np.abs(scaled_b.sum(axis=0) - scaled_a.sum(axis=0))))
    gap = min(float(scaled_b[t, target].min() - np.delete(scaled_b[t], target).max())
              for t, target in enumerate(targets) if target is not None)
    actual_sets = [np.lexsort((np.arange(a.shape[1]), -row))[:len(target)].tolist() if target is not None else None
                   for row, target in zip(b, targets)]
    realized = all(target is None or set(target) == set(actual) for target, actual in zip(targets, actual_sets))
    tv = float(0.5 * np.abs(scaled_b - scaled_a).sum())
    objective_error = None if solver_tv_scaled is None else abs(tv - solver_tv_scaled)
    passed = bool(b.min() >= 0.0 and row_error <= tolerance and column_error <= tolerance
                  and gap >= margin - 1e-8 and realized
                  and (objective_error is None or objective_error <= objective_tolerance))
    return {"certified": passed, "scale": scale, "nonnegative": bool(b.min() >= 0.0),
        "minimum_affinity_scaled": float(scaled_b.min()),
        "row_sum_max_abs_error_scaled": row_error, "column_sum_max_abs_error_scaled": column_error,
        "rank_gap_min_scaled": gap, "rank_margin_scaled": margin, "actual_top_sets_realized": realized,
        "tv_scaled": tv, "tv_original": tv * scale,
        "relative_l1_movement": float(np.abs(b - a).sum() / a.sum()),
        "solver_tv_scaled": solver_tv_scaled, "objective_error_scaled": objective_error,
        "target_sets": targets, "prediction_evaluated": False, "post_solve_clipping": False}


def minimum_tv(native, targets, margin=1e-6, seconds=30.0, tolerance=1e-7, objective_tolerance=1e-5):
    native = np.asarray(native, dtype=np.float64)
    if (native.ndim != 2 or len(targets) != native.shape[0] or not np.isfinite(native).all()
            or (native < 0).any() or native.max() <= 0 or (native.sum(axis=1) <= 0).any()
            or margin <= 0 or seconds <= 0 or not any(t is not None for t in targets)):
        raise ValueError("Invalid projection inputs.")
    k, v = native.shape
    for target in targets:
        if target is not None and (not 0 < len(target) < v or len(set(target)) != len(target)
                                   or any(not isinstance(w, (int, np.integer)) or w < 0 or w >= v for w in target)):
            raise ValueError("Invalid complete rank target.")
    scale = float(native.max())
    original = native / scale
    n = k * v
    # Variables: affinity, absolute movement, one threshold for each topic.
    identity = sparse.eye(n, format="csr")
    zeros = sparse.csr_matrix((n, k))
    abs_constraints = sparse.vstack([sparse.hstack([identity, -identity, zeros]),
                                     sparse.hstack([-identity, -identity, zeros])], format="csr")
    rr, cc, vv, rank_rhs = [], [], [], []
    for topic, target in enumerate(targets):
        if target is None:
            continue
        selected = set(target)
        for word in range(v):
            row = len(rank_rhs)
            rr.extend([row, row]); cc.extend([topic * v + word, 2 * n + topic])
            vv.extend([-1.0, 1.0] if word in selected else [1.0, -1.0])
            rank_rhs.append(-margin if word in selected else 0.0)
    rank_constraints = sparse.coo_matrix((vv, (rr, cc)), shape=(len(rank_rhs), 2 * n + k)).tocsr()
    inequalities = sparse.vstack([abs_constraints, rank_constraints], format="csr")
    rhs = np.r_[original.ravel(), -original.ravel(), rank_rhs]
    er, ec, ev, equalities = [], [], [], []
    for topic in range(k):
        row = len(equalities); equalities.append(float(original[topic].sum()))
        er.extend([row] * v); ec.extend(range(topic * v, (topic + 1) * v)); ev.extend([1.0] * v)
    # One redundant column equality is omitted in the solver but independently
    # checked, together with every other column, in the final certificate.
    for word in range(v - 1):
        row = len(equalities); equalities.append(float(original[:, word].sum()))
        er.extend([row] * k); ec.extend([t * v + word for t in range(k)]); ev.extend([1.0] * k)
    equality_matrix = sparse.coo_matrix((ev, (er, ec)), shape=(len(equalities), 2 * n + k)).tocsr()
    objective = np.r_[np.zeros(n), np.full(n, 0.5), np.zeros(k)]
    started = time.perf_counter()
    attempts = []
    report = {"success": False, "certified": False, "solver_status": 1,
              "solver_message": "Solver budget exhausted before a verified solution was available.",
              "prediction_evaluated": False, "target_sets": targets, "post_solve_clipping": False}
    for method in ["highs", "highs-ipm"]:
        remaining = seconds - (time.perf_counter() - started)
        if remaining <= 0:
            break
        solved = linprog(objective, A_ub=inequalities, b_ub=rhs,
                         A_eq=equality_matrix, b_eq=equalities, bounds=(0.0, None), method=method,
                         options={"time_limit": remaining, "dual_feasibility_tolerance": 1e-8,
                                  "primal_feasibility_tolerance": 1e-8})
        report = {"success": bool(solved.success), "certified": False, "solver_status": int(solved.status),
                  "solver_message": str(solved.message), "seconds": time.perf_counter() - started,
                  "prediction_evaluated": False, "target_sets": targets, "post_solve_clipping": False,
                  "solver_method": method}
        if not solved.success:
            attempts.append({"method": method, "solver_status": int(solved.status), "success": False})
            report["solver_attempts"] = attempts
            return None, report
        refined = solved.x[:n].reshape(k, v) * scale
        checked = certify(native, refined, targets, margin, tolerance, float(solved.fun), objective_tolerance)
        report.update(checked, primary_optimal=bool(solved.status == 0))
        attempts.append({"method": method, "solver_status": int(solved.status),
                         "success": True, "certified": checked["certified"],
                         "minimum_affinity_scaled": checked["minimum_affinity_scaled"]})
        report["solver_attempts"] = attempts
        if checked["certified"]:
            return refined, report
        # This is a numerical retry of exactly the same constraints/objective,
        # under the remaining 30-second budget, before any metric is inspected.
        # No clipping, constraint relaxation, or extra acquisition is performed.
        tiny_negative_only = (not checked["nonnegative"] and checked["minimum_affinity_scaled"] >= -1e-8
            and checked["row_sum_max_abs_error_scaled"] <= tolerance
            and checked["column_sum_max_abs_error_scaled"] <= tolerance
            and checked["rank_gap_min_scaled"] >= margin - 1e-8
            and checked["actual_top_sets_realized"] and checked["objective_error_scaled"] <= objective_tolerance)
        if method == "highs" and tiny_negative_only:
            continue
        break
    report.update(success=False, rejection="Projected readout failed independent numerical verification.",
                  seconds=time.perf_counter() - started, solver_attempts=attempts)
    return None, report
