"""Equivalent displacement LPs with exact separation of range constraints.

Affinities are never clipped. The primary TV objective, rank margin, native
row/column totals, fitting contexts and secondary objective are unchanged.
"""
from __future__ import annotations
import time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from .minimum_tv import certify


def displacement_system(native, targets, margin):
    a = np.asarray(native, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all() or (a < 0).any() or a.max() <= 0:
        raise ValueError("Invalid native readout.")
    k, v = a.shape
    if len(targets) != k or margin <= 0:
        raise ValueError("Invalid rank specification.")
    original, scale, n = a / a.max(), float(a.max()), a.size
    # delta = positive movement - negative movement; this removes cancellation
    # of large native logits from the range-LP right hand sides.
    identity = sparse.eye(n, format="csr")
    nonnegative = sparse.hstack([-identity, identity, sparse.csr_matrix((n, k))], format="csr")
    rr, cc, values, rhs = [], [], [], []
    for topic, target in enumerate(targets):
        if target is None:
            continue
        if not 0 < len(set(target)) == len(target) < v or any(w < 0 or w >= v for w in target):
            raise ValueError("Invalid complete rank target.")
        chosen = set(target)
        for word in range(v):
            row = len(rhs); j = topic * v + word
            sign = -1. if word in chosen else 1.
            rr.extend([row] * 3); cc.extend([j, n + j, 2 * n + topic]); values.extend([sign, -sign, -sign])
            rhs.append(original[topic, word] - margin if sign < 0 else -original[topic, word])
    rank = sparse.coo_matrix((values, (rr, cc)), shape=(len(rhs), 2 * n + k)).tocsr()
    aub = sparse.vstack([nonnegative, rank], format="csr")
    bub = np.r_[original.ravel(), rhs]
    rows = sparse.kron(sparse.eye(k), np.ones((1, v)), format="csr")
    columns = sparse.hstack([sparse.eye(v)] * k, format="csr")[:v - 1]
    totals = sparse.vstack([rows, columns], format="csr")
    aeq = sparse.hstack([totals, -totals, sparse.csr_matrix((totals.shape[0], k))], format="csr")
    return original, scale, aub, bub, aeq, np.zeros(totals.shape[0])


def solve_ipm(c, aub, bub, aeq, beq, bounds, seconds):
    return linprog(c, A_ub=aub, b_ub=bub, A_eq=aeq, b_eq=beq, bounds=bounds, method="highs-ipm",
        options={"time_limit": max(.001, seconds), "primal_feasibility_tolerance": 1e-9,
                 "dual_feasibility_tolerance": 1e-9, "ipm_optimality_tolerance": 1e-10})


def decode_movement(solution, original, scale):
    n = original.size
    return (original + (solution.x[:n] - solution.x[n:2 * n]).reshape(original.shape)) * scale


def primary_tv(native, targets, margin=1e-6, seconds=180.):
    original, scale, aub, bub, aeq, beq = displacement_system(native, targets, margin)
    n, k = original.size, original.shape[0]
    started = time.perf_counter()
    solved = solve_ipm(np.r_[np.full(2 * n, .5), np.zeros(k)], aub, bub, aeq, beq,
                       [(0., None)] * (2 * n + k), seconds)
    report = {"success": False, "certified": False, "solver": "highs-ipm_displacement",
        "status": int(solved.status), "message": solved.message, "seconds": time.perf_counter() - started,
        "same_primary_objective": True, "same_feasible_set": True, "post_solve_clipping": False}
    if not solved.success:
        return None, report
    result = decode_movement(solved, original, scale)
    check = certify(native, result, targets, margin=margin, solver_tv_scaled=float(solved.fun))
    report.update(check, success=check["certified"])
    return (result if report["success"] else None), report


def index_secondary(native, targets, primary_value, margin=1e-6, slack=1e-5, seconds=180.):
    original, scale, aub, bub, aeq, beq = displacement_system(native, targets, margin)
    n, k = original.size, original.shape[0]
    cap = primary_value + slack
    aub = sparse.vstack([aub, sparse.csr_matrix(np.r_[np.full(2 * n, .5), np.zeros(k)][None, :])], format="csr")
    bub = np.r_[bub, cap]
    weights = np.arange(1, n + 1, dtype=float) / n
    started = time.perf_counter()
    solved = solve_ipm(np.r_[weights, -weights, np.zeros(k)], aub, bub, aeq, beq,
                       [(0., None)] * (2 * n + k), seconds)
    report = {"success": False, "certified": False, "solver": "highs-ipm_displacement",
        "mode": "index", "status": int(solved.status), "message": solved.message,
        "tv_cap_scaled": cap, "seconds": time.perf_counter() - started, "same_feasible_set": True}
    if not solved.success:
        return None, report
    result = decode_movement(solved, original, scale)
    check = certify(native, result, targets, margin=margin)
    success = check["certified"] and check["tv_scaled"] <= cap + 1e-7
    report.update(check, success=bool(success), secondary_objective=float((result / scale).ravel() @ weights))
    return (result if success else None), report


def range_secondary(native, targets, primary_value, contexts, decoder_gain, primary_readout,
                    margin=1e-6, slack=1e-5, seconds=180., maximum_iterations=50,
                    separation_tolerance=1e-8, log=None):
    original, scale, aub, bub, aeq, beq = displacement_system(native, targets, margin)
    contexts, gain = np.asarray(contexts, dtype=float), np.asarray(decoder_gain, dtype=float)
    k, v = original.shape
    if contexts.ndim != 2 or contexts.shape[1] != k or len(contexts) == 0 or gain.shape != (v,):
        raise ValueError("Invalid fitting-only range contexts.")
    n, count, variables = original.size, len(contexts), 2 * original.size + k
    if not np.isfinite(contexts).all() or not np.isfinite(gain).all():
        raise ValueError("Nonfinite range input.")
    cap = primary_value + slack
    aub = sparse.vstack([aub, sparse.csr_matrix(np.r_[np.full(2 * n, .5), np.zeros(k)][None, :])], format="csr")
    bub = np.r_[bub, cap]
    base = sparse.hstack([aub, sparse.csr_matrix((aub.shape[0], 2 * count))], format="csr")
    equality = sparse.hstack([aeq, sparse.csr_matrix((aeq.shape[0], 2 * count))], format="csr")
    objective = np.r_[np.zeros(variables), -np.ones(count) / count, np.ones(count) / count]
    bounds = [(0., None)] * variables + [(None, None)] * (2 * count)
    initial = (contexts @ (primary_readout - native)) * gain
    upper_words = [{int(w)} for w in np.argmax(initial, axis=1)]
    lower_words = [{int(w)} for w in np.argmin(initial, axis=1)]
    started, history = time.perf_counter(), []
    report = {"success": False, "certified": False, "mode": "range",
        "solver": "highs-ipm_exact_constraint_generation", "same_secondary_objective": True,
        "same_feasible_set": True, "tv_cap_scaled": cap, "post_solve_clipping": False,
        "full_range_constraints_count": 2 * count * v}
    for iteration in range(1, maximum_iterations + 1):
        remaining = seconds - (time.perf_counter() - started)
        if remaining <= 0:
            break
        rr, cc, values, rhs = [], [], [], []
        for d in range(count):
            for sign, words, bound in [(1., upper_words[d], variables + count + d),
                                       (-1., lower_words[d], variables + d)]:
                for word in sorted(words):
                    row = len(rhs)
                    coeff = contexts[d] * gain[word] * scale * sign
                    columns = np.arange(k) * v + word
                    rr.extend([row] * (2 * k + 1)); cc.extend(columns.tolist() + (n + columns).tolist() + [bound])
                    values.extend(coeff.tolist() + (-coeff).tolist() + [-sign]); rhs.append(0.)
        cuts = sparse.coo_matrix((values, (rr, cc)), shape=(len(rhs), variables + 2 * count)).tocsr()
        solved = solve_ipm(objective, sparse.vstack([base, cuts], format="csr"), np.r_[bub, rhs],
                           equality, beq, bounds, remaining)
        item = {"iteration": iteration, "active_range_constraints": len(rhs), "status": int(solved.status),
                "seconds_total": time.perf_counter() - started}
        if not solved.success:
            item["message"] = solved.message; history.append(item)
            break
        result = decode_movement(solved, original, scale)
        changes = (contexts @ (result - native)) * gain
        minimum, maximum = changes.min(axis=1), changes.max(axis=1)
        low, high = solved.x[variables:variables + count], solved.x[variables + count:]
        violation = float(max(0., np.max(low - minimum), np.max(maximum - high)))
        actual_objective = float(np.mean(maximum - minimum))
        item.update(lower_bound=float(solved.fun), full_objective=actual_objective,
                    full_constraint_max_violation=violation)
        history.append(item)
        if log:
            log.info("RANGE SEPARATION | iteration=%d | active=%d/%d | gap=%.3g | seconds=%.2f",
                     iteration, len(rhs), 2 * count * v, actual_objective - solved.fun, item["seconds_total"])
        if violation <= separation_tolerance:
            check = certify(native, result, targets, margin=margin)
            success = (check["certified"] and check["tv_scaled"] <= cap + 1e-7
                       and abs(actual_objective - solved.fun) <= max(1e-7, 2 * separation_tolerance))
            report.update(check, success=bool(success), history=history, seconds=time.perf_counter() - started,
                active_range_constraints=len(rhs), secondary_objective=actual_objective,
                master_lower_bound=float(solved.fun), optimality_gap=abs(actual_objective - float(solved.fun)),
                full_constraint_max_violation=violation,
                bound_covers="same 64 fitting contexts before correction and backoff only")
            return (result if success else None), report
        added = 0
        for d in range(count):
            word = int(np.argmax(changes[d]))
            if maximum[d] > high[d] + separation_tolerance and word not in upper_words[d]:
                upper_words[d].add(word); added += 1
            word = int(np.argmin(changes[d]))
            if low[d] > minimum[d] + separation_tolerance and word not in lower_words[d]:
                lower_words[d].add(word); added += 1
        if not added:
            report["failure"] = "full-range separation stalled at numerical tolerance"
            break
    report.update(history=history, seconds=time.perf_counter() - started,
                  failure=report.get("failure", "range solver exhausted the declared numerical budget"))
    return None, report
