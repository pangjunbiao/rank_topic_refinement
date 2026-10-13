"""Secondary LP controls and an L2 QP on the original feasible set."""
from __future__ import annotations
import time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from .minimum_tv import certify


def constraint_system(native, targets, margin):
    native = np.asarray(native, dtype=np.float64)
    k, v = native.shape
    n = k * v
    scale = float(native.max())
    original = native / scale
    identity = sparse.eye(n, format="csr")
    zeros = sparse.csr_matrix((n, k))
    absolute = sparse.vstack([sparse.hstack([identity, -identity, zeros]),
                              sparse.hstack([-identity, -identity, zeros])], format="csr")
    rr, cc, values, rhs = [], [], [], []
    for topic, target in enumerate(targets):
        if target is None:
            continue
        selected = set(target)
        for word in range(v):
            row = len(rhs)
            rr.extend([row, row]); cc.extend([topic * v + word, 2 * n + topic])
            values.extend([-1., 1.] if word in selected else [1., -1.])
            rhs.append(-margin if word in selected else 0.)
    ranks = sparse.coo_matrix((values, (rr, cc)), shape=(len(rhs), 2 * n + k)).tocsr()
    aub = sparse.vstack([absolute, ranks], format="csr")
    bub = np.r_[original.ravel(), -original.ravel(), rhs]
    rows = sparse.kron(sparse.eye(k), np.ones((1, v)), format="csr")
    cols = sparse.hstack([sparse.eye(v)] * k, format="csr")[:v - 1]
    equal = sparse.vstack([rows, cols], format="csr")
    aeq = sparse.hstack([equal, sparse.csr_matrix((k + v - 1, n + k))], format="csr")
    beq = np.r_[original.sum(axis=1), original.sum(axis=0)[:-1]]
    return original, scale, aub, bub, aeq, beq


def _lp(c, aub, bub, aeq, beq, bounds, seconds):
    started, attempts = time.perf_counter(), []
    for method in ["highs", "highs-ipm"]:
        remaining = seconds - (time.perf_counter() - started)
        if remaining <= 0:
            break
        solution = linprog(c, A_ub=aub, b_ub=bub, A_eq=aeq, b_eq=beq, bounds=bounds,
            method=method, options={"time_limit": remaining, "primal_feasibility_tolerance": 1e-9,
                                    "dual_feasibility_tolerance": 1e-9})
        attempts.append({"method": method, "status": int(solution.status), "message": solution.message})
        if solution.success and solution.x.min() >= 0:
            return solution, attempts
        if solution.success:
            # L/U bounds can legitimately be negative. The caller independently
            # certifies affinities, which are the only nonnegative scientific data.
            return solution, attempts
        if solution.status != 4:
            return solution, attempts
    return None, attempts


def secondary_lp(native, targets, primary_tv, margin, slack, seconds,
                 mode="range", contexts=None, decoder_gain=None):
    original, scale, aub, bub, aeq, beq = constraint_system(native, targets, margin)
    k, v = original.shape
    n, variables = k * v, 2 * k * v + k
    cap = primary_tv + slack
    cap_row = sparse.csr_matrix((np.full(n, .5), (np.zeros(n, dtype=int), np.arange(n, 2 * n))), shape=(1, variables))
    aub = sparse.vstack([aub, cap_row], format="csr")
    bub = np.r_[bub, cap]
    bounds = [(0., None)] * variables
    c = np.zeros(variables)
    if mode == "index":
        c[:n] = np.arange(1, n + 1, dtype=float) / n
    elif mode == "range":
        contexts = np.asarray(contexts, dtype=float)
        decoder_gain = np.asarray(decoder_gain, dtype=float)
        if contexts.ndim != 2 or contexts.shape[1] != k or decoder_gain.shape != (v,):
            raise ValueError("Range LP requires fitting latents and affine decoder coefficients.")
        count = len(contexts)
        size = count * v
        row_ids = np.repeat(np.arange(size), k)
        word_ids = np.tile(np.arange(v), count)
        col_ids = np.tile(np.arange(k), size) * v + np.repeat(word_ids, k)
        coeff = (contexts[:, None, :] * (decoder_gain * scale)[None, :, None]).ravel()
        q = sparse.coo_matrix((coeff, (row_ids, col_ids)), shape=(size, variables)).tocsr()
        d = np.repeat(np.arange(count), v)
        upper = sparse.coo_matrix((-np.ones(size), (np.arange(size), count + d)), shape=(size, 2 * count)).tocsr()
        lower = sparse.coo_matrix((np.ones(size), (np.arange(size), d)), shape=(size, 2 * count)).tocsr()
        baseline = ((contexts @ original) * decoder_gain * scale).ravel()
        aub = sparse.vstack([sparse.hstack([aub, sparse.csr_matrix((aub.shape[0], 2 * count))]),
                             sparse.hstack([q, upper]), sparse.hstack([-q, lower])], format="csr")
        bub = np.r_[bub, baseline, -baseline]
        aeq = sparse.hstack([aeq, sparse.csr_matrix((aeq.shape[0], 2 * count))], format="csr")
        c = np.r_[c, -np.ones(count) / count, np.ones(count) / count]
        bounds += [(None, None)] * (2 * count)
    else:
        raise ValueError("Unknown secondary LP mode.")
    started = time.perf_counter()
    solved, attempts = _lp(c, aub, bub, aeq, beq, bounds, seconds)
    report = {"mode": mode, "success": False, "certified": False, "attempts": attempts,
              "primary_tv_scaled": primary_tv, "tv_cap_scaled": cap, "post_solve_clipping": False}
    if solved is None or not solved.success:
        report.update(seconds=time.perf_counter() - started, failure="secondary LP did not solve")
        return None, report
    result = solved.x[:n].reshape(k, v) * scale
    verified = certify(native, result, targets, margin=margin)
    if (not verified["nonnegative"] and verified["minimum_affinity_scaled"] >= -1e-8):
        # Resolve exactly the same LP using IPM if the first simplex solution
        # contains tiny negative affinities. No clipping or budget relaxation.
        remaining = seconds - (time.perf_counter() - started)
        if remaining > 0:
            retry = linprog(c, A_ub=aub, b_ub=bub, A_eq=aeq, b_eq=beq, bounds=bounds,
                method="highs-ipm", options={"time_limit": remaining, "primal_feasibility_tolerance": 1e-9,
                                            "dual_feasibility_tolerance": 1e-9})
            report["attempts"].append({"method": "highs-ipm-certificate-retry", "status": int(retry.status)})
            if retry.success:
                solved = retry
                result = retry.x[:n].reshape(k, v) * scale
                verified = certify(native, result, targets, margin=margin)
    success = verified["certified"] and verified["tv_scaled"] <= cap + 1e-7
    report.update(verified, success=bool(success), secondary_objective=float(solved.fun),
                  seconds=time.perf_counter() - started)
    if mode == "range":
        changes = (contexts @ (result - native)) * decoder_gain
        report["measured_fitting_logit_range_mean"] = float(np.ptp(changes, axis=1).mean())
        report["bound_covers"] = "selected fitting contexts before moment correction and backoff only"
    return (result if success else None), report


def l2_projection(native, targets, margin, seconds, eps):
    import osqp
    original, scale, aub, bub, aeq, beq = constraint_system(native, targets, margin)
    k, v = original.shape
    n = k * v
    keep = np.r_[np.arange(n), np.arange(2 * n, 2 * n + k)]
    rank = aub[2 * n:, keep]
    eq = aeq[:, keep]
    size = n + k
    constraints = sparse.vstack([sparse.eye(size), eq, rank], format="csc")
    low = np.r_[np.zeros(size), beq, np.full(rank.shape[0], -np.inf)]
    high = np.r_[np.full(size, np.inf), beq, bub[2 * n:]]
    p = sparse.diags(np.r_[np.ones(n), np.zeros(k)], format="csc")
    q = np.r_[-original.ravel(), np.zeros(k)]
    solver = osqp.OSQP()
    started = time.perf_counter()
    solver.setup(P=p, q=q, A=constraints, l=low, u=high, eps_abs=eps, eps_rel=eps,
                 max_iter=100000, time_limit=seconds, polishing=True, verbose=False)
    solved = solver.solve(raise_error=False)
    report = {"success": False, "certified": False, "objective": "half_squared_l2_scaled",
        "solver": "OSQP", "solver_status": solved.info.status, "solver_iterations": int(solved.info.iter),
        "original_feasible_set": True, "tv_cap_applied": False, "post_solve_clipping": False}
    if solved.x is None or solved.info.status_val != 1 or not np.isfinite(solved.x).all():
        report["seconds"] = time.perf_counter() - started
        return None, report
    candidate_scaled = solved.x[:n].reshape(k, v)
    result = candidate_scaled * scale
    verified = certify(native, result, targets, margin=margin)
    if not verified["certified"]:
        # QP arithmetic can put a coordinate just below zero. A separate LP
        # finds the nearest feasible point under the ORIGINAL totals and ranks.
        # This is an explicitly reported numerical repair, never clipping.
        remaining = seconds - (time.perf_counter() - started)
        repair_rhs = bub.copy()
        repair_rhs[:2 * n] = np.r_[candidate_scaled.ravel(), -candidate_scaled.ravel()]
        objective = np.r_[np.zeros(n), np.full(n, .5), np.zeros(k)]
        repair, attempts = _lp(objective, aub, repair_rhs, aeq, beq, [(0., None)] * (2 * n + k), max(remaining, .001))
        report["feasibility_repair_attempts"] = attempts
        if repair is not None and repair.success:
            repaired = repair.x[:n].reshape(k, v) * scale
            correction = float(np.abs(repaired / scale - candidate_scaled).sum())
            report["feasibility_repair_l1_scaled"] = correction
            if correction / max(original.sum(), 1.) <= 1e-5:
                result = repaired
                verified = certify(native, result, targets, margin=margin)
    report.update(verified, success=verified["certified"],
        l2_objective_scaled=float(.5 * ((result / scale - original) ** 2).sum()),
        numerical_control_is_approximate=bool("feasibility_repair_l1_scaled" in report),
        seconds=time.perf_counter() - started)
    return (result if verified["certified"] else None), report
