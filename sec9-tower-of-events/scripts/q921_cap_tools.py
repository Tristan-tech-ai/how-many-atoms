"""q921 (7 Oct 2026): solvers on top of q920_cap_flint.Cap (instrument for T61, recorded in an amendment).
newton_fixed_c: Newton at fixed centre mass c on the unknowns (u, w, we) with the full Jacobian of every iteration and a damped step (feasibility and decrease of the 2-norm).
born_centre: at fixed A, from a K 60 state (c = 0) raise c by predictor (dv/dc from the Jacobian) and corrector steps, taking the Newton step in c on G0(c) = D(0) - D(A) whenever it is shorter than the adaptive step, until |G0| < 1e-30.
newton_full: Newton on all unknowns (u, w, we, wc) of the odd system with a damped step.
step_in_A: predictor du/dA = -J^-1 b and corrector newton_full at a new A (a new Cap)."""

import time, copy
from flint import arb, arb_mat
from q920_cap_flint import Cap, col, vec, maxabs, norm2


def solve_lin(Jsub, rhs):
    n = Jsub.nrows()
    return vec(Jsub.solve(col(rhs)), n)


def newton_fixed_c(cap, vfix, c, say, tol=arb("1e-38"), maxit=12):
    n = cap.n
    v = list(vfix)
    for it in range(maxit):
        res, J, _ = cap.res_jac(v + [c], True, False)
        r = res[:n]
        rn = norm2(r)
        if maxabs(r) < tol:
            return v, res[n], True
        Jsub = arb_mat(n, n, [J[i, k] for i in range(n) for k in range(n)])
        dv = solve_lin(Jsub, [-t for t in r])
        lam = arb(1)
        acc = False
        for _ in range(14):
            vn = [(v[i] + lam * dv[i]).mid() for i in range(n)]
            if cap.feasible(vn + [c] if c > 0 else vn + [arb(1)]):
                rr, _, _ = cap.res_jac(vn + [c], False, False)
                if norm2(rr[:n]) < rn:
                    v = vn
                    acc = True
                    break
            lam = lam / 2
        if not acc:
            return v, res[n], False
    res, _, _ = cap.res_jac(v + [c], False, False)
    return v, res[n], maxabs(res[:n]) < tol


def born_centre(cap, v0, say, dc0=arb("1e-5"), dcmax=arb("1e-3"), cmax=arb("0.05")):
    """v0 = (u, w, we) of the K 60 state, c = 0. Returns (v, c) with G0(c) = 0."""
    n = cap.n
    v = list(v0)
    c = arb(0)
    dc = dc0
    while True:
        res, J, _ = cap.res_jac(v + [c if c > 0 else arb(0)], True, False)
        g0 = res[n]
        Jsub = arb_mat(n, n, [J[i, k] for i in range(n) for k in range(n)])
        dvdc = solve_lin(Jsub, [-J[i, n] for i in range(n)])
        dG = J[n, n] + sum((J[n, k] * dvdc[k] for k in range(n)), arb(0))
        say(
            f"c = {c.str(8, radius=False)}: G0 = {g0.str(6, radius=False)}, dG0/dc = {dG.str(6, radius=False)}, step bound {dc.str(3, radius=False)}"
        )
        if abs(g0) < arb("1e-30") and c > 0:
            return v, c
        dcn = (-g0 / dG).mid()
        newton_short = bool(abs(dcn) <= dc)
        step = dcn if newton_short else (dc if dcn > 0 else -dc)
        cn = c + step
        vp = [(v[i] + step * dvdc[i]).mid() for i in range(n)]
        vn, gn, ok = newton_fixed_c(cap, vp, cn, say)
        if not ok:
            dc = dc / 2
            say(
                f"   corrector failed at c = {cn.str(8, radius=False)}; step bound -> {dc.str(3, radius=False)}"
            )
            if dc < arb("1e-14"):
                raise RuntimeError("born_centre: step below 1e-14")
            continue
        v, c = vn, cn
        if (not newton_short) and dc < dcmax:
            dc = min(dc * arb("1.6"), dcmax)
        if c > cmax:
            raise RuntimeError("born_centre: c above cmax")


def newton_full(cap, v, say, tol=arb("1e-36"), maxit=40):
    """Newton on all unknowns with FULL steps whenever the next point is feasible (the norm of the residual may rise for a few iterations: the monotone line search of the first version rejected steps that converge); steps are halved only for feasibility; a rise of the norm above 1e4 times the initial one or nan counts as failure."""
    nn = cap.nn
    rn0 = None
    for it in range(maxit):
        res, J, _ = cap.res_jac(v, True, False)
        rn = norm2(res)
        if maxabs(res) < tol:
            return v, res, True
        if rn0 is None:
            rn0 = rn
        if not (rn < 1e4 * rn0):
            return v, res, False
        dv = solve_lin(J, [-t for t in res])
        lam = arb(1)
        acc = False
        for _ in range(14):
            vn = [(v[i] + lam * dv[i]).mid() for i in range(nn)]
            if cap.feasible(vn):
                v = vn
                acc = True
                break
            lam = lam / 2
        if not acc:
            return v, res, False
    res, _, _ = cap.res_jac(v, False, False)
    return v, res, maxabs(res) < tol


def step_in_A(cap_old, v, A_new_str, say):
    """predictor du/dA at (cap_old, v) and corrector at A_new; returns (cap_new, v_new, res, ok)"""
    res, J, b = cap_old.res_jac(v, True, True)
    du = solve_lin(J, [-t for t in b])
    dA = arb(A_new_str) - cap_old.A
    cap_new = copy.copy(cap_old)
    cap_new.A = arb(A_new_str)
    vp = [(v[i] + dA * du[i]).mid() for i in range(cap_old.nn)]
    vn, rn, ok = newton_full(cap_new, vp, say)
    return cap_new, vn, rn, ok
