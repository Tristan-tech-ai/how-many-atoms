"""Q486: least-favourable priors for the bounded normal mean (squared error), SYMMETRIC continuation with the two
centre transitions handled explicitly, so that every transition is located by bisection on a sign condition.

WHY. q484 (exchange algorithm) split atoms beyond m = 6.9; q485 (Newton after L-BFGS) converged between transitions but
accumulated junk atoms at the centre births beyond m = 13.7. The capacity problem taught the structure (Barletta-Dytso
arXiv 2609.10039 for the Gaussian channel; q471 reproduced their 1.6659 and 2.9075): the support changes only at the
centre, by one atom, in two ways - an atom APPEARS at 0 when the KKT function at 0 reaches the level r, and the centre
atom SPLITS into a pair when the KKT function's curvature at 0 changes sign. q484/q485 show the same centre births here.
Here that structure is used, not assumed: an off-support scan of R - r on the whole interval is printed at every step.

STATE. Atoms +-a_j (j = 1..J, a_J = m pinned) with weight w_j each, and optionally a centre atom with weight c.
Unknowns: a_1..a_{J-1}, w_1..w_J, [c]. Equations: R(a_j) = R(m) (j < J), R'(a_j) = 0 (j < J), [R(0) = R(m)],
2 sum w + c = 1. K = 2J (+1). R, R' from q484's Prob (trapezoid step 0.01), R'' analytic:
R''(t) = int [2 phi - 4 e phi u + e^2 phi (u^2 - 1)] dx, e = delta(x) - t, u = x - t.
CONTINUATION in m by DM: scale, Newton (central-difference Jacobian, damped) to max |F| < 1e-13, then
  no centre: s0 = R(0) - r; if s0 > 0 bisect m for s0 = 0 with the no-centre family, record the transition,
             add the centre atom (weight 1e-3) and re-solve;
  centre:    c must stay positive; s2 = R''(0); if s2 > 0 bisect m for s2 = 0, record the split, replace the
             centre by +-0.02 with weight c/2 each and re-solve.
PRE-REGISTERED: (C) the first transition (2 -> 3, centre appears) at 1.056744 +- 1e-4 (q484's bisection, Johnstone
1.057). (M) transitions m_K for K to m = MMAX; every accepted state has max |F| < 1e-12 and an off-support scan
max(R - r) <= 1e-10 away from the atoms, else it is flagged. Fits on the transition locations over m >= 8:
K = c m^(4/3) + b, c m + b, c m log m + b, and the free exponent; the bulk profile rho^3 against distance d
from the edge is reported at the largest m for the comparison with capacity's rho^3 = b d, b = 0.018997722.
v2 (03:10): a new family starts at m_t + 1e-3 from a scan of seeds (centre weight 1e-4..1e-2; pair +-0.01..0.4), every
step that fails to converge is split into 4, 16, 64 substeps, and no unconverged state may trigger a transition
(v1 seeded the split pair at +-0.02 at the grid point and ran away after the first split).
v3 (03:35): the split family failed to start at 8.3535 from seeds 1e-3 past the transition; now offsets 1e-3..5e-2 past
m_t, pair seeds 0.01..0.7, an L-BFGS fallback, and RESUME=1 continues from the last row of the json.
v4 (04:05): at 10.4496 the appear is near-degenerate (R''(0) = -3e-7) and the full Newton stalls; fallback with the centre
weight c as the parameter: inner Newton at fixed c, outer root of R(0) - R(m) in c.
v5 (04:10): split fallback with the inner pair position eps as the parameter (inner Newton without R'(a_1) = 0,
outer root of R'(eps)), after the split at 15.017851 failed to start.
v6 (05:05): the new family is solved directly at the target grid point by the natural parameter (after the split at
18.678365 jumped branches when stepped from m_t + 1e-3).
v7 (05:25): a young split family is marched to the grid point through m_t + 1e-3, 3e-3, 1e-2, ... with eps predicted
by sqrt(m - m_t) and re-rooted at each point (v6's direct solve at the grid point still failed at 18.678365).
v8 (05:40): the young split family is continued in eps with m as an unknown (v7's march in m failed at m_t + 1e-2).
v9 (05:55): the young appear family is continued in the centre weight c with m as an unknown (the appear at 19.270655
failed to start by v8's direct solve).
v10 (06:30): acceptance residual TOLR (env, default 1e-10) in place of 1e-11, after a plain step stopped at 3.7e-11.
Usage: py q486_lfp_sym.py [MMAX] [DM]"""

import sys, time, json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
say = lambda *a: print(*a, flush=True)
_argv = sys.argv
sys.argv = [sys.argv[0]]
import q484_lfp_bnm as Q

sys.argv = _argv
MMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
DM = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
import os

TOLR = float(
    os.environ.get("TOLR", "1e-10")
)  # v10: acceptance residual (was 1e-11; the double floor reached 3.7e-11 at m = 20.45)


def full(a, w, c):
    th = np.concatenate([-a[::-1], a])
    ww = np.concatenate([w[::-1], w])
    if c is not None:
        th = np.concatenate([-a[::-1], [0.0], a])
        ww = np.concatenate([w[::-1], [c], w])
    return th, ww


def R2(P, t, d):
    u = P.x - t
    e = d - t
    phi = Q.S2P * np.exp(-0.5 * u * u)
    return Q.H * (2 * phi - 4 * e * phi * u + e * e * phi * (u * u - 1)).sum()


def resid(P, m, v, J, centre):
    a = np.concatenate([v[: J - 1], [m]])
    w = v[J - 1 : 2 * J - 1]
    c = v[2 * J - 1] if centre else None
    th, ww = full(a, w, c)
    d = P.rule(th, ww)
    pts = np.concatenate([a, [0.0]]) if centre else a
    R, dR = P.risk(pts, d)
    F = list(R[: J - 1] - R[J - 1]) + list(dR[: J - 1])
    if centre:
        F.append(R[J] - R[J - 1])
    F.append(2 * w.sum() + (c if centre else 0.0) - 1.0)
    return np.array(F), d, th, ww


def jac_analytic(P, m, v, J, centre):
    """v14: the Jacobian of resid in closed form on the same trapezoid grid. The finite-difference Jacobian carries
    about 1e-10 noise per entry, and with cond 1e10 to 4e12 (m >= 20; the weak direction is the innermost atoms,
    where R is flat to 1e-8) Newton stalled at res 1e-9 (23.30 -> 23.35). d delta/d w_k = phi_k (theta_k - delta)/Z,
    d delta/d theta_k = pi_k [1 + (x - theta_k)(theta_k - delta)]; a_j moves the pair +-a_j, w_j weighs both;
    dR(t)/dp = int 2 e d_p delta phi(x - t), dR'(t)/dp = int d_p delta (2 e u - 2) phi(x - t), plus R'(a_j) and
    R''(a_j) for the evaluation point t = a_j itself."""
    a = np.concatenate([v[: J - 1], [m]])
    w = v[J - 1 : 2 * J - 1]
    c = v[2 * J - 1] if centre else None
    th, ww = full(a, w, c)
    cen = 1 if centre else 0
    Ph = P.phi(th)
    Z = ww @ Ph
    d = ((ww * th) @ Ph) / Z
    g = th[:, None] - d[None, :]
    dw = Ph * g / Z  # d delta / d w_k
    dth = (ww[:, None] * Ph / Z) * (1 + (P.x[None, :] - th[:, None]) * g)  # d delta / d theta_k
    ip = lambda j: J - 1 + j + cen  # index of +a_j in th (j = 1..J); -a_j is J - j
    D = [dth[ip(j)] - dth[J - j] for j in range(1, J)] + [dw[ip(j)] + dw[J - j] for j in range(1, J + 1)]
    if centre:
        D.append(dw[J])
    D = np.array(D)  # (n, nx)
    pts = np.concatenate([a, [0.0]]) if centre else a
    u = P.x[None, :] - pts[:, None]
    e = d[None, :] - pts[:, None]
    ph = Q.S2P * np.exp(-0.5 * u * u)
    GR = Q.H * (D @ (2 * e * ph).T)  # (n, nT): dR(t_i)/dp
    GD = Q.H * (D @ ((2 * e * u - 2) * ph).T)  # dR'(t_i)/dp
    dR = Q.H * ((-2 * e + e * e * u) * ph).sum(1)
    d2R = Q.H * ((2 - 4 * e * u + e * e * (u * u - 1)) * ph).sum(1)
    n = len(v)
    Jm = np.zeros((n, n))
    for i in range(J - 1):  # rows R(a_i) - R(m), then R'(a_i)
        Jm[i] = GR[:, i] - GR[:, J - 1]
        Jm[i, i] += dR[i]
        Jm[J - 1 + i] = GD[:, i]
        Jm[J - 1 + i, i] += d2R[i]
    k = 2 * J - 2
    if centre:
        Jm[k] = GR[:, J] - GR[:, J - 1]
        k += 1
    Jm[k, J - 1 : 2 * J - 1] = 2.0
    if centre:
        Jm[k, 2 * J - 1] = 1.0
    return Jm


AJAC = os.environ.get("AJAC", "1") == "1"


def newton(P, m, v, J, centre, steps=50):
    F, *_ = resid(P, m, v, J, centre)
    nF = np.abs(F).max()
    for it in range(steps):
        if nF < 1e-13:
            break
        n = len(v)
        if AJAC:
            Jm = jac_analytic(P, m, v, J, centre)
        else:
            Jm = np.zeros((n, n))
            for j in range(n):
                h = 1e-6 if j < J - 1 else 1e-7
                vp = v.copy()
                vp[j] += h
                vm = v.copy()
                vm[j] -= h
                Jm[:, j] = (resid(P, m, vp, J, centre)[0] - resid(P, m, vm, J, centre)[0]) / (2 * h)
        try:
            dv = np.linalg.solve(Jm, F)
        except np.linalg.LinAlgError:
            dv = np.linalg.lstsq(Jm, F, rcond=None)[0]
        lam = 1.0
        ok = False
        while lam > 1e-5:
            vn = v - lam * dv
            a = np.concatenate([vn[: J - 1], [m]])
            w = vn[J - 1 : 2 * J - 1]
            good = (
                np.all(np.diff(np.concatenate([[0.0], a])) > 1e-7)
                and w.min() > 0
                and (not centre or vn[-1] > 0)
            )
            if good:
                Fn = resid(P, m, vn, J, centre)[0]
                if np.abs(Fn).max() < nF:
                    v, F, nF, ok = vn, Fn, np.abs(Fn).max(), True
                    break
            lam /= 2
        if not ok:
            break
    return v, nF


def signals(P, m, v, J, centre):
    F, d, th, ww = resid(P, m, v, J, centre)
    Ra, _ = P.risk(th, d)
    r = ww @ Ra
    R0, _ = P.risk(np.array([0.0]), d)
    grid = np.linspace(0, m, 4001)
    Rg, _ = P.risk(grid, d)
    a = np.abs(th)
    far = np.array([np.min(np.abs(a - g)) > 0.05 for g in grid])
    off = (Rg - r)[far].max() if far.any() else -1.0
    return R0[0] - r, R2(P, 0.0, d), off, r


def solve_at(m, v, J, centre):
    P = Q.Prob(m)
    v = v.copy()
    v[: J - 1] = v[: J - 1] * 1.0
    v, nF = newton(P, m, v, J, centre)
    if nF > 1e-12 and AJAC:
        # v14c: Levenberg-Marquardt with the analytic Jacobian when Newton stalls in the flat central directions
        # (23.35: Newton 6e-9, LM 9.6e-13 in 21 evaluations from the same point)
        from scipy.optimize import least_squares

        a_ok = (
            lambda vv: np.all(np.diff(np.concatenate([[0.0], vv[: J - 1], [m]])) > 1e-7)
            and vv[J - 1 :].min() > 0
        )
        try:
            sol = least_squares(
                lambda vv: resid(P, m, vv, J, centre)[0],
                v,
                jac=lambda vv: jac_analytic(P, m, vv, J, centre),
                method="lm",
                xtol=1e-15,
                ftol=1e-15,
                gtol=1e-15,
                max_nfev=400,
            )
            nL = np.abs(resid(P, m, sol.x, J, centre)[0]).max()
            if nL < nF and a_ok(sol.x):
                v2, n2 = newton(P, m, sol.x.copy(), J, centre)
                # v14d: LM lands on local minima of |F|^2 with res near 1e-9 (19.30 -> 19.35: c 0.0256, a1 0.7885, res
                # 9.3e-10, accepted under TOLR 1e-9; the true state has c 0.0350, a1 0.8846). Accept the LM point only
                # when Newton from it converges (res < 1e-13); otherwise the step fails and the caller subdivides.
                if min(n2, nL) < 1e-13:
                    v, nF = (v2, n2) if n2 < nL else (sol.x, nL)
        except Exception:
            pass
    return P, v, nF


def rescale(v, J, m_old, m_new):
    v = v.copy()
    v[: J - 1] *= m_new / m_old
    return v


def bisect(fun, lo, hi, tol=1e-6):
    flo = fun(lo)
    for _ in range(60):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        fm = fun(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def advance(m0, v0, J, centre, m1, nsub=1):
    """continue a converged family from m0 to m1 in nsub equal substeps; returns (v, nF) at m1"""
    v = v0.copy()
    mm = m0
    for k in range(nsub):
        mn = m0 + (m1 - m0) * (k + 1) / nsub
        P, v, nF = solve_at(mn, rescale(v, J, mm, mn), J, centre)
        mm = mn
        if nF > TOLR:
            return v, nF
    return v, nF


def robust_advance(m0, v0, J, centre, m1):
    for nsub in (1, 4, 16, 64):
        v, nF = advance(m0, v0, J, centre, m1, nsub)
        if nF <= TOLR:
            return v, nF
    return v, nF


def resid_fixc(P, m, v, J, c):
    """centre weight c held fixed: unknowns a_1..a_{J-1}, w_1..w_J; the equation R(0) = R(m) is left out"""
    a = np.concatenate([v[: J - 1], [m]])
    w = v[J - 1 : 2 * J - 1]
    th, ww = full(a, w, c)
    d = P.rule(th, ww)
    R, dR = P.risk(np.concatenate([a, [0.0]]), d)
    F = np.array(list(R[: J - 1] - R[J - 1]) + list(dR[: J - 1]) + [2 * w.sum() + c - 1.0])
    return F, R[J] - R[J - 1]


def newton_fixc(P, m, v, J, c, steps=60):
    F, s = resid_fixc(P, m, v, J, c)
    nF = np.abs(F).max()
    for it in range(steps):
        if nF < 1e-14:
            break
        n = len(v)
        Jm = np.zeros((n, n))
        for j in range(n):
            h = 1e-6 if j < J - 1 else 1e-7
            vp = v.copy()
            vp[j] += h
            vm = v.copy()
            vm[j] -= h
            Jm[:, j] = (resid_fixc(P, m, vp, J, c)[0] - resid_fixc(P, m, vm, J, c)[0]) / (2 * h)
        dv = np.linalg.lstsq(Jm, F, rcond=None)[0]
        lam = 1.0
        ok = False
        while lam > 1e-5:
            vn = v - lam * dv
            a = np.concatenate([vn[: J - 1], [m]])
            if np.all(np.diff(np.concatenate([[0.0], a])) > 1e-7) and vn[J - 1 :].min() > 0:
                Fn, sn = resid_fixc(P, m, vn, J, c)
                if np.abs(Fn).max() < nF:
                    v, F, s, nF, ok = vn, Fn, sn, np.abs(Fn).max(), True
                    break
            lam /= 2
        if not ok:
            break
    return v, nF, s


def appear_by_c(mt, a0, w0, J, ms):
    """centre weight c as the parameter at fixed m = ms: inner Newton at fixed c, outer root of s(c) = R(0) - R(m).
    v9: c marched up by x1.5 with a WARM start from the previous c (cold starts failed for c >= 2e-3 at 19.270655).
    """
    P = Q.Prob(ms)
    v = np.concatenate([a0 * ms / mt, w0 * (1 - 1e-7) / (2 * w0.sum())])
    c = 1e-7
    v, nF, s = newton_fixc(P, ms, v, J, c)
    if s <= 0 or nF > TOLR:
        return None
    lo = (c, v, s)
    hi = None
    while c < 0.3:
        c2 = c * 1.5
        v2 = v.copy()
        v2[J - 1 :] = v[J - 1 :] * (1 - c2) / (1 - c)
        v2, n2, s2 = newton_fixc(P, ms, v2, J, c2)
        if n2 > TOLR:
            return None
        if s2 <= 0:
            hi = (c2, v2, s2)
            break
        c, v, s = c2, v2, s2
        lo = (c, v, s)
    if hi is None:
        return None
    (cl, vl, sl), (ch, vh, sh) = lo, hi
    for _ in range(80):
        cm = 0.5 * (cl + ch)
        vm = vl.copy()
        vm[J - 1 :] = vl[J - 1 :] * (1 - cm) / (1 - cl)
        vm, nm, sm = newton_fixc(P, ms, vm, J, cm)
        if nm > TOLR:
            break
        if sm > 0:
            cl, vl, sl = cm, vm, sm
        else:
            ch, vh, sh = cm, vm, sm
        if ch - cl < 1e-15 + 1e-11 * ch:
            break
    c = 0.5 * (cl + ch)
    v = vl.copy()
    v[J - 1 :] = vl[J - 1 :] * (1 - c) / (1 - cl)
    v, nF, s = newton_fixc(P, ms, v, J, c)
    vv = np.concatenate([v, [c]])
    P, vv2, nF2 = solve_at(ms, vv, J, True)
    return (vv2, nF2) if nF2 <= TOLR else ((vv, max(nF, abs(s))) if max(nF, abs(s)) <= TOLR else None)


def resid_fixeps(P, m, u, J2, eps):
    """split family with the inner pair position a_1 = eps held fixed; unknowns a_2..a_{J2-1}, w_1..w_{J2};
    the equation R'(a_1) = 0 is left out and returned as the outer function"""
    a = np.concatenate([[eps], u[: J2 - 2], [m]])
    w = u[J2 - 2 : 2 * J2 - 2]
    th, ww = full(a, w, None)
    d = P.rule(th, ww)
    R, dR = P.risk(a, d)
    F = np.array(list(R[: J2 - 1] - R[J2 - 1]) + list(dR[1 : J2 - 1]) + [2 * w.sum() - 1.0])
    return F, dR[0]


def newton_fixeps(P, m, u, J2, eps, steps=60):
    F, s = resid_fixeps(P, m, u, J2, eps)
    nF = np.abs(F).max()
    for it in range(steps):
        if nF < 1e-14:
            break
        n = len(u)
        Jm = np.zeros((n, n))
        for j in range(n):
            h = 1e-6 if j < J2 - 2 else 1e-7
            up = u.copy()
            up[j] += h
            um = u.copy()
            um[j] -= h
            Jm[:, j] = (resid_fixeps(P, m, up, J2, eps)[0] - resid_fixeps(P, m, um, J2, eps)[0]) / (2 * h)
        du = np.linalg.lstsq(Jm, F, rcond=None)[0]
        lam = 1.0
        ok = False
        while lam > 1e-5:
            un = u - lam * du
            a = np.concatenate([[eps], un[: J2 - 2], [m]])
            if np.all(np.diff(a) > 1e-7) and un[J2 - 2 :].min() > 0:
                Fn, sn = resid_fixeps(P, m, un, J2, eps)
                if np.abs(Fn).max() < nF:
                    u, F, s, nF, ok = un, Fn, sn, np.abs(Fn).max(), True
                    break
            lam /= 2
        if not ok:
            break
    return u, nF, s


def split_by_eps(mt, v_old, J, ms):
    """the inner pair position eps as the parameter at fixed m = ms: root of R'(eps) = 0.
    v11: eps marched up by x1.2 from 0.003 with a WARM start from the previous eps, then bisection with warm starts
    (a cold-start grid missed the root near 0.06 at 21.0212, where the inner solve fails cold beyond eps = 0.05).
    """
    P = Q.Prob(ms)
    J2 = J + 1
    a0 = v_old[: J - 1]
    w0 = v_old[J - 1 : 2 * J - 1]
    c = v_old[2 * J - 1]
    u = np.concatenate([a0 * ms / mt, [c / 2], w0])
    e = 0.003
    u, nF, sv = newton_fixeps(P, ms, u, J2, e)
    if nF > TOLR or sv <= 0:
        return None
    lo = (e, u, sv)
    hi = None
    amax = (a0[0] * ms / mt if len(a0) else ms) - 1e-3
    while e * 1.2 < amax:
        e2 = e * 1.2
        u2, n2, s2 = newton_fixeps(P, ms, u.copy(), J2, e2)
        if n2 > TOLR:
            return None
        if s2 <= 0:
            hi = (e2, u2, s2)
            break
        e, u, sv = e2, u2, s2
        lo = (e, u, sv)
    if hi is None:
        return None
    (el, ul, sl), (eh, uh, sh) = lo, hi
    for _ in range(60):
        em = 0.5 * (el + eh)
        um, nm, sm = newton_fixeps(P, ms, ul.copy(), J2, em)
        if nm > TOLR:
            break
        if sm > 0:
            el, ul, sl = em, um, sm
        else:
            eh, uh, sh = em, um, sm
        if eh - el < 1e-12:
            break
    eps = 0.5 * (el + eh)
    u, nF, sv = newton_fixeps(P, ms, ul.copy(), J2, eps)
    v = np.concatenate([[eps], u])
    P2, v2, nF2 = solve_at(ms, v, J2, False)
    if nF2 <= TOLR:
        return v2, nF2
    return (v, max(nF, abs(sv))) if max(nF, abs(sv)) <= TOLR else None


def eps_root(P, ms, base, J2, eps_guess):
    """root of R'(eps) = 0 near eps_guess with the inner Newton seeded by base (u without eps)"""
    grid = sorted(set([eps_guess * f for f in (0.5, 0.7, 0.85, 1.0, 1.2, 1.45, 1.8, 2.3)]))
    vals = []
    for e in grid:
        u, nF, sv = newton_fixeps(P, ms, base.copy(), J2, e)
        vals.append((e, sv, nF, u))
    for (e1, s1, n1, u1), (e2, s2, n2, u2) in zip(vals, vals[1:]):
        if n1 <= TOLR and n2 <= TOLR and (s1 > 0) != (s2 > 0):
            lo, hi, slo, ub = e1, e2, s1, u1
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                sm, um, nm = newton_fixeps(P, ms, ub.copy(), J2, mid)[2], None, None
                u_m, n_m, s_m = newton_fixeps(P, ms, ub.copy(), J2, mid)
                if (s_m > 0) == (slo > 0):
                    lo, slo, ub = mid, s_m, u_m
                else:
                    hi = mid
                if hi - lo < 1e-12:
                    break
            eps = 0.5 * (lo + hi)
            u, nF, sv = newton_fixeps(P, ms, ub.copy(), J2, eps)
            if nF <= TOLR:
                return eps, u, nF, sv
    return None


def split_march(mt, v_old, J, m_target):
    """v7: march the young split family from m_t + 1e-3 to m_target through m_t + 3e-3, 1e-2, 3e-2 ..., predicting
    eps ~ sqrt(m - m_t) at each point and re-solving the eps root with the previous inner state as seed"""
    J2 = J + 1
    ms = mt + 1e-3
    got = split_by_eps(mt, v_old, J, ms)
    if got is None:
        return None
    v = got[0]
    eps = v[0]
    u = v[1:]
    mprev = ms
    pts = [mt + x for x in (3e-3, 1e-2, 3e-2, 6e-2, 0.1, 0.15) if mt + x < m_target - 1e-9] + [m_target]
    for mn in pts:
        P = Q.Prob(mn)
        eg = eps * np.sqrt((mn - mt) / (mprev - mt))
        base = u.copy()
        base[: J2 - 2] *= mn / mprev
        r = eps_root(P, mn, base, J2, eg)
        if r is None:
            say(f"      split_march: no eps root at m = {mn:.5f} (guess {eg:.4f})")
            return None
        eps, u, nF, sv = r
        mprev = mn
        say(f"      split_march: m = {mn:.5f}, eps = {eps:.5f}, inner res {nF:.1e}")
    v = np.concatenate([[eps], u])
    P2, v2, nF2 = solve_at(m_target, v, J2, False)
    if nF2 <= TOLR:
        return v2, nF2
    return (v, max(nF, abs(sv))) if max(nF, abs(sv)) <= TOLR else None


def split_eps_continuation(mt, v_old, J, m_target):
    """v8: natural-parameter continuation of the young split family in eps. Unknowns (u, m) at fixed eps, equations
    resid_fixeps plus R'(eps) = 0; eps marched up by 10 per cent with a secant predictor until m passes m_target;
    one x-grid (that of m_target + 0.2) for every m, so the residual is smooth in m."""
    J2 = J + 1
    P = Q.Prob(m_target + 0.2)
    ms = mt + min(
        1e-3, 0.5 * (m_target - mt)
    )  # v12: a transition within 1e-3 below the grid point (35 -> 36) left one
    got = split_by_eps(mt, v_old, J, ms)  # continuation point and hist[-2] raised IndexError
    if got is None:
        say(f"      split continuation: split_by_eps found no root at m_t + {ms - mt:.1e} = {ms:.6f}")
        return None
    v = got[0]
    eps = v[0]
    z = np.concatenate([v[1:], [ms]])

    def Fz(zz, e):
        F, sv = resid_fixeps(P, zz[-1], zz[:-1], J2, e)
        return np.concatenate([F, [sv]])

    def newton_z(zz, e, steps=40):
        F = Fz(zz, e)
        nF = np.abs(F).max()
        for it in range(steps):
            if nF < 1e-14:
                break
            n = len(zz)
            Jm = np.zeros((n, n))
            for j in range(n):
                h = 1e-6 if (j < J2 - 2 or j == n - 1) else 1e-7
                zp = zz.copy()
                zp[j] += h
                zm = zz.copy()
                zm[j] -= h
                Jm[:, j] = (Fz(zp, e) - Fz(zm, e)) / (2 * h)
            dz = np.linalg.lstsq(Jm, F, rcond=None)[0]
            lam = 1.0
            ok = False
            while lam > 1e-5:
                zn = zz - lam * dz
                a = np.concatenate([[e], zn[: J2 - 2], [zn[-1]]])
                if np.all(np.diff(a) > 1e-7) and zn[J2 - 2 : -1].min() > 0:
                    Fn = Fz(zn, e)
                    if np.abs(Fn).max() < nF:
                        zz, F, nF, ok = zn, Fn, np.abs(Fn).max(), True
                        break
                lam /= 2
            if not ok:
                break
        return zz, nF

    z, nF = newton_z(z, eps)
    if nF > TOLR:
        say(f"      split continuation: first (u, m) solve failed, res {nF:.1e}")
        return None
    hist = [(eps, z.copy())]
    while z[-1] < m_target:
        e_new = eps * 1.1
        if len(hist) >= 2:
            (e1, z1), (e2, z2) = hist[-2], hist[-1]
            zp = z2 + (z2 - z1) * (e_new - e2) / (e2 - e1)
        else:
            zp = z.copy()
        zn, nF = newton_z(zp, e_new)
        if nF > TOLR:
            zn, nF = newton_z(z.copy(), e_new)
        for ratio in (1.05, 1.02):
            if nF <= TOLR:
                break
            e_new = eps * ratio
            zn, nF = newton_z(z.copy(), e_new)
        if nF > TOLR:
            if m_target - z[-1] < 0.01:
                v = np.concatenate([[eps], z[:-1]])
                v[: J2 - 1] *= m_target / z[-1]
                P2, v2, nF2 = solve_at(m_target, v, J2, False)
                if nF2 <= TOLR:
                    return v2, nF2
            say(
                f"      split continuation: step to eps = {e_new:.4f} failed (res {nF:.1e}) at m = {z[-1]:.5f}"
            )
            return None
        eps, z = e_new, zn
        hist.append((eps, z.copy()))
        if len(hist) > 200:
            return None
    # the state at m_target: interpolate between the last two points in m, then the full symmetric Newton
    if len(hist) < 2:  # v12 guard
        v = np.concatenate([[hist[0][0]], hist[0][1][:-1]])
        v[: J2 - 1] *= m_target / hist[0][1][-1]
        P2, v2, nF2 = solve_at(m_target, v, J2, False)
        say(
            f"      split continuation: one eps point, eps at the grid point {v2[0]:.5f}, full Newton res {nF2:.1e}"
        )
        return (v2, nF2) if nF2 <= TOLR else None
    (e1, z1), (e2, z2) = hist[-2], hist[-1]
    f = (m_target - z1[-1]) / (z2[-1] - z1[-1])
    e_t = e1 + f * (e2 - e1)
    z_t = z1 + f * (z2 - z1)
    v = np.concatenate([[e_t], z_t[:-1]])
    P2, v2, nF2 = solve_at(m_target, v, J2, False)
    say(
        f"      split continuation: {len(hist)} eps steps, eps at the grid point {v2[0]:.5f}, full Newton res {nF2:.1e}"
    )
    return (v2, nF2) if nF2 <= TOLR else None


def appear_c_continuation(mt, a0, w0, J, m_target):
    """v9: natural-parameter continuation of the young appear family in the centre weight c, with m as an unknown:
    unknowns (a_1..a_{J-1}, w_1..w_J, m) at fixed c, equations resid_fixc plus R(0) - R(m) = 0; c marched up by
    20 per cent with a secant predictor until m passes m_target; one x-grid for every m."""
    P = Q.Prob(m_target + 0.2)
    ms = mt + min(1e-3, 0.5 * (m_target - mt))  # v12, as in split_eps_continuation
    got = appear_by_c(mt, a0, w0, J, ms)
    if got is None:
        return None
    v = got[0]
    c = v[-1]
    z = np.concatenate([v[:-1], [ms]])

    def Fz(zz, cc):
        F, sv = resid_fixc(P, zz[-1], zz[:-1], J, cc)
        return np.concatenate([F, [sv]])

    def newton_z(zz, cc, steps=40):
        F = Fz(zz, cc)
        nF = np.abs(F).max()
        for it in range(steps):
            if nF < 1e-14:
                break
            n = len(zz)
            Jm = np.zeros((n, n))
            for j in range(n):
                h = 1e-6 if (j < J - 1 or j == n - 1) else 1e-7
                zp = zz.copy()
                zp[j] += h
                zm = zz.copy()
                zm[j] -= h
                Jm[:, j] = (Fz(zp, cc) - Fz(zm, cc)) / (2 * h)
            dz = np.linalg.lstsq(Jm, F, rcond=None)[0]
            lam = 1.0
            ok = False
            while lam > 1e-5:
                zn = zz - lam * dz
                a = np.concatenate([[0.0], zn[: J - 1], [zn[-1]]])
                if np.all(np.diff(a) > 1e-7) and zn[J - 1 : -1].min() > 0:
                    Fn = Fz(zn, cc)
                    if np.abs(Fn).max() < nF:
                        zz, F, nF, ok = zn, Fn, np.abs(Fn).max(), True
                        break
                lam /= 2
            if not ok:
                break
        return zz, nF

    z, nF = newton_z(z, c)
    if nF > TOLR:
        say(f"      appear continuation: first (u, m) solve failed, res {nF:.1e}")
        return None
    hist = [(c, z.copy())]
    while z[-1] < m_target:
        c_new = c * 1.2
        if len(hist) >= 2:
            (c1, z1), (c2, z2) = hist[-2], hist[-1]
            zp = z2 + (z2 - z1) * (c_new - c2) / (c2 - c1)
        else:
            zp = z.copy()
        zn, nF = newton_z(zp, c_new)
        if nF > TOLR:
            zn, nF = newton_z(z.copy(), c_new)
        for ratio in (1.1, 1.05, 1.02):
            if nF <= TOLR:
                break
            c_new = c * ratio
            zn, nF = newton_z(z.copy(), c_new)
        if nF > TOLR:
            if m_target - z[-1] < 0.01:  # close enough: the full Newton from the last point
                v = np.concatenate([z[:-1], [c]])
                v[: J - 1] *= m_target / z[-1]
                P2, v2, nF2 = solve_at(m_target, v, J, True)
                if nF2 <= TOLR:
                    say(
                        f"      appear continuation: finished by the full Newton from m = {z[-1]:.5f}, res {nF2:.1e}"
                    )
                    return v2, nF2
            say(
                f"      appear continuation: step to c = {c_new:.3e} failed (res {nF:.1e}) at m = {z[-1]:.5f}"
            )
            return None
        c, z = c_new, zn
        hist.append((c, z.copy()))
        if len(hist) > 200:
            return None
    if len(hist) < 2:  # v12 guard
        v = np.concatenate([hist[0][1][:-1], [hist[0][0]]])
        v[: J - 1] *= m_target / hist[0][1][-1]
        P2, v2, nF2 = solve_at(m_target, v, J, True)
        say(
            f"      appear continuation: one c point, c at the grid point {v2[-1]:.4e}, full Newton res {nF2:.1e}"
        )
        return (v2, nF2) if nF2 <= TOLR else None
    (c1, z1), (c2, z2) = hist[-2], hist[-1]
    f = (m_target - z1[-1]) / (z2[-1] - z1[-1])
    c_t = c1 + f * (c2 - c1)
    z_t = z1 + f * (z2 - z1)
    v = np.concatenate([z_t[:-1], [c_t]])
    P2, v2, nF2 = solve_at(m_target, v, J, True)
    say(
        f"      appear continuation: {len(hist)} c steps, c at the grid point {v2[-1]:.4e}, full Newton res {nF2:.1e}"
    )
    return (v2, nF2) if nF2 <= TOLR else None


def start_family(mt, v_old, J, centre_old, kind, m_target):
    """new structure just past the transition mt, then continued to m_target. v3: several offsets past mt and several
    seeds; for a split, L-BFGS on the full problem (q484) as a fallback, then the symmetric Newton."""
    a0 = v_old[: J - 1]
    w0 = v_old[J - 1 : 2 * J - 1]
    # v6: solve the new family directly at the target grid point with the natural parameter (eps or c); stepping from
    # m_t + 1e-3 with a linear predictor jumped branches at 18.678365, where eps grows like 1.75 sqrt(m - m_t)
    got = (
        split_eps_continuation(mt, v_old, J, m_target)
        if kind == "split"
        else appear_c_continuation(mt, a0, w0, J, m_target)
    )
    if got is not None and got[1] <= TOLR:
        J2, c2 = (J + 1, False) if kind == "split" else (J, True)
        PP = Q.Prob(m_target)
        sg = signals(PP, m_target, got[0], J2, c2)
        if sg[2] <= 1e-10:
            return got[0], got[1], J2, c2
    for off in (1e-3, 5e-3, 2e-2, min(5e-2, 0.9 * (m_target - mt)) if m_target - mt > 2e-2 else 2e-2):
        ms = mt + off
        best = None
        if kind == "appear":
            for c0 in (1e-4, 1e-3, 1e-2, 3e-2):
                v = np.concatenate([a0 * ms / mt, w0 * (1 - c0), [c0]])
                P, vv, nF = solve_at(ms, v, J, True)
                if nF <= TOLR and vv[-1] > 0 and (best is None or nF < best[1]):
                    best = (vv, nF, J, True)
            if best is None:
                got = appear_by_c(mt, a0, w0, J, ms)
                if got is not None:
                    best = (got[0], got[1], J, True)
        else:
            c = v_old[2 * J - 1]
            for eps in (0.01, 0.02, 0.04, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7):
                v = np.concatenate([[eps], a0 * ms / mt, [c / 2], w0])
                P, vv, nF = solve_at(ms, v, J + 1, False)
                if nF <= TOLR and vv[0] > 1e-6 and (best is None or nF < best[1]):
                    best = (vv, nF, J + 1, False)
            if best is None:
                got = split_by_eps(mt, v_old, J, ms)
                if got is not None:
                    best = (got[0], got[1], J + 1, False)
            if best is None:  # fallback: unconstrained L-BFGS from the widest seed
                for eps in (0.1, 0.3, 0.5):
                    a = np.concatenate([[eps], a0 * ms / mt, [ms]])
                    w = np.concatenate([[c / 2], w0])
                    th, ww = full(a, w, None)
                    P = Q.Prob(ms)
                    th2, ww2, _ = P.solve(th, ww, iters=20000)
                    th2, ww2 = Q.clean(th2, ww2)
                    pos = np.sort(th2[th2 > 1e-9])
                    if len(pos) != J + 1:
                        continue
                    wpos = np.array([ww2[np.argmin(np.abs(th2 - p_))] for p_ in pos])
                    v = np.concatenate([pos[:-1], wpos / (2 * wpos.sum())])
                    P, vv, nF = solve_at(ms, v, J + 1, False)
                    if nF <= TOLR and vv[0] > 1e-6:
                        best = (vv, nF, J + 1, False)
                        break
        if best is not None:
            vv, nF, J2, c2 = best
            v2, nF2 = robust_advance(ms, vv, J2, c2, m_target) if m_target > ms else (vv, nF)
            if nF2 <= TOLR:
                return v2, nF2, J2, c2
    return None


if __name__ == "__main__":
    t0 = time.time()
    say(f"q486 v2: symmetric continuation, bounded normal mean, squared error; m to {MMAX} by {DM}")
    m = 0.5
    J = 1
    centre = False
    P, v, nF = solve_at(m, np.array([0.5]), J, centre)
    trans = []
    rows = []
    import os

    if os.environ.get("RESUME") == "1":
        dd = json.load(open("results_q486_lfp_sym.json"))
        rows = dd["rows"]
        trans = [tuple(t) for t in dd["transitions"]]
        lr = rows[-1]
        m = lr["m"]
        J = lr["J"]
        centre = lr["centre"]
        v = np.array(lr["v"])
        say(f"   RESUME from m = {m}, K = {lr['K']}, {len(trans)} transitions so far")
    while m < MMAX - 1e-9:
        m1 = round(m + DM, 10)
        # v14: when the current state is within 0.02 of its transition (the 35 -> 36 split lies 2e-4 below 23.30),
        # the rescale predictor cannot follow eps ~ 1.75 sqrt(m - m_t); with the analytic Jacobian it could also land
        # on another stationary point, so the young-family restart goes FIRST there
        young = bool(trans) and (m - trans[-1][0]) < 0.02
        v1, nF = (v, np.inf) if young else robust_advance(m, v, J, centre, m1)
        if nF > TOLR and trans and m1 - trans[-1][0] < 0.3:
            # v13: a young family (split pair eps ~ 1.75 sqrt(m - m_t), or a small centre weight) is not reached by
            # the rescale predictor: at 23.30 -> 23.35 Newton stalled near eps = 0.03 (res 1e-9 to 1e-5). Restart it
            # from its transition with the natural-parameter continuation straight to m1.
            mt_, Kb_, Ka_, kind_ = trans[-1]
            rb = [rr for rr in rows if rr["m"] < mt_ and rr["K"] == Kb_][-1]
            vb, nb = robust_advance(rb["m"], np.array(rb["v"]), rb["J"], rb["centre"], mt_)
            if nb <= TOLR:
                Jb = rb["J"]
                got = (
                    split_eps_continuation(mt_, vb, Jb, m1)
                    if kind_ == "split"
                    else appear_c_continuation(mt_, vb[: Jb - 1], vb[Jb - 1 : 2 * Jb - 1], Jb, m1)
                )
                if got is not None and got[1] <= TOLR and len(got[0]) == len(v):
                    v1, nF = got
                    say(
                        f"      v13: young {kind_} family restarted from m_t = {mt_:.6f} to m = {m1:.2f}, res {nF:.1e}, first entry {v1[0]:.5f}"
                    )
        if nF > TOLR and young:
            v1, nF = robust_advance(m, v, J, centre, m1)
        if nF > TOLR:
            say(f"   STOP: no convergence from m = {m:.4f} to {m1:.4f} even with 64 substeps (res {nF:.1e})")
            break
        P = Q.Prob(m1)
        s0, s2, off, r = signals(P, m1, v1, J, centre)
        kind = "appear" if (not centre and s0 > 0) else ("split" if (centre and s2 > 0) else None)
        if kind:
            K = 2 * J + (1 if centre else 0)
            idx = 0 if kind == "appear" else 1

            def gfun(mm, _m=m, _v=v, _J=J, _c=centre, _i=idx):
                vv, nn = robust_advance(_m, _v, _J, _c, mm)
                return signals(Q.Prob(mm), mm, vv, _J, _c)[_i]

            mt = bisect(gfun, m, m1)
            vt, _ = robust_advance(m, v, J, centre, mt)
            res = start_family(mt, vt, J, centre, kind, m1)
            if res is None:
                say(f"   STOP: could not start the {kind} family at m = {mt:.6f}")
                break
            v1, nF, J, centre = res
            trans.append((mt, K, K + 1, kind))
            say(f"   TRANSITION {K} -> {K + 1} ({kind}) at m = {mt:.6f}   [{time.time() - t0:.0f}s]")
            P = Q.Prob(m1)
            s0, s2, off, r = signals(P, m1, v1, J, centre)
        m, v = m1, v1
        K = 2 * J + (1 if centre else 0)
        flag = "" if (nF <= TOLR and off <= 1e-10) else f"  FLAG(res {nF:.1e}, off {off:+.1e})"
        sig = s2 if centre else s0  # v14c: the transition signal against the residual it sits on
        if abs(sig) < 10 * max(nF, 1e-14):
            flag += f"  AMBIG(signal {sig:+.1e} vs res {nF:.1e})"
        rows.append(
            dict(
                m=round(m, 6),
                K=K,
                r=r,
                res=nF,
                off=off,
                s0=s0,
                s2=s2,
                v=[float(t) for t in v],
                J=J,
                centre=centre,
            )
        )
        if abs(m - round(m)) < 1e-9 or flag:
            a = np.concatenate([v[: J - 1], [m]])
            th, _ = full(a, v[J - 1 : 2 * J - 1], v[2 * J - 1] if centre else None)
            g = np.diff(np.sort(th))
            say(
                f"m = {m:6.2f}: K = {K:3d}, r = {r:.12f}, res {nF:.1e}, off-support {off:+.1e}, R(0)-r {s0:+.2e}, R''(0) {s2:+.2e}, "
                f"edge gaps {np.round(g[-4:][::-1], 4)}, centre gap {g[len(g)//2]:.4f}{flag}   [{time.time() - t0:.0f}s]"
            )
        json.dump(dict(transitions=trans, rows=rows), open("results_q486_lfp_sym.json", "w"))
    tm = np.array([t[0] for t in trans])
    Kt = np.array([t[2] for t in trans])
    say("transitions: " + ", ".join(f"{a}->{b} {kind} {mt:.5f}" for mt, a, b, kind in trans))
    sel = tm >= 8
    if sel.sum() >= 4:
        for name, f in (
            ("m^(4/3)", tm[sel] ** (4 / 3)),
            ("m", tm[sel]),
            ("m log m", tm[sel] * np.log(tm[sel])),
        ):
            A = np.vstack([f, np.ones_like(f)]).T
            (cc, bb), *_ = np.linalg.lstsq(A, Kt[sel], rcond=None)
            say(
                f"   K = c {name} + b: c {cc:.5f}, b {bb:+.4f}, rms {np.sqrt(np.mean((Kt[sel] - A @ [cc, bb])**2)):.4f}"
            )
        e, _ = np.polyfit(np.log(tm[sel]), np.log(Kt[sel]), 1)
        say(f"   free exponent (no offset): {e:.4f}")
    say(f"[{time.time() - t0:.0f}s]")
