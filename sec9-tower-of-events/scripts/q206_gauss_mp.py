"""Q206: the forty-digit mirror-reduced KKT solver for the amplitude-constrained GAUSSIAN channel Y = X + Z,
Z ~ N(0, 1), X in [-A, A] (Amendment 283 item 3: the decisive instrument - the Gaussian channel's own spacing
profile at K = 40-60 against (a) the binomial profile 3.61 d^-0.324 and (b) the Abbott-Machta boundary-value
solution with their coefficient). Structure as q179: right-half interior atoms u_j (0 < u_j < A) with weights w_j,
mirrors at -u_j, the wall atoms at +-A with weight we each, an optional centre atom (odd K) with weight wc;
unknowns v = (u, w, we[, wc]); residuals D(u_j) - D(A) = 0, D'(u_j) = 0, 2 (sum w + we) + wc = 1.
Quadrature in y: trapezoidal on [-A - 9, A + 9] with step H (exponentially accurate for these smooth integrands).
Analytic Jacobian; 'check' mode compares it with finite differences. Modes:
   py q206_gauss_mp.py check <A> <K> [dps]            (guess from q192's machinery, one Newton, Jacobian check)
   py q206_gauss_mp.py solve <A> <K> [dps] [seed.json] (Newton with 2-norm backtracking; writes results_q206_gauss_K{K}_A{A}.json)
"""

import os, sys, json, time
from mpmath import mp, mpf, exp, log, sqrt, pi, matrix, lu_solve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
say = lambda *a: print(*a, flush=True)
t0 = time.time()
H = mpf("0.02")
L2P = None


def grid(A):
    n = int((2 * A + 18) / H) + 1
    return [-A - 9 + H * i for i in range(n)]


def phi_row(y, u):
    """log phi(y|u) and phi(y|u) over the grid, with (y - u)"""
    Ls = [-((yy - u) ** 2) / 2 - L2P for yy in y]
    return Ls, [exp(v) for v in Ls], [yy - u for yy in y]


def tables(A, y, u, w, we, wc):
    m = len(u)
    Q = [mpf(0)] * len(y)
    rows = []
    for uj, wj in zip(u, w):
        Lp, Pp, Sp = phi_row(y, uj)
        Lm, Pm, Sm = phi_row(y, -uj)
        rows.append((Lp, Pp, Sp, Lm, Pm, Sm))
        for i in range(len(y)):
            Q[i] += wj * (Pp[i] + Pm[i])
    LA, PA, SA = phi_row(y, A)
    LmA, PmA, SmA = phi_row(y, -A)
    for i in range(len(y)):
        Q[i] += we * (PA[i] + PmA[i])
    Lc = Pc = Sc = None
    if wc is not None:
        Lc, Pc, Sc = phi_row(y, mpf(0))
        for i in range(len(y)):
            Q[i] += wc * Pc[i]
    lQ = [log(q) for q in Q]
    return rows, (LA, PA, SA), (LmA, PmA, SmA), (Lc, Pc, Sc), Q, lQ


def D_of(Lr, Pr, lQ):
    return H * sum(Pr[i] * (Lr[i] - lQ[i]) for i in range(len(Pr)))


def Dp_of(Lr, Pr, Sr, lQ):
    return H * sum(Pr[i] * Sr[i] * (Lr[i] - lQ[i] + 1) for i in range(len(Pr)))


def Dpp_of(Lr, Pr, Sr, lQ):
    return H * sum(Pr[i] * (Sr[i] ** 2 * (Lr[i] - lQ[i] + 2) - (Lr[i] - lQ[i] + 1)) for i in range(len(Pr)))


def residual_and_jacobian(A, y, u, w, we, wc, want_J=True):
    m = len(u)
    odd = wc is not None
    rows, wallp, wallm, cen, Q, lQ = tables(A, y, u, w, we, wc)
    LA, PA, SA = wallp
    D_u = [D_of(r[0], r[1], lQ) for r in rows]
    D_A = D_of(LA, PA, lQ)
    Dp_u = [Dp_of(r[0], r[1], r[2], lQ) for r in rows]
    res = [D_u[j] - D_A for j in range(m)] + Dp_u + [2 * (sum(w) + we) + (wc if odd else 0) - 1]
    if odd:  # the centre atom's own KKT condition D(0) = C (its position is fixed by symmetry)
        res.append(D_of(cen[0], cen[1], lQ) - D_A)
    if not want_J:
        return res, None
    nunk = 2 * m + 1 + (1 if odd else 0)
    J = matrix(nunk, nunk)

    # kernel-weighted sums helper: for an evaluation row (Lr, Pr, Sr) and a source atom k with (Pk_plus, Pk_minus, Sk_plus, Sk_minus)
    def dD_dw(Pr, Pk_sum):
        return -H * sum(Pr[i] * Pk_sum[i] / Q[i] for i in range(len(Pr)))

    def dDp_dw(Pr, Sr, Pk_sum):
        return -H * sum(Pr[i] * Sr[i] * Pk_sum[i] / Q[i] for i in range(len(Pr)))

    def dD_du(Pr, dQ_du):
        return -H * sum(Pr[i] * dQ_du[i] / Q[i] for i in range(len(Pr)))

    def dDp_du(Pr, Sr, dQ_du):
        return -H * sum(Pr[i] * Sr[i] * dQ_du[i] / Q[i] for i in range(len(Pr)))

    # source derivatives: dQ/dw_k = P_k(+) + P_k(-);  dQ/du_k = w_k [P_k(+) S_k(+) - P_k(-) S_k(-)]  (mirror moves opposite)
    src_w = [[r[1][i] + r[4][i] for i in range(len(y))] for r in rows]
    src_u = [
        [w[k] * (rows[k][1][i] * rows[k][2][i] - rows[k][4][i] * rows[k][5][i]) for i in range(len(y))]
        for k in range(m)
    ]
    src_we = [wallp[1][i] + wallm[1][i] for i in range(len(y))]
    src_wc = [cen[1][i] for i in range(len(y))] if odd else None
    Dpp_u = [Dpp_of(r[0], r[1], r[2], lQ) for r in rows]
    for j in range(m):  # rows: D(u_j) - D(A)
        Lr, Pr, Sr = rows[j][0], rows[j][1], rows[j][2]
        for k in range(m):
            J[j, k] = dD_du(Pr, src_u[k]) - dD_du(PA, src_u[k]) + (Dp_u[j] if k == j else 0)
            J[j, m + k] = dD_dw(Pr, src_w[k]) - dD_dw(PA, src_w[k])
        J[j, 2 * m] = dD_dw(Pr, src_we) - dD_dw(PA, src_we)
        if odd:
            J[j, 2 * m + 1] = dD_dw(Pr, src_wc) - dD_dw(PA, src_wc)
    for j in range(m):  # rows: D'(u_j)
        Lr, Pr, Sr = rows[j][0], rows[j][1], rows[j][2]
        for k in range(m):
            J[m + j, k] = dDp_du(Pr, Sr, src_u[k]) + (Dpp_u[j] if k == j else 0)
            J[m + j, m + k] = dDp_dw(Pr, Sr, src_w[k])
        J[m + j, 2 * m] = dDp_dw(Pr, Sr, src_we)
        if odd:
            J[m + j, 2 * m + 1] = dDp_dw(Pr, Sr, src_wc)
    for k in range(m):
        J[2 * m, m + k] = mpf(2)
    J[2 * m, 2 * m] = mpf(2)
    if odd:
        J[2 * m, 2 * m + 1] = mpf(1)
        Pc = cen[1]
        rr = 2 * m + 1  # row: D(0) - D(A)
        for k in range(m):
            J[rr, k] = dD_du(Pc, src_u[k]) - dD_du(PA, src_u[k])
            J[rr, m + k] = dD_dw(Pc, src_w[k]) - dD_dw(PA, src_w[k])
        J[rr, 2 * m] = dD_dw(Pc, src_we) - dD_dw(PA, src_we)
        J[rr, 2 * m + 1] = dD_dw(Pc, src_wc) - dD_dw(PA, src_wc)
    return res, J


def feasible(A, u, w, we, wc):
    m = len(u)
    return (
        all(u[i] < u[i + 1] for i in range(m - 1))
        and u[0] > 0
        and u[-1] < A
        and all(t > 0 for t in w)
        and we > 0
        and (wc is None or wc > 0)
    )


SOLVER = os.environ.get("SOLVER", "lu")
CAP = mpf(os.environ.get("CAP", "0.05"))
CKPT = os.environ.get("CKPT", "")


def solve_step(J, r):
    """Newton step. SOLVER=lu: forty-digit LU. SOLVER=svdcap: float SVD of J; the coefficient along each right-singular
    vector is (u_i . -r)/s_i, and any coefficient beyond CAP in magnitude is clipped to CAP (the soft central modes of
    Amendment 302, whose linearised step is unphysical); when nothing is clipped the exact LU step is used instead.
    """
    if SOLVER not in ("svdcap", "svdls", "svdq", "svdnest", "svdnestq"):
        return lu_solve(J, matrix([-t for t in r])), 0, []
    import numpy as np

    Jf = np.array([[float(J[i, j]) for j in range(J.cols)] for i in range(J.rows)])
    rf = -np.array([float(t) for t in r])
    U, s, Vt = np.linalg.svd(Jf)
    c = (U.T @ rf) / s
    cap = float(CAP)
    capped = np.abs(c) > cap
    ncap = int(capped.sum())
    if ncap == 0:
        return (
            lu_solve(J, matrix([-t for t in r])),
            0,
            [(U[:, i], Vt[i]) for i in range(len(s) - 1, len(s) - 4, -1)],
        )
    c = np.clip(c, -cap, cap)
    dv = Vt.T @ c
    soft = [(U[:, i], Vt[i]) for i in np.where(capped)[0][::-1]]  # softest first
    return [mpf(repr(float(v))) for v in dv], ncap, soft


def soft_quadratic_step(A, y, v, unpack, soft, h, say, trace):
    """SOLVER=svdq (Amendment 305, pre-registered remedy for the 1e-9 wall): a full quadratic model of the residual
    components p_i(delta) = u_i . r(v + sum_j delta_j v_j) in the subspace of the s <= 3 softest capped modes, fitted
    exactly on the 1 + 2s + s(s-1)/2 design points 0, +-h e_j, h(e_j + e_k); the model's root nearest zero (Newton
    on the model from 0), |delta| <= 3h, then evaluated; accepted if the 2-norm of r falls, else halved up to four
    times. Captures the curvature and the coupling that the one-dimensional search (svdls) cannot."""
    import numpy as np, itertools

    soft = soft[:3]
    sdim = len(soft)
    V = [[mpf(repr(float(t))) for t in vi] for _, vi in soft]
    Um = [[mpf(repr(float(t))) for t in ui] for ui, _ in soft]

    def r_at(delta):
        vn = [v[i] + sum(mpf(repr(float(delta[j]))) * V[j][i] for j in range(sdim)) for i in range(len(v))]
        if not feasible(A, *unpack(vn)):
            return None, None, None
        rn, _ = residual_and_jacobian(A, y, *unpack(vn), want_J=False)
        return (
            np.array([float(sum(Um[j][i] * rn[i] for i in range(len(rn)))) for j in range(sdim)]),
            sqrt(sum(t * t for t in rn)),
            vn,
        )

    pts = [np.zeros(sdim)]
    for j in range(sdim):
        pts += [h * np.eye(sdim)[j], -h * np.eye(sdim)[j]]
    for j, k in itertools.combinations(range(sdim), 2):
        pts.append(h * (np.eye(sdim)[j] + np.eye(sdim)[k]))
    quad_idx = list(itertools.combinations_with_replacement(range(sdim), 2))
    phi = lambda d: np.concatenate([[1.0], d, [d[j] * d[k] for j, k in quad_idx]])
    P = []
    n0 = None
    for d in pts:
        pv, nn, _ = r_at(d)
        if pv is None:
            return v, 0
        P.append(pv)
        n0 = nn if n0 is None else n0
    Phi = np.array([phi(d) for d in pts])
    C = np.linalg.solve(Phi, np.array(P))  # C[:, i] = coefficients of p_i

    def model(d):
        return phi(d) @ C

    def jac(d):
        Jm = np.zeros((sdim, sdim))
        for i in range(sdim):
            for j in range(sdim):
                Jm[i, j] = C[1 + j, i] + sum(
                    C[1 + sdim + q, i] * ((d[b] if a == j else 0) + (d[a] if b == j else 0))
                    for q, (a, b) in enumerate(quad_idx)
                )
        return Jm

    d = np.zeros(sdim)
    for _ in range(30):
        try:
            step = np.linalg.solve(jac(d), -model(d))
        except np.linalg.LinAlgError:
            break
        d = d + step
        if np.linalg.norm(step) < 1e-14:
            break
    if not np.all(np.isfinite(d)):
        return v, 0
    scale = np.max(np.abs(d)) / (3 * h)
    if scale > 1:
        d = d / scale
    for _ in range(5):
        pv, nn, vn = r_at(d)
        if pv is not None and nn < n0:
            if trace:
                say(
                    f"         soft-subspace quadratic step (s = {sdim}): delta {np.round(d, 4).tolist()} -> 2-norm {mp.nstr(nn, 3)}"
                )
            return vn, 1
        d = d / 2
    return v, 0


NEST_S = int(os.environ.get("NEST_S", "3"))
NEST_H = float(os.environ.get("NEST_H", "0.02"))
NEST_AUTO = os.environ.get("NEST_AUTO", "1") == "1"
NEST_C2 = float(os.environ.get("NEST_C2", "2e-6"))


def nested_cycle(A, y, v, unpack, J, r, say, trace):
    """SOLVER=svdnest (Amendment 310, pre-registered): one outer cycle of the nested solve. Float SVD of the current
    Jacobian; the S softest right-singular directions are the outer coordinates delta, the rest the stiff subspace.
    inner(v): chord iteration v <- v + V_stiff (-U_stiff . r / s_stiff) with exact residuals until the stiff
    components are below 1 % of the soft ones (or 8 iterations). Outer: p(delta) = U_soft . r(inner(v + V_soft delta));
    finite-difference Jacobian at scale NEST_H, Newton step on delta, |delta| <= 5 NEST_H, halved up to three times
    until the 2-norm of r falls. Returns the new v (or v unchanged) and whether it moved."""
    import numpy as np

    S = NEST_S
    h = NEST_H
    Jf = np.array([[float(J[i, j]) for j in range(J.cols)] for i in range(J.rows)])
    U, sv, Vt = np.linalg.svd(Jf)
    nS = len(sv)
    soft_idx = list(range(nS - S, nS))
    stiff_idx = list(range(0, nS - S))
    Us = U[:, stiff_idx]
    Vs = Vt[stiff_idx].T
    ss = sv[stiff_idx]
    Uo = U[:, soft_idx]
    Vo = Vt[soft_idx].T

    def res(vv):
        rr, _ = residual_and_jacobian(A, y, *unpack(vv), want_J=False)
        return rr

    def inner(vv):
        rr = res(vv)
        for _ in range(8):
            rf = np.array([float(t) for t in rr])
            ps = Us.T @ rf
            po = Uo.T @ rf
            if np.linalg.norm(ps) < max(0.01 * np.linalg.norm(po), 1e-30):
                break
            dv = Vs @ (-ps / ss)
            vn = [vv[i] + mpf(repr(float(dv[i]))) for i in range(len(vv))]
            if not feasible(A, *unpack(vn)):
                break
            vv = vn
            rr = res(vv)
        return vv, rr

    n_orig = sqrt(sum(t * t for t in r))
    v0, r0 = inner(v)
    n0 = sqrt(sum(t * t for t in r0))
    if n0 > n_orig:
        v0, r0, n0 = (
            v,
            r,
            n_orig,
        )  # a divergent inner solve (too few soft modes for this system): keep the original point
    p0 = Uo.T @ np.array([float(t) for t in r0])
    if NEST_AUTO:
        h = float(
            max(1e-5, min(NEST_H, np.sqrt(np.abs(p0).max() / NEST_C2)))
        )  # secant scale = the quadratic root scale sqrt(|p|/c2) (Amendment 315)
    M = np.zeros((S, S))
    for j in range(S):
        dv = Vo[:, j] * h
        vj = [v0[i] + mpf(repr(float(dv[i]))) for i in range(len(v0))]
        if not feasible(A, *unpack(vj)):
            return v, False
        _, rj = inner(vj)
        M[:, j] = (Uo.T @ np.array([float(t) for t in rj]) - p0) / h
    try:
        delta = np.linalg.solve(M, -p0)
    except np.linalg.LinAlgError:
        return v, False
    scale = np.max(np.abs(delta)) / (5 * h)
    if scale > 1:
        delta = delta / scale
    for _ in range(4):
        dv = Vo @ delta
        vn = [v0[i] + mpf(repr(float(dv[i]))) for i in range(len(v0))]
        if feasible(A, *unpack(vn)):
            vn, rn = inner(vn)
            nn = sqrt(sum(t * t for t in rn))
            if nn < n0:
                if trace:
                    say(
                        f"         nested cycle (h = {h:.1e}): delta {np.round(delta, 6).tolist()} -> 2-norm {mp.nstr(nn, 3)} (soft components before {np.abs(p0).max():.1e})"
                    )
                return vn, True
        delta = delta / 2
    if n0 < n_orig:
        if trace:
            say(f"         nested cycle: only the inner (stiff) solve moved -> 2-norm {mp.nstr(n0, 3)}")
        return v0, True
    return v, False


NESTQ_CLAMP = float(os.environ.get("NESTQ_CLAMP", "0.2"))


def nested_quadratic_cycle(A, y, v, unpack, J, r, say, trace):
    """SOLVER=svdnestq (Amendment 316, pre-registered): the full quadratic model of the soft residual components on the
    stiff-relaxed manifold. Soft/stiff split by float SVD as in nested_cycle; every design point (0, +-h e_j, h(e_j+e_k))
    is inner-solved (chord iteration on the stiff subspace) before its soft components are read; the S-dimensional
    quadratic model is fitted exactly, its root nearest zero taken (Newton on the model), |delta| <= NESTQ_CLAMP, then
    inner-solved and accepted if the 2-norm of r falls (halved up to four times)."""
    import numpy as np, itertools

    S = NEST_S
    h = NEST_H
    Jf = np.array([[float(J[i, j]) for j in range(J.cols)] for i in range(J.rows)])
    U, sv, Vt = np.linalg.svd(Jf)
    nS = len(sv)
    Us = U[:, : nS - S]
    Vs = Vt[: nS - S].T
    ss = sv[: nS - S]
    Uo = U[:, nS - S :]
    Vo = Vt[nS - S :].T

    def res(vv):
        rr, _ = residual_and_jacobian(A, y, *unpack(vv), want_J=False)
        return rr

    def inner(vv):
        rr = res(vv)
        for _ in range(8):
            rf = np.array([float(t) for t in rr])
            ps = Us.T @ rf
            po = Uo.T @ rf
            if np.linalg.norm(ps) < max(0.01 * np.linalg.norm(po), 1e-30):
                break
            dv = Vs @ (-ps / ss)
            vn = [vv[i] + mpf(repr(float(dv[i]))) for i in range(len(vv))]
            if not feasible(A, *unpack(vn)):
                break
            vv = vn
            rr = res(vv)
        return vv, rr

    n_orig = sqrt(sum(t * t for t in r))
    v0, r0 = inner(v)
    n0 = sqrt(sum(t * t for t in r0))
    if n0 > n_orig:
        v0, r0, n0 = v, r, n_orig
    comp = lambda rr: Uo.T @ np.array([float(t) for t in rr])
    pts = [np.zeros(S)]
    for j in range(S):
        pts += [h * np.eye(S)[j], -h * np.eye(S)[j]]
    for j, k in itertools.combinations(range(S), 2):
        pts.append(h * (np.eye(S)[j] + np.eye(S)[k]))
    quad_idx = list(itertools.combinations_with_replacement(range(S), 2))
    phi = lambda d: np.concatenate([[1.0], d, [d[a] * d[b] for a, b in quad_idx]])
    P = [comp(r0)]
    for d in pts[1:]:
        dv = Vo @ d
        vd = [v0[i] + mpf(repr(float(dv[i]))) for i in range(len(v0))]
        if not feasible(A, *unpack(vd)):
            return v, False
        _, rd = inner(vd)
        P.append(comp(rd))
    Phi = np.array([phi(d) for d in pts])
    C = np.linalg.solve(Phi, np.array(P))

    def model(d):
        return phi(d) @ C

    def jac(d):
        Jm = np.zeros((S, S))
        for i in range(S):
            for j in range(S):
                Jm[i, j] = C[1 + j, i] + sum(
                    C[1 + S + q, i] * ((d[b] if a == j else 0) + (d[a] if b == j else 0))
                    for q, (a, b) in enumerate(quad_idx)
                )
        return Jm

    d = np.zeros(S)
    for _ in range(40):
        try:
            step = np.linalg.solve(jac(d), -model(d))
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(step)) > NESTQ_CLAMP:
            step = step * NESTQ_CLAMP / np.max(np.abs(step))
        d = d + step
        if np.linalg.norm(step) < 1e-14:
            break
    if not np.all(np.isfinite(d)):
        return v, False
    if np.max(np.abs(d)) > NESTQ_CLAMP:
        d = d * NESTQ_CLAMP / np.max(np.abs(d))
    for _ in range(5):
        dv = Vo @ d
        vn = [v0[i] + mpf(repr(float(dv[i]))) for i in range(len(v0))]
        if feasible(A, *unpack(vn)):
            vn, rn = inner(vn)
            nn = sqrt(sum(t * t for t in rn))
            if nn < n0:
                if trace:
                    say(
                        f"         nested quadratic cycle (h = {h:.1e}, S = {S}): delta {np.round(d, 5).tolist()} -> 2-norm {mp.nstr(nn, 3)} (soft before {np.abs(P[0]).max():.1e}, model root residual {np.abs(model(d)).max():.1e})"
                    )
                return vn, True
        d = d / 2
    if n0 < n_orig:
        return v0, True
    return v, False


def soft_line_search(A, y, v, unpack, soft, h, say, trace):
    """SOLVER=svdls: along each capped right-singular vector v_i the linear model is invalid (Amendment 302); fit the
    residual component p(delta) = u_i . r(v + delta v_i) by a parabola through delta = -h, 0, h and step to its root
    nearest zero (vertex if no real root), |delta| <= 3h, accepted only if the 2-norm of r falls."""
    import numpy as np

    r0, _ = residual_and_jacobian(A, y, *unpack(v), want_J=False)
    n0 = sqrt(sum(t * t for t in r0))
    moved = 0
    for ui, vi in soft[:3]:
        vim = [mpf(repr(float(t))) for t in vi]
        uim = [mpf(repr(float(t))) for t in ui]

        def comp(delta):
            vn = [v[i] + delta * vim[i] for i in range(len(v))]
            if not feasible(A, *unpack(vn)):
                return None, None, None
            rn, _ = residual_and_jacobian(A, y, *unpack(vn), want_J=False)
            return sum(uim[i] * rn[i] for i in range(len(rn))), sqrt(sum(t * t for t in rn)), vn

        p0 = sum(uim[i] * r0[i] for i in range(len(r0)))
        pp, _, _ = comp(mpf(h))
        pm, _, _ = comp(mpf(-h))
        if pp is None or pm is None:
            continue
        a2 = (pp - 2 * p0 + pm) / (2 * mpf(h) ** 2)
        a1 = (pp - pm) / (2 * mpf(h))
        if a2 == 0:
            delta = -p0 / a1 if a1 != 0 else mpf(0)
        else:
            disc = a1 * a1 - 4 * a2 * p0
            delta = (
                (-a1 / (2 * a2))
                if disc < 0
                else min(((-a1 + sqrt(disc)) / (2 * a2), (-a1 - sqrt(disc)) / (2 * a2)), key=abs)
            )
        delta = max(min(delta, 3 * mpf(h)), -3 * mpf(h))
        pn, nn, vn = comp(delta)
        if pn is not None and nn < n0:
            v = vn
            r0, _ = residual_and_jacobian(A, y, *unpack(v), want_J=False)
            n0 = nn
            moved += 1
            if trace:
                say(f"         soft-mode search: delta {mp.nstr(delta, 3)} -> 2-norm {mp.nstr(nn, 3)}")
    return v, moved


def newton(A, y, u, w, we, wc, steps=80, tol=mpf("1e-20"), trace=True):
    m = len(u)
    odd = wc is not None
    v = list(u) + list(w) + [we] + ([wc] if odd else [])
    unpack = lambda x: (x[:m], x[m : 2 * m], x[2 * m], (x[2 * m + 1] if odd else None))
    r, J = residual_and_jacobian(A, y, *unpack(v))
    res = max(abs(t) for t in r)
    k = 0
    while res > tol and k < steps:
        dv, ncap, soft = solve_step(J, r)
        res2 = sqrt(sum(t * t for t in r))
        lam = mpf(1)
        acc = False
        for _ in range(12):
            vn = [v[i] + lam * dv[i] for i in range(len(v))]
            if feasible(A, *unpack(vn)):
                rn, _ = residual_and_jacobian(A, y, *unpack(vn), want_J=False)
                res2n = sqrt(sum(t * t for t in rn))
                if res2n < res2 * (1 - mpf("1e-4") * lam) or max(abs(t) for t in rn) < tol:
                    v = vn
                    acc = True
                    break
            lam /= 2
        if not acc:  # Levenberg-Marquardt fallback on the same Jacobian (as q189)
            JT = J.T
            A0 = JT * J
            g = JT * matrix([-t for t in r])
            Dg = matrix(A0.rows, A0.cols)
            for i in range(A0.rows):
                Dg[i, i] = A0[i, i]
            mu = LM_MU[0]
            for _ in range(14):
                dv2 = lu_solve(A0 + mu * Dg, g)
                vn = [v[i] + dv2[i] for i in range(len(v))]
                if feasible(A, *unpack(vn)):
                    rn, _ = residual_and_jacobian(A, y, *unpack(vn), want_J=False)
                    res2n = sqrt(sum(t * t for t in rn))
                    if res2n < res2:
                        v = vn
                        acc = True
                        LM_MU[0] = max(mu / 10, mpf("1e-14"))
                        lam = "LM"
                        break
                mu *= 10
            if not acc:
                if trace:
                    say(f"      it {k+1}: no decrease (Newton and LM); stop")
                    break
        if SOLVER == "svdls" and soft and (ncap > 0 or (not isinstance(lam, str) and lam < 1)):
            v, _ = soft_line_search(A, y, v, unpack, soft, float(CAP) / 2, say, trace)
        if SOLVER == "svdq" and soft:
            v, _ = soft_quadratic_step(A, y, v, unpack, soft, float(CAP) / 2, say, trace)
        if SOLVER == "svdnest" and (res < mpf("1e-8") or (not isinstance(lam, str) and lam < 1)):
            r_cur, J_cur = residual_and_jacobian(A, y, *unpack(v))
            v, _ = nested_cycle(A, y, v, unpack, J_cur, r_cur, say, trace)
        if SOLVER == "svdnestq" and (res < mpf("1e-8") or (not isinstance(lam, str) and lam < 1)):
            r_cur, J_cur = residual_and_jacobian(A, y, *unpack(v))
            v, _ = nested_quadratic_cycle(A, y, v, unpack, J_cur, r_cur, say, trace)
        if CKPT:
            json.dump(
                {"A": mp.nstr(A, 45), "m": m, "odd": odd, "v": [mp.nstr(t, 45) for t in v]}, open(CKPT, "w")
            )
        r, J = residual_and_jacobian(A, y, *unpack(v))
        res = max(abs(t) for t in r)
        k += 1
        if trace:
            say(
                f"      it {k}: lam {lam if isinstance(lam, str) else mp.nstr(lam, 3)} residual {mp.nstr(res, 3)}"
                + (f" [capped {ncap}]" if ncap else "")
                + f"   [{time.time()-t0:.0f}s]"
            )
    return (*unpack(v), res, k)


LM_MU = [mpf("1e-3")]


def signals(A, y, u, w, we, wc, grid_pts=400):
    rows, wallp, wallm, cen, Q, lQ = tables(A, y, u, w, we, wc)
    C = D_of(wallp[0], wallp[1], lQ)

    def Dx(x):
        Lr, Pr, Sr = phi_row(y, x)
        return D_of(Lr, Pr, lQ)

    if wc is None:
        centre = Dx(mpf(0)) - C
    else:
        d = mpf("0.01")
        centre = (Dx(-d) - 2 * Dx(mpf(0)) + Dx(d)) / d**2
    atoms = list(u) + [A] + ([mpf(0)] if wc is not None else [])
    best = None
    where = None
    for i in range(1, grid_pts):
        x = A * i / grid_pts
        if min(abs(x - a) for a in atoms) < mpf("0.05"):
            continue
        v = Dx(x) - C
        if best is None or v > best:
            best, where = v, x
    return centre, best, where, C


def guess_from_q192(A, K):
    import q192_gauss_profile as g192

    Af = float(A)
    u0, w0, wc0 = g192.guess(Af, K)
    u, w, wc, res, ok = g192.solve(Af, u0, w0, wc0)
    say(f"float guess (q192 machinery): residual {res:.1e}, clean {ok}")
    return (
        [mpf(repr(float(v))) for v in u],
        [mpf(repr(float(v))) for v in w[:-1]],
        mpf(repr(float(w[-1]))),
        (None if wc is None else mpf(repr(float(wc)))),
    )


if __name__ == "__main__":
    mode = sys.argv[1]
    A = mpf(sys.argv[2])
    K = int(sys.argv[3])
    mp.dps = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    L2P = log(2 * pi) / 2
    y = grid(A)
    say(f"Gaussian channel, A = {A}, K = {K}, dps {mp.dps}, grid {len(y)} points, step {H}")
    if len(sys.argv) > 5 and sys.argv[5].endswith(".json"):
        d = json.load(open(os.path.join(HERE, sys.argv[5])))
        u = [mpf(v) for v in d["u"]]
        w = [mpf(v) for v in d["w"]]
        we = mpf(d["we"])
        wc = None if d["wc"] is None else mpf(d["wc"])
    else:
        u, w, we, wc = guess_from_q192(A, K)
    if mode == "check":
        r, J = residual_and_jacobian(A, y, u, w, we, wc)
        say(f"residual at the guess: {mp.nstr(max(abs(t) for t in r), 3)}   [{time.time()-t0:.0f}s]")
        m = len(u)
        odd = wc is not None
        v = list(u) + list(w) + [we] + ([wc] if odd else [])
        unpack = lambda x: (x[:m], x[m : 2 * m], x[2 * m], (x[2 * m + 1] if odd else None))
        h = mpf("1e-12")
        worst = mpf(0)
        where = None
        for i in range(len(v)):
            vp = list(v)
            vp[i] += h
            rp, _ = residual_and_jacobian(A, y, *unpack(vp), want_J=False)
            for jj in range(len(r)):
                fd = (rp[jj] - r[jj]) / h
                err = abs(fd - J[jj, i]) / max(1, abs(J[jj, i]))
                if err > worst:
                    worst, where = err, (jj, i, mp.nstr(fd, 8), mp.nstr(J[jj, i], 8))
        say(
            f"largest relative disagreement analytic vs finite difference: {mp.nstr(worst, 3)} at (row, col, fd, analytic) = {where}   [{time.time()-t0:.0f}s]"
        )
        sys.exit(0)
    uu, ww, wwe, wwc, res, k = newton(A, y, u, w, we, wc)
    cen, off, where, C = signals(A, y, uu, ww, wwe, wwc)
    ordered = all(uu[i] < uu[i + 1] for i in range(len(uu) - 1))
    wmin = min(list(ww) + [wwe] + ([wwc] if wwc is not None else []))
    ok = res < mpf("1e-20")
    optimal = ok and ordered and wmin > 0 and cen <= 0 and off <= 0
    verdict = (
        "OPTIMAL"
        if optimal
        else ("not converged" if not ok else ("centre positive" if cen > 0 else "off-atom positive"))
    )
    say(
        f"A = {A}, K = {K}: residual {mp.nstr(res, 3)} in {k} steps, min w {mp.nstr(wmin, 4)}, ordered {ordered}, centre signal {mp.nstr(cen, 4)}, off-atom max {mp.nstr(off, 4)} at x = {mp.nstr(where, 5)}, C = {mp.nstr(C, 12)} -> {verdict}   [{time.time()-t0:.0f}s]"
    )
    xs = sorted([-x for x in uu] + list(uu) + [A, -A] + ([mpf(0)] if wwc is not None else []))
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    say("gaps from the wall inward: " + " ".join(mp.nstr(g, 6) for g in gaps[::-1][:10]))
    json.dump(
        {
            "A": mp.nstr(A, 20),
            "K": K,
            "residual": float(res),
            "verdict": verdict,
            "C": mp.nstr(C, 30),
            "centre_signal": float(cen),
            "off_atom_max": float(off),
            "u": [mp.nstr(v, 40) for v in uu],
            "w": [mp.nstr(v, 40) for v in ww],
            "we": mp.nstr(wwe, 40),
            "wc": (None if wwc is None else mp.nstr(wwc, 40)),
        },
        open(os.path.join(HERE, f"results_q206_gauss_K{K}_A{mp.nstr(A, 8)}.json"), "w"),
        indent=1,
    )
