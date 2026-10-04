"""q847 (03 Oct 2026, Amendments 1485-1486): TEXT COPY of q846 with the data uniform on the annulus R1 <= |y| <= R (env R1, default 0;
R1 = 0 is q846). Prints the edge values L = log(m/f0) + 1 at both edges, f0 = 1/(pi (R^2 - R1^2)). States q847_state_R1_<R1>_R<R>.json.
q846's docstring follows.
q846 (03 Oct 2026, Amendment 1481, candidate C2): the disc NPMLE of q845 solved by the constrained Newton method (CNM, Wang 2007,
J. R. Stat. Soc. B 69, 185-198; UNCHECKED citation) on a fine grid of ring radii, then a continuous polish of the locations.
q845's damped Newton with vertex insertions stalls where a newborn ring or centre atom has a tiny mass (R >= 20); CNM handles tiny masses.
Kernel functions are TEXT COPIES of q845: unit-mass ring density at |y| = rho, k(rho; r) = exp(-(rho - r)^2/2) i0e(rho r)/(2 pi);
KKT function D(r) = int_0^R (2 rho/R^2) k(rho; r)/m(rho) d rho <= 1, equality on the support.
CNM step: support S, weights w; add the local maxima of D on the grid (step 0.002 to R + 4) that exceed 1; solve
min || sqrt(q) (S_mat v - 2) ||, v >= 0 (S_mat_ij = k_ij/m_i) by NNLS, normalise, backtrack on the likelihood; drop zero weights.
Stop when max D - 1 < 1e-11 on the grid. Then: merge grid neighbours into rings (mass-weighted), and polish (r_j, log w_j) by Newton on
D(r_j) = 1, D'(r_j) = 0 (centre atom: D(0) = 1 only). Acceptance as registered in 1481. Prints as q845; states q846_state_R<R>.json.
Usage: py q846_npmle_disc_cnm.py R1,R2,... [NQ]"""

import sys, json, math, os
import numpy as np
from scipy.special import i0e, i1e
from scipy.optimize import nnls

RS = [float(x) for x in sys.argv[1].split(",")]
NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
D1 = -0.304944111846
DBG = os.environ.get("DEBUG")
R1 = float(os.environ.get("R1", "0"))


def kern(rho, r):
    return np.exp(-((rho[:, None] - r[None, :]) ** 2) / 2) * i0e(rho[:, None] * r[None, :]) / (2 * np.pi)


def dkern(rho, r):
    x = rho[:, None] * r[None, :]
    return (
        np.exp(-((rho[:, None] - r[None, :]) ** 2) / 2)
        * (rho[:, None] * i1e(x) - r[None, :] * i0e(x))
        / (2 * np.pi)
    )


def setup(R):
    x, w = np.polynomial.legendre.leggauss(NQ)
    rho = R1 + (x + 1) * (R - R1) / 2
    q = w * (R - R1) / 2 * 2 * rho / (R**2 - R1**2)
    return rho, q


def cnm(R, rho, q, maxit=3000, grid=None, start=None):
    if grid is None:
        grid = np.arange(0, R + 4 + 1e-9, 0.002)
    KG = kern(rho, grid)
    if start is None:
        S = [int(np.argmin(np.abs(grid - x))) for x in np.linspace(0, R, 12)]
        w = np.full(len(S), 1 / len(S))
    else:
        S = sorted(set(int(np.argmin(np.abs(grid - x))) for x in start[0]))
        w = np.full(len(S), 1 / len(S))
    for it in range(maxit):
        m = KG[:, S] @ w
        D = KG.T @ (q / m)
        viol = D.max() - 1
        if DBG and it % 50 == 0:
            print(f"  cnm it {it}: support {len(S)}, KKT max - 1 {viol:.2e}", flush=True)
        if viol < 1e-11:
            break
        lm = [i for i in range(1, len(grid) - 1) if D[i] > 1 and D[i] >= D[i - 1] and D[i] >= D[i + 1]]
        if D[0] > 1 and D[0] >= D[1]:
            lm = [0] + lm
        wold = dict(zip(S, w))
        S = sorted(set(S) | set(lm))  # new support points enter with weight 0
        w = np.array([wold.get(s, 0.0) for s in S])
        w = w / w.sum()
        m = KG[:, S] @ w
        # quadratic model of L on the simplex: min sum_i q_i (s_i v - 2)^2, v >= 0, sum v = 1 (the plain NNLS has the trivial
        # solution v = 2w because s_i w = 1; the simplex row, weight 1e4, removes it)
        A = np.vstack([np.sqrt(q)[:, None] * KG[:, S] / m[:, None], 100.0 * np.ones((1, len(S)))])
        b = np.concatenate([2 * np.sqrt(q), [100.0]])
        v, _ = nnls(A, b)
        v = v / v.sum()
        L0 = np.sum(q * np.log(m))
        lam = 1.0
        while lam > 1e-10:
            wn = (1 - lam) * w + lam * v
            if (
                np.sum(q * np.log(KG[:, S] @ wn))
                >= L0 + 0.333 * lam * np.sum((v - w) * (KG[:, S].T @ (q / m))) - 1e-18
            ):
                break
            lam /= 2
        w = wn
        keep = w > 0
        S = [s for s, k in zip(S, keep) if k]
        w = w[keep] / w[keep].sum()
    return grid[S], w, viol


def rings_from(rg, wg):
    """merge grid points closer than 0.4 into rings, mass-weighted (CNM spreads one ring over neighbouring grid points; rings are >= 1 apart)"""
    r, w = [], []
    for a, b in zip(rg, wg):
        if r and a - r[-1][-1] <= 0.4:
            r[-1].append(a)
            w[-1].append(b)
        else:
            r.append([a])
            w.append([b])
    rr = np.array([np.dot(x, y) / np.sum(y) if x[0] > 0 else 0.0 for x, y in zip(r, w)])
    ww = np.array([np.sum(y) for y in w])
    return rr, ww / ww.sum()


def resid(r, w, centre, rho, q):
    m = kern(rho, r) @ w
    Dq = (q / m)[:, None]
    D = (kern(rho, r) * Dq).sum(0)
    Dp = (dkern(rho, r) * Dq).sum(0)
    return np.array(list(D - 1) + [Dp[j] for j in range(len(r)) if not (centre and j == 0)])


def d2kern(rho, r):
    """second radial derivative of the ring kernel in r (closed form; i1e(x)/x -> 1/2 at x = 0)"""
    x = rho[:, None] * r[None, :]
    e = np.exp(-((rho[:, None] - r[None, :]) ** 2) / 2)
    I0, I1 = i0e(x), i1e(x)
    I1x = np.where(x > 1e-12, I1 / np.where(x > 1e-12, x, 1.0), 0.5)
    R_, P = r[None, :], rho[:, None]
    return (
        e * ((P - R_) * (P * I1 - R_ * I0) + P**2 * (I0 - I1x - I1) - I0 - R_ * P * (I1 - I0)) / (2 * np.pi)
    )


def jac_analytic(r, w, centre, rho, q):
    """Jacobian of resid in the variables (r without a centre atom's 0, log w)."""
    K, K1, K2 = kern(rho, r), dkern(rho, r), d2kern(rho, r)
    m = K @ w
    a = q / m
    b = q / m**2
    n = len(r)
    dD_dw = -(K * b[:, None]).T @ K
    dD_dr = -(K * b[:, None]).T @ (K1 * w[None, :]) + np.diag((K1 * a[:, None]).sum(0))
    dP_dw = -(K1 * b[:, None]).T @ K
    dP_dr = -(K1 * b[:, None]).T @ (K1 * w[None, :]) + np.diag((K2 * a[:, None]).sum(0))
    rows_P = [j for j in range(n) if not (centre and j == 0)]
    cols_r = [k for k in range(n) if not (centre and k == 0)]
    top = np.hstack([dD_dr[:, cols_r], dD_dw * w[None, :]])
    bot = np.hstack([dP_dr[np.ix_(rows_P, cols_r)], (dP_dw * w[None, :])[rows_P, :]])
    return np.vstack([top, bot])


def polish(r, w, rho, q, it=100):
    centre = r[0] < 1e-9
    n = len(r)
    pack = lambda r, w: np.concatenate([r[1:] if centre else r, np.log(w)])
    unpack = lambda x: (
        (np.concatenate([[0.0], x[: n - 1]]), np.exp(x[n - 1 :])) if centre else (x[:n], np.exp(x[n:]))
    )
    x = pack(r, w)
    for _ in range(it):
        Fv = resid(*unpack(x), centre, rho, q)
        nF = np.linalg.norm(Fv)
        if np.max(np.abs(Fv)) < 1e-13:
            break
        J = jac_analytic(*unpack(x), centre, rho, q)
        st = np.linalg.lstsq(J, -Fv, rcond=None)[0]
        lam = 1.0
        while lam > 1e-8:
            xn = x + lam * st
            rr, _ = unpack(xn)
            if (
                np.all(np.diff(rr) > 0)
                and rr[0] >= 0
                and np.linalg.norm(resid(*unpack(xn), centre, rho, q)) < nF
            ):
                x = xn
                break
            lam /= 2
        else:
            break
    rr, ww = unpack(x)
    return rr, ww, np.max(np.abs(resid(rr, ww, centre, rho, q)))


for R in RS:
    rho, q = setup(R)
    rg, wg, vg = cnm(R, rho, q)
    r, w = rings_from(rg, wg)
    # second stage: CNM on a fine local grid (step 5e-5 within +-0.005 of each ring) plus the coarse grid far from the rings
    fine = np.unique(
        np.concatenate(
            [np.arange(max(0.0, x - 0.005), x + 0.005, 5e-5) for x in r]
            + [[0.0]]
            + [np.arange(0, R + 4 + 1e-9, 0.002)]
        )
    )
    rg, wg, vg = cnm(R, rho, q, grid=fine, start=(r, w))
    r, w = rings_from(rg, wg)
    if DBG:
        print(
            f"  CNM: grid KKT {vg:.1e}; rings {len(r)}: r {np.round(r[:5], 4).tolist()} w {np.round(w[:5], 8).tolist()}",
            flush=True,
        )
    r, w, res = polish(r, w, rho, q)
    m = kern(rho, r) @ w
    g = np.arange(0, R + 4, 0.001)
    viol = ((kern(rho, g) * (q / m)[:, None]).sum(0)).max() - 1
    f0 = 1 / (np.pi * (R**2 - R1**2))
    mR = (kern(np.array([R]), r) @ w)[0]
    Gt = np.sum(q * np.log(m / f0)) * np.pi * (R**2 - R1**2)
    mR1 = (kern(np.array([R1]), r) @ w)[0]
    acc = (res < 1e-10) and (viol < 1e-9) and abs(w.sum() - 1) < 1e-12
    print(
        f"R {R}: rings {len(r)} (centre atom {'yes' if r[0] < 1e-9 else 'no'}); residual {res:.1e}; KKT max - 1 {viol:+.1e}; mass - 1 "
        f"{w.sum() - 1:+.1e}; {'ACCEPTED' if acc else 'NOT ACCEPTED'}; m(R)/f0 {mR/f0:.10f}; log(m(R)/f0) + 1 {math.log(mR/f0) + 1:+.8e}; "
        f"R (log(m(R)/f0) + 1) {R*(math.log(mR/f0) + 1):+.6f} (D = {D1:+.6f}); R1 {R1}: log(m(R1)/f0) + 1 {math.log(mR1/f0) + 1:+.8e}, "
        f"R1 (log(m(R1)/f0) + 1) {R1*(math.log(mR1/f0) + 1):+.6f}; Gt {Gt:+.8f}; inner rings {np.round(r[:3], 5).tolist()} "
        f"masses {np.round(w[:3], 8).tolist()}",
        flush=True,
    )
    json.dump(
        {"R": R, "r": r.tolist(), "w": w.tolist(), "res": res, "viol": viol, "NQ": NQ},
        open(f"q847_state_R1_{R1}_R{R}.json", "x"),
    )
