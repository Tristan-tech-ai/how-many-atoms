"""q854 (03 Oct 2026, Amendment 1517): TEXT COPY of q851 that also prints Def(t) = int f log(m/f) dx (f the normalised data
density) and Dslope = [Def - D (f(a) + f(-a))]/(t (f(a) - f(-a))), D = -0.304944111846. q851's docstring follows.
q851 (03 Oct 2026, Amendment 1502): population NPMLE in 1D with tilted data, f(x) proportional to exp(t x) on [-a, a], unit Gaussian
kernel. KKT: D(theta) = int f(x) phi(x - theta)/m(x) dx <= 1, equality on the support. Method as q846 (constrained Newton method on a
grid of step 0.002 with the simplex enforced in the NNLS step; atoms = grid clusters merged within 0.4; Newton on (positions, log masses)
with the analytic Jacobian; KKT on a 0.001 grid). Prints depth_L = x_min + a, depth_R = a - x_max and M = (depth_L - depth_R)/(2t).
Usage: py q851_npmle1d_tilt.py a t1,t2,... [NQ panels per unit: default 2 panels of 20 nodes]"""

import sys, math
import numpy as np
from scipy.optimize import nnls

a = float(sys.argv[1])
TS = [float(x) for x in sys.argv[2].split(",")]
xg, wg = np.polynomial.legendre.leggauss(20)
edges = np.arange(-a, a + 1e-9, 0.5)
X = np.concatenate([(l + r) / 2 + (r - l) / 2 * xg for l, r in zip(edges[:-1], edges[1:])])
WX = np.concatenate([(r - l) / 2 * wg for l, r in zip(edges[:-1], edges[1:])])
phi = lambda u: np.exp(-(u**2) / 2) / math.sqrt(2 * math.pi)


def kern(th):
    return phi(X[:, None] - th[None, :])


def dkern(th):
    u = X[:, None] - th[None, :]
    return u * phi(u)


def d2kern(th):
    u = X[:, None] - th[None, :]
    return (u**2 - 1) * phi(u)


def cnm(q, maxit=3000):
    grid = np.arange(-a - 1, a + 1 + 1e-9, 0.002)
    KG = kern(grid)
    S = [int(np.argmin(np.abs(grid - z))) for z in np.linspace(-a, a, 12)]
    w = np.full(len(S), 1 / len(S))
    for it in range(maxit):
        m = KG[:, S] @ w
        D = KG.T @ (q / m)
        viol = D.max() - 1
        if viol < 1e-12:
            break
        lm = [i for i in range(1, len(grid) - 1) if D[i] > 1 and D[i] >= D[i - 1] and D[i] >= D[i + 1]]
        wold = dict(zip(S, w))
        S = sorted(set(S) | set(lm))
        w = np.array([wold.get(i, 0.0) for i in S])
        w /= w.sum()
        m = KG[:, S] @ w
        Am = np.vstack([np.sqrt(q)[:, None] * KG[:, S] / m[:, None], 100.0 * np.ones((1, len(S)))])
        b = np.concatenate([2 * np.sqrt(q), [100.0]])
        v, _ = nnls(Am, b)
        v = v / v.sum()
        L0 = np.sum(q * np.log(m))
        g = np.sum((v - w) * (KG[:, S].T @ (q / m)))
        lam = 1.0
        while lam > 1e-12:
            wn = (1 - lam) * w + lam * v
            if np.sum(q * np.log(KG[:, S] @ wn)) >= L0 + 0.3 * lam * g - 1e-18:
                break
            lam /= 2
        w = wn
        keep = w > 0
        S = [i for i, k in zip(S, keep) if k]
        w = w[keep] / w[keep].sum()
    rg, wg_ = grid[S], w
    r, ws = [], []
    for z, ww in zip(rg, wg_):
        if r and z - r[-1][-1] <= 0.4:
            r[-1].append(z)
            ws[-1].append(ww)
        else:
            r.append([z])
            ws.append([ww])
    th = np.array([np.dot(z, y) / np.sum(y) for z, y in zip(r, ws)])
    wt = np.array([np.sum(y) for y in ws])
    return th, wt / wt.sum(), viol


def resid(th, w, q):
    K = kern(th)
    m = K @ w
    D = (K * (q / m)[:, None]).sum(0)
    Dp = (dkern(th) * (q / m)[:, None]).sum(0)
    return np.concatenate([D - 1, Dp])


def jac(th, w, q):
    K, K1, K2 = kern(th), dkern(th), d2kern(th)
    m = K @ w
    aa = q / m
    bb = q / m**2
    dD_dw = -(K * bb[:, None]).T @ K
    dD_dt = -(K * bb[:, None]).T @ (K1 * w[None, :]) + np.diag((K1 * aa[:, None]).sum(0))
    dP_dw = -(K1 * bb[:, None]).T @ K
    dP_dt = -(K1 * bb[:, None]).T @ (K1 * w[None, :]) + np.diag((K2 * aa[:, None]).sum(0))
    return np.vstack([np.hstack([dD_dt, dD_dw * w[None, :]]), np.hstack([dP_dt, dP_dw * w[None, :]])])


def polish(th, w, q, it=100):
    n = len(th)
    z = np.concatenate([th, np.log(w)])
    for _ in range(it):
        F = resid(z[:n], np.exp(z[n:]), q)
        nF = np.linalg.norm(F)
        if np.max(np.abs(F)) < 1e-14:
            break
        st = np.linalg.lstsq(jac(z[:n], np.exp(z[n:]), q), -F, rcond=None)[0]
        lam = 1.0
        while lam > 1e-9:
            zn = z + lam * st
            if np.all(np.diff(zn[:n]) > 0) and np.linalg.norm(resid(zn[:n], np.exp(zn[n:]), q)) < nF:
                z = zn
                break
            lam /= 2
        else:
            break
    th, w = z[:n], np.exp(z[n:])
    return th, w, np.max(np.abs(resid(th, w, q)))


for t in TS:
    f = np.exp(t * X)
    q = WX * f / np.sum(WX * f)
    th, w, vg = cnm(q)
    th, w, res = polish(th, w, q)
    m = kern(th) @ w
    gq = np.arange(-a - 2, a + 2 + 1e-9, 0.001)
    kkt = (kern(gq) * (q / m)[:, None]).sum(0).max() - 1
    dL, dR = th[0] + a, a - th[-1]
    Zf = np.sum(WX * f)
    fn = f / Zf
    Def = float(np.sum(WX * fn * np.log(m / fn)))
    fa, fma = math.exp(t * a) / Zf, math.exp(-t * a) / Zf
    Dslope = (Def - (-0.304944111846) * (fa + fma)) / (t * (fa - fma)) if t else float("nan")
    print(
        f"   defect: Def = {Def:+.12f}; f(a) = {fa:.10f}, f(-a) = {fma:.10f}; Def/(f(a) + f(-a)) = {Def/(fa + fma):+.9f}; Dslope = {Dslope:+.6f}",
        flush=True,
    )
    acc = res < 1e-10 and kkt < 1e-9 and abs(w.sum() - 1) < 1e-12
    print(
        f"a {a} t {t}: atoms {len(th)}; residual {res:.1e}; KKT max - 1 {kkt:+.1e}; {'ACCEPTED' if acc else 'NOT ACCEPTED'}; depth_L {dL:.7f}, "
        f"depth_R {dR:.7f}; M = (dL - dR)/(2t) {(dL - dR)/(2*t) if t else float('nan'):+.6f}; mean depth {(dL + dR)/2:.7f}",
        flush=True,
    )
