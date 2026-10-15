"""q850 (03 Oct 2026, Amendment 1499): 1D amplitude-constrained Gaussian channel with a linear input reward,
maximise I(X; Y) + t E[X] over X in [-A, A], Y = X + Z, Z ~ N(0, 1). KKT: g(x) = i(x) + t x - C' <= 0 on [-A, A], equality on the
support, i(x) = D(N(x, 1) || p_Y) = -log(2 pi e)/2 - int phi(y - x) log p(y) dy. The walls +-A carry atoms.
Method: Blahut-Arimoto with the reward (P <- P exp(i + t x), normalised) on a grid of step 0.005 for the support; clusters -> atoms;
Newton on (interior positions, log masses of all atoms, C') with g = 0 at every atom, g' = 0 at interior atoms, sum of masses 1;
KKT max of g on a 0.001 grid. Output: depth_R = A - (largest interior atom), depth_L = (smallest interior atom) + A.
Usage: py q850_cap1d_tilt.py A t1,t2,...  [BA iterations, default 4000]"""

import sys, math
import numpy as np

A = float(sys.argv[1])
TS = [float(x) for x in sys.argv[2].split(",")]
NBA = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
# output quadrature: composite Gauss-Legendre, panels of width 0.5 on [-A - 10, A + 10]
xg, wg = np.polynomial.legendre.leggauss(20)
edges = np.arange(-A - 10, A + 10 + 1e-9, 0.5)
Y = np.concatenate([(a + b) / 2 + (b - a) / 2 * xg for a, b in zip(edges[:-1], edges[1:])])
WY = np.concatenate([(b - a) / 2 * wg for a, b in zip(edges[:-1], edges[1:])])
phi = lambda u: np.exp(-(u**2) / 2) / math.sqrt(2 * math.pi)
H0 = 0.5 * math.log(2 * math.pi * math.e)


def p_of(x, w):
    return phi(Y[:, None] - x[None, :]) @ w


def i_of(xq, p):
    return -H0 - (phi(Y[:, None] - xq[None, :]) * (WY * np.log(p))[:, None]).sum(0)


def di_of(xq, p):
    return -(((Y[:, None] - xq[None, :]) * phi(Y[:, None] - xq[None, :])) * (WY * np.log(p))[:, None]).sum(0)


def ba(t):
    xs = np.linspace(-A, A, int(round(2 * A / 0.005)) + 1)
    K = phi(Y[:, None] - xs[None, :])
    P = np.full(len(xs), 1 / len(xs))
    for it in range(NBA):
        p = K @ P
        iv = -H0 - K.T @ (WY * np.log(p))
        P = P * np.exp(iv + t * xs - (iv + t * xs).max())
        P /= P.sum()
    return xs, P


def atoms_from(xs, P):
    pk = [
        i
        for i in range(len(xs))
        if P[i] > 1e-6 * P.max() and (i == 0 or P[i] >= P[i - 1]) and (i == len(xs) - 1 or P[i] > P[i + 1])
    ]
    cuts = [0] + [pk[k] + int(np.argmin(P[pk[k] : pk[k + 1] + 1])) for k in range(len(pk) - 1)] + [len(xs)]
    x = np.array(
        [
            np.sum(xs[cuts[k] : cuts[k + 1]] * P[cuts[k] : cuts[k + 1]]) / np.sum(P[cuts[k] : cuts[k + 1]])
            for k in range(len(pk))
        ]
    )
    w = np.array([np.sum(P[cuts[k] : cuts[k + 1]]) for k in range(len(pk))])
    x[0], x[-1] = -A, A  # the wall atoms sit on the walls
    return x, w / w.sum()


def resid(z, n, t):
    xin = z[: n - 2]
    w = np.exp(z[n - 2 : 2 * n - 2])
    C = z[-1]
    x = np.concatenate([[-A], xin, [A]])
    p = p_of(x, w)
    g = i_of(x, p) + t * x - C
    gp = di_of(xin, p) + t
    return np.concatenate([g, gp, [w.sum() - 1]])


def newton(x, w, t):
    n = len(x)
    p = p_of(x, w)
    C = np.mean(i_of(x, p) + t * x)
    z = np.concatenate([x[1:-1], np.log(w), [C]])
    for it in range(80):
        F = resid(z, n, t)
        nF = np.linalg.norm(F)
        if np.max(np.abs(F)) < 1e-13:
            break
        J = np.zeros((len(F), len(z)))
        for j in range(len(z)):
            zp = z.copy()
            h = 1e-7 * max(1.0, abs(z[j]))
            zp[j] += h
            J[:, j] = (resid(zp, n, t) - F) / h
        st = np.linalg.lstsq(J, -F, rcond=None)[0]
        lam = 1.0
        while lam > 1e-9:
            zn = z + lam * st
            xin = zn[: n - 2]
            if np.all(np.diff(np.concatenate([[-A], xin, [A]])) > 0) and np.linalg.norm(resid(zn, n, t)) < nF:
                z = zn
                break
            lam /= 2
        else:
            break
    xin = z[: n - 2]
    w = np.exp(z[n - 2 : 2 * n - 2])
    C = z[-1]
    return np.concatenate([[-A], xin, [A]]), w, C, np.max(np.abs(resid(z, n, t)))


import os, json

SEEDK = os.environ.get(
    "SEEDK"
)  # seed from the record's 1D capacity state c60_K<K>.json (entry nearest A), then continue in t
prev = None
if SEEDK:
    h = json.load(open(f"c60_K{SEEDK}.json"))["hist"]
    e = min(h, key=lambda q: abs(q["A"] - A))
    pp = np.array(e["p"]) * A / e["A"]
    xin = np.concatenate([-pp[::-1], [0.0] if int(SEEDK) % 2 else [], pp])
    x0 = np.concatenate([[-A], xin, [A]])
    w0 = np.full(len(x0), 1 / len(x0))
    K0 = phi(Y[:, None] - x0[None, :])
    for it in range(3000):  # Blahut-Arimoto on the fixed atoms for the masses
        iv = -H0 - K0.T @ (WY * np.log(K0 @ w0))
        w0 = w0 * np.exp(iv - iv.max())
        w0 /= w0.sum()
    prev = (x0, w0)
    print(f"seed from c60_K{SEEDK}.json entry A = {e['A']:.6f} (positions rescaled to A = {A})", flush=True)
for t in TS:
    if prev is not None:
        x, w = prev
    else:
        xs, P = ba(t)
        x, w = atoms_from(xs, P)
    gq = np.linspace(-A, A, int(round(2 * A / 0.001)) + 1)
    for rnd in range(12):
        x, w, C, res = newton(x, w, t)
        p = p_of(x, w)
        g = i_of(gq, p) + t * gq - C
        kkt = g.max()
        if kkt < 1e-9 or res > 1e-10:
            break
        # insert an interior atom at the largest violation with a small mass taken evenly from the others
        xn = gq[int(np.argmax(g))]
        if np.min(np.abs(x - xn)) < 1e-3:
            break
        k = int(np.searchsorted(x, xn))
        x = np.insert(x, k, xn)
        w = np.insert(w * (1 - 1e-3), k, 1e-3)
    dR = A - x[-2]
    dL = x[1] + A
    if res < 1e-10 and kkt < 1e-9:
        prev = (x, w)
    print(
        f"A {A} t {t}: atoms {len(x)} (walls w {w[0]:.6f}, {w[-1]:.6f}); residual {res:.1e}; KKT max {kkt:+.1e}; I + tE = {C:.10f}; "
        f"depth_L {dL:.6f}, depth_R {dR:.6f}; (dR - dL)/(2t) {(dR - dL)/(2*t) if t else float('nan'):+.5f}; mean depth {(dR + dL)/2:.6f}",
        flush=True,
    )
