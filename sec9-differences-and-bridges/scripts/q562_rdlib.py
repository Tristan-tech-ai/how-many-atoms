"""Q562 tool: certified NPMLE (= rate-distortion for squared error, = rational inattention with quadratic loss and Shannon cost) on a
source in noise units, with the test-channel distortion D_noise and mutual information I (nats) of the optimum. Written 30 Sep 15:31:53
(date stamp) for the blind bridge test; q535b's solver copied as text (grid EG, mass peaks, L-BFGS descent, Newton, KKT certificate).
Sources: 'unif' uniform on [-a, a]; 'tnorm' normal with sd a/3 truncated to [-a, a].
Mapping: RD slope beta (nats per unit squared error) on a source of half-width A gives kernel exp(-beta (x - y)^2), noise sd
sigma = 1/sqrt(2 beta), half-length a = A/sigma; D (source units) = D_noise sigma^2 = D_noise A^2/a^2."""

import sys, time, json, os
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import root, minimize

FUNC = "npmle"
TOL = 1e-10
t0 = time.time()
say = lambda *a: print(*a, flush=True)
MODE = "unif"


def setup(a, n=3000):
    xg, wg = leggauss(n)
    x = a * xg
    w = a * wg
    if MODE == "tnorm":
        w = w * np.exp(-x * x / (2 * (a / 3) ** 2))
    return x, w / w.sum()


def gfun(ys, ws, y, xg, qx):
    Pt = (ws[:, None] * np.exp(-((xg[None, :] - ys[:, None]) ** 2) / 2)).sum(0)
    E = np.exp(-((xg[None, :] - np.asarray(y)[:, None]) ** 2) / 2)
    d = xg[None, :] - np.asarray(y)[:, None]
    W = qx / Pt**2 if FUNC == "immune" else qx / Pt
    return (
        (E * W).sum(1),
        (E * d * W).sum(1),
        (E * (d * d - 1) * W).sum(1),
        ((qx / Pt).sum() if FUNC == "immune" else 1.0),
    )


def polish(ys, ws, xg, qx):
    K = len(ys)

    def F(u):
        y, w = u[:K], np.concatenate([u[K:], [1 - u[K:].sum()]])
        if np.any(w <= 0) or np.any(np.diff(y) <= 1e-6):
            return np.full(len(u), 1e3)
        g0, g1, _, C = gfun(y, w, y, xg, qx)
        return np.concatenate([g1 / C, g0[:-1] / C - 1])

    u = np.concatenate([ys, ws[:-1]])
    sol = root(F, u, method="hybr", tol=1e-14)
    u = sol.x
    f = F(u)
    r = np.abs(f).max()
    for _ in range(6):
        if r < 1e-13 or r > 1e-3:
            break
        J = np.empty((len(u), len(u)))
        for i in range(len(u)):
            du = np.zeros_like(u)
            du[i] = 1e-7 * max(1.0, abs(u[i]))
            J[:, i] = (F(u + du) - F(u - du)) / (2 * du[i])
        un = u + np.linalg.lstsq(J, -f, rcond=None)[0]
        fn = F(un)
        rn = np.abs(fn).max()
        if rn >= r:
            break
        u, f, r = un, fn, rn
    return u[:K], np.concatenate([u[K:], [1 - u[K:].sum()]]), r


def signals(a, ys, ws, xg, qx):
    g0, g1, g2, C = gfun(ys, ws, ys, xg, qx)
    yy = np.arange(-a - 3, a + 3, 0.01)
    G0, _, _, _ = gfun(ys, ws, yy, xg, qx)
    ex = 0.3 * np.min(np.diff(ys)) if len(ys) > 1 else 0.3
    far = np.min(np.abs(yy[:, None] - ys[None, :]), 1) > ex
    j = int(np.argmax(G0[far]))
    k = int(np.argmax(g2))
    return G0[far][j] / C - 1, yy[far][j], g2[k] / C, k, C


def grid_solve(a, xg, qx, iters=20000, tmax=600):
    y = np.arange(-a - 1.5, a + 1.5 + 1e-9, 0.04)
    Phi = np.exp(-((xg[None, :] - y[:, None]) ** 2) / 2)
    Fn_ = (lambda P: (qx / P).sum()) if FUNC == "immune" else (lambda P: -(qx * np.log(P)).sum())
    w = np.full(len(y), 1 / len(y))
    P = w @ Phi
    F = Fn_(P)
    eta = 1.0
    t = time.time()
    for it in range(iters):
        g = Phi @ (qx / P**2) if FUNC == "immune" else Phi @ (qx / P)
        Cg = F if FUNC == "immune" else 1.0
        while True:
            wn = w * np.exp(eta * np.clip(g / Cg - 1, -50, 50))
            wn /= wn.sum()
            Pn = wn @ Phi
            Fn = Fn_(Pn)
            if Fn <= F or eta < 1e-8:
                break
            eta /= 2
        w, P, F = wn, Pn, Fn
        eta = min(eta * 1.2, 50.0)
        if time.time() - t > tmax:
            break
    return y, w, F, it


def descend(ys, ws, xg, qx, iters=6000):
    K = len(ys)

    def fg(u):
        y, th = u[:K], u[K:]
        w = np.exp(th - th.max())
        w /= w.sum()
        d = xg[None, :] - y[:, None]
        E = np.exp(-d * d / 2)
        P = w @ E
        W = qx / P**2 if FUNC == "immune" else qx / P
        g0 = E @ W
        g1 = (E * d) @ W
        return ((qx / P).sum() if FUNC == "immune" else -(qx * np.log(P)).sum()), np.concatenate(
            [-w * g1, w * (-g0 + (w * g0).sum())]
        )

    u = np.concatenate([ys, np.log(np.maximum(ws, 1e-300))])
    r = minimize(fg, u, jac=True, method="L-BFGS-B", options={"maxiter": iters, "ftol": 1e-16, "gtol": 1e-13})
    y, th = r.x[:K], r.x[K:]
    w = np.exp(th - th.max())
    w /= w.sum()
    keep = w > 1e-7
    o = np.argsort(y[keep])
    return y[keep][o], (w[keep] / w[keep].sum())[o]


def peaks(y, w):
    m = w > 1e-6 * w.max()
    out = []
    for i in range(1, len(w) - 1):
        if m[i] and w[i] >= w[i - 1] and w[i] > w[i + 1]:
            s = slice(max(0, i - 3), i + 4)
            out.append(((w[s] * y[s]).sum() / w[s].sum(), w[s].sum()))
    ys = np.array([o[0] for o in out])
    ws = np.array([o[1] for o in out])
    return ys, ws / ws.sum()


def solve(a, mode="unif"):
    global MODE
    MODE = mode
    xg, qx = setup(a)
    y, w, F, it = grid_solve(a, xg, qx, tmax=120)
    ys, ws = peaks(y, w)
    ok = False
    for ch in range(15):
        ys, ws = descend(ys, ws, xg, qx)
        y2, w2, r = polish(ys, ws, xg, qx)
        if r > 1e-9:
            ys, ws = descend(ys, ws, xg, qx, iters=30000)
            y2, w2, r = polish(ys, ws, xg, qx)
        if r > 1e-9:
            break
        ys, ws = y2, w2
        b, wb, sp, k, C = signals(a, ys, ws, xg, qx)
        if b <= TOL and sp <= TOL:
            ok = True
            break
        if b >= sp:
            yn = np.concatenate([ys, [wb]])
            wn = np.concatenate([ws * 0.99, [0.01]])
        else:
            yn = np.concatenate([np.delete(ys, k), [ys[k] - 0.15, ys[k] + 0.15]])
            wn = np.concatenate([np.delete(ws, k), [ws[k] / 2] * 2])
        o = np.argsort(yn)
        ys, ws = yn[o], wn[o]
    E = np.exp(-((xg[None, :] - ys[:, None]) ** 2) / 2) * ws[:, None]
    P = E.sum(0)
    post = E / P
    Dn = float((qx * (post * (xg[None, :] - ys[:, None]) ** 2).sum(0)).sum())
    I = float((qx * (post * np.log(np.maximum(post, 1e-300) / ws[:, None])).sum(0)).sum())
    return len(ys), ok, ys, ws, Dn, I
