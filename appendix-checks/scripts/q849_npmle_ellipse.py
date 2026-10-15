"""q849 (03 Oct 2026, BACKLOG 71): the population NPMLE for data uniform on the filled ellipse x^2/A^2 + y^2/B^2 <= 1 in the plane, unit
Gaussian noise, without rotation symmetry. D2 symmetry is used: a candidate point c = (x, y) in the closed quarter x, y >= 0 stands for
its mirror orbit {(+-x, +-y)} with the mass spread evenly over the orbit. Data quadrature on the quarter in elliptic-polar coordinates
x = s A cos t, y = s B sin t (Gauss-Legendre NS x NT, Jacobian s A B; weights q sum to 1). Candidates: a square grid of step H inside
s <= 1. Solver: the constrained Newton method of q846 (simplex-enforced NNLS step, backtracking on the likelihood), stopped when
max KKT - 1 < TOL on the candidate grid; then a second stage on a fine local grid (step H2 within +-H of each support point) together
with the coarse grid. The output density m is unique (Gaussian convolution is injective), so a grid-supported optimum gives m up to the
grid's resolution; validate on the disc (A = B).
Prints: support size, KKT on the final grid, L = log(m/f0) + 1 at the major vertex (A, 0) and the minor vertex (0, B), f0 = 1/(pi A B).
Usage: py q849_npmle_ellipse.py A B [H H2 NS NT]   (defaults H 0.15, H2 0.03, NS 120, NT 120)"""

import sys, math
import numpy as np
from scipy.optimize import nnls

A, B = float(sys.argv[1]), float(sys.argv[2])
H = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
H2 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.03
NS = int(sys.argv[5]) if len(sys.argv) > 5 else 120
NT = int(sys.argv[6]) if len(sys.argv) > 6 else 120
f0 = 1 / (math.pi * A * B)
TOL = 1e-10

xs, ws = np.polynomial.legendre.leggauss(NS)
s = (xs + 1) / 2
ws = ws / 2
xt, wt = np.polynomial.legendre.leggauss(NT)
t = (xt + 1) * np.pi / 4
wt = wt * np.pi / 4
S_, T_ = np.meshgrid(s, t, indexing="ij")
YX = (S_ * A * np.cos(T_)).ravel()
YY = (S_ * B * np.sin(T_)).ravel()
Q = ((4 / np.pi) * np.outer(ws * s, wt)).ravel()  # f0 * 4 * (s A B ds dt) = (4/pi) s ds dt


def orbit_kernel(cx, cy, px, py):
    """unit-mass orbit density of candidates (cx, cy) evaluated at points (px, py): matrix len(p) x len(c)"""
    out = np.zeros((len(px), len(cx)))
    for sx in (1, -1):
        for sy in (1, -1):
            out += np.exp(
                -((px[:, None] - sx * cx[None, :]) ** 2 + (py[:, None] - sy * cy[None, :]) ** 2) / 2
            )
    # images that coincide (on an axis) were counted twice or four times: dividing by 4 spreads unit mass over the orbit
    return out / (4 * 2 * np.pi)


def cnm(cx, cy, w0=None, maxit=2000):
    K = orbit_kernel(cx, cy, YX, YY)
    n = len(cx)
    S = list(range(0, n, max(1, n // 40))) if w0 is None else [i for i in range(n) if w0[i] > 0]
    w = np.full(len(S), 1 / len(S)) if w0 is None else w0[S] / w0[S].sum()
    for it in range(maxit):
        m = K[:, S] @ w
        D = K.T @ (Q / m)
        viol = D.max() - 1
        if viol < TOL:
            break
        # add the largest violators that are not adjacent to each other (at most 60 per step)
        cand = np.argsort(-D)[:600]
        add = []
        for i in cand:
            if D[i] <= 1:
                break
            if all((cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2 > (2.5 * H) ** 2 for j in add):
                add.append(i)
            if len(add) >= 60:
                break
        wold = dict(zip(S, w))
        S = sorted(set(S) | set(add))
        w = np.array([wold.get(i, 0.0) for i in S])
        m = K[:, S] @ w
        Am = np.vstack([np.sqrt(Q)[:, None] * K[:, S] / m[:, None], 100.0 * np.ones((1, len(S)))])
        b = np.concatenate([2 * np.sqrt(Q), [100.0]])
        v, _ = nnls(Am, b, maxiter=50 * len(S))
        v = v / v.sum()
        L0 = np.sum(Q * np.log(m))
        g = np.sum((v - w) * (K[:, S].T @ (Q / m)))
        lam = 1.0
        while lam > 1e-12:
            wn = (1 - lam) * w + lam * v
            if np.sum(Q * np.log(K[:, S] @ wn)) >= L0 + 0.3 * lam * g - 1e-17:
                break
            lam /= 2
        w = wn
        keep = w > 0
        S = [i for i, k in zip(S, keep) if k]
        w = w[keep] / w[keep].sum()
        if it % 25 == 0:
            print(f"   cnm it {it}: support {len(S)}, KKT max - 1 {viol:.2e}", flush=True)
    full = np.zeros(n)
    full[S] = w
    return full, viol, K


def grid(step, inside=1.0):
    g = np.arange(0, max(A, B) + step, step)
    X, Y = np.meshgrid(g, g, indexing="ij")
    k = (X / A) ** 2 + (Y / B) ** 2 <= inside**2
    return X[k], Y[k]


cx, cy = grid(H)
print(
    f"A {A} B {B}: data {len(Q)}, coarse candidates {len(cx)}, kernel {len(Q)*len(cx)*8/1e9:.2f} GB",
    flush=True,
)
w, viol, K = cnm(cx, cy)
sup = w > 0
fx = [cx[sup]]
fy = [cy[sup]]
for x0, y0 in zip(cx[sup], cy[sup]):
    gx = np.arange(max(0.0, x0 - H), x0 + H + 1e-9, H2)
    gy = np.arange(max(0.0, y0 - H), y0 + H + 1e-9, H2)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    k = (X / A) ** 2 + (Y / B) ** 2 <= 1.0
    fx.append(X[k])
    fy.append(Y[k])
F = np.unique(np.round(np.vstack([np.concatenate(fx), np.concatenate(fy)]).T, 9), axis=0)
cx2, cy2 = F[:, 0], F[:, 1]
w0 = np.zeros(len(cx2))
idx = {(round(a, 9), round(b, 9)): i for i, (a, b) in enumerate(zip(cx2, cy2))}
for a, b, ww in zip(cx[sup], cy[sup], w[sup]):
    w0[idx[(round(a, 9), round(b, 9))]] = ww
print(f"  stage 2: fine candidates {len(cx2)} (coarse support {sup.sum()})", flush=True)
w2, viol2, K2 = cnm(cx2, cy2, w0=w0)
vx = np.array([A, 0.0])
vy = np.array([0.0, B])
mv = orbit_kernel(cx2[w2 > 0], cy2[w2 > 0], vx, vy) @ w2[w2 > 0]
Lmaj, Lmin = math.log(mv[0] / f0) + 1, math.log(mv[1] / f0) + 1
print(
    f"A {A} B {B} H {H} H2 {H2} NS {NS} NT {NT}: support {int((w2 > 0).sum())} orbits; KKT max - 1 (fine grid) {viol2:+.1e}; "
    f"L major vertex {Lmaj:+.8e}; L minor vertex {Lmin:+.8e}; kappa major {A/B**2:.6f}, minor {B/A**2:.6f}; "
    f"L/kappa major {Lmaj/(A/B**2):+.6f}, minor {Lmin/(B/A**2):+.6f} (D = -0.304944)",
    flush=True,
)
