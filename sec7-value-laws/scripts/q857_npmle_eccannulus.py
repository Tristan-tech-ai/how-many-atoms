"""q857 (04 Oct 2026, BACKLOG 80): population NPMLE for data uniform on an ECCENTRIC annulus in the plane: the disc |y| <= R minus the
disc |y - (e, 0)| < r (inner circle centred at (e, 0)), unit Gaussian noise. Only the mirror symmetry y -> -y is used: a candidate point
c = (x, y), y >= 0, stands for {(x, y), (x, -y)} with its mass spread evenly. Data quadrature on the upper half in polar coordinates about
the inner centre: rho from r to rho_out(phi) (ray from (e, 0) at angle phi to the outer circle), phi in [0, pi], Gauss-Legendre NS x NT,
Jacobian rho; weights Q sum to 1. Candidates: a square grid of step H, y >= 0, inside |y| <= R and outside the disc of radius r - 1 about
the inner centre. Solver: the constrained Newton method of q852/q849, copied as text (simplex-enforced NNLS step, backtracking on the
likelihood), stopped at max KKT - 1 < TOL on the candidate grid. Stage 2 (fine local grid) only if STAGE2=1 (memory: see BACKLOG 80).
Prints support size, KKT, Gt = (L - log f0)/f0, f0 = 1/(pi (R^2 - r^2)).
Usage: py q857_npmle_eccannulus.py R r e [H NS NT]   (defaults H 0.2, NS 40, NT 360)"""

import sys, math, os
import numpy as np
from scipy.optimize import nnls

R, r, e = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
H = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
NS = int(sys.argv[5]) if len(sys.argv) > 5 else 40
NT = int(sys.argv[6]) if len(sys.argv) > 6 else 360
H2 = 0.04
f0 = 1 / (math.pi * (R**2 - r**2))
TOL = 1e-10
assert r + abs(e) < R

xs, ws = np.polynomial.legendre.leggauss(NS)
xt, wt = np.polynomial.legendre.leggauss(NT)
phi = (xt + 1) * np.pi / 2
wphi = wt * np.pi / 2
rho_out = -e * np.cos(phi) + np.sqrt(R**2 - (e * np.sin(phi)) ** 2)  # |(e,0) + rho (cos, sin)| = R
YX, YY, Q = [], [], []
for p, wp, ro in zip(phi, wphi, rho_out):
    rr = r + (xs + 1) * (ro - r) / 2
    wr = ws * (ro - r) / 2
    YX.append(e + rr * np.cos(p))
    YY.append(rr * np.sin(p))
    Q.append(2 * f0 * rr * wr * wp)
YX = np.concatenate(YX)
YY = np.concatenate(YY)
Q = np.concatenate(Q)


def orbit_kernel(cx, cy, px, py):
    """unit-mass orbit density of candidates (cx, cy) at points (px, py): matrix len(p) x len(c); a point on the axis counts twice/2"""
    dx2 = (px[:, None] - cx[None, :]) ** 2
    out = np.exp(-(dx2 + (py[:, None] - cy[None, :]) ** 2) / 2)
    out += np.exp(-(dx2 + (py[:, None] + cy[None, :]) ** 2) / 2)
    return out / (2 * 2 * np.pi)


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


def grid(step):
    gx = np.arange(-R, R + step / 2, step)
    gy = np.arange(0, R + step / 2, step)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    k = (X**2 + Y**2 <= R**2) & ((X - e) ** 2 + Y**2 >= (r - 1) ** 2)
    return X[k], Y[k]


cx, cy = grid(H)
print(
    f"R {R} r {r} e {e} H {H}: data {len(Q)} (sum Q {Q.sum():.12f}), candidates {len(cx)}, kernel {len(Q)*len(cx)*8/1e9:.2f} GB",
    flush=True,
)
w, viol, K = cnm(cx, cy)
if os.environ.get("STAGE2") == "1":
    sup = w > 0
    fx = [cx[sup]]
    fy = [cy[sup]]
    for x0, y0 in zip(cx[sup], cy[sup]):
        gx = np.arange(x0 - H / 2, x0 + H / 2 + 1e-9, H2)
        gy = np.arange(max(0.0, y0 - H / 2), y0 + H / 2 + 1e-9, H2)
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        k = X**2 + Y**2 <= R**2
        fx.append(X[k])
        fy.append(Y[k])
    F = np.unique(np.round(np.vstack([np.concatenate(fx), np.concatenate(fy)]).T, 9), axis=0)
    cx2, cy2 = F[:, 0], F[:, 1]
    w0 = np.zeros(len(cx2))
    idx = {(round(a, 9), round(b, 9)): i for i, (a, b) in enumerate(zip(cx2, cy2))}
    for a, b, ww in zip(cx[sup], cy[sup], w[sup]):
        w0[idx[(round(a, 9), round(b, 9))]] = ww
    print(f"  stage 2: fine candidates {len(cx2)}", flush=True)
    w, viol, K = cnm(cx2, cy2, w0=w0)
Lopt = float(np.sum(Q * np.log(K @ w)))
Gt = (Lopt - math.log(f0)) / f0
print(
    f"R {R} r {r} e {e} H {H} NS {NS} NT {NT} STAGE2 {os.environ.get('STAGE2', '0')}: support {int((w > 0).sum())} orbits; "
    f"KKT max - 1 {viol:+.1e}; L = {Lopt:.12f}; Gt = {Gt:+.6f}",
    flush=True,
)
