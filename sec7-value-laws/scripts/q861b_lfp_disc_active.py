"""q861b = q861 + active-set loop (see the comment block after polish). q861 (04 Oct 2026): the least favourable prior (LFP) for the 2D bounded normal mean on the disc |theta| <= m, unit noise, squared loss
(sum over both coordinates), restricted to rotation-invariant priors (rings). For rings (radii r_k, masses w_k) the posterior mean is
radial, delta(y) = d(|y|) y/|y| with
   d(rho) = sum_k w_k r_k e^{-(rho - r_k)^2/2} i1e(r_k rho) / sum_k w_k e^{-(rho - r_k)^2/2} i0e(r_k rho),
and the risk at radius tau is the 1D integral
   r(tau) = int_0^inf rho e^{-(rho - tau)^2/2} [ (d(rho)^2 + tau^2) i0e(rho tau) - 2 d(rho) tau i1e(rho tau) ] d rho.
The Bayes risk B = sum_k w_k r(r_k) is concave in the prior with gradient r(tau); LFP KKT: r(tau) <= B on [0, m], equality on the
support. Stage 1: exponentiated-gradient ascent on a radial grid (step HC); stage 2: rings = grid clusters, Newton on (interior radii,
masses, B) with r(r_k) = B, r'(r_k) = 0 (interior rings), sum w = 1; KKT checked on a 0.002 grid.
Usage: py q861_lfp_disc.py m [HC NRHO]   (prints rings, B, KKT; state q861_state_m<m>.json)"""

import sys, json, math
import numpy as np
from scipy.special import i0e, i1e

m = float(sys.argv[1])
HC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
NRHO = int(sys.argv[3]) if len(sys.argv) > 3 else 600
xg, wg = np.polynomial.legendre.leggauss(NRHO)
RMAX = m + 9.0
rho = (xg + 1) * RMAX / 2
wrho = wg * RMAX / 2


def dfun(r, w):
    E = np.exp(-((rho[:, None] - r[None, :]) ** 2) / 2)
    num = (E * i1e(rho[:, None] * r[None, :])) @ (w * r)
    den = (E * i0e(rho[:, None] * r[None, :])) @ w
    return num / den


def risk(tau, d):
    E = np.exp(-((rho[:, None] - tau[None, :]) ** 2) / 2)
    z = rho[:, None] * tau[None, :]
    integ = (
        rho[:, None]
        * E
        * ((d[:, None] ** 2 + tau[None, :] ** 2) * i0e(z) - 2 * d[:, None] * tau[None, :] * i1e(z))
    )
    return wrho @ integ


def stage1(grid, iters=4000):
    w = np.full(len(grid), 1 / len(grid))
    eta = 2.0
    for it in range(iters):
        d = dfun(grid, w)
        r = risk(grid, d)
        B = w @ r
        w = w * np.exp(eta * (r - r.max()))
        w /= w.sum()
        w[w < 1e-14] = 0
        w /= w.sum()
    return w, r.max() - B


def clusters(grid, w, gap=0.3):
    idx = np.where(w > 1e-9)[0]
    out = []
    cur = [idx[0]]
    for i in idx[1:]:
        if grid[i] - grid[cur[-1]] > gap:
            out.append(cur)
            cur = [i]
        else:
            cur.append(i)
    out.append(cur)
    return [(float(np.sum(grid[c] * w[c]) / np.sum(w[c])), float(np.sum(w[c]))) for c in out]


def polish(r, w):
    wall = abs(r[-1] - m) < 0.05
    if wall:
        r[-1] = m
    free = [k for k in range(len(r)) if not (wall and k == len(r) - 1) and r[k] > 0.05]
    ctr = [k for k in range(len(r)) if r[k] <= 0.05]  # a cluster at the origin is a centre atom (r = 0)
    for k in ctr:
        r[k] = 0.0

    def unpack(z):
        rr = r.copy()
        rr[free] = z[: len(free)]
        ww = np.exp(z[len(free) : len(free) + len(r)])
        return rr, ww / ww.sum(), z[-1]

    def F(z):
        rr, ww, B = unpack(z)
        d = dfun(rr, ww)
        h = 1e-5
        rv = risk(rr, d)
        res = list(rv - B)
        for k in free:
            res.append((risk(np.array([rr[k] + h]), d)[0] - risk(np.array([rr[k] - h]), d)[0]) / (2 * h))
        return np.array(res)

    z = np.concatenate([r[free], np.log(w), [w @ risk(r, dfun(r, w))]])
    for it in range(60):
        f = F(z)
        nf = np.max(np.abs(f))
        if nf < 1e-12:
            break
        J = np.zeros((len(f), len(z)))
        e = 1e-7
        for j in range(len(z)):
            zp = z.copy()
            zp[j] += e
            J[:, j] = (F(zp) - f) / e
        st = np.linalg.lstsq(J, -f, rcond=None)[0]
        lam = 1.0
        while lam > 1e-6:
            zn = z + lam * st
            rr = unpack(zn)[0]
            if np.all(rr[free] > 0) and np.all(rr <= m + 1e-12) and np.max(np.abs(F(zn))) < nf:
                z = zn
                break
            lam /= 2
        else:
            break
    rr, ww, B = unpack(z)
    return rr, ww, B, np.max(np.abs(F(z)))


# ---- q861b (04 Oct 2026): TEXT COPY of q861 above (functions unchanged) with an active-set loop in place of the single polish:
# a free ring whose radius falls below 0.02 becomes the centre atom; a ring whose mass falls below 1e-5 while the polish stalls is
# dropped; while the KKT function r - B exceeds 1e-9 on a 0.002 grid, a ring (or the centre atom, below 0.03) is inserted at the worst
# point with mass 1e-3. States q861b_state_m<m>.json.
grid = np.arange(0, m + HC / 2, HC)
grid[-1] = m
w, gap1 = stage1(grid)
cl = clusters(grid, w)
r = np.array([c[0] for c in cl])
wv = np.array([c[1] for c in cl])
if abs(r[-1] - m) < 0.05:
    r[-1] = m
g = np.arange(0, m + 1e-12, 0.002)
g[-1] = m
for rnd in range(16):
    r, wv, B, res = polish(r.copy(), wv.copy())
    small = [k for k in range(len(r)) if 0 < r[k] < 0.02]
    if small:
        k = small[0]
        r[k] = 0.0
        if np.sum(r == 0.0) > 1:
            z0 = np.where(r == 0.0)[0]
            wv[z0[0]] += wv[z0[1:]].sum()
            keep = np.ones(len(r), bool)
            keep[z0[1:]] = False
            r, wv = r[keep], wv[keep]
        continue
    if res > 1e-10 and wv.min() < 1e-5 and len(r) > 1:
        keep = wv > wv.min()
        r, wv = r[keep], wv[keep] / wv[keep].sum()
        continue
    rv = risk(g, dfun(r, wv)) - B
    kkt = rv.max()
    if kkt < 1e-9 and res < 1e-10:
        break
    t = g[int(np.argmax(rv))]
    if t < 0.03 and not np.any(r == 0.0):
        r = np.concatenate([[0.0], r])
        wv = np.concatenate([[1e-3], wv])
    elif t >= 0.03:
        k = int(np.searchsorted(r, t))
        r = np.insert(r, k, t)
        wv = np.insert(wv, k, 1e-3)
    else:
        break
    wv = wv / wv.sum()
kkt = (risk(g, dfun(r, wv)) - B).max()
acc = res < 1e-10 and kkt < 1e-9
print(
    f"m {m}: rings {len(r)} radii {np.round(r, 6).tolist()} masses {np.round(wv, 8).tolist()}; B {B:.12f}; 2 - B {2 - B:.8f}; "
    f"residual {res:.1e}; KKT max r - B {kkt:+.1e}; stage-1 gap {gap1:.1e}; rounds {rnd + 1}; {'ACCEPTED' if acc else 'NOT ACCEPTED'}",
    flush=True,
)
json.dump(
    {"m": m, "r": r.tolist(), "w": wv.tolist(), "B": B, "res": res, "kkt": float(kkt)},
    open(f"q861b_state_m{m}.json", "w"),
)
