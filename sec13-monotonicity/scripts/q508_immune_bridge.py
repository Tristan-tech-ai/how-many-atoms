"""Q508: bridge 3, IMMUNOLOGY. The optimal immune repertoire of Mayer, Balasubramanian, Mora, Walczak (PNAS 2015; arXiv 1407.6888, text
extracted in papers/mayer_2015_immune.txt): minimise the expected cost of infections
    <F> = sum_a Q_a Fbar_a(Ptilde_a),   Ptilde_a = sum_r P_r f(r - a),   f(r - a) = exp(-(r - a)^2/(2 sigma^2)),
convex in P (their App. B); for the linear cost F(m) = m, Fbar = 1/Ptilde (their eq. B1). That is the record's class: a convex
(concave after a sign) functional of the Gaussian-smoothed measure, so the common lemma (NPMLE_DERIVATION.md section 1) gives the same
leading phase law rho^2 rho' = s0 = 1/(16 pi^2), independent of F. Mayer et al. report, from numerics on random environments, that "the
peaks are spaced evenly, at roughly the cross-reactivity scale sigma".

THE TEST MODEL. One-dimensional shape space, antigen density Q uniform on [-a, a], sigma = 1 (L = a in units of sigma), F(m) = m:
minimise C(P) = int Q(x)/Ptilde(x) dx. KKT: g(r) = int Q(x) f(r - x)/Ptilde(x)^2 dx <= C, equality on the support. Receptors may sit on
[-a - 3, a + 3]. Solver written afresh (exponentiated-gradient seed on a grid, mass-peak clustering, Newton polish of g(y_k) = C,
g'(y_k) = 0, KKT check), no import of the record's solvers.

PRE-REGISTERED, before the first run:
 P1 (discreteness): at a = 12, 18 and 24 the optimum is a finite set of atoms with the KKT inequality holding (max off-support g - C < 0
    and g'' < 0 at every atom, residual < 1e-10).
 P2 (the phase law, parameter-free): in the bulk (atoms at depth d >= 3 from the nearest end and farther than 1.5 from the centre), the
    chirp constant s = 16 pi^2 rho^2 rho' measured from consecutive atom spacings (rho = 1/gap at gap midpoints, rho' by central
    differences of rho) has its median in [0.8, 1.5] at a = 18 and 24. Uniform spacing ("evenly spaced") would give s = 0.
 P3 (the count): K(a) at a = 12, 18, 24 exceeds 0.4002 a^(4/3) (the leading term) and the local log-slope of K between a = 12 and 24 is
    above 1.0 (faster than evenly spaced peaks, whose count would grow linearly).
 PASS needs P1 and P2; P3 is reported. FAIL of P2 with P1 holding means the phase law does not carry to this functional (label M) unless
 the diagnosis finds an instrument fault (label B).
Usage: py q508_immune_bridge.py [a1 a2 ...]"""

import sys, time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import root

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
say = lambda *a: print(*a, flush=True)
t0 = time.time()
AS = [float(v) for v in sys.argv[1:]] or [12.0, 18.0, 24.0]


def setup(a, nx=4000):
    xg, wg = leggauss(nx)
    return a * xg, a * wg / (2 * a)  # Q uniform: density 1/(2a)


def gfun(ys, ws, y, xg, qx):
    Pt = (ws[:, None] * np.exp(-((xg[None, :] - ys[:, None]) ** 2) / 2)).sum(0)
    E = np.exp(-((xg[None, :] - np.asarray(y)[:, None]) ** 2) / 2)
    d = xg[None, :] - np.asarray(y)[:, None]
    w = qx / Pt**2
    return (E * w).sum(1), (E * d * w).sum(1), (E * (d * d - 1) * w).sum(1), (qx / Pt).sum()


def seed(a, xg_unused, qx_unused, n=601, iters=8000, eta=0.5, nx=900):
    xg, qx = setup(a, nx)  # coarse rule for the seed only; the polish uses the fine rule
    y = np.linspace(-a - 3, a + 3, n)
    P = np.full(n, 1.0 / n)
    K = np.exp(-((xg[None, :] - y[:, None]) ** 2) / 2)
    for it in range(iters):
        Pt = P @ K
        g = (K * (qx / Pt**2)).sum(1)
        C = (qx / Pt).sum()
        P = P * (g / C) ** eta
        P /= P.sum()
        if it % 200 == 0 and g.max() / C - 1 < 1e-9:
            break
    return y, P, it


def peaks(y, P, thr=1e-7):
    pk = [
        i
        for i in range(len(P))
        if P[i] > thr * P.max() and (i == 0 or P[i] >= P[i - 1]) and (i == len(P) - 1 or P[i] > P[i + 1])
    ]
    lab = np.argmin(np.abs(y[:, None] - y[pk][None, :]), 1)
    return (
        np.array([(y[lab == k] * P[lab == k]).sum() / P[lab == k].sum() for k in range(len(pk))]),
        np.array([P[lab == k].sum() for k in range(len(pk))]),
    )


def polish(ys, ws, xg, qx):
    K = len(ys)

    def F(v):
        yy, ww = v[:K], v[K:]
        if np.any(ww <= 0):
            return np.full(2 * K, 1e3)
        g0, g1, _, C = gfun(yy, ww / ww.sum(), yy, xg, qx)
        return np.concatenate([g0 / C - 1, g1 / C])

    sol = root(F, np.concatenate([ys, ws]), method="hybr", tol=1e-14)
    v = sol.x
    ys, ws = v[:K], v[K:] / v[K:].sum()
    o = np.argsort(ys)
    return ys[o], ws[o], np.abs(F(sol.x)).max()


if __name__ == "__main__":
    say(
        f"q508 immune-repertoire bridge: Q uniform on [-a, a], Gaussian cross-reactivity sigma = 1, F(m) = m; a in {AS}"
    )
    counts = {}
    for a in AS:
        xg, qx = setup(a)
        y, P, it = seed(a, xg, qx)
        ys, ws = peaks(y, P)
        ys, ws, res = polish(ys, ws, xg, qx)
        for rnd in range(6):  # KKT refinement with residual check
            g0, g1, g2, C = gfun(ys, ws, ys, xg, qx)
            yy = np.linspace(-a - 3, a + 3, 8001)
            G0, _, _, _ = gfun(ys, ws, yy, xg, qx)
            far = np.min(np.abs(yy[:, None] - ys[None, :]), 1) > 0.2
            viol = (G0[far] / C - 1).max()
            if res < 1e-10 and viol < 0 and (g2 / C).max() < 0:
                break
            if (g2 / C).max() > 0:
                j = int(np.argmax(g2))
                ys = np.concatenate([np.delete(ys, j), [ys[j] - 0.15, ys[j] + 0.15]])
                ws = np.concatenate([np.delete(ws, j), [ws[j] / 2] * 2])
            elif viol > 0:
                ys = np.concatenate([ys, [yy[far][np.argmax(G0[far])]]])
                ws = np.concatenate([ws * 0.97, [0.03]])
            yp, wp, rp = polish(ys, ws, xg, qx)
            if rp < res or rp < 1e-10:
                ys, ws, res = yp, wp, rp
            else:
                say(f"   refinement at a = {a} did not improve (residual {rp:.1e}); stopping refinement")
                break
        g0, g1, g2, C = gfun(ys, ws, ys, xg, qx)
        yy = np.linspace(-a - 3, a + 3, 8001)
        G0, _, _, _ = gfun(ys, ws, yy, xg, qx)
        far = np.min(np.abs(yy[:, None] - ys[None, :]), 1) > 0.2
        viol = (G0[far] / C - 1).max()
        K = len(ys)
        counts[a] = K
        say(
            f"\na = {a}: seed {it} iterations; K = {K} atoms, residual {res:.1e}, max off-support g/C - 1 = {viol:+.2e}, max g''/C at atoms {(g2/C).max():+.2e}   [{time.time() - t0:.0f}s]"
        )
        say(f"   atoms {np.round(ys, 3)}")
        gaps = np.diff(ys)
        mid = 0.5 * (ys[1:] + ys[:-1])
        rho = 1 / gaps
        rp = np.gradient(rho, mid)
        dpt = a - np.abs(mid)
        bulk = (dpt >= 3) & (np.abs(mid) > 1.5)
        s = 16 * np.pi**2 * rho**2 * rp * np.sign(-mid)  # density rises inward on both sides
        say(f"   gaps {np.round(gaps, 3)}")
        if bulk.sum() >= 2:
            say(
                f"   bulk chirp s = 16 pi^2 rho^2 rho' (inward): {np.round(s[bulk], 3)}; median {np.median(s[bulk]):.3f}  (law: about 1; evenly spaced: 0)"
            )
        say(f"   K = {K} against the leading term 0.4002 a^(4/3) = {0.4002442498*a**(4/3):.2f}")
    if len(AS) >= 2:
        a1, a2 = min(AS), max(AS)
        say(
            f"\nlog-slope of K between a = {a1} and {a2}: {np.log(counts[a2]/counts[a1])/np.log(a2/a1):.3f}  (evenly spaced peaks: 1.0; law 4/3 asymptotically)"
        )
    say(f"[{time.time() - t0:.0f}s]")
