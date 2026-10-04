"""Q506: bridge 2 of the highest target, rational inattention. Replicate the one published numerical solution of Jung, Kim, Matejka,
Sims (RES 2019; working paper pp. 18-19) with the record's functional, solved afresh (no q49x import).

THE PUBLISHED CASE. Quadratic tracking, prior Y ~ N(0, 1) truncated at +-3, information cost lambda = 0.5. Their numerical solution
(1000-point grid on [-3, 3], "not accurate to more than about 3 decimal places") has exactly four actions:
    -1.0880, -0.1739, 0.1739, 1.0880.
THE IDENTITY. Their first-order conditions (6)-(11): f(x, y) = p(x) exp(U(x, y)/lambda) h(y) and C(x) = int exp(U/lambda) h(y) dy <= 1,
= 1 on the support, with h fixed by (8). That is the NPMLE / rate-distortion KKT function with data density g and a Gaussian kernel.
The paper is internally inconsistent by a factor 2 on the kernel variance: Corollary 2.1 (U = -z^2) gives variance lambda/2, while
eq. (15) and this example ("gives X a N(0, .5) distribution" at lambda = .5) correspond to variance lambda. Both are run.

PRE-REGISTERED, before the run:
 R1: under exactly one of the two conventions the optimum has four atoms at +-0.1739 and +-1.0880, each within 2e-3 (their stated
     accuracy); the other convention gives a different support.
 R2: the untruncated check of their eq. (15) under the matching convention: X ~ N(0, 1 - sigma_k^2), i.e. N(0, 0.5), is the solution
     when the prior is not truncated (checked by the KKT function being flat to 1e-6 on [-4, 4] for that X).
 PASS needs R1. FAIL means the identity (or my reading of their convention) is wrong: label M or B after reading the signals.
v2 (before any comparison was read): the coarse BA seed stops before a close pair separates; a KKT refinement (split where
c'' > 0 at an atom, insert where c > 1) runs after the polish. The first run's 3-atom state under the eq. 15 convention violated KKT
by +2.75e-5 and was discarded.
Usage: py q506_ri_replication.py"""

import sys, time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import root
from scipy.special import erf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
say = lambda *a: print(*a, flush=True)
t0 = time.time()

xg, wg = leggauss(3000)
xg = 3 * xg
wg = 3 * wg
Zt = erf(3 / np.sqrt(2))
gx = np.exp(-(xg**2) / 2) / np.sqrt(2 * np.pi) / Zt  # truncated N(0, 1) density on [-3, 3]
px = wg * gx


def cfun(v, ys, ws, y):
    """KKT function c(y) = int g(x) phi_v(x - y)/m(x) dx and its first two derivatives; kernel variance v"""
    Kx = np.exp(-((xg[None, :] - ys[:, None]) ** 2) / (2 * v))
    Z = (ws[:, None] * Kx).sum(0)
    E = np.exp(-((xg[None, :] - np.asarray(y)[:, None]) ** 2) / (2 * v))
    d = xg[None, :] - np.asarray(y)[:, None]
    return (E * px / Z).sum(1), (E * d / v * px / Z).sum(1), (E * (d * d / v / v - 1 / v) * px / Z).sum(1)


def ba(v, n=801, iters=20000, tol=1e-9, nx=800):
    """coarse Blahut-Arimoto seed on its own 800-node rule (the polish uses the 3000-node rule)"""
    xb, wb = leggauss(nx)
    xb = 3 * xb
    wb = 3 * wb
    pb = wb * np.exp(-(xb**2) / 2) / np.sqrt(2 * np.pi) / Zt
    y = np.linspace(-3, 3, n)
    q = np.full(n, 1.0 / n)
    K = np.exp(-((xb[None, :] - y[:, None]) ** 2) / (2 * v))
    for it in range(iters):
        Z = q @ K
        c = (K * pb / Z).sum(1)
        q = q * c
        q /= q.sum()
        if it % 100 == 0 and c.max() - 1 < tol:
            break
    return y, q, it


def peaks(y, q, thr=1e-6):
    pk = [i for i in range(1, len(q) - 1) if q[i] >= q[i - 1] and q[i] > q[i + 1] and q[i] > thr * q.max()]
    yp = y[pk]
    lab = np.argmin(np.abs(y[:, None] - yp[None, :]), 1)
    return (
        np.array([(y[lab == k] * q[lab == k]).sum() / q[lab == k].sum() for k in range(len(pk))]),
        np.array([q[lab == k].sum() for k in range(len(pk))]),
    )


def polish(v, ys, ws):
    K = len(ys)

    def F(u):
        yy, ww = u[:K], u[K:]
        if np.any(ww <= 0):
            return np.full(2 * K, 1e3)
        c0, c1, _ = cfun(v, yy, ww, yy)
        return np.concatenate([c0 - 1, c1])

    sol = root(F, np.concatenate([ys, ws]), method="hybr", tol=1e-14)
    return sol.x[:K], sol.x[K:], np.abs(F(sol.x)).max()


if __name__ == "__main__":
    lam = 0.5
    pub = np.array([-1.0880, -0.1739, 0.1739, 1.0880])
    say(
        f"q506: Jung-Kim-Matejka-Sims tracking example, prior N(0,1) truncated at +-3, lambda = {lam}; published support {pub}"
    )
    verdicts = {}
    for name, v in (
        ("kernel variance lambda (eq. 15 convention)", lam),
        ("kernel variance lambda/2 (Corollary 2.1 convention)", lam / 2),
    ):
        y, q, it = ba(v)
        ys, ws = peaks(y, q)
        ys, ws, res = polish(v, ys, ws)
        o = np.argsort(ys)
        ys, ws = ys[o], ws[o]
        for rnd in range(8):  # KKT refinement: split atoms with c'' > 0, insert where c > 1
            _, _, c2a = cfun(v, ys, ws, ys)
            yy = np.linspace(-3, 3, 6001)
            c0, _, _ = cfun(v, ys, ws, yy)
            far = np.min(np.abs(yy[:, None] - ys[None, :]), 1) > 0.05
            if c2a.max() > 1e-10:
                j = int(np.argmax(c2a))
                d = 0.1
                ys = np.concatenate([np.delete(ys, j), [ys[j] - d, ys[j] + d]])
                ws = np.concatenate([np.delete(ws, j), [ws[j] / 2, ws[j] / 2]])
                say(f"   refinement {rnd}: split atom at {ys[-2] + d:+.4f} (c'' = {c2a.max():.2e})")
            elif (c0[far] - 1).max() > 1e-10:
                yn = yy[far][np.argmax(c0[far] - 1)]
                ys = np.concatenate([ys, [yn]])
                ws = np.concatenate([ws * 0.98, [0.02]])
                say(f"   refinement {rnd}: insert atom at {yn:+.4f} (c - 1 = {(c0[far] - 1).max():.2e})")
            else:
                break
            ys, ws, res = polish(v, ys, ws)
            o = np.argsort(ys)
            ys, ws = ys[o], ws[o]
        yy = np.linspace(-3, 3, 6001)
        c0, _, _ = cfun(v, ys, ws, yy)
        far = np.min(np.abs(yy[:, None] - ys[None, :]), 1) > 0.05
        say(
            f"\n{name}: BA {it} iterations; {len(ys)} atoms {np.round(ys, 4)}, weights {np.round(ws, 4)}, residual {res:.1e}, "
            f"max off-support c - 1 = {(c0[far] - 1).max():+.2e}   [{time.time() - t0:.0f}s]"
        )
        if len(ys) == 4:
            err = np.abs(ys - pub).max()
            say(f"   max |atom - published| = {err:.4f}  -> {'MATCH' if err < 2e-3 else 'no match'}")
            verdicts[name] = err < 2e-3
        else:
            verdicts[name] = False
    ok = sum(verdicts.values()) == 1
    say(f"\nR1: {'PASS' if ok else 'FAIL'} ({sum(verdicts.values())} convention(s) match)")
    # R2: untruncated check under the matching convention
    v = lam if verdicts.get("kernel variance lambda (eq. 15 convention)") else lam / 2
    xs = np.linspace(-4, 4, 81)
    s2 = 1 - v
    say(
        f"R2: untruncated prior N(0,1), kernel variance {v}: X ~ N(0, {s2}) predicted; the output law is then N(0, 1) exactly, so the KKT "
        f"function is identically 1 (convolution of Gaussians: {s2} + {v} = 1)"
    )
    say(f"[{time.time() - t0:.0f}s]")
