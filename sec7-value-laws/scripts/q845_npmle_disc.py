"""q845 (03 Oct 2026, Amendment 1481, candidate C2): the population NPMLE for data uniform on the disc of radius R, unit Gaussian noise.
By rotation symmetry the mixing law is a set of rings (radius r_j, mass w_j; r = 0 is a centre atom). Unit-mass ring density at |y| = rho:
k(rho; r) = exp(-(rho - r)^2/2) i0e(rho r)/(2 pi). Output m(rho) = sum_j w_j k(rho; r_j). KKT function
D(r) = int_0^R (2 rho/R^2) k(rho; r)/m(rho) d rho <= 1, with equality on the rings. Seeds: EM on a radial grid (step 0.01, capped),
rings = mass clusters; polish: damped Newton on D(r_j) = 1, D'(r_j) = 0 (centre atom: D(0) = 1 only); KKT check on a grid to R + 4,
a violation adds a ring there and re-polishes. Prints for each R: ring count, acceptance numbers, m(R)/f0, R (log(m(R)/f0) + 1) and
Gt(R) = int over the disc of log(m/f0). States go to q845_state_R<R>.json (new files only).
Usage: py q845_npmle_disc.py R1,R2,... [NQ]   (R list may contain a:b:step ranges; env SAVE = comma list of R whose rows are
printed and states saved, default all; a step that is not accepted is retried from a fresh EM seed)"""

import sys, json, math, os
import numpy as np
from scipy.special import i0e, i1e

RS = []
for tok in sys.argv[1].split(","):
    if ":" in tok:
        a_, b_, st_ = map(float, tok.split(":"))
        RS += [round(a_ + k * st_, 6) for k in range(int(round((b_ - a_) / st_)) + 1)]
    else:
        RS.append(float(tok))
import os as _os

SAVE = [float(x) for x in _os.environ["SAVE"].split(",")] if _os.environ.get("SAVE") else None
NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
D1 = -0.304944111846  # 1D per-edge defect (702, 706)


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
    rho = (x + 1) * R / 2
    q = w * R / 2 * 2 * rho / R**2
    return rho, q


def Dfun(r, rho, q, m, deriv=False):
    D = (kern(rho, r) * (q / m)[:, None]).sum(0)
    if not deriv:
        return D
    return D, (dkern(rho, r) * (q / m)[:, None]).sum(0)


def em_seed(R, rho, q, iters=4000):
    grid = np.arange(0, R + 1.0 + 1e-9, 0.01)
    K = kern(rho, grid)
    w = np.full(len(grid), 1 / len(grid))
    em = lambda v: v * (K.T @ (q / (K @ v)))
    ll = lambda v: np.sum(q * np.log(K @ v))
    for _ in range(iters):  # SQUAREM (Varadhan-Roland) cycles with a monotone fallback
        w1 = em(w)
        w2 = em(w1)
        rv = w1 - w
        vv = w2 - w1 - rv
        nv = np.linalg.norm(vv)
        if nv == 0:
            w = w2
            break
        a = -np.linalg.norm(rv) / nv
        wx = w - 2 * a * rv + a * a * vv
        if a > -1 or np.any(wx <= 0):
            wx = w2
        wn = em(np.clip(wx, 1e-300, None))
        w = wn if ll(wn) >= ll(w2) else w2
    w /= w.sum()
    if os.environ.get("DEBUG"):
        mm = K @ w
        print("EM seed: KKT max - 1 on the grid", (K.T @ (q / mm)).max() - 1, flush=True)
    # clusters: one per local maximum of w (above 1e-5 of the largest), split at the minima between maxima
    pk = [
        i
        for i in range(len(grid))
        if w[i] > 1e-5 * w.max() and (i == 0 or w[i] >= w[i - 1]) and (i == len(grid) - 1 or w[i] > w[i + 1])
    ]
    cuts = [0] + [pk[k] + int(np.argmin(w[pk[k] : pk[k + 1] + 1])) for k in range(len(pk) - 1)] + [len(grid)]
    r = np.array(
        [
            np.sum(grid[cuts[k] : cuts[k + 1]] * w[cuts[k] : cuts[k + 1]]) / np.sum(w[cuts[k] : cuts[k + 1]])
            for k in range(len(pk))
        ]
    )
    m_ = np.array([np.sum(w[cuts[k] : cuts[k + 1]]) for k in range(len(pk))])
    if pk and pk[0] == 0:
        r[0] = 0.0
    keep = m_ > 1e-7
    return r[keep], m_[keep] / m_[keep].sum()


def resid(r, w, centre, rho, q):
    m = kern(rho, r) @ w
    D, Dp = Dfun(r, rho, q, m, True)
    e = list(D - 1) + [Dp[j] for j in range(len(r)) if not (centre and j == 0)]
    return np.array(e)


def pack(r, w, centre):
    return np.concatenate(
        [r[1:] if centre else r, np.log(w)]
    )  # Newton works in log-mass (positivity built in)


def unpack(x, n, centre):
    if centre:
        return np.concatenate([[0.0], x[: n - 1]]), np.exp(x[n - 1 :])
    return x[:n], np.exp(x[n:])


def newton(r, w, rho, q, it=150):
    centre = r[0] < 1e-9
    n = len(r)
    x = pack(r, w, centre)
    for _ in range(it):
        rr, ww = unpack(x, n, centre)
        Fv = resid(rr, ww, centre, rho, q)
        nF = np.linalg.norm(Fv)
        if np.max(np.abs(Fv)) < 1e-13:
            break
        J = np.zeros((len(Fv), len(x)))
        for j in range(len(x)):
            xp = x.copy()
            h = 1e-7 * max(1.0, abs(x[j]))
            xp[j] += h
            J[:, j] = (resid(*unpack(xp, n, centre), centre, rho, q) - Fv) / h
        st = np.linalg.lstsq(J, -Fv, rcond=None)[0]
        lam = 1.0
        while lam > 1e-8:
            xn = x + lam * st
            rr, ww = unpack(xn, n, centre)
            if np.all(np.diff(rr) > 0) and rr[0] >= 0:
                if np.linalg.norm(resid(rr, ww, centre, rho, q)) < nF:
                    x = xn
                    break
            lam /= 2
        else:
            break
    rr, ww = unpack(x, n, centre)
    return rr, ww, np.max(np.abs(resid(rr, ww, centre, rho, q)))


def ascend(r, w, rho, q):
    """Maximise L = sum q log m over ring radii and softmax masses (L-BFGS-B, analytic gradient); a centre atom stays at 0."""
    from scipy.optimize import minimize

    n = len(r)
    centre = r[0] < 1e-9

    def fg(x):
        rr, z = x[:n], x[n:]
        ww = np.exp(z - z.max())
        ww /= ww.sum()
        m = kern(rho, rr) @ ww
        D, Dp = Dfun(rr, rho, q, m, True)
        return -np.sum(q * np.log(m)), -np.concatenate([ww * Dp, ww * (D - 1)])

    x0 = np.concatenate([r, np.log(w)])
    bnds = [(0.0, None)] * n + [(None, None)] * n  # a centre atom may lift off into a ring
    out = minimize(
        fg,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bnds,
        options={"maxiter": 5000, "gtol": 1e-13, "ftol": 1e-16},
    )
    if os.environ.get("DEBUG"):
        print(
            "      ascend:",
            out.nit,
            "it,",
            str(out.message)[:60],
            "; r",
            np.round(np.sort(out.x[:n])[:4], 4).tolist(),
            flush=True,
        )
    rr, z = out.x[:n], out.x[n:]
    ww = np.exp(z - z.max())
    ww /= ww.sum()
    o = np.argsort(rr)
    rr, ww = rr[o], ww[o]
    rr[rr < 1e-6] = 0.0
    keep_r, keep_w = [], []  # merge rings that met, drop empty ones
    for a, b in zip(rr, ww):
        if b < 1e-12:
            continue
        if keep_r and a - keep_r[-1] < 1e-3:
            keep_r[-1] = (keep_r[-1] * keep_w[-1] + a * b) / (keep_w[-1] + b) if keep_r[-1] > 0 else 0.0
            keep_w[-1] += b
        else:
            keep_r.append(a)
            keep_w.append(b)
    ww = np.array(keep_w)
    return np.array(keep_r), ww / ww.sum()


def solve(R, seed=None):
    rho, q = setup(R)
    r, w = seed if seed is not None else em_seed(R, rho, q)
    import os

    DBG = os.environ.get("DEBUG")
    if DBG:
        print("seed", np.round(r, 4).tolist(), np.round(w, 5).tolist(), flush=True)
    for rnd in range(15):
        r, w = ascend(r, w, rho, q)
        r, w, res = newton(r, w, rho, q)
        if res > 1e-10 and len(r) > 2:
            # repair near the centre (a centre atom and a small ring are nearly degenerate): drop one of the two innermost atoms,
            # re-ascend and re-polish; keep the first variant that polishes
            for drop in (0, 1):
                r2 = np.delete(r, drop)
                w2 = np.delete(w, drop)
                w2 = w2 / w2.sum()
                r2, w2 = ascend(r2, w2, rho, q)
                r2, w2, res2 = newton(r2, w2, rho, q)
                if DBG:
                    print("   repair drop", drop, "res", res2, "r", np.round(r2[:4], 4).tolist(), flush=True)
                if res2 < 1e-10:
                    r, w, res = r2, w2, res2
                    break
        if DBG:
            print("round", rnd, "res", res, "r", np.round(r, 4).tolist(), flush=True)
        m = kern(rho, r) @ w
        g = np.arange(0, R + 4, 0.005)
        Dg = Dfun(g, rho, q, m)
        viol = Dg.max() - 1
        if viol < 1e-9 and np.all(w > 1e-12):
            break
        if np.any(w <= 1e-12):
            k = w > 1e-12
            r, w = r[k], w[k] / w[k].sum()
            continue
        # add a ring at the largest violation (local maximum away from existing rings)
        i = int(np.argmax(Dg))
        if np.min(np.abs(r - g[i])) < 1e-3:
            break  # violation on an existing ring (within 1e-3): do not duplicate
        if r[0] < 1e-9 and len(r) > 1 and g[i] < r[1]:
            # lift-off: the centre atom is unstable (D rises off 0); move it out to a small ring carrying its own mass
            r3 = r.copy()
            r3[0] = 0.5 * g[i]
            r3, w3 = ascend(r3, w.copy(), rho, q)
            r3, w3, res3 = newton(r3, w3, rho, q)
            m3 = kern(rho, r3) @ w3
            v3 = Dfun(g, rho, q, m3).max() - 1
            if DBG:
                print("   lift-off try: res", res3, "kkt", v3, "r", np.round(r3[:4], 4).tolist(), flush=True)
            if res3 < 1e-10:
                r, w = r3, w3
                continue
        kn = kern(rho, np.array([g[i]]))[:, 0]
        lo, hi = 0.0, 0.5  # vertex-direction step: best mass t along (1 - t) G + t delta
        for _ in range(60):
            t = (lo + hi) / 2
            if np.sum(q * (kn - m) / ((1 - t) * m + t * kn)) > 0:
                lo = t
            else:
                hi = t
        t = max(lo, 1e-6)
        r = np.sort(np.concatenate([r, [g[i]]]))
        idx = int(np.searchsorted(r, g[i]))
        w = np.insert(w * (1 - t), idx, t)
    return r, w, res, viol, rho, q, m


prev = None
for R in RS:
    seed = None
    if prev is not None:
        r0, w0, R0 = prev
        seed = (r0 * R / R0, w0.copy())
    r, w, res, viol, rho, q, m = solve(R, seed)
    if not ((res < 1e-10) and (viol < 1e-9)) and seed is not None:
        r2, w2, res2, viol2, rho, q, m2 = solve(R, None)
        if res2 < res:
            r, w, res, viol, m = r2, w2, res2, viol2, m2
    f0 = 1 / (np.pi * R**2)
    mR = (kern(np.array([R]), r) @ w)[0]
    Gt = (
        np.sum(q * np.log(m / f0)) * np.pi * R**2
    )  # q already carries f0 * 2 pi rho; times pi R^2 gives the plain area integral
    acc = (res < 1e-10) and (viol < 1e-9) and abs(w.sum() - 1) < 1e-12
    if SAVE is not None and not any(abs(R - x) < 1e-9 for x in SAVE):
        prev = (r, w, R) if (res < 1e-10 and viol < 1e-9) else prev
        print(f"  step R {R}: rings {len(r)} res {res:.1e} kkt {viol:+.1e}", flush=True)
        continue
    print(
        f"R {R}: rings {len(r)} (centre atom {'yes' if r[0] < 1e-9 else 'no'}); residual {res:.1e}; KKT max - 1 {viol:+.1e}; mass - 1 "
        f"{w.sum() - 1:+.1e}; {'ACCEPTED' if acc else 'NOT ACCEPTED'}; m(R)/f0 {mR/f0:.10f}; log(m(R)/f0) + 1 {math.log(mR/f0) + 1:+.8e}; "
        f"R (log(m(R)/f0) + 1) {R*(math.log(mR/f0) + 1):+.6f} (D = {D1:+.6f}); Gt {Gt:+.8f}; outer rings {np.round(r[-3:], 5).tolist()}",
        flush=True,
    )
    json.dump(
        {"R": R, "r": r.tolist(), "w": w.tolist(), "res": res, "viol": viol, "NQ": NQ},
        open(f"q845_state_R{R}.json", "x"),
    )
    prev = (r, w, R)
