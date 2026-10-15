"""Q556: SYMMETRIC fixed-a solver for mirror-symmetric data (immune functional), written 30 Sep 14:02:26 by date stamp. The free solver
q535 fails to certify soft symmetric states at a = 20 (q551: hole H = 0.01 stopped at residual 1e-8 after an asymmetric path). Here
the state is kept symmetric (pairs +-p with pair masses v, optional centre atom), so the asymmetric soft modes are gone:
grid EG seeds (q535 v2 grid_solve and peaks, copied as text) -> symmetrised -> L-BFGS descent (q535 v2 descend, copied) ->
symmetrised -> q508c's spolish and signals (imported; q508c's main is guarded and q531/q534 import it the same way). Events: a birth
at |y| < 0.05 adds a centre atom (mass 1e-3), elsewhere a pair; a split of the centre atom makes a pair at +-0.1, a split of a pair
makes two pairs at p +- 0.1. Up to 20 changes; certified when both signals are below 1e-10 with polish residual below 1e-10.
Data modes: unif, hole H (piecewise Gauss-Legendre as q535).
VALIDATION (must pass before any score): a = 16 unif K = 18; a = 16 hole 0.1 K = 16; a = 20 unif K = 24 (all certified by q535).
Usage: py q556_symfixed.py A MODE [PARAM]"""

import sys, time, json
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize

A = float(sys.argv[1])
MODE = sys.argv[2]
PAR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
sys.argv = [sys.argv[0]]
import q508c_immune_symmetric as S

say = lambda *a: print(*a, flush=True)
t0 = time.time()
TOL = 1e-10


def setup(a, n=3000):
    xg, wg = leggauss(n)
    if MODE == "hole":
        lo, hi = PAR, a
        x1 = lo + (hi - lo) * (xg + 1) / 2
        w1 = (hi - lo) / 2 * wg
        x = np.concatenate([-x1[::-1], x1])
        w = np.concatenate([w1[::-1], w1])
    else:
        x = a * xg
        w = a * wg
    return x, w / w.sum()


def gfun(ys, ws, y, xg, qx):
    Pt = (ws[:, None] * np.exp(-((xg[None, :] - ys[:, None]) ** 2) / 2)).sum(0)
    E = np.exp(-((xg[None, :] - np.asarray(y)[:, None]) ** 2) / 2)
    d = xg[None, :] - np.asarray(y)[:, None]
    W = qx / Pt**2
    return (E * W).sum(1), (E * d * W).sum(1), (E * (d * d - 1) * W).sum(1), (qx / Pt).sum()


def grid_solve(a, xg, qx, iters=20000, tmax=600):
    y = np.arange(-a - 1.5, a + 1.5 + 1e-9, 0.04)
    Phi = np.exp(-((xg[None, :] - y[:, None]) ** 2) / 2)
    w = np.full(len(y), 1 / len(y))
    P = w @ Phi
    F = (qx / P).sum()
    eta = 1.0
    t = time.time()
    for it in range(iters):
        g = Phi @ (qx / P**2)
        while True:
            wn = w * np.exp(eta * np.clip(g / F - 1, -50, 50))
            wn /= wn.sum()
            Pn = wn @ Phi
            Fn = (qx / Pn).sum()
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
        W = qx / P**2
        g0 = E @ W
        g1 = (E * d) @ W
        return (qx / P).sum(), np.concatenate([-w * g1, w * (-g0 + (w * g0).sum())])

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


def symmetrise(ys, ws):
    c = np.abs(ys) < 0.05
    pos = ys > 0.05
    neg = ys < -0.05
    p = np.sort(ys[pos])
    pn = np.sort(-ys[neg])
    n = min(len(p), len(pn))
    p = (p[:n] + pn[:n]) / 2 if n else np.array([])
    vp = np.array([ws[pos][np.argsort(ys[pos])][i] + ws[neg][np.argsort(-ys[neg])][i] for i in range(n)])
    centre = bool(c.any())
    v0 = ws[c].sum() if centre else 0.0
    tot = vp.sum() + v0
    vp = vp / tot
    return p, vp, centre


def apply_event(kind, p, v, centre, wb, ksp):
    p, v = p.copy(), v.copy()
    if kind.startswith("birth"):
        m = {"birth": 1e-3, "birth2": 0.03, "birth3": 0.1}[kind]  # v3: heavier newborns as fallbacks
        if wb < 0.05 and not centre:
            centre = True
            v = v * (1 - m)
        else:
            j = np.searchsorted(p, wb)
            p = np.insert(p, j, wb)
            v = np.insert(v * (1 - m), j, m)
    else:
        if abs(ksp) < 0.05 and centre:
            v0 = 1 - v.sum()
            centre = False
            p = np.insert(p, 0, 0.1)
            v = np.insert(v, 0, v0)
        else:
            j = int(np.argmin(np.abs(p - abs(ksp))))
            pj, vj = p[j], v[j]
            p = np.concatenate([p[:j], [pj - 0.1, pj + 0.1], p[j + 1 :]])
            v = np.concatenate([v[:j], [vj / 2, vj / 2], v[j + 1 :]])
    return p, v, centre


if (
    __name__ == "__main__"
):  # v3 heavier birth fallbacks; v2 (15:15:49): when the polish after an event fails, return to the last
    a = A
    xg, qx = setup(a)
    say(f"q556 v2 symmetric, a = {a}, mode {MODE} {PAR}")  # polished state and try the other event
    y, w, F, it = grid_solve(a, xg, qx)
    ys, ws = peaks(y, w)
    say(f"grid: {len(ys)} mass peaks   [{time.time() - t0:.0f}s]")
    ys, ws = descend(ys, ws, xg, qx)
    p, v, centre = symmetrise(ys, ws)
    ok = False
    last = None
    alts = []
    for ch in range(25):
        yf, wf = S.full(p, v, centre)
        yf, wf = descend(yf, wf, xg, qx, iters=20000)
        p, v, centre = symmetrise(yf, wf)
        pp, vv, r = S.spolish(p, v, centre, xg, qx)
        if r > 1e-10:
            say(f"   polish residual {r:.1e} at K = {len(S.full(p, v, centre)[0])}")
            if last is not None and alts:
                kind = alts.pop(0)
                lp, lv, lc, lwb, lksp = last
                p, v, centre = apply_event(kind, lp, lv, lc, lwb, lksp)
                say(f"   back to the last polished state; trying the {kind} instead")
                continue
            say("   NOT CERTIFIED")
            break
        p, v = pp, vv
        b, wb, sp, ksp = S.signals(a, p, v, centre, xg, qx)
        K = len(S.full(p, v, centre)[0])
        say(
            f"K = {K}: residual {r:.1e}, birth {b:+.2e} at |y| = {wb:.3f}, split {sp:+.2e} at {ksp:+.3f}, centre {centre}   [{time.time() - t0:.0f}s]"
        )
        if b <= TOL and sp <= TOL:
            ok = True
            break
        order = [k for k, sgn in sorted((("birth", b), ("split", sp)), key=lambda t: -t[1]) if sgn > TOL]
        if b > TOL:
            order = order + ["birth2", "birth3"]
        last = (p.copy(), v.copy(), centre, wb, ksp)
        alts = order[1:]
        p, v, centre = apply_event(order[0], p, v, centre, wb, ksp)
    yf, wf = S.full(p, v, centre)
    K = len(yf)
    say(
        f"RESULT a = {a} {MODE} {PAR}: K = {K} {'KKT-CERTIFIED' if ok else 'not certified'}; positive atoms {np.round(yf[yf > 1e-9], 3).tolist()}; centre {centre}"
    )
    json.dump(
        {"a": a, "mode": MODE, "par": PAR, "K": K, "ok": ok, "ys": yf.tolist(), "ws": wf.tolist()},
        open(f"results_q556_{MODE}{PAR}_a{a}.json", "w"),
        indent=1,
    )
