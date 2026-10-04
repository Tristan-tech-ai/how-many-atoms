"""q862 (04 Oct 2026): the 1D least favourable prior for the bounded normal mean |theta| <= m (unit noise, squared loss), symmetric
priors. Stage 1: exponentiated-gradient ascent of the Bayes risk on a grid (gives an initial active set). Stage 2: an active-set loop:
Newton on (interior positive atoms, masses, B) with r(x_k) = B and r'(x_k) = 0 at interior atoms (wall atom at m fixed; centre atom at 0
if present); then the KKT function r - B on a 0.002 grid; if it is violated by more than 1e-9, an atom is inserted at the worst point
(the centre if it is within 0.03 of 0, else a symmetric pair) and the loop repeats; a pair that collapses below 0.02 becomes the centre
atom. Posterior mean delta(y) = sum w_k x_k phi(y - x_k)/sum w_k phi(y - x_k); risk r(theta) = int phi(y - theta)(delta(y) - theta)^2 dy.
Usage: py q862_lfp1d.py m [HC]; mode "thr": the two-point -> three-point transition (validation: 1.0567, Casella-Strawderman).
"""

import sys, math
import numpy as np

NY = 800


def setup(m):
    xg, wg = np.polynomial.legendre.leggauss(NY)
    L = m + 9.0
    return xg * L, wg * L


def risk(theta, x, w, Y, WYq):
    P = np.exp(-((Y[:, None] - x[None, :]) ** 2) / 2)
    p = P @ w
    d = (P @ (w * x)) / p
    K = np.exp(-((Y[:, None] - theta[None, :]) ** 2) / 2) / math.sqrt(2 * math.pi)
    return WYq @ (K * (d[:, None] - theta[None, :]) ** 2)


if sys.argv[1] == "thr":
    from scipy.optimize import brentq

    def ex(m):
        Y, WYq = setup(m)
        x = np.array([-m, m])
        w = np.array([0.5, 0.5])
        r = risk(np.array([0.0, m]), x, w, Y, WYq)
        return r[0] - r[1]

    print(f"two-point prior +-m stops being least favourable at m = {brentq(ex, 0.8, 1.4, xtol=1e-12):.8f}")
    sys.exit()
m = float(sys.argv[1])
HC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
Y, WYq = setup(m)
g = np.arange(0, m + HC / 2, HC)
g[-1] = m
xs = np.concatenate([-g[::-1][:-1], g])
w = np.full(len(xs), 1 / len(xs))
for it in range(4000):
    r = risk(xs, xs, w, Y, WYq)
    B = w @ r
    w = w * np.exp(2.0 * (r - r.max()))
    w /= w.sum()
    w[w < 1e-14] = 0
    w /= w.sum()
gap1 = r.max() - B
pos = xs >= -1e-12
idx = np.where((w > 1e-9) & pos)[0]
cl = []
cur = [idx[0]]
for i in idx[1:]:
    if xs[i] - xs[cur[-1]] > 0.3:
        cl.append(cur)
        cur = [i]
    else:
        cur.append(i)
cl.append(cur)
half = [(float(np.sum(xs[c] * w[c]) / np.sum(w[c])), float(np.sum(w[c]))) for c in cl]
centre = half[0][0] < 0.1
cm = half[0][1] if centre else 0.0
pp = [h[0] for h in (half[1:] if centre else half)]
pw = [h[1] for h in (half[1:] if centre else half)]
if not pp:
    pp, pw = [m], [0.5]
pp[-1] = m


def solve(centre, cm, pp, pw):
    nI = len(pp) - 1

    def build(z):
        p = np.array(list(z[:nI]) + [m])
        ww = np.exp(z[nI:-1])
        B_ = z[-1]
        if centre:
            x = np.concatenate([-p[::-1], [0.0], p])
            w_ = np.concatenate([ww[1:][::-1], [ww[0]], ww[1:]])
        else:
            x = np.concatenate([-p[::-1], p])
            w_ = np.concatenate([ww[::-1], ww])
        return p, x, w_ / w_.sum(), B_

    def F(z):
        p, x, w_, B_ = build(z)
        h = 1e-5
        th = np.concatenate([[0.0], p]) if centre else p
        res = list(risk(th, x, w_, Y, WYq) - B_)
        for k in range(nI):
            res.append(
                (risk(np.array([p[k] + h]), x, w_, Y, WYq)[0] - risk(np.array([p[k] - h]), x, w_, Y, WYq)[0])
                / (2 * h)
            )
        return np.array(res)

    p0, x0, w0, _ = build(np.array(pp[:nI] + list(np.log(([cm] if centre else []) + pw)) + [0.0]))
    B0 = w0 @ risk(x0, x0, w0, Y, WYq)
    z = np.array(pp[:nI] + list(np.log(([cm] if centre else []) + pw)) + [B0])
    for it in range(80):
        f = F(z)
        nf = np.max(np.abs(f))
        if nf < 1e-13:
            break
        J = np.zeros((len(f), len(z)))
        for j in range(len(z)):
            zp = z.copy()
            zp[j] += 1e-7
            J[:, j] = (F(zp) - f) / 1e-7
        st = np.linalg.lstsq(J, -f, rcond=None)[0]
        lam = 1.0
        while lam > 1e-8:
            zn = z + lam * st
            pn = build(zn)[0]
            if np.all(np.diff(np.concatenate([[0.0], pn])) > 0) and np.max(np.abs(F(zn))) < nf:
                z = zn
                break
            lam /= 2
        else:
            break
    p, x, w_, B_ = build(z)
    ws = w_[len(p) :] if not centre else w_[len(p) :]  # masses of the positive side (centre first if present)
    return p, x, w_, B_, np.max(np.abs(F(z))), ws


for rnd in range(12):
    p, x, w_, B, res, ws = solve(centre, cm, pp, pw)
    if res > 1e-10:  # a mass heading to zero: drop that atom and re-solve
        cmass = float(w_[len(p)]) if centre else None
        pm = [float(w_[np.argmin(np.abs(x - v))]) for v in p]
        cands = ([("c", cmass)] if centre else []) + [(k, v) for k, v in enumerate(pm[:-1])]
        if cands:
            which, mv = min(cands, key=lambda q: q[1])
            if mv < 1e-5:
                if which == "c":
                    centre, cm = False, 0.0
                    pp = list(p)
                    pw = pm
                else:
                    pp = [v for k, v in enumerate(p) if k != which]
                    pw = [v for k, v in enumerate(pm) if k != which]
                    cm = cmass if centre else cm
                continue
    # collapse a pair at the origin into the centre atom
    if len(p) > 1 and p[0] < 0.02:
        mass_pair = 2 * w_[len(p) - 1] if not centre else None
        centre, cm = True, float(2 * w_[np.argmin(np.abs(x - p[0]))]) if not centre else cm
        pp = list(p[1:])
        pw = [float(w_[np.argmin(np.abs(x - v))]) for v in pp]
        continue
    gq = np.arange(0, m + 1e-12, 0.002)
    gq[-1] = m
    rv = risk(gq, x, w_, Y, WYq) - B
    kkt = rv.max()
    if kkt < 1e-9 and res < 1e-10:
        break
    t = gq[int(np.argmax(rv))]
    cm = float(w_[len(p)]) if centre else cm
    pw = [float(w_[np.argmin(np.abs(x - v))]) for v in p]
    pp = list(p)
    if t < 0.03 and not centre:
        centre, cm = True, 1e-3
    else:
        k = int(np.searchsorted(pp, t))
        pp.insert(k, float(t))
        pw.insert(k, 1e-3)
gq = np.arange(0, m + 1e-12, 0.002)
gq[-1] = m
kkt = (risk(gq, x, w_, Y, WYq) - B).max()
K = len(x)
acc = res < 1e-10 and kkt < 1e-9
print(
    f"m {m}: K {K} (centre atom {'yes' if centre else 'no'}); positive atoms {np.round(p, 6).tolist()}; B {B:.10f}; residual {res:.1e}; "
    f"KKT {kkt:+.1e}; stage-1 gap {gap1:.1e}; rounds {rnd + 1}; {'ACCEPTED' if acc else 'NOT ACCEPTED'}",
    flush=True,
)
