"""q868: q867 for any dimension d (radial kernel Gamma(d/2) (2/x)^nu ive(nu, x), nu = d/2 - 1; mean cosine ive(d/2)/ive(d/2 - 1);
density of |Y| c_d rho^(d-1) e^{-(rho - t)^2/2} K(t rho), c_d = 1/(2^(d/2 - 1) Gamma(d/2))); d 1, 2, 3 reproduce q866/q867.

e1 (birth):    single wall shell at m stops being least favourable where r(0) = r(m)
e2 (lift-off): prior {centre w0, wall shell}, equalizer weight; the centre stops being a local maximum of the risk where r''(0) = 0
e3 (birth):    prior {shell a, wall shell} with r(a) = r(m), r'(a) = 0; a centre atom is needed where r(0) = r(m)
Radial risk for |theta| = t: r(t) = int f_t(rho) [D(rho)^2 - 2 t D(rho) A(t rho) + t^2] drho, with f_t the density of |Y| and A the
mean cosine (d 2: i1e/i0e, d 3: Langevin); D the radial posterior mean. Composite Gauss-Legendre in rho.
usage: py q868_lfp_events_dimd.py <d> <event e1|e2|e3> [lo hi]
"""

import sys
import numpy as np
from scipy.special import ive, gammaln
from scipy.optimize import brentq, fsolve


def kern(d, x):  # (radial kernel e^{-x} * sphere average of e^{x cos}, mean cosine)
    x = np.asarray(x, float)
    nu = d / 2 - 1
    small = x < 1e-4
    xs = np.where(small, 1.0, x)
    K = np.exp(gammaln(d / 2) + nu * np.log(2 / xs)) * ive(nu, xs)
    A = ive(d / 2, xs) / ive(nu, xs)
    return (
        np.where(small, np.exp(-x) * (1 + x * x / (2 * d)), K),
        np.where(small, x / d - x**3 / (d * d * (d + 2)), A),
    )


def cd(d):
    return np.exp(-((d / 2 - 1) * np.log(2) + gammaln(d / 2)))


def nodes(hi, panels=60, n=24):
    g, gw = np.polynomial.legendre.leggauss(n)
    e = np.linspace(0, hi, panels + 1)
    x = np.concatenate([(e[k] + e[k + 1]) / 2 + (e[k + 1] - e[k]) / 2 * g for k in range(panels)])
    w = np.concatenate([(e[k + 1] - e[k]) / 2 * gw for k in range(panels)])
    return x, w


def Dpost(d, rho, at, w):
    num = 0.0
    den = 0.0
    for r, wi in zip(at, w):
        K, A = kern(d, r * rho)
        e = wi * np.exp(-((rho - r) ** 2) / 2) * K
        num = num + e * r * A
        den = den + e
    return num / den


def risk(d, t, at, w):
    x, qw = nodes(max(t, max(at)) + 13.0 + np.sqrt(d))
    K, A = kern(d, t * x)
    f = cd(d) * x ** (d - 1) * np.exp(-((x - t) ** 2) / 2) * K
    D = Dpost(d, x, at, w)
    return float(np.sum(qw * f * (D * D - 2 * t * D * A + t * t)))


def d1(d, t, at, w, h=1e-3):
    return (risk(d, t + h, at, w) - risk(d, t - h, at, w)) / (2 * h)


def d2c(d, at, w, h=5e-3):
    return 2 * (risk(d, h, at, w) - risk(d, 0.0, at, w)) / h**2  # r even in t


def interior_excess(d, at, w, m):
    B = risk(d, m, at, w)
    ts = np.linspace(0, m, 161)
    return max(risk(d, t, at, w) for t in ts) - B


def e1(d, lo, hi):
    f = lambda m: risk(d, 0.0, [m], [1.0]) - risk(d, m, [m], [1.0])
    return brentq(f, lo, hi, xtol=1e-11)


def two(d, m):
    at = [0.0, m]
    g = lambda w0: risk(d, 0.0, at, [w0, 1 - w0]) - risk(d, m, at, [w0, 1 - w0])
    w0 = brentq(g, 1e-10, 0.95, xtol=1e-14)
    return at, [w0, 1 - w0]


def e2(d, lo, hi):
    return brentq(lambda m: d2c(d, *two(d, m)), lo, hi, xtol=1e-9)


def ringwall(d, m, guess):
    def eqs(x):
        a, wa = x
        at = [a, m]
        w = [wa, 1 - wa]
        return [risk(d, a, at, w) - risk(d, m, at, w), d1(d, a, at, w)]

    sol, info, ier, msg = fsolve(eqs, guess, full_output=True, xtol=1e-13)
    return sol, np.max(np.abs(info["fvec"]))


def e3(d, lo, hi):
    st = {"g": [0.47 * lo, 0.35]}

    def f(m):
        (a, wa), res = ringwall(d, m, st["g"])
        st["g"] = [a, wa]
        return risk(d, 0.0, [a, m], [wa, 1 - wa]) - risk(d, m, [a, m], [wa, 1 - wa]), a, wa, res

    ms = np.linspace(lo, hi, 11)
    vals = []
    for m in ms:
        v, a, wa, res = f(m)
        vals.append(v)
        print(
            f"  e3 scan m {m:.3f}: r(0) - r(m) {v:+.3e}; a {a:.5f}; w_a {wa:.5f}; residual {res:.1e}",
            flush=True,
        )
    i = next(k for k in range(len(vals) - 1) if vals[k] < 0 <= vals[k + 1])
    st["g"] = list(ringwall(d, ms[i], [0.47 * ms[i], 0.35])[0])
    return brentq(lambda m: f(m)[0], ms[i], ms[i + 1], xtol=1e-9)


def solve_inner(d, m, a, w, w0):
    """inner shells a (k), weights w (k), optional centre weight w0 (None = no centre); wall weight the rest.
    Equations: r(a_i) = r(m), r'(a_i) = 0, and r(0) = r(m) when the centre is present."""
    k = len(a)
    cen = w0 is not None

    def unpack(x):
        aa = list(x[:k])
        ww = list(x[k : 2 * k])
        c = x[2 * k] if cen else 0.0
        at = ([0.0] if cen else []) + aa + [m]
        wt = ([c] if cen else []) + ww + [1 - c - sum(ww)]
        return at, wt, aa

    def eqs(x):
        at, wt, aa = unpack(x)
        B = risk(d, m, at, wt)
        out = [risk(d, ai, at, wt) - B for ai in aa] + [d1(d, ai, at, wt) for ai in aa]
        return out + ([risk(d, 0.0, at, wt) - B] if cen else [])

    x0 = list(a) + list(w) + ([w0] if cen else [])
    sol, info, ier, msg = fsolve(eqs, x0, full_output=True, xtol=1e-13)
    at, wt, aa = unpack(sol)
    return sol, at, wt, float(np.max(np.abs(info["fvec"])))


def track(d, m0, m1, a, w, w0, kind, step=0.05):
    """continue the state from m0 upward; kind 'birth' (no centre: r(0) - r(m)) or 'split' (centre: r''(0)); return the root."""
    k = len(a)
    x = list(a) + list(w) + ([w0] if w0 is not None else [])

    def F(m, x):
        sol, at, wt, res = solve_inner(d, m, x[:k], x[k : 2 * k], (x[2 * k] if w0 is not None else None))
        v = (risk(d, 0.0, at, wt) - risk(d, m, at, wt)) if kind == "birth" else d2c(d, at, wt)
        return v, list(sol), at, wt, res

    m = m0
    v, x, at, wt, res = F(m, x)
    prev = (m, v, x)
    print(
        f"  {kind} track d {d} m {m:.3f}: f {v:+.3e}; support {np.round(at, 5)}; weights {np.round(wt, 5)}; residual {res:.1e}",
        flush=True,
    )
    while m < m1:
        m = round(m + step, 10)
        v, x, at, wt, res = F(m, x)
        print(
            f"  {kind} track d {d} m {m:.3f}: f {v:+.3e}; support {np.round(at, 5)}; weights {np.round(wt, 5)}; residual {res:.1e}",
            flush=True,
        )
        if prev[1] < 0 <= v:
            xs = prev[2]
            root = brentq(lambda mm: F(mm, xs)[0], prev[0], m, xtol=1e-9)
            v, xr, at, wt, res = F(root - 0.01, xs)
            print(
                f"  state at root - 0.01: support {np.round(at, 6)}; weights {np.round(wt, 6)}; residual {res:.1e}; "
                f"interior excess {interior_excess(d, at, wt, root - 0.01):+.2e}",
                flush=True,
            )
            return root, F(root, xs)
        prev = (m, v, x)
    return None, None


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "track":
    import json

    d = int(sys.argv[1])
    spec = json.loads(sys.argv[3])
    root, st = track(
        d, spec["m0"], spec["m1"], spec["a"], spec["w"], spec.get("w0"), spec["kind"], spec.get("step", 0.05)
    )
    if root is None:
        print("no crossing")
    else:
        print(
            f"d {d} {spec['kind']} event: m {root:.8f}; support {np.round(st[2], 6)}; weights {np.round(st[3], 6)}; residual {st[4]:.1e}"
        )
    sys.exit()

if __name__ == "__main__":
    d = int(sys.argv[1])
    ev = sys.argv[2]
    lo, hi = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (1.0, 3.0)
    if ev == "e1":
        m = e1(d, lo, hi)
        print(
            f"d {d} e1 (birth of the centre): m {m:.8f}; interior excess just below "
            f"{interior_excess(d, [m - 1e-3], [1.0], m - 1e-3):+.2e}"
        )
    elif ev == "e2":
        m = e2(d, lo, hi)
        at, w = two(d, m)
        print(
            f"d {d} e2 (lift-off of the centre): m {m:.8f}; w0 {w[0]:.6f}; r''(0) at m -+ 0.01 "
            f"{d2c(d, *two(d, m - 0.01)):+.3e}, {d2c(d, *two(d, m + 0.01)):+.3e}; "
            f"interior excess at m - 0.01 {interior_excess(d, *two(d, m - 0.01), m - 0.01):+.2e}"
        )
    elif ev == "e3":
        m = e3(d, lo, hi)
        (a, wa), res = ringwall(d, m - 0.01, [0.47 * m, 0.35])
        print(
            f"d {d} e3 (birth of a new centre): m {m:.8f}; at m - 0.01: shell a {a:.5f}, w_a {wa:.5f}, residual {res:.1e}, "
            f"interior excess {interior_excess(d, [a, m - 0.01], [wa, 1 - wa], m - 0.01):+.2e}"
        )
