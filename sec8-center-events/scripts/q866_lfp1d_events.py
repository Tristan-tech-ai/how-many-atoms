"""q866: the first 1D LFP events on [-m, m] read as thresholds of explicit symmetric priors (no EG stage).

e1 (birth):  prior +-m stops being least favourable where r(0) = r(m)            (classical 1.0567; q862 validation 1.05674351)
e2 (split):  prior {-m, 0, m} with equalizer weight; the centre stops being a local maximum of the risk where r''(0) = 0
e3 (birth):  prior {+-a, +-m} with r(a) = r(m), r'(a) = 0; a centre atom is needed where r(0) = r(m)
Risk r(theta) = int (delta(theta + z) - theta)^2 phi(z) dz by the trapezoid rule on z in [-14, 14] (spectral for these integrands).
"""

import sys
import numpy as np
from scipy.optimize import brentq, fsolve

Z = np.linspace(-14, 14, 8001)
DZ = Z[1] - Z[0]
PHI = np.exp(-(Z**2) / 2) / np.sqrt(2 * np.pi)


def post_mean(y, at, w):
    lw = np.log(w)[None, :] - (y[:, None] - at[None, :]) ** 2 / 2
    lw -= lw.max(axis=1, keepdims=True)
    e = np.exp(lw)
    return (e @ at) / e.sum(axis=1)


def risk(th, at, w):
    return float(np.sum((post_mean(th + Z, at, w) - th) ** 2 * PHI) * DZ)


def d1(th, at, w, h=1e-3):
    return (risk(th + h, at, w) - risk(th - h, at, w)) / (2 * h)


def d2(th, at, w, h=2e-3):
    return (risk(th + h, at, w) - 2 * risk(th, at, w) + risk(th - h, at, w)) / h**2


def e1():
    f = lambda m: risk(0.0, np.array([-m, m]), np.array([0.5, 0.5])) - risk(
        m, np.array([-m, m]), np.array([0.5, 0.5])
    )
    return brentq(f, 0.8, 1.3, xtol=1e-12)


def three(m):
    at = np.array([-m, 0.0, m])
    g = lambda w0: risk(0.0, at, np.array([(1 - w0) / 2, w0, (1 - w0) / 2])) - risk(
        m, at, np.array([(1 - w0) / 2, w0, (1 - w0) / 2])
    )
    w0 = brentq(g, 1e-9, 0.9, xtol=1e-14)
    return at, np.array([(1 - w0) / 2, w0, (1 - w0) / 2])


def e2():
    def f(m):
        at, w = three(m)
        return d2(0.0, at, w)

    return brentq(f, 1.6, 2.6, xtol=1e-9)


def four(m, guess):
    def eqs(x):
        a, wa = x
        at = np.array([-m, -a, a, m])
        w = np.array([0.5 - wa, wa, wa, 0.5 - wa])
        return [risk(a, at, w) - risk(m, at, w), d1(a, at, w)]

    sol, info, ier, msg = fsolve(eqs, guess, full_output=True, xtol=1e-13)
    return sol, ier, np.max(np.abs(info["fvec"]))


def e3(lo=2.5, hi=3.5):
    state = {"g": None}

    def f(m):
        g = state["g"] if state["g"] is not None else [0.45 * m, 0.25]
        (a, wa), ier, res = four(m, g)
        state["g"] = [a, wa]
        at = np.array([-m, -a, a, m])
        w = np.array([0.5 - wa, wa, wa, 0.5 - wa])
        return risk(0.0, at, w) - risk(m, at, w)

    ms = np.linspace(lo, hi, 11)
    vals = []
    for m in ms:
        vals.append(f(m))
        print(f"  e3 scan m {m:.2f}: r(0) - r(m) {vals[-1]:+.3e}; a, w_a {state['g']}", flush=True)
    i = next(k for k in range(len(vals) - 1) if vals[k] < 0 <= vals[k + 1])
    state["g"] = None
    f(ms[i])
    return brentq(f, ms[i], ms[i + 1], xtol=1e-9)


def five(m, guess):
    def eqs(x):
        a, wa, w0 = x
        wm = (1 - w0) / 2 - wa
        at = np.array([-m, -a, 0.0, a, m])
        w = np.array([wm, wa, w0, wa, wm])
        return [risk(0.0, at, w) - risk(m, at, w), risk(a, at, w) - risk(m, at, w), d1(a, at, w)]

    sol, info, ier, msg = fsolve(eqs, guess, full_output=True, xtol=1e-13)
    a, wa, w0 = sol
    wm = (1 - w0) / 2 - wa
    return np.array([-m, -a, 0.0, a, m]), np.array([wm, wa, w0, wa, wm]), sol, np.max(np.abs(info["fvec"]))


def e4(lo=3.0, hi=4.2):
    st = {"g": [1.72 * lo / 3.6, 0.3, 0.05]}

    def f(m):
        at, w, sol, res = five(m, st["g"])
        st["g"] = list(sol)
        return d2(0.0, at, w), sol, res

    ms = np.linspace(lo, hi, 13)
    vals = []
    for m in ms:
        v, sol, res = f(m)
        vals.append(v)
        print(
            f"  e4 scan m {m:.2f}: r''(0) {v:+.3e}; a, w_a, w_0 {np.round(sol, 6)}; residual {res:.1e}",
            flush=True,
        )
    i = next(k for k in range(len(vals) - 1) if vals[k] < 0 <= vals[k + 1])
    st["g"] = list(five(ms[i], st["g"])[2])
    return brentq(lambda m: f(m)[0], ms[i], ms[i + 1], xtol=1e-9)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "e4":
        print(f"e4 (split of the second centre): {e4():.8f}")
        sys.exit()
    m1 = e1()
    print(f"e1 (birth, +-m): {m1:.8f}", flush=True)
    m2 = e2()
    at, w = three(m2)
    print(
        f"e2 (split of the centre): {m2:.8f}; w0 {w[1]:.6f}; r''(0) check at m2 -+ 0.01: "
        f"{d2(0.0, *three(m2 - 0.01)):+.3e}, {d2(0.0, *three(m2 + 0.01)):+.3e}",
        flush=True,
    )
    m3 = e3()
    print(f"e3 (birth of a new centre): {m3:.8f}", flush=True)
