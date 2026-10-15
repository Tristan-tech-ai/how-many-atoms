"""q1673 (10 Oct 2026, READING, no verdict word, predictions written at 09:57): shape of the lower-K differences d(K) = a_pred - A of q1671 (bulk law through c28, stored series to order 18, pair of q1669), K_b 61..82 without K_b 73: is ln|d| linear in v^2 (an exponentially small sector; the recorded decay of ln Da is 0.2087 per unit v^2 from the control values Da(115.0) = 0.21082 and Da(66.3) = 5480 of q1362) or in ln K (a power law)?
Predictions:
 P1 ln|d| against v^2 is linear with correlation magnitude above 0.99 and slope magnitude between 0.10 and 0.25;
 P2 ln|d| against ln K is also linear with correlation magnitude above 0.99 (the script prints both correlations);
 P3 the slope magnitude against v^2 is within 30 percent of 0.2087.
Control: a synthetic d = -1e-11 exp(-0.2 (v^2 - 140)) with relative noise 2 percent returns the slope 0.2 within 5 percent through the same fitting code.
Usage: py q1673_lower_k_shape.py (from small_gap_q206/; seconds)."""

import json, os, re
import numpy as np
import sympy as sp
import mpmath as mp
from scipy.optimize import brentq
import births_law as B

H = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(H, "..", "scratch")
S = json.load(open(os.path.join(H, "..", "centre_scripts", "d22_out_18.json")))
E = {n: float(S["e%d" % n]) for n in range(2, 19)}
txt = open(os.path.join(H, "q1557_stdout_ORD28.txt"), encoding="utf-8").read()
f = {}
for n in (26, 28):
    m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, txt, re.M)
    f[n] = float(mp.mpf(sp.N(sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi}), 30)))
B.F = ((B.F + [-2.420924146814851e-10, -8.973800969707301e-11])[:13] + [f[26], f[28]])[:15]
B.NG = 14
B.g = [1.0]
for n in range(1, 15):
    B.g.append(-sum(B.F[j] * B.g[n - j] for j in range(1, n + 1)))
NU, D0 = -0.0937736508347, 0.0011599160063
ex = [E[n] for n in range(6, 19)]


def meas(K):
    fn = (
        os.path.join(A, "tower_K%d_birth_out.json" % (K + 1))
        if K % 2 == 0
        else os.path.join(A, "tower_K%d_As2.json" % (K + 1))
    )
    d = json.load(open(fn))
    return d["A_cross"] if K % 2 == 0 else d["A_s"]


def v2(K, a):
    Hs = 1e-3
    rho = lambda x: brentq(lambda r: B.dlaw(r) + D0 - x, 0.05, 6.0, xtol=1e-15)
    Nn = lambda x: B.Nlaw(rho(x)) + 0.5 - NU
    rp = (Nn(a + Hs) - Nn(a - Hs)) / (2 * Hs)
    return (4 * np.pi * rp) ** 2


Ks = [K for K in range(61, 83) if K != 73]
d = {}
V = {}
for K in Ks:
    a = B.a_pred(K, NU, D0, ex)
    d[K] = a - meas(K)
    V[K] = v2(K, a)
x = np.array([V[K] for K in Ks])
y = np.log(np.abs([d[K] for K in Ks]))
lk = np.log(Ks)


def fit(xv, yv):
    c = np.polyfit(xv, yv, 1)
    return c[0], float(np.corrcoef(xv, yv)[0, 1])


s1, r1 = fit(x, y)
s2, r2 = fit(lk, y)
print("v^2 at K_b 61, 70, 82: %.1f, %.1f, %.1f" % (V[61], V[70], V[82]))
print(
    "ln|d| against v^2: slope %.4f, correlation %+.4f; ln|d| against ln K: slope %.3f, correlation %+.4f"
    % (s1, r1, s2, r2)
)
res = y - np.polyval(np.polyfit(x, y, 1), x)
res2 = y - np.polyval(np.polyfit(lk, y, 1), lk)
print(
    "rms of the residual of ln|d|: against v^2 %.4f, against ln K %.4f"
    % (np.sqrt((res**2).mean()), np.sqrt((res2**2).mean()))
)
rng = np.random.default_rng(5)
xs = np.linspace(140, 162, 21)
ys = np.log(1e-11 * np.exp(-0.2 * (xs - 140)) * (1 + 0.02 * rng.standard_normal(21)))
sc, _ = fit(xs, ys)
lab = lambda q: "holds" if q else "does not hold"
p1 = abs(r1) > 0.99 and 0.10 <= abs(s1) <= 0.25
p2 = abs(r2) > 0.99
p3 = abs(abs(s1) / 0.2087 - 1) < 0.30
ctl = abs(abs(sc) / 0.2 - 1) < 0.05
print(
    "P1 linear in v^2 with slope in [0.10, 0.25]: %s; P2 also linear in ln K: %s; P3 slope within 30 percent of 0.2087: %s; control (synthetic slope %.4f): %s"
    % (lab(p1), lab(p2), lab(p3), sc, lab(ctl))
)
print("READING q1673 done (labels by the script; no verdict word)")
