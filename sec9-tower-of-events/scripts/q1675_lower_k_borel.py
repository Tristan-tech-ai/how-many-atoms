"""q1675 (10 Oct 2026, READING, no verdict word, predictions written at 09:57): the births law with the median Borel sum of the centre series (machine of q1220: weight Gamma(n + 1.5), diagonal Pade [8/8] of the coefficients e_2..e_18 of d22_out_18.json, rays theta = +-0.3, S_med = Re (S_+ + S_-)/2) in place of the truncated series. The bulk law runs through c28 and the pair is refitted on K_b 83..130 (two linearisation rounds). S_med(x), x = 1/v^2, is tabulated on a Chebyshev grid of 24 nodes for x in [0.0045, 0.0075] (v^2 from 133 to 222).
Predictions:
 P1 the largest size of a_pred - A over K_b 61..69 shrinks by more than a factor 2 relative to q1671 (1.055e-11), that is it is below 5.3e-12;
 P2 the in-sample rms over K_b 83..130 with the pair refitted changes by less than 30 percent of 1.5200e-13 (q1669, nmax 18);
 P3 the median sum differs from the truncation at nmax = 20 at K_b 61 (as a change of a_pred with the pair of q1669 fixed) by more than 1e-12 in size.
Control: the table reproduces the direct S_med at three points to 1e-13 relative to the point values, and at x = 1/12.608284652941046^2 the median sum equals the sum to n = 18 within 1e-11 (control C4 of q1220).
Usage: py q1675_lower_k_borel.py (from small_gap_q206/; several minutes)."""

import io, contextlib, json, os, re, sys
import numpy as np
import sympy as sp
import mpmath as mp

H = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(H, "..", "centre_scripts")
A = os.path.join(H, "..", "scratch")
cwd0 = os.getcwd()
os.chdir(C)
sys.path.insert(0, C)
sys.argv = ["x", "--none"]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import q1220_borel_pade_births_law as T
os.chdir(H)
sys.path.insert(0, H)
import births_law as B

mp.mp.dps = 30
Bf, p, q = T.machine(T.coef_series(), 8, 8)
smed = lambda x: float(T.smed(Bf, mp.mpf(x), 0.3))
nodes = [(0.0045 + 0.0075) / 2 + (0.0075 - 0.0045) / 2 * np.cos(np.pi * (j + 0.5) / 24) for j in range(24)]
ys = [smed(x) for x in nodes]
TAB = np.polynomial.chebyshev.Chebyshev.fit(nodes, ys, 23, domain=[0.0045, 0.0075])
S18 = json.load(open(os.path.join(C, "d22_out_18.json")))
S20 = json.load(open(os.path.join(C, "d22_out_20.json")))
E = {n: float(S20["e%d" % n]) for n in range(2, 21)}
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
from scipy.optimize import brentq


def a_pred_med(K, nu, d0, Hs=1e-3):
    rho = lambda a: brentq(lambda r: B.dlaw(r) + d0 - a, 0.05, 6.0, xtol=1e-15)
    Nn = lambda a: B.Nlaw(rho(a)) + 0.5 - nu

    def dd(r):
        v = 4 * np.pi * r
        return float(TAB(1.0 / v**2))

    return brentq(
        lambda a: 2 * Nn(a) - K - 1 / 6 - dd((Nn(a + Hs) - Nn(a - Hs)) / (2 * Hs)), 30, 85, xtol=1e-14
    )


def meas(K):
    fn = (
        os.path.join(A, "tower_K%d_birth_out.json" % (K + 1))
        if K % 2 == 0
        else os.path.join(A, "tower_K%d_As2.json" % (K + 1))
    )
    d = json.load(open(fn))
    return d["A_cross"] if K % 2 == 0 else d["A_s"]


Kin = [K for K in range(83, 131) if B.measured(K) is not None]
min_ = np.array([B.measured(K) for K in Kin])
h = 1e-7
nu, d0 = -0.0937736508347, 0.0011599160063
for rnd in range(2):
    p0 = np.array([a_pred_med(K, nu, d0) for K in Kin])
    M = np.stack(
        [
            [(a_pred_med(K, nu + h, d0) - a_pred_med(K, nu - h, d0)) / (2 * h) for K in Kin],
            [(a_pred_med(K, nu, d0 + h) - a_pred_med(K, nu, d0 - h)) / (2 * h) for K in Kin],
        ],
        1,
    )
    sol, *_ = np.linalg.lstsq(M, -(p0 - min_), rcond=None)
    nu, d0 = nu + sol[0], d0 + sol[1]
rin = float(np.sqrt(((np.array([a_pred_med(K, nu, d0) for K in Kin]) - min_) ** 2).mean()))
dl = {K: a_pred_med(K, nu, d0) - meas(K) for K in range(61, 83)}
print(
    "median Borel sum: refitted pair (%.13f, %.13f), in-sample rms over K_b 83..130 %.4e (truncation nmax 18: 1.5200e-13)"
    % (nu, d0, rin)
)
print(
    "differences (1e-12) K_b 61..69: "
    + ", ".join("%+.1f" % (dl[K] * 1e12) for K in range(61, 70))
    + "; largest size 61..69 %.3e, 70..82 %.3e"
    % (max(abs(dl[K]) for K in range(61, 70)), max(abs(dl[K]) for K in range(70, 83)))
)
ex20 = [E[n] for n in range(6, 21)]
ex18 = [E[n] for n in range(6, 19)]
P18 = (-0.0937736508347, 0.0011599160063)
a_med = a_pred_med(61, *P18)
a_t20 = B.a_pred(61, *P18, ex20)
a_t18 = B.a_pred(61, *P18, ex18)
print(
    "at K_b 61 with the pair of q1669 fixed: median sum minus truncation nmax 20 %+.3e; minus truncation nmax 18 %+.3e"
    % (a_med - a_t20, a_med - a_t18)
)
chk = [abs(TAB(x) - smed(x)) for x in (0.0052, 0.0061, 0.0069)]
x4 = 1 / 12.608284652941046**2
c4 = abs(smed(x4) - sum(float(S18["e%d" % n]) * x4**n for n in range(2, 19)))
lab = lambda q: "holds" if q else "does not hold"
p1 = max(abs(dl[K]) for K in range(61, 70)) < 5.3e-12
p2 = abs(rin / 1.5200e-13 - 1) < 0.30
p3 = abs(a_med - a_t20) > 1e-12
ctl = max(chk) < 1e-13 * 0 + 1e-13 and c4 < 1e-11
print("control: table minus direct %s, C4 %.2e" % (", ".join("%.1e" % c for c in chk), c4))
print(
    "P1 largest size over K_b 61..69 below 5.3e-12: %s; P2 in-sample rms within 30 percent: %s; P3 median sum differs from the nmax 20 truncation at K_b 61 by more than 1e-12: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(ctl))
)
print("READING q1675 done (labels by the script; no verdict word)")
