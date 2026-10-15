"""q1718 (10 Oct 2026, READING, no verdict word): the shape fits of q1673 (ln|d| against v^2 and against ln K, d = a_pred - A of the births model q1671) redone for all 22 births K_b 61..82 with the amplitudes re-measured by the arb walker (q1712_res.json, q1717_res.json), K_b 73 included. Copy of q1673 with three changes: the amplitudes, the event list and the output.
NOT BLIND: the corrected differences (reading 175) were read before the predictions were written. The registration text P1 ('rms falls by a factor 3 against the fit with the stored K_b 73') cannot be evaluated, since q1673 excluded K_b 73, and is replaced here before any output:
 P1 the rms of the residual of ln|d| against v^2 for the 22 corrected events is below 0.08 (q1673: 0.1196 for 21 stored events);
 P2 the slope against v^2 stays within 10 percent of -0.1266 (q1673);
 P3 the exponent against ln K stays within 10 percent of -9.514 (q1673);
 P4 the rms against ln K is larger than the rms against v^2, as in q1673 (0.1315 against 0.1196).
Control: the same fitting code on the 21 stored events (K_b 73 excluded, stored amplitudes) reproduces the q1673 slopes -0.1266 and -9.514 to 0.001.
Usage: py q1718_lower_k_shape_corrected.py (seconds)."""

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


R = json.load(open(os.path.join(H, "q1712_res.json")))
R.update(json.load(open(os.path.join(H, "q1717_res.json"))))
NEW = {int(k): float(v["new"]) for k, v in R.items() if v.get("status") == "ok"}
Ks = list(range(61, 83))
d = {}
V = {}
d0s = {}
for K in Ks:
    a = B.a_pred(K, NU, D0, ex)
    d[K] = a - NEW[K]
    d0s[K] = a - meas(K)
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
Kc = [K for K in Ks if K != 73]
xc = np.array([V[K] for K in Kc])
yc = np.log(np.abs([d0s[K] for K in Kc]))
lc = np.log(Kc)
sc1, _ = fit(xc, yc)
sc2, _ = fit(lc, yc)
lab = lambda q: "holds" if q else "does not hold"
rms1 = float(np.sqrt((res**2).mean()))
rms2 = float(np.sqrt((res2**2).mean()))
print("control on the 21 stored events: slope against v^2 %.4f, exponent against ln K %.3f" % (sc1, sc2))
p1 = rms1 < 0.08
p2 = abs(s1 / -0.1266 - 1) < 0.10
p3 = abs(s2 / -9.514 - 1) < 0.10
p4 = rms2 > rms1
ctl = abs(sc1 + 0.1266) < 0.001 and abs(sc2 + 9.514) < 0.001
print(
    "P1 rms against v^2 below 0.08: %s; P2 slope within 10 percent of -0.1266: %s; P3 exponent within 10 percent of -9.514: %s; P4 rms against ln K larger: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(p4), lab(ctl))
)
print("READING q1718 done (labels by the script; no verdict word)")
