"""q1746 (10 Oct 2026, READING, no verdict word): is the inner deviation of the exact atoms of the size of the Stokes level of the optimally truncated bulk series, S(k) = 16 pi^2 t_min(k), where t_min(k) is the smallest term |g_n| rho_k^(3-2n) / |3-2n| (n = 1..60) of the ratio-model series of q1740 at the radius rho_k? The residual column of q1727_out.txt (rows k = 30, 32, ..., 60, the residual of the exact block against the c28 law after the window fit of the added bulk term, reading 186) is parsed by code and set beside S(k).
NOT BLIND: the q1727 table and the q1740 terms were seen. Predictions (written at 13:30):
 P1 S(30) lies between 2e-12 and 4e-12;
 P2 S(k) falls monotonically with k over k = 28..60;
 P3 the shape does not match: the k of the largest |residual| over the printed rows k = 30..60 differs from the k of the largest S(k) (k = 30).
Control (otherwise VOID): t_min at k = 30 and k = 60 reproduces q1740 (1.92e-14 at n = 29 and 2.29e-19 at n = 41, 3 digits).
Usage: py q1746_stokes_level_vs_residual.py (from small_gap_q206/; seconds)."""

import os, re, sys
import numpy as np
import sympy as sp
import mpmath as mp
from scipy.optimize import brentq
import births_law as B

mp.mp.dps = 50
H = os.path.dirname(os.path.abspath(__file__))


def parse(fn, ns):
    t = open(os.path.join(H, fn), encoding="utf-8").read()
    out = {}
    for n in ns:
        m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, t, re.M)
        out[n] = mp.mpf(sp.N(sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi}), 40))
    return out


f = parse("q1557_stdout_ORD30.txt", (26, 28, 30))
F0 = [mp.mpf(x) for x in B.F]  # F_0..F_10 exact forms of births_law
F15 = F0 + [mp.mpf("-2.420924146814851e-10"), mp.mpf("-8.973800969707301e-11")]
F15 = F15[:13] + [f[26], f[28], f[30]]  # F_0..F_15 as in q1736 (positions 11, 12 are f22, f24)
Aq, cq = mp.mpf("32.307"), mp.mpf("-0.023")


def Fmodel(top):
    G = list(F15)
    for j in range(16, top + 1):
        G.append(G[-1] * (j + cq) / Aq)
    return G


def gser(G, top):
    g = [mp.mpf(1)]
    for n in range(1, top + 1):
        g.append(-sum(G[j] * g[n - j] for j in range(1, n + 1)))
    return g


TOP = 60
G = Fmodel(TOP)
g = gser(G, TOP)
c1 = max(abs((g[n] - mp.mpf(B.g[n])) / g[n]) for n in range(0, 11) if g[n] != 0) if False else None
G10 = [mp.mpf(x) for x in B.F]
g10 = gser(G10, 10)
c1 = max(abs(g10[n] - mp.mpf(float(B.g[n]))) / abs(g10[n]) for n in range(1, 11))
print("C1 recursion at NG = 10 against births_law.g: largest relative difference %.2e" % float(c1))
NU0 = mp.mpf("-0.0937736508483")
NGL = 18


def Nl(r, g, NG):
    r = mp.mpf(r)
    return (
        16
        * mp.pi**2
        * (
            r**4 / 4
            + g[1] * r**2 / 2
            + g[2] * mp.log(r)
            + sum(g[n] * r ** (4 - 2 * n) / (4 - 2 * n) for n in range(3, NG + 1))
        )
    )


def dl(r, g, NG):
    r = mp.mpf(r)
    return 16 * mp.pi**2 * (r**3 / 3 + sum(g[n] * r ** (3 - 2 * n) / (3 - 2 * n) for n in range(1, NG + 1)))


def tmin(k):
    r = mp.findroot(lambda x: Nl(x, g, NGL) - (k + NU0), 1.0)
    t = {n: abs(g[n]) * r ** (3 - 2 * n) / abs(3 - 2 * n) for n in range(1, TOP + 1)}
    n = min(t, key=lambda m: t[m])
    return r, n, t[n]


res = {}
for k in range(28, 61):
    res[k] = tmin(k)
ctl = (
    res[30][1] == 29
    and abs(float(res[30][2]) / 1.92e-14 - 1) < 3e-3
    and res[60][1] == 41
    and abs(float(res[60][2]) / 2.29e-19 - 1) < 3e-3
)
print("CONTROL q1740 smallest terms reproduced: %s" % ("holds" if ctl else "does not hold"))
if not ctl:
    print("VOID: control failed")
    sys.exit(2)
S = {k: 16 * mp.pi**2 * res[k][2] for k in res}
rows = {}
for ln in open(os.path.join(H, "q1727_out.txt"), encoding="utf-8"):
    tok = ln.split()
    if len(tok) == 6 and tok[0].isdigit() and int(tok[0]) >= 30:
        rows[int(tok[0])] = float(tok[4])
if len(rows) < 10:
    print("VOID: residual rows not parsed")
    sys.exit(2)
print("k, rho_k, n of the smallest term, S(k) = 16 pi^2 t_min, |residual| of q1727, ratio |residual| / S(k):")
for k in sorted(rows):
    print(
        "  k = %2d: rho %.5f, n* = %2d, S = %s, |residual| = %.3e, ratio = %s"
        % (k, float(res[k][0]), res[k][1], mp.nstr(S[k], 4), abs(rows[k]), mp.nstr(abs(rows[k]) / S[k], 3))
    )
mono = all(S[k + 1] < S[k] for k in range(28, 60))
kres = max(rows, key=lambda k: abs(rows[k]))
kS = max(S, key=lambda k: S[k])
ratios = [float(abs(rows[k]) / S[k]) for k in sorted(rows)]
print(
    "S(28) = %s, S(30) = %s; largest |residual| at k = %d, largest S at k = %d; the ratio |residual| / S(k) runs from %.3g to %.3g over the printed rows (spread factor %.3g)"
    % (mp.nstr(S[28], 4), mp.nstr(S[30], 4), kres, kS, min(ratios), max(ratios), max(ratios) / min(ratios))
)
lab = lambda q: "holds" if q else "does not hold"
print(
    "P1 S(30) between 2e-12 and 4e-12: %s; P2 S(k) falls monotonically over k = 28..60: %s; P3 largest |residual| at a k different from the k of the largest S: %s"
    % (lab(2e-12 <= float(S[30]) <= 4e-12), lab(mono), lab(kres != kS))
)
print("READING q1746 done (labels by the script; no verdict word)")
