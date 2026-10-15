"""q1745 (10 Oct 2026, READING, no verdict word): the tail of the ratio-model bulk series beyond n = 18 at the inner atoms. Setup of q1740 unchanged (F_0..F_15 stored through f30, model E above, recursion for g_n at 50 digits, rho_k from N_law(rho) = k + nu0 at NG = 18). Tail sum S = sum of t_n for n = 19 up to the index of the smallest term (29 at k = 30, 41 at k = 60 in q1740), with t_n = |g_n| rho^(3-2n) / |3-2n| (no factor 16 pi^2).
NOT BLIND: the single terms of q1740 were seen (1.49e-13 at n = 19 and k = 30). Predictions (written at 13:22):
 P1 the sum of t_n for n = 19..29 at k = 30 lies between 3e-13 and 1.5e-12;
 P2 the sum of t_n for n = 19..41 at k = 60 lies below 1e-15;
 P3 the ratio of the k = 30 tail sum to the term n = 18 lies between 3 and 12.
Control (otherwise VOID): the radii and the terms of q1740 are reproduced: rho_30 = 0.9399318052 and rho_60 = 1.11556582 to 9 digits, the term n = 19 at k = 30 is 1.486e-13 and the smallest terms 1.92e-14 and 2.29e-19 (3 digits).
Usage: py q1745_bulk_tail_sums.py (from small_gap_q206/; seconds)."""

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


rho = {k: mp.findroot(lambda r: Nl(r, g, NGL) - (k + NU0), 1.0) for k in (30, 60)}
print(
    "rho_30 = %s, rho_60 = %s (N_law(rho) = k + nu0, law to NG = 18)"
    % (mp.nstr(rho[30], 10), mp.nstr(rho[60], 10))
)
# C2: same law in births_law
B.F = [float(x) for x in G[: NGL + 1]]
B.NG = NGL
B.g = [float(x) for x in g[: NGL + 1]]
c2 = max(abs(float(dl(rho[k], g, NGL)) / float(B.dlaw(float(rho[k]))) - 1) for k in (30, 60))
print(
    "C2 position law from the terms against births_law.dlaw at NG = 18: largest relative difference %.2e" % c2
)
if not (c1 < 1e-12 and c2 < 1e-12):
    print("VOID: control failed")
    sys.exit(2)


def terms(r):
    return {n: abs(g[n]) * mp.mpf(r) ** (3 - 2 * n) / abs(3 - 2 * n) for n in range(1, TOP + 1)}


T = {k: terms(rho[k]) for k in (30, 60)}
nstar = {30: 29, 60: 41}
ctl = (
    abs(rho[30] - mp.mpf("0.9399318052")) < 1e-9
    and abs(rho[60] - mp.mpf("1.11556582")) < 1e-8
    and abs(float(T[30][19]) / 1.486e-13 - 1) < 1e-3
    and abs(float(T[30][29]) / 1.92e-14 - 1) < 3e-3
    and abs(float(T[60][41]) / 2.29e-19 - 1) < 3e-3
)
print("CONTROL q1740 values reproduced: %s" % ("holds" if ctl else "does not hold"))
if not ctl:
    print("VOID: control failed")
    sys.exit(2)
S30 = sum(T[30][n] for n in range(19, 30))
S60 = sum(T[60][n] for n in range(19, 42))
S30b = sum(T[30][n] for n in range(19, 41))
print(
    "k = 30: t_18 = %s, tail sum n = 19..29 = %s (n = 19..40: %s), ratio tail / t_18 = %s, tail / t_19 = %s"
    % (
        mp.nstr(T[30][18], 4),
        mp.nstr(S30, 4),
        mp.nstr(S30b, 4),
        mp.nstr(S30 / T[30][18], 4),
        mp.nstr(S30 / T[30][19], 4),
    )
)
print(
    "k = 60: t_18 = %s, tail sum n = 19..41 = %s, ratio tail / t_18 = %s"
    % (mp.nstr(T[60][18], 4), mp.nstr(S60, 4), mp.nstr(S60 / T[60][18], 4))
)
print("tail terms k = 30, n = 19..29: " + " ".join(mp.nstr(T[30][n], 3) for n in range(19, 30)))
lab = lambda q: "holds" if q else "does not hold"
print(
    "P1 k = 30 sum n = 19..29 between 3e-13 and 1.5e-12: %s (%.3e); P2 k = 60 sum n = 19..41 below 1e-15: %s (%.3e); P3 ratio tail / t_18 at k = 30 between 3 and 12: %s (%.3f)"
    % (
        lab(3e-13 <= float(S30) <= 1.5e-12),
        float(S30),
        lab(float(S60) < 1e-15),
        float(S60),
        lab(3 <= float(S30 / T[30][18]) <= 12),
        float(S30 / T[30][18]),
    )
)
print("READING q1745 done (labels by the script; no verdict word)")
