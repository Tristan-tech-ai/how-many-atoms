"""q1740 (10 Oct 2026, READING, no verdict word): where is the smallest term of the bulk series at the inner atoms? With the bulk coefficients F_j of the stored law through f30 (F_0..F_15, as in q1736) and model E above (F_j = F_(j-1) (j + c) / A for j >= 16, A = 32.307, c = -0.023, the least-squares fit of q1732), the coefficients g_n of the density law are computed by the recursion g_n = -sum_j F_j g_(n-j) to n = 60 in mpmath at 50 digits. The term of the position law at the radius rho is t_n(rho) = |g_n| rho^(3-2n) / |3-2n| (the factor 16 pi^2 is left out). rho_k solves N_law(rho) = k + nu0 with the law truncated at NG = 18 and nu0 = -0.0937736508483 (model B pair of q1736).
NOT BLIND: readings 188 to 197 were read. Predictions (written at 13:05):
 P1 the smallest term at k = 30 sits at n > 18;
 P2 the index of the smallest term at k = 60 exceeds that at k = 30 by at least 5;
 P3 the term at n = 19 at k = 30 lies between 2e-13 and 2e-11.
The registration said 'to n = 40'; the series is evaluated to n = 60 so that the minimum is not at the edge, and the indices are reported for n <= 40 and n <= 60.
Controls (otherwise the run is VOID): C1 the recursion at NG = 10 equals births_law.g[0..10] to a relative 1e-12; C2 16 pi^2 [rho^3/3 + sum_(n<=18) g_n rho^(3-2n)/(3-2n)] equals births_law.dlaw(rho) with the same law (NG = 18) to a relative 1e-12 at rho_30 and rho_60.
Usage: py q1740_bulk_term_sizes.py (from small_gap_q206/; seconds)."""

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


res = {}
for k in (30, 60):
    t = terms(rho[k])
    res[k] = t
    n40 = min(range(1, 41), key=lambda n: t[n])
    n60 = min(range(1, TOP + 1), key=lambda n: t[n])
    print(
        "k = %d: smallest term for n <= 40 at n = %d (%s), for n <= 60 at n = %d (%s); terms n = 15, 17, 19, 21, 25, 30: %s"
        % (
            k,
            n40,
            mp.nstr(t[n40], 3),
            n60,
            mp.nstr(t[n60], 3),
            " ".join(mp.nstr(t[n], 3) for n in (15, 17, 19, 21, 25, 30)),
        )
    )
n30 = min(range(1, TOP + 1), key=lambda n: res[30][n])
n60 = min(range(1, TOP + 1), key=lambda n: res[60][n])
print("A rho^2 at k = 30 and 60: %s, %s" % (mp.nstr(Aq * rho[30] ** 2, 5), mp.nstr(Aq * rho[60] ** 2, 5)))
t19 = float(res[30][19])
lab = lambda q: "holds" if q else "does not hold"
print(
    "P1 smallest term at k = 30 at n > 18: %s; P2 index at k = 60 exceeds that at k = 30 by at least 5: %s; P3 term n = 19 at k = 30 is %.3e, between 2e-13 and 2e-11: %s"
    % (lab(n30 > 18), lab(n60 - n30 >= 5), t19, lab(2e-13 <= t19 <= 2e-11))
)
print("READING q1740 done (labels by the script; no verdict word)")
