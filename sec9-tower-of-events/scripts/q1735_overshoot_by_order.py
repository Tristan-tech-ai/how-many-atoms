"""q1735 (10 Oct 2026, READING, no verdict word): where does the fit-free extension of the bulk law overshoot at the lower births? a_pred - A at K_b 61, 65, 70, 76, 82 for the bulk law NG = 14..24 (coefficients beyond f30 from model B of q1729), with the limit pair of model B (-0.0937736508483, +0.0011599160185), and the NG of the smallest size per K.
NOT BLIND: reading 191 was read.
Predictions:
 P1 at K_b 61 the best NG lies between 15 and 18;
 P2 the best size at K_b 61 is below 1.5e-12;
 P3 for K_b 82 the best NG is above 20 and the size there is below 1e-13.
Usage: py q1735_overshoot_by_order.py (from small_gap_q206/; about 1 minute)."""

import json, os, re
import numpy as np
import sympy as sp
import mpmath as mp
import births_law as B

H = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(H, "..", "centre_scripts")
S20 = json.load(open(os.path.join(C, "d22_out_20.json")))
E = {n: float(S20["e%d" % n]) for n in range(2, 21)}


def parse(fn, ns):
    t = open(os.path.join(H, fn), encoding="utf-8").read()
    out = {}
    for n in ns:
        m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, t, re.M)
        out[n] = float(
            mp.mpf(sp.N(sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi}), 30))
        )
    return out


f = parse("q1557_stdout_ORD30.txt", (26, 28, 30))
F15 = (B.F + [-2.420924146814851e-10, -8.973800969707301e-11])[:13] + [f[26], f[28], f[30]]


def extended(inc, top=24):
    G = list(F15)
    r15 = G[15] / G[14]
    for j in range(16, top + 1):
        G.append(G[-1] * (r15 + (j - 15) * inc))
    return G


def setlaw(G, NG):
    B.F = G[: NG + 1]
    B.NG = NG
    B.g = [1.0]
    for n in range(1, NG + 1):
        B.g.append(-sum(B.F[j] * B.g[n - j] for j in range(1, n + 1)))


R = json.load(open(os.path.join(H, "q1712_res.json")))
R.update(json.load(open(os.path.join(H, "q1717_res.json"))))
NEW = {int(k): float(v["new"]) for k, v in R.items() if v.get("status") == "ok"}


def meas(K):
    return B.measured(K) if K >= 83 else NEW[K]


ex = [E[n] for n in range(6, 19)]
Klow = list(range(61, 83))
Kmid = [K for K in range(83, 131) if B.measured(K) is not None]
Kout = [K for K in range(131, 143) if B.measured(K) is not None]


def diffs(G, NG, pair, Ks):
    setlaw(G, NG)
    return {K: B.a_pred(K, pair[0], pair[1], ex) - meas(K) for K in Ks}


PB = (-0.0937736508483, 0.0011599160185)
GB = extended(0.0318)
Ks = [61, 65, 70, 76, 82]
tab = {NG: diffs(GB, NG, PB, Ks) for NG in range(14, 25)}
print("a_pred - A (1e-12) by NG:    " + "   ".join("K_b %d" % K for K in Ks))
for NG in range(14, 25):
    print("  NG = %2d  " % NG + "   ".join("%+8.2f" % (tab[NG][K] * 1e12) for K in Ks))
best = {K: min(range(14, 25), key=lambda NG: abs(tab[NG][K])) for K in Ks}
print(
    "best NG per K: "
    + ", ".join("K_b %d: NG %d (size %.2e)" % (K, best[K], abs(tab[best[K]][K])) for K in Ks)
)
lab = lambda q: "holds" if q else "does not hold"
p1 = 15 <= best[61] <= 18
p2 = abs(tab[best[61]][61]) < 1.5e-12
p3 = best[82] > 20 and abs(tab[best[82]][82]) < 1e-13
print(
    "P1 best NG at K_b 61 in [15, 18]: %s; P2 best size at K_b 61 below 1.5e-12: %s; P3 K_b 82 best NG above 20 with size below 1e-13: %s"
    % (lab(p1), lab(p2), lab(p3))
)
print("READING q1735 done (labels by the script; no verdict word)")
