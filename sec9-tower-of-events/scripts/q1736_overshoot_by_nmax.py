"""q1736 (10 Oct 2026, READING, no verdict word): the table of q1735 repeated for the stored centre series to order nmax = 16, 18 and 20, to separate the truncation of the centre series from that of the bulk law. Same pair (model B limit pair), same bulk extension. Original docstring of q1735: (10 Oct 2026, READING, no verdict word): where does the fit-free extension of the bulk law overshoot at the lower births? a_pred - A at K_b 61, 65, 70, 76, 82 for the bulk law NG = 14..24 (coefficients beyond f30 from model B of q1729), with the limit pair of model B (-0.0937736508483, +0.0011599160185), and the NG of the smallest size per K.
NOT BLIND: reading 191 was read.
Predictions of q1736 (replace those of q1735): NOT BLIND, reading 192 read.
 P1 with nmax = 20 the best size at K_b 61 over NG = 14..24 is below the 6.4e-13 of nmax = 18;
 P2 with nmax = 20 the best size at K_b 76 and K_b 82 stays below 5e-14;
 P3 the best NG at K_b 61 and K_b 82 does not move by more than one order between nmax = 16, 18 and 20.
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
import copy

res_by_nmax = {}
for nmax in (16, 18, 20):
    ex[:] = [E[n] for n in range(6, nmax + 1)]
    tab = {NG: diffs(GB, NG, PB, Ks) for NG in range(14, 25)}
    best = {K: min(range(14, 25), key=lambda NG: abs(tab[NG][K])) for K in Ks}
    res_by_nmax[nmax] = (tab, best)
    print(
        "nmax = %d: best NG per K: " % nmax
        + ", ".join("K_b %d: NG %d (size %.2e)" % (K, best[K], abs(tab[best[K]][K])) for K in Ks)
    )
tab, best = res_by_nmax[20]
print(
    "nmax = 20, a_pred - A (1e-12) by NG, K_b 61 and K_b 82: "
    + "; ".join("NG %d %+.2f %+.2f" % (NG, tab[NG][61] * 1e12, tab[NG][82] * 1e12) for NG in range(14, 25))
)
lab = lambda q: "holds" if q else "does not hold"
b16, b18, b20 = res_by_nmax[16][1], res_by_nmax[18][1], res_by_nmax[20][1]
t18, t20 = res_by_nmax[18][0], res_by_nmax[20][0]
p1 = abs(t20[b20[61]][61]) < abs(t18[b18[61]][61])
p2 = abs(t20[b20[76]][76]) < 5e-14 and abs(t20[b20[82]][82]) < 5e-14
p3 = all(max(b16[K], b18[K], b20[K]) - min(b16[K], b18[K], b20[K]) <= 1 for K in (61, 82))
print(
    "P1 best size at K_b 61 with nmax 20 below that with nmax 18: %s; P2 best sizes at K_b 76 and 82 with nmax 20 below 5e-14: %s; P3 best NG moves by at most one order between nmax 16, 18, 20 at K_b 61 and 82: %s"
    % (lab(p1), lab(p2), lab(p3))
)
print("READING q1736 done (labels by the script; no verdict word)")
