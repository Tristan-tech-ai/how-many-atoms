"""q1737 (10 Oct 2026, READING, no verdict word): out-of-selection check of the fit-free choice (NG = 18, nmax = 20) that q1736 found best for the lower births K_b 61..82. The same prediction (limit pair of model B, bulk law NG = 17, 18, 19 with the coefficient model B, stored centre series to order 20, no fitted number) is evaluated on the events that did not enter the choice: K_b 83..130 and K_b 131..143.
NOT BLIND: the choice of NG = 18 and nmax = 20 was made on the lower births (reading 194 follows), so the later events are the independent check; no amplitude of K_b >= 83 was used in the choice.
Predictions:
 P1 the rms of a_pred - A over K_b 83..130 with (NG = 18, nmax = 20) is below 1.5e-13;
 P2 |a_pred - A| < 2e-13 for every K_b 131..143 that exists;
 P3 the rms over K_b 83..130 with (NG = 18, nmax = 20) is not larger than with (NG = 24, nmax = 18) of q1734 (8.1e-14) by more than a factor 2.
Usage: py q1737_fit_free_nmax20.py (from small_gap_q206/; about 2 minutes)."""

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
rms = lambda d, Ks: float(np.sqrt(np.mean([d[K] ** 2 for K in Ks])))


def diffs(G, NG, pair, Ks):
    setlaw(G, NG)
    return {K: B.a_pred(K, pair[0], pair[1], ex) - meas(K) for K in Ks}


GB = extended(0.0318)
PB = (-0.0937736508483, 0.0011599160185)
Kout = [K for K in range(131, 144) if B.measured(K) is not None]
out = {}
for nmax in (18, 20):
    ex[:] = [E[n] for n in range(6, nmax + 1)]
    for NG in (17, 18, 19, 24):
        d = diffs(GB, NG, PB, Klow + Kmid + Kout)
        out[(nmax, NG)] = d
        print(
            "nmax %d NG %2d: K_b 61..82 largest %.2e; rms over K_b 83..130 %.2e; K_b 131..143 largest %.2e"
            % (nmax, NG, max(abs(d[K]) for K in Klow), rms(d, Kmid), max(abs(d[K]) for K in Kout))
        )
d = out[(20, 18)]
print(
    "nmax 20 NG 18 a_pred - A (1e-12), K_b 131..143: "
    + ", ".join("%d %+.2f" % (K, d[K] * 1e12) for K in Kout)
)
lab = lambda q: "holds" if q else "does not hold"
p1 = rms(d, Kmid) < 1.5e-13
p2 = all(abs(d[K]) < 2e-13 for K in Kout)
p3 = rms(d, Kmid) <= 2 * 8.14e-14
print(
    "P1 rms over K_b 83..130 below 1.5e-13: %s; P2 K_b 131..143 below 2e-13: %s; P3 not worse than 2 x 8.1e-14: %s"
    % (lab(p1), lab(p2), lab(p3))
)
print("READING q1737 done (labels by the script; no verdict word)")
