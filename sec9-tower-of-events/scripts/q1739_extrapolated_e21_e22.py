"""q1739 (10 Oct 2026, READING, no verdict word): the centre series extended beyond the stored order 20 by a guess, to see whether the best bulk order NG moves up with nmax as reading 194 and reading 196 suggest. e_21 and e_22 are NOT derived: the ratio r_n = e_n / e_(n-1) of the stored e_14..e_20 is fitted by a straight line in n (variant A, the registered one), e_21 = r_21 e_20 and e_22 = r_22 e_21. Variant B (sensitivity, no label): a quadratic in n through the same seven ratios. The fit and its residual are printed first. Same setup as q1736 (limit pair of model B, bulk coefficient model B with increment 0.0318, re-measured lower births).
NOT BLIND: readings 194 to 196 were read; the ratios r_n and the line fit were looked at before this script was written (max residual 0.033 of ratios near 90; the increments r_n - r_(n-1) rise from 4.209 to 4.304, so a straight line underestimates the next ratio by about 0.06).
Predictions (written at 13:05):
 P1 with nmax = 21 (variant A) the best NG at K_b 61 over NG = 14..24 is 18 or 19;
 P2 with nmax = 21 (variant A) the best size at K_b 61 stays below 2.5e-13;
 P3 with nmax = 22 (variant A) the best NG at K_b 61 is between 18 and 20.
Control (otherwise the run is VOID): nmax = 20 reproduces q1736: best NG 18 at K_b 61 and K_b 82 with sizes 2.49e-13 and 2.84e-14 (within 1 percent).
Usage: py q1739_extrapolated_e21_e22.py (from small_gap_q206/; about 1 minute)."""

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
ns = np.arange(14, 21)
y = np.array([float(E[n] / E[n - 1]) for n in ns])
pl = np.polyfit(ns, y, 1)
res = y - np.polyval(pl, ns)
pq = np.polyfit(ns, y, 2)
print(
    "ratio line r_n = %.6f n + %.6f on n = 14..20; residuals %s; largest %.3f (relative %.1e)"
    % (pl[0], pl[1], np.array2string(res, precision=3), abs(res).max(), abs(res / y).max())
)
EA = dict(E)
EB = dict(E)
for n in (21, 22):
    EA[n] = EA[n - 1] * float(np.polyval(pl, n))
    EB[n] = EB[n - 1] * float(np.polyval(pq, n))
print(
    "e_21: A %.6e, B %.6e (relative difference %.1e); e_22: A %.6e, B %.6e (relative difference %.1e)"
    % (EA[21], EB[21], abs(EB[21] / EA[21] - 1), EA[22], EB[22], abs(EB[22] / EA[22] - 1))
)
cases = [("A", 20, E), ("A", 21, EA), ("A", 22, EA), ("B", 21, EB), ("B", 22, EB)]
best = {}
for tag, nmax, Ed in cases:
    ex[:] = [Ed[n] for n in range(6, nmax + 1)]
    tab = {NG: diffs(GB, NG, PB, Ks) for NG in range(14, 25)}
    b = {K: min(range(14, 25), key=lambda NG: abs(tab[NG][K])) for K in Ks}
    best[(tag, nmax)] = (tab, b)
    print(
        "variant %s nmax %d: best NG per K: " % (tag, nmax)
        + ", ".join("K_b %d: NG %d (size %.2e)" % (K, b[K], abs(tab[b[K]][K])) for K in Ks)
    )
    print(
        "   K_b 61 a_pred - A (1e-12) by NG 16..22: "
        + " ".join("%d %+.2f" % (NG, tab[NG][61] * 1e12) for NG in range(16, 23))
    )
lab = lambda q: "holds" if q else "does not hold"
t0, b0 = best[("A", 20)]
ctl = (
    b0[61] == 18
    and b0[82] == 18
    and abs(abs(t0[18][61]) / 2.49e-13 - 1) < 0.01
    and abs(abs(t0[18][82]) / 2.84e-14 - 1) < 0.01
)
print("CONTROL nmax 20 reproduces q1736 (best NG 18 at K_b 61 and 82, sizes within 1 percent): %s" % lab(ctl))
if not ctl:
    print("VOID: control failed")
    raise SystemExit(2)
t21, b21 = best[("A", 21)]
t22, b22 = best[("A", 22)]
p1 = b21[61] in (18, 19)
p2 = abs(t21[b21[61]][61]) < 2.5e-13
p3 = 18 <= b22[61] <= 20
print(
    "P1 nmax 21 best NG at K_b 61 is 18 or 19: %s; P2 nmax 21 best size at K_b 61 below 2.5e-13: %s; P3 nmax 22 best NG at K_b 61 in 18..20: %s"
    % (lab(p1), lab(p2), lab(p3))
)
print("READING q1739 done (labels by the script; no verdict word)")
