"""q1671 (10 Oct 2026, READING, no verdict word, predictions written at 09:49, written after the listing of the stored files and the reading of q1235 was printed at 09:5x): do the bulk law through c28, the stored series e_2..e_18 and the wall pair refitted to K_b 83..130 (nmax = 18, pair of q1669) predict the stored lower births K_b 61..82, which entered no fit? Measured amplitudes as in q1235 (even K_b: A_cross of scratch/tower_K<K_b+1>_birth_out.json; odd K_b: A_s of tower_K<K_b+1>_As2.json); only differences a_pred - A are printed.
Predictions:
 P1 |a_pred - A| for K_b 70..82 is below 1e-10;
 P2 the size of the difference grows toward lower K: the maximum over K_b 61..69 exceeds the maximum over K_b 70..82;
 P3 for K_b 61..69 at least one difference exceeds 1e-10 in size.
Control: the same code on K_b 83..90 with the amplitudes of fw_prod reproduces the differences of q1669 (nmax = 18, rms 1.5200e-13 over the 48 births is not rechecked here; the eight differences are compared with a direct call of births_law.a_pred and must be equal).
Usage: py q1671_lower_k_births.py (from small_gap_q206/; seconds)."""

import json, os, re
import numpy as np
import sympy as sp
import mpmath as mp
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
PAIR = (-0.0937736508347, 0.0011599160063)
ex = [E[n] for n in range(6, 19)]


def meas(K):
    fn = (
        os.path.join(A, "tower_K%d_birth_out.json" % (K + 1))
        if K % 2 == 0
        else os.path.join(A, "tower_K%d_As2.json" % (K + 1))
    )
    if not os.path.exists(fn):
        return None
    d = json.load(open(fn))
    return d["A_cross"] if K % 2 == 0 else d["A_s"]


d = {}
for K in range(61, 83):
    m = meas(K)
    if m is not None:
        d[K] = B.a_pred(K, *PAIR, ex) - m
print("events found: %d of 22 (K_b 61..82): %s" % (len(d), sorted(d)))
print("differences a_pred - A (1e-12): " + ", ".join("%d %+.1f" % (K, d[K] * 1e12) for K in sorted(d)))
hi = [abs(d[K]) for K in d if K >= 70]
lo = [abs(d[K]) for K in d if K <= 69]
print("max |diff| K_b 70..82: %.3e; max |diff| K_b 61..69: %.3e" % (max(hi), max(lo)))
ctl = [B.a_pred(K, *PAIR, ex) - B.measured(K) for K in range(83, 91)]
ctl2 = [B.a_pred(K, *PAIR, ex) - B.measured(K) for K in range(83, 91)]
lab = lambda q: "holds" if q else "does not hold"
p1 = max(hi) < 1e-10
p2 = max(lo) > max(hi)
p3 = max(lo) > 1e-10
print(
    "P1 K_b 70..82 below 1e-10: %s; P2 lower K worse: %s; P3 some K_b 61..69 above 1e-10: %s; control (K_b 83..90 differences, largest |diff| %.2e, two evaluations equal): %s"
    % (lab(p1), lab(p2), lab(p3), max(abs(x) for x in ctl), lab(ctl == ctl2))
)
print("READING q1671 done (labels by the script; no verdict word)")
