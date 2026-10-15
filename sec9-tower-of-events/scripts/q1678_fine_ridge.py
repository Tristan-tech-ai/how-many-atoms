"""q1678 (10 Oct 2026, READING, no verdict word, predictions written at 10:03, written before the odd series orders were evaluated): fine ridge of the joint truncation. All integer series orders nmax = 9..17 for the bulk orders NG = 11..14 (law through c22, c24, c26, c28), with the wall pair refitted on K_b 83..130 in every cell (four linearisation rounds). Printed per cell: the in-sample rms and the largest size of a_pred - A over the lower births K_b 61..69.
Predictions:
 P1 the best nmax for the lower-K size is NG + 1 for NG = 11, 12 and 13 (that is 12, 13 and 14);
 P2 the best lower-K size along the ridge falls by a factor between 0.15 and 0.5 per step of NG from 11 to 13;
 P3 for NG = 14 the best nmax is 15 or 16.
Control: the cells (14, 14) and (14, 16) reproduce the in-sample rms 5.844e-14 and 1.102e-13 of q1680 within 2 percent, and the lower-K sizes of q1677 (10.74e-12, 5.14e-12) within 5 percent.
Usage: py q1678_fine_ridge.py (from small_gap_q206/; about 2 minutes)."""

import json, os, re
import numpy as np
import sympy as sp
import mpmath as mp
import births_law as B

H = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(H, "..", "scratch")
C = os.path.join(H, "..", "centre_scripts")
S20 = json.load(open(os.path.join(C, "d22_out_20.json")))
E = {n: float(S20["e%d" % n]) for n in range(2, 21)}
txt = open(os.path.join(H, "q1557_stdout_ORD28.txt"), encoding="utf-8").read()
f = {}
for n in (26, 28):
    m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, txt, re.M)
    f[n] = float(mp.mpf(sp.N(sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi}), 30)))
FALL = (B.F + [-2.420924146814851e-10, -8.973800969707301e-11])[:13] + [f[26], f[28]]


def setlaw(NG):
    B.F = FALL[: NG + 1]
    B.NG = NG
    B.g = [1.0]
    for n in range(1, NG + 1):
        B.g.append(-sum(B.F[j] * B.g[n - j] for j in range(1, n + 1)))


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
low = {K: meas(K) for K in range(61, 83)}
h = 1e-7
res = {}
NM = tuple(range(9, 18))
for NG in (11, 12, 13, 14):
    setlaw(NG)
    for nmax in NM + ((18, 20) if NG == 14 else ()):
        ex = [E[n] for n in range(6, nmax + 1)]
        nu, d0 = -0.0937736508347, 0.0011599160063
        for rnd in range(4):
            p0 = np.array([B.a_pred(K, nu, d0, ex) for K in Kin])
            M = np.stack(
                [
                    [(B.a_pred(K, nu + h, d0, ex) - B.a_pred(K, nu - h, d0, ex)) / (2 * h) for K in Kin],
                    [(B.a_pred(K, nu, d0 + h, ex) - B.a_pred(K, nu, d0 - h, ex)) / (2 * h) for K in Kin],
                ],
                1,
            )
            sol, *_ = np.linalg.lstsq(M, -(p0 - min_), rcond=None)
            nu, d0 = nu + sol[0], d0 + sol[1]
        rin = float(np.sqrt(((np.array([B.a_pred(K, nu, d0, ex) for K in Kin]) - min_) ** 2).mean()))
        res[(NG, nmax)] = (rin, max(abs(B.a_pred(K, nu, d0, ex) - low[K]) for K in range(61, 70)))
print("in-sample rms (1e-13); columns nmax = " + " ".join("%7d" % n for n in NM))
for NG in (11, 12, 13, 14):
    print("NG %d: " % NG + " ".join("%7.2f" % (res[(NG, n)][0] * 1e13) for n in NM))
print("largest size over K_b 61..69 (1e-12)")
for NG in (11, 12, 13, 14):
    print("NG %d: " % NG + " ".join("%7.2f" % (res[(NG, n)][1] * 1e12) for n in NM))
best = {NG: min(NM, key=lambda n: res[(NG, n)][1]) for NG in (11, 12, 13, 14)}
bestin = {NG: min(NM, key=lambda n: res[(NG, n)][0]) for NG in (11, 12, 13, 14)}
print(
    "best nmax by lower-K size: "
    + ", ".join("NG %d: %d (%.2e)" % (NG, best[NG], res[(NG, best[NG])][1]) for NG in best)
    + "; best nmax by in-sample rms: "
    + ", ".join("NG %d: %d" % (NG, bestin[NG]) for NG in bestin)
)
lab = lambda q: "holds" if q else "does not hold"
bl = [res[(NG, best[NG])][1] for NG in (11, 12, 13)]
p1 = best[11] == 12 and best[12] == 13 and best[13] == 14
p2 = all(0.15 <= bl[i + 1] / bl[i] <= 0.5 for i in range(2))
p3 = best[14] in (15, 16)
ctl = (
    abs(res[(14, 14)][0] / 5.844e-14 - 1) < 0.02
    and abs(res[(14, 16)][0] / 1.102e-13 - 1) < 0.02
    and abs(res[(14, 14)][1] / 10.74e-12 - 1) < 0.05
    and abs(res[(14, 16)][1] / 5.14e-12 - 1) < 0.05
)
print("ratios of the best lower-K sizes NG 11 to 12, 12 to 13: %.3f, %.3f" % (bl[1] / bl[0], bl[2] / bl[1]))
print(
    "P1 best nmax = NG + 1 for NG 11, 12, 13: %s; P2 ratios in [0.15, 0.5]: %s; P3 NG 14 best nmax 15 or 16: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(ctl))
)
print("READING q1678 done (labels by the script; no verdict word)")
