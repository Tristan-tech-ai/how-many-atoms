"""q1676 (10 Oct 2026, READING, no verdict word, predictions written at 10:00): joint scan of the bulk order NG = 11..14 (law through c22, c24, c26, c28) and the series order nmax = 10, 12, 14, 16, 18, 20, with the wall pair refitted on K_b 83..130 in every cell (two linearisation rounds from the q1669 pair). Printed per cell: the in-sample rms over K_b 83..130 and the largest sizes of a_pred - A over the lower births K_b 61..69 and K_b 70..82 (K_b 73 included).
Predictions:
 P1 the smallest value of the largest size over K_b 61..69 in the grid is below 7e-12 and is reached at nmax <= 16;
 P2 the cell with the smallest in-sample rms is not the cell with the smallest lower-K size over K_b 61..69;
 P3 for every NG the lower-K size over K_b 61..69 is smallest at nmax <= 16.
Control: the cell NG = 14, nmax = 18 reproduces 1.5200e-13 and 1.055e-11 of q1669 and q1671, and the cell NG = 14, nmax = 20 reproduces 1.6407e-13 and 1.269e-11 of q1669 and q1674.
Usage: py q1676_joint_scan.py (from small_gap_q206/; about 2 minutes)."""

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
for NG in (11, 12, 13, 14):
    setlaw(NG)
    for nmax in (10, 12, 14, 16, 18, 20):
        ex = [E[n] for n in range(6, nmax + 1)]
        nu, d0 = -0.0937736508347, 0.0011599160063
        for rnd in range(2):
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
        dl = {K: B.a_pred(K, nu, d0, ex) - low[K] for K in low}
        res[(NG, nmax)] = (
            rin,
            max(abs(dl[K]) for K in range(61, 70)),
            max(abs(dl[K]) for K in range(70, 83)),
        )
print("in-sample rms over K_b 83..130 (1e-13); rows NG = 11..14, columns nmax = 10, 12, 14, 16, 18, 20")
for NG in (11, 12, 13, 14):
    print("NG %d: " % NG + "  ".join("%9.2f" % (res[(NG, n)][0] * 1e13) for n in (10, 12, 14, 16, 18, 20)))
print("largest size over K_b 61..69 (1e-12)")
for NG in (11, 12, 13, 14):
    print("NG %d: " % NG + "  ".join("%9.2f" % (res[(NG, n)][1] * 1e12) for n in (10, 12, 14, 16, 18, 20)))
print("largest size over K_b 70..82 (1e-12)")
for NG in (11, 12, 13, 14):
    print("NG %d: " % NG + "  ".join("%9.2f" % (res[(NG, n)][2] * 1e12) for n in (10, 12, 14, 16, 18, 20)))
best_low = min(res, key=lambda c: res[c][1])
best_in = min(res, key=lambda c: res[c][0])
print(
    "smallest lower-K size over K_b 61..69: %.3e at NG %d, nmax %d; smallest in-sample rms: %.3e at NG %d, nmax %d"
    % (res[best_low][1], *best_low, res[best_in][0], *best_in)
)
lab = lambda q: "holds" if q else "does not hold"
p1 = res[best_low][1] < 7e-12 and best_low[1] <= 16
p2 = best_low != best_in
p3 = all(min((res[(NG, n)][1], n) for n in (10, 12, 14, 16, 18, 20))[1] <= 16 for NG in (11, 12, 13, 14))
ctl = (
    abs(res[(14, 18)][0] - 1.5200e-13) < 5e-16
    and abs(res[(14, 18)][1] - 1.055e-11) < 5e-15
    and abs(res[(14, 20)][0] - 1.6407e-13) < 5e-16
    and abs(res[(14, 20)][1] - 1.269e-11) < 5e-15
)
print(
    "P1 smallest lower-K size below 7e-12 at nmax <= 16: %s; P2 best in-sample cell differs from best lower-K cell: %s; P3 for every NG the smallest lower-K size is at nmax <= 16: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(ctl))
)
print("READING q1676 done (labels by the script; no verdict word)")
