"""q1678b (10 Oct 2026, READING, post hoc, no predictions, written after q1678 was read): the SIGNED differences a_pred - A at K_b 61 and K_b 69 for NG = 11..14 and nmax = 10..18, same cells and refit as q1678 (four rounds, pair refitted on K_b 83..130). Purpose: to see whether the minimum of the lower-K size along nmax is a zero crossing of a monotone sequence or a true minimum.
Usage: py q1678b_signed_lower_k.py (from small_gap_q206/; about 2 minutes)."""

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
NM = tuple(range(10, 19))
for NG in (11, 12, 13, 14):
    setlaw(NG)
    for nmax in NM:
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
        res[(NG, nmax)] = (
            B.a_pred(61, nu, d0, ex) - low[61],
            B.a_pred(69, nu, d0, ex) - low[69],
            B.a_pred(82, nu, d0, ex) - low[82],
        )
for k, name in ((0, "K_b 61"), (1, "K_b 69"), (2, "K_b 82")):
    print("signed a_pred - A at %s (1e-12); columns nmax = %s" % (name, " ".join("%7d" % n for n in NM)))
    for NG in (11, 12, 13, 14):
        print("NG %d: " % NG + " ".join("%7.2f" % (res[(NG, n)][k] * 1e12) for n in NM))
print("READING q1678b done (post hoc, no labels)")
