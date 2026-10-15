"""q1688 (10 Oct 2026, READING, no verdict word, predictions written at 10:26): Aitken extrapolation over the bulk order. For each event the predictions a12, a13, a14 of the bulk laws through c24, c26, c28 at nmax = 18, each with its own wall pair refitted on K_b 83..130 (four rounds), give a_inf = a14 + (a14 - a13) rho/(1 - rho) with rho = (a14 - a13)/(a13 - a12). The amplitude A cancels in the differences used for rho. Compared with the measured events: lower births K_b 61..82, in-sample K_b 83..130, and the later events K_b 131..135, 137 and 138 with the pairs frozen from the in-sample refits.
Predictions (partly known: hand arithmetic on the printed signed tables of q1678b gave a_inf - A of +0.7e-12 at K_b 61, +0.06e-12 at K_b 69 and +0.01e-12 at K_b 82):
 P1 the largest |a_inf - A| over K_b 61..69 is below 2e-12;
 P2 the largest |a_inf - A| over K_b 70..82 without K_b 73 is below 1e-12;
 P3 the rms of a_inf - A over K_b 83..130 is below 3e-13;
 P4 |a_inf - A| < 5e-13 for each of K_b 131..135, 137, 138.
Control: rho lies between 0.3 and 0.6 at every event used, and the extrapolation of an exact geometric sequence 1 + 0.4^N for N = 12, 13, 14 returns the limit 1 within 1e-12.
Usage: py q1688_aitken_bulk.py (from small_gap_q206/; about 2 minutes)."""

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
    if K >= 83:
        return B.measured(K)
    fn = (
        os.path.join(A, "tower_K%d_birth_out.json" % (K + 1))
        if K % 2 == 0
        else os.path.join(A, "tower_K%d_As2.json" % (K + 1))
    )
    d = json.load(open(fn))
    return d["A_cross"] if K % 2 == 0 else d["A_s"]


Kin = [K for K in range(83, 131) if B.measured(K) is not None]
min_ = np.array([B.measured(K) for K in Kin])
h = 1e-7
Kall = (
    list(range(61, 83)) + Kin + [K for K in (131, 132, 133, 134, 135, 137, 138) if B.measured(K) is not None]
)
ex = [E[n] for n in range(6, 19)]
pred = {}
for NG in (12, 13, 14):
    setlaw(NG)
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
    pred[NG] = {K: B.a_pred(K, nu, d0, ex) for K in Kall}


def ext(a12, a13, a14):
    rho = (a14 - a13) / (a13 - a12)
    return a14 + (a14 - a13) * rho / (1 - rho), rho


D = {}
RHO = {}
for K in Kall:
    a, r = ext(pred[12][K], pred[13][K], pred[14][K])
    D[K] = a - meas(K)
    RHO[K] = r
d14 = {K: pred[14][K] - meas(K) for K in Kall}
print("a_inf - A (1e-12), K_b 61..69: " + ", ".join("%d %+.2f" % (K, D[K] * 1e12) for K in range(61, 70)))
print("a_inf - A (1e-12), K_b 70..82: " + ", ".join("%d %+.2f" % (K, D[K] * 1e12) for K in range(70, 83)))
print(
    "a_inf - A (1e-12), K_b 131..138: "
    + ", ".join("%d %+.3f" % (K, D[K] * 1e12) for K in Kall if K >= 131)
    + "; the law through c28 alone: "
    + ", ".join("%d %+.3f" % (K, d14[K] * 1e12) for K in Kall if K >= 131)
)
rin = float(np.sqrt(np.mean([D[K] ** 2 for K in Kin])))
r14 = float(np.sqrt(np.mean([d14[K] ** 2 for K in Kin])))
print(
    "rms over K_b 83..130: a_inf %.3e, law through c28 alone %.3e; rho range over all events %.3f to %.3f (K_b 61..82: %.3f to %.3f)"
    % (
        rin,
        r14,
        min(RHO.values()),
        max(RHO.values()),
        min(RHO[K] for K in range(61, 83)),
        max(RHO[K] for K in range(61, 83)),
    )
)
geo = [1 + 0.4**N for N in (12, 13, 14)]
ag, _ = ext(*geo)
lab = lambda q: "holds" if q else "does not hold"
p1 = max(abs(D[K]) for K in range(61, 70)) < 2e-12
p2 = max(abs(D[K]) for K in range(70, 83) if K != 73) < 1e-12
p3 = rin < 3e-13
p4 = all(abs(D[K]) < 5e-13 for K in Kall if K >= 131)
ctl = all(0.3 < r < 0.6 for r in RHO.values()) and abs(ag - 1) < 1e-12
print(
    "P1 K_b 61..69 below 2e-12: %s; P2 K_b 70..82 without 73 below 1e-12: %s; P3 rms over K_b 83..130 below 3e-13: %s; P4 K_b 131..138 within 5e-13: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(p4), lab(ctl))
)
print("READING q1688 done (labels by the script; no verdict word)")
