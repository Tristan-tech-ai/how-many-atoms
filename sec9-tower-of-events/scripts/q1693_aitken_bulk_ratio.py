"""q1693 (10 Oct 2026, READING, no verdict word): q1692 showed that the Aitken extrapolation over (NG = 13, 14, 15) with the fixed ratio rho_bar = 0.4404 leaves 1.634e-12 at the lower births (G1 of q1692 did not hold), worse than the extrapolation over (12, 13, 14) with rho_bar = 0.4130 (0.45e-12). The per-event ratio (a15 - a14)/(a14 - a13) rises from 0.39-0.43 to 0.40-0.46 with the order, as the bulk coefficient ratios f_(2n+2)/f_(2n) do (0.3386, 0.3707, 0.4036, 0.4336, 0.4620). New variant, written after q1692 was read: a two-point extrapolation a_inf = a15 + (a15 - a14) r/(1 - r) with r = f30/f28 = 0.46196, the last known bulk ratio, a model quantity that uses no amplitude; compared with the same extrapolation with r = 0.4404 and with a15 alone.
Predictions:
 P1 with r = f30/f28 the largest |a_inf - A| over K_b 61..82 without K_b 73 is below 1.0e-12;
 P2 it is smaller than the 1.634e-12 of the fixed-ratio extrapolation over (13, 14, 15) in q1692;
 P3 the later events K_b 131..135, 137, 138 satisfy |a_inf - A| < 0.1e-12 each.
Control: with r = 0 the extrapolation reproduces a15 and the largest |a15 - A| over K_b 61..69 equals 3.72e-12 of q1692.
Usage: py q1693_aitken_bulk_ratio.py (from small_gap_q206/; about 1 minute)."""

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
txt = open(os.path.join(H, "q1557_stdout_ORD30.txt"), encoding="utf-8").read()
f = {}
for n in (26, 28, 30):
    m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, txt, re.M)
    f[n] = mp.mpf(sp.N(sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi}), 30))
r30 = float(f[30] / f[28])
FALL = (B.F + [-2.420924146814851e-10, -8.973800969707301e-11])[:13] + [
    float(f[26]),
    float(f[28]),
    float(f[30]),
]


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
Klow = list(range(61, 83))
Kout = [K for K in (131, 132, 133, 134, 135, 137, 138) if B.measured(K) is not None]
ex = [E[n] for n in range(6, 19)]
P = {}
for NG in (14, 15):
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
    P[NG] = {K: B.a_pred(K, nu, d0, ex) for K in Klow + Kin + Kout}


def D(r):
    return {K: P[15][K] + (P[15][K] - P[14][K]) * r / (1 - r) - meas(K) for K in Klow + Kin + Kout}


Dr, D44, D0 = D(r30), D(0.4404), D(0.0)
mx = lambda d, ks: max(abs(d[K]) for K in ks)
low = [K for K in Klow if K != 73]
print(
    "r = f30/f28 = %.5f: largest |a_inf - A| over K_b 61..82 without 73: %.3e (K_b 73 %+.2e); rms K_b 83..130 %.3e; K_b 131..138 (1e-12): %s"
    % (
        r30,
        mx(Dr, low),
        Dr[73],
        float(np.sqrt(np.mean([Dr[K] ** 2 for K in Kin]))),
        ", ".join("%d %+.3f" % (K, Dr[K] * 1e12) for K in Kout),
    )
)
print(
    "r = 0.4404: largest over K_b 61..82 without 73: %.3e; no extrapolation (a15): largest over K_b 61..69 %.3e, over K_b 70..82 without 73 %.3e"
    % (mx(D44, low), mx(D0, range(61, 70)), mx(D0, [K for K in range(70, 83) if K != 73]))
)
lab = lambda q: "holds" if q else "does not hold"
p1 = mx(Dr, low) < 1.0e-12
p2 = mx(Dr, low) < 1.634e-12
p3 = all(abs(Dr[K]) < 0.1e-12 for K in Kout)
ctl = abs(mx(D0, range(61, 70)) - 3.72e-12) < 5e-14
print(
    "P1 below 1.0e-12: %s; P2 smaller than the fixed-ratio value 1.634e-12: %s; P3 later events within 0.1e-12: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(ctl))
)
print("READING q1693 done (labels by the script; no verdict word)")
