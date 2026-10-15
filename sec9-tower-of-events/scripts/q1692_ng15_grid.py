"""q1692 (10 Oct 2026, READING, no verdict word): evaluation of F12 to F19 of q1666_prereg_f30.txt (registered by code at 09:53, 10:02, 10:11 and 10:14, each after a grep showed no line 'f30 = ') and of the Aitken extrapolation over (NG = 13, 14, 15), on the ORD = 30 output (f30 = -7.25424082052e-12, found by q1667 at 10:32). The wall pair is refitted on K_b 83..130 (four rounds) in each cell; the lower births K_b 61..82 (stored tower files) and the later events K_b 131..135, 137, 138 are compared by differences only.
Registered text (copied): F12 (NG = 15, nmax = 18) the largest size of a_pred - A over K_b 61..69 lies between 3.2e-12 and 6.3e-12; F13 the difference at K_b 73 stays above +5e-12; F14 the best nmax for the lower births K_b 61..69 (over nmax = 10..18) is 16 or 18 and its largest size lies between 0.8e-12 and 4.1e-12; F15 at nmax = 14 the largest size over K_b 61..69 is larger than at nmax = 16; F16 (nmax = 18) the signed difference lies in [-5.5e-12, -3.3e-12] at K_b 61, in [-2.0e-12, -1.2e-12] at K_b 69 and in [-0.28e-12, -0.16e-12] at K_b 82; F17 the signed difference at K_b 61 decreases monotonically with nmax from 10 to 18; F18 f30/f28 lies between 0.455 and 0.475; F19 the refit pair at nmax = 18 has nu0 in [-0.0937736508450, -0.0937736508390] and d0 in [0.0011599160100, 0.0011599160150].
New predictions, written after q1667 was read (pair and rms of NG = 15 printed, nothing about the lower births at NG = 15) and before this script was run:
 G1 the Aitken extrapolation over (13, 14, 15) with a fixed ratio rho_bar (median over K_b 61..82) gives a largest |a_inf - A| over K_b 61..82 without K_b 73 below 0.45e-12 (the value of the extrapolation over (12, 13, 14));
 G2 rho_bar lies between 0.38 and 0.45;
 G3 the later events K_b 131..135, 137, 138 satisfy |a_inf - A| < 0.1e-12 each.
Control: the (NG = 14, nmax = 18) cell reproduces the signed differences of q1678b at K_b 61 (-10.53e-12) to 0.05e-12.
Usage: py q1692_ng15_grid.py (from small_gap_q206/; about 2 minutes)."""

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


def parse(fn, ns):
    txt = open(os.path.join(H, fn), encoding="utf-8").read()
    out = {}
    for n in ns:
        m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, txt, re.M)
        out[n] = mp.mpf(sp.N(sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi}), 30))
    return out


f = parse("q1557_stdout_ORD30.txt", (26, 28, 30))
f28 = f[30] / f[28]
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


def fit(NG, nmax):
    setlaw(NG)
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
    return (nu, d0), {K: B.a_pred(K, nu, d0, ex) for K in Klow + Kin + Kout}


cell = {}
for nmax in range(10, 19):
    cell[nmax] = fit(15, nmax)
sd = {nmax: {K: cell[nmax][1][K] - meas(K) for K in (61, 69, 73, 82)} for nmax in cell}
sz = {nmax: max(abs(cell[nmax][1][K] - meas(K)) for K in range(61, 70)) for nmax in cell}
print(
    "NG 15 signed a_pred - A at K_b 61 (1e-12), nmax 10..18: "
    + " ".join("%8.2f" % (sd[n][61] * 1e12) for n in range(10, 19))
)
print(
    "NG 15 signed a_pred - A at K_b 69: "
    + " ".join("%8.2f" % (sd[n][69] * 1e12) for n in range(10, 19))
    + "; at K_b 82: "
    + " ".join("%7.2f" % (sd[n][82] * 1e12) for n in range(10, 19))
)
print(
    "NG 15 largest size over K_b 61..69 (1e-12), nmax 10..18: "
    + " ".join("%8.2f" % (sz[n] * 1e12) for n in range(10, 19))
    + "; K_b 73 at nmax 18: %+.2f" % (sd[18][73] * 1e12)
)
best = min(sz, key=sz.get)
pair = cell[18][0]
print(
    "f30/f28 = %.5f; pair (NG 15, nmax 18) = (%.13f, %.13f); best nmax by lower-K size %d (%.3e)"
    % (f28, *pair, best, sz[best])
)
# Aitken over (13, 14, 15)
P = {NG: fit(NG, 18)[1] for NG in (13, 14, 15)}
rho = {K: (P[15][K] - P[14][K]) / (P[14][K] - P[13][K]) for K in Klow}
rb = float(np.median(list(rho.values())))
D = {K: P[15][K] + (P[15][K] - P[14][K]) * rb / (1 - rb) - meas(K) for K in Klow + Kin + Kout}
g1v = max(abs(D[K]) for K in Klow if K != 73)
rin = float(np.sqrt(np.mean([D[K] ** 2 for K in Kin])))
print(
    "Aitken over (13, 14, 15): rho_bar %.4f (range %.3f to %.3f); largest |a_inf - A| over K_b 61..82 without 73: %.3e; K_b 73: %+.2e; rms K_b 83..130: %.3e; K_b 131..138 (1e-12): %s"
    % (
        rb,
        min(rho.values()),
        max(rho.values()),
        g1v,
        D[73],
        rin,
        ", ".join("%d %+.3f" % (K, D[K] * 1e12) for K in Kout),
    )
)
lab = lambda q: "holds" if q else "does not hold"
F12 = 3.2e-12 <= sz[18] <= 6.3e-12
F13 = sd[18][73] > 5e-12
F14 = best in (16, 18) and 0.8e-12 <= sz[best] <= 4.1e-12
F15 = sz[14] > sz[16]
F16 = (
    -5.5e-12 <= sd[18][61] <= -3.3e-12
    and -2.0e-12 <= sd[18][69] <= -1.2e-12
    and -0.28e-12 <= sd[18][82] <= -0.16e-12
)
F17 = all(sd[n + 1][61] < sd[n][61] for n in range(10, 18))
F18 = 0.455 <= f28 <= 0.475
F19 = -0.0937736508450 <= pair[0] <= -0.0937736508390 and 0.0011599160100 <= pair[1] <= 0.0011599160150
G1 = g1v < 0.45e-12
G2 = 0.38 <= rb <= 0.45
G3 = all(abs(D[K]) < 0.1e-12 for K in Kout)
ctl = abs(fit(14, 18)[1][61] - meas(61) + 10.53e-12) < 0.05e-12
print(
    "F12 %s; F13 %s; F14 %s; F15 %s; F16 %s; F17 %s; F18 %s; F19 %s; G1 %s; G2 %s; G3 %s; control %s"
    % tuple(lab(x) for x in (F12, F13, F14, F15, F16, F17, F18, F19, G1, G2, G3, ctl))
)
print("READING q1692 done (labels by the script; no verdict word)")
