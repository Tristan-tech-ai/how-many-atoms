"""q1720 (10 Oct 2026, READING, no verdict word): the Aitken extrapolation over the bulk order of q1689 (fixed ratio rho_bar, bulk laws through c24, c26, c28, stored series to order 18) with the lower-birth amplitudes K_b 61..82 replaced by the amplitudes re-measured with the arb walker (q1712_res.json and q1717_res.json). Copy of q1689 with the function meas changed for K < 83 and with new labels.
NOT BLIND: the re-measured amplitudes and the corrected differences of the c28 law (readings 174, 175) were read before the predictions were written; the extrapolated differences had not been computed.
Predictions:
 P1 the largest |a_inf - A| over K_b 61..69 is below 2e-12 (it was below this with the stored amplitudes);
 P2 the largest |a_inf - A| over K_b 70..82, K_b 73 included, is below 1e-12;
 P3 |a_inf - A| at K_b 73 is below 1e-12 (it was +11.1e-12 with the stored amplitude);
 P4 the rms of a_inf - A over K_b 83..130 is below 2e-13 (the data are unchanged there: a control of the code).
Control: rho_bar lies between 0.38 and 0.45 and the extrapolation of the exact geometric sequence 1 + 0.4^N for N = 12, 13, 14 with rho_bar = 0.4 returns the limit 1 within 1e-12.
Usage: py q1720_aitken_remeasured_lower_births.py (from small_gap_q206/; about 1 minute)."""

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


R = json.load(open(os.path.join(H, "q1712_res.json")))
R.update(json.load(open(os.path.join(H, "q1717_res.json"))))
NEW = {int(k): float(v["new"]) for k, v in R.items() if v.get("status") == "ok"}


def meas(K):
    if K >= 83:
        return B.measured(K)
    return NEW[K]


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
RHO = {K: (pred[14][K] - pred[13][K]) / (pred[13][K] - pred[12][K]) for K in range(61, 83)}
rb = float(np.median(list(RHO.values())))


def ext(a12, a13, a14, rho=None):
    rho = rb if rho is None else rho
    return a14 + (a14 - a13) * rho / (1 - rho)


D = {K: ext(pred[12][K], pred[13][K], pred[14][K]) - meas(K) for K in Kall}
d14 = {K: pred[14][K] - meas(K) for K in Kall}
print(
    "fixed ratio rho_bar (median over K_b 61..82) = %.4f; range there %.3f to %.3f"
    % (rb, min(RHO.values()), max(RHO.values()))
)
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
print("rms over K_b 83..130: a_inf %.3e, law through c28 alone %.3e" % (rin, r14))
geo = [1 + 0.4**N for N in (12, 13, 14)]
ag = ext(*geo, rho=0.4)
lab = lambda q: "holds" if q else "does not hold"
p1 = max(abs(D[K]) for K in range(61, 70)) < 2e-12
p2 = max(abs(D[K]) for K in range(70, 83)) < 1e-12
p3 = abs(D[73]) < 1e-12
p4 = rin < 2e-13
ctl = 0.38 < rb < 0.45 and abs(ag - 1) < 1e-12
print(
    "P1 K_b 61..69 below 2e-12: %s; P2 K_b 70..82 with 73 below 1e-12: %s; P3 K_b 73 below 1e-12: %s; P4 rms over K_b 83..130 below 2e-13: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(p4), lab(ctl))
)
print("READING q1720 done (labels by the script; no verdict word)")
