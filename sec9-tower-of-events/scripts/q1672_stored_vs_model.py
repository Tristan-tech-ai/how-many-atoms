"""q1672 (10 Oct 2026, READING, no verdict word, predictions written at 09:52; the stored e_19 and e_20 and the q1217 predictions were printed at 09:5x before this script was written, so P3 is not blind): are the stored coefficients above order 14 smooth, and does replacing e_15..e_20 by a smooth model extrapolation change the births rms?
Model: r_n = e_(n+1)/e_n = alpha n + beta + gamma/n fitted by least squares on n = 10..14 (five points) from d22_out_20.json; e_15 = e_14 r_14, e_16 = e_15 r_15, and so on up to e_20.
Predictions:
 P1 the stored ratios r_n for n = 10..19 lie within 0.3 of the q1217 model (alpha 4.445013827229242, beta 6.513159723811139, gamma 38.82665045972515, fitted on n = 10..17; for n = 18, 19 these are its registered predictions);
 P2 replacing e_15..e_20 by the extrapolation of the n = 10..14 fit changes the in-sample rms (bulk law through c28, pair refitted, nmax = 20) by less than 30 percent of the stored value 1.6407e-13;
 P3 the stored e_19 and e_20 agree with the q1217 predictions -1.4977196802674226e+29 and -1.3930579859998422e+31 to within 3 percent.
Control: with the stored values the nmax = 20 in-sample rms equals 1.6407e-13 of q1669 to 3 digits.
Usage: py q1672_stored_vs_model.py (from small_gap_q206/; about 2 minutes)."""

import json, os, re
import numpy as np
import sympy as sp
import mpmath as mp
import births_law as B

H = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(H, "..", "centre_scripts")
S = json.load(open(os.path.join(C, "d22_out_20.json")))
E = {n: float(S["e%d" % n]) for n in range(2, 21)}
al, be, ga = 4.445013827229242, 6.513159723811139, 38.82665045972515
r = {n: E[n + 1] / E[n] for n in range(10, 20)}
dev = {n: r[n] - (al * n + be + ga / n) for n in r}
print(
    "stored ratio minus q1217 model for n = 10..19: "
    + ", ".join("%d: %+.4f" % (n, dev[n]) for n in sorted(dev))
)
A = np.array([[n, 1, 1.0 / n] for n in range(10, 15)])
coef, *_ = np.linalg.lstsq(A, np.array([r[n] for n in range(10, 15)]), rcond=None)
Em = dict(E)
for n in range(15, 20):
    Em[n + 1] = Em[n] * (coef[0] * n + coef[1] + coef[2] / n)
# e_15 itself is E[15] stored? replace e_15..e_20 : e_15 = e_14 r_14
Em[15] = E[14] * (coef[0] * 14 + coef[1] + coef[2] / 14)
for n in range(15, 20):
    Em[n + 1] = Em[n] * (coef[0] * n + coef[1] + coef[2] / n)
print(
    "model coefficients (alpha, beta, gamma) from n = 10..14: %.4f, %.4f, %.4f; model/stored for e_15..e_20: %s"
    % (*coef, ", ".join("%d: %.5f" % (n, Em[n] / E[n]) for n in range(15, 21)))
)
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
Ks = [K for K in range(83, 131) if B.measured(K) is not None]
meas = np.array([B.measured(K) for K in Ks])
h = 1e-7


def refit(Ed):
    ex = [Ed[n] for n in range(6, 21)]
    nu, d0 = -0.0937736508344, 0.0011599160061
    for rnd in range(2):
        p0 = np.array([B.a_pred(K, nu, d0, ex) for K in Ks])
        M = np.stack(
            [
                [(B.a_pred(K, nu + h, d0, ex) - B.a_pred(K, nu - h, d0, ex)) / (2 * h) for K in Ks],
                [(B.a_pred(K, nu, d0 + h, ex) - B.a_pred(K, nu, d0 - h, ex)) / (2 * h) for K in Ks],
            ],
            1,
        )
        sol, *_ = np.linalg.lstsq(M, -(p0 - meas), rcond=None)
        nu, d0 = nu + sol[0], d0 + sol[1]
    return float(np.sqrt(((np.array([B.a_pred(K, nu, d0, ex) for K in Ks]) - meas) ** 2).mean()))


rs, rm = refit(E), refit(Em)
print(
    "in-sample rms at nmax = 20: stored %.4e, with e_15..e_20 from the n = 10..14 extrapolation %.4e (ratio %.3f)"
    % (rs, rm, rm / rs)
)
q19, q20 = -1.4977196802674226e29, -1.3930579859998422e31
print(
    "stored e_19 / prediction - 1 = %+.4f, e_20 / prediction - 1 = %+.4f" % (E[19] / q19 - 1, E[20] / q20 - 1)
)
lab = lambda q: "holds" if q else "does not hold"
p1 = all(abs(v) < 0.3 for v in dev.values())
p2 = abs(rm / rs - 1) < 0.30
p3 = abs(E[19] / q19 - 1) < 0.03 and abs(E[20] / q20 - 1) < 0.03
ctl = abs(rs - 1.6407e-13) < 5e-16
print(
    "P1 stored ratios within 0.3 of the q1217 model: %s; P2 model extrapolation changes the rms by less than 30 percent: %s; P3 e_19 and e_20 within 3 percent of the predictions: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(ctl))
)
print("READING q1672 done (labels by the script; no verdict word)")
