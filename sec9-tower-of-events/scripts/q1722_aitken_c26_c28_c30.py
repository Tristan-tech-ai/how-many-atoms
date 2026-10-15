"""q1722 (10 Oct 2026, READING, no verdict word): the Aitken extrapolation over the bulk order with the laws c26, c28, c30 (NG = 13, 14, 15; f30 from q1557_stdout_ORD30.txt) against the one with c24, c26, c28 (NG = 12, 13, 14), both with the lower-birth amplitudes re-measured by the arb walker (q1712_res.json, q1717_res.json). The pair of each law is fitted on K_b 83..130 (four rounds), the ratio rho_bar is the median over K_b 61..82 of the ratios of the model predictions.
NOT BLIND: reading 157 had found the (13, 14, 15) version worse at the lower births with the stored amplitudes, and the corrected data of the (12, 13, 14) version were read (reading 180).
Predictions:
 P1 the largest |a_inf - A| over K_b 61..82 with (13, 14, 15) is above the 0.21e-12 of (12, 13, 14);
 P2 the rms of a_inf - A over K_b 83..130 with (13, 14, 15) is above the 3.123e-14 of (12, 13, 14);
 P3 rho_bar of (13, 14, 15) is above 0.43.
Control: the (12, 13, 14) run in this script reproduces 0.21e-12 (largest over K_b 61..82, to 0.01e-12) and 3.123e-14 (to 1 percent).
Usage: py q1722_aitken_c26_c28_c30.py (from small_gap_q206/; about 2 minutes)."""

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
FALL = (B.F + [-2.420924146814851e-10, -8.973800969707301e-11])[:13] + [f[26], f[28], f[30]]


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
    return B.measured(K) if K >= 83 else NEW[K]


ex = [E[n] for n in range(6, 19)]
h = 1e-7
Kin = [K for K in range(83, 131) if B.measured(K) is not None]
Klow = list(range(61, 83))
mf = np.array([B.measured(K) for K in Kin])


def run(triple):
    pred = {}
    for NG in triple:
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
            sol, *_ = np.linalg.lstsq(M, -(p0 - mf), rcond=None)
            nu, d0 = nu + sol[0], d0 + sol[1]
        pred[NG] = {K: B.a_pred(K, nu, d0, ex) for K in Klow + Kin}
    a, b, c = triple
    rb = float(np.median([(pred[c][K] - pred[b][K]) / (pred[b][K] - pred[a][K]) for K in Klow]))
    inf = {K: pred[c][K] + (pred[c][K] - pred[b][K]) * rb / (1 - rb) for K in Klow + Kin}
    low = max(abs(inf[K] - meas(K)) for K in Klow)
    rms = float(np.sqrt(np.mean([(inf[K] - meas(K)) ** 2 for K in Kin])))
    return rb, low, rms, inf


r1 = run((12, 13, 14))
r2 = run((13, 14, 15))
for name, r in (("(12, 13, 14)", r1), ("(13, 14, 15)", r2)):
    print(
        "%s: rho_bar %.4f; largest |a_inf - A| over K_b 61..82 %.3e; rms over K_b 83..130 %.3e"
        % (name, r[0], r[1], r[2])
    )
print(
    "(13, 14, 15) a_inf - A (1e-12), K_b 61..69: "
    + ", ".join("%d %+.2f" % (K, (r2[3][K] - meas(K)) * 1e12) for K in range(61, 70))
)
lab = lambda q: "holds" if q else "does not hold"
p1 = r2[1] > 0.21e-12
p2 = r2[2] > 3.123e-14
p3 = r2[0] > 0.43
ctl = abs(r1[1] - 0.21e-12) < 0.01e-12 and abs(r1[2] / 3.123e-14 - 1) < 0.01
print(
    "P1 largest over K_b 61..82 above 0.21e-12: %s; P2 rms over K_b 83..130 above 3.123e-14: %s; P3 rho_bar above 0.43: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(ctl))
)
print("READING q1722 done (labels by the script; no verdict word)")
