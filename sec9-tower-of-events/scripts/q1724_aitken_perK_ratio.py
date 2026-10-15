"""q1724 (10 Oct 2026, READING, no verdict word): Aitken extrapolation over the bulk order with the ratio of each K taken from its own three predictions (no fixed rho_bar), for the lower births K_b 61..82 with the re-measured amplitudes, for the triples (12, 13, 14) and (13, 14, 15); the pair of each law is fitted on K_b 83..130 as in q1722. Also the Aitken value of the two per-K extrapolations with the four predictions (12, 13, 14, 15): the second Aitken step over the sequence of the two first-step values and the c30 value.
NOT BLIND: readings 180 and 182 were read before the predictions were written.
Predictions:
 P1 the per-K Aitken value of (12, 13, 14) differs from the re-measured amplitude by less than 0.3e-12 for every K_b 61..82;
 P2 the per-K Aitken value of (13, 14, 15) also differs by less than 0.3e-12 for every K_b 61..82 (the fixed ratio gave 1.634e-12);
 P3 the second Aitken step (over the first-step values of the two triples and the c30 value) does not lower the largest difference below that of P1.
Control: the per-K ratio of the exact geometric sequence 1 + 0.4^N recovers the limit within 1e-12.
Usage: py q1724_aitken_perK_ratio.py (from small_gap_q206/; about 2 minutes)."""

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
    return rb, low, rms, inf, pred


import numpy as np


def aitken(x0, x1, x2):
    return x2 - (x2 - x1) ** 2 / ((x2 - x1) - (x1 - x0))


_, _, _, _, pa = run((12, 13, 14))
_, _, _, _, pb = run((13, 14, 15))
v1 = {K: aitken(pa[12][K], pa[13][K], pa[14][K]) for K in Klow}
v2 = {K: aitken(pb[13][K], pb[14][K], pb[15][K]) for K in Klow}
# second step over the sequence v1(12-14), v2(13-15) and the c30 value: three terms of a sequence that should converge faster
v3 = {K: aitken(v1[K], v2[K], pb[15][K]) for K in Klow}
e1 = {K: v1[K] - meas(K) for K in Klow}
e2 = {K: v2[K] - meas(K) for K in Klow}
e3 = {K: v3[K] - meas(K) for K in Klow}
print("per-K Aitken (12, 13, 14) minus A (1e-12): " + ", ".join("%d %+.2f" % (K, e1[K] * 1e12) for K in Klow))
print("per-K Aitken (13, 14, 15) minus A (1e-12): " + ", ".join("%d %+.2f" % (K, e2[K] * 1e12) for K in Klow))
print("second step minus A (1e-12): " + ", ".join("%d %+.2f" % (K, e3[K] * 1e12) for K in Klow))
m1, m2, m3 = (
    max(abs(v) for v in e1.values()),
    max(abs(v) for v in e2.values()),
    max(abs(v) for v in e3.values()),
)
print("largest sizes: (12, 13, 14) %.3e; (13, 14, 15) %.3e; second step %.3e" % (m1, m2, m3))
geo = [1 + 0.4**N for N in (12, 13, 14)]
ctl = abs(aitken(*geo) - 1) < 1e-12
lab = lambda q: "holds" if q else "does not hold"
print("P1 %s; P2 %s; P3 %s; control: %s" % (lab(m1 < 0.3e-12), lab(m2 < 0.3e-12), lab(m3 >= m1), lab(ctl)))
print("READING q1724 done (labels by the script; no verdict word)")
