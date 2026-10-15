"""q1721 (10 Oct 2026, READING, no verdict word): the wall pair of the Aitken-extrapolated births model refitted on the widest event set, K_b 61..130 (the 22 lower births with the amplitudes re-measured by the arb walker, q1712_res.json and q1717_res.json, and the 48 births K_b 83..130), against the pair fitted on K_b 83..130 alone, with predictions for K_b 131..142.
NOT BLIND: the lower births are predicted to 0.21e-12 with the pair of K_b 83..130 (reading 180), which was read before the predictions were written.
Predictions:
 P1 the refitted pair (law through c28, four rounds) moves by less than 3e-12 in nu0 and in d0 against the pair of K_b 83..130;
 P2 the rms of a_inf - A over all 70 fitted events is below 6e-14;
 P3 |a_inf - A| for K_b 131..142 (the events that exist) stays within 0.1e-12 with the refitted pair.
Control: the 83..130 fit reproduces the rms 3.123e-14 of q1720 over K_b 83..130 within 1 percent.
Usage: py q1721_pair_all_events.py (from small_gap_q206/; about 1 to 2 minutes)."""

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
    return B.measured(K) if K >= 83 else NEW[K]


ex = [E[n] for n in range(6, 19)]
h = 1e-7
Kmid = [K for K in range(83, 131) if B.measured(K) is not None]
Klow = list(range(61, 83))
Kout = [K for K in range(131, 143) if B.measured(K) is not None]


def fit(Kfit):
    mf = np.array([meas(K) for K in Kfit])
    pairs = {}
    pred = {}
    for NG in (12, 13, 14):
        setlaw(NG)
        nu, d0 = -0.0937736508347, 0.0011599160063
        for rnd in range(4):
            p0 = np.array([B.a_pred(K, nu, d0, ex) for K in Kfit])
            M = np.stack(
                [
                    [(B.a_pred(K, nu + h, d0, ex) - B.a_pred(K, nu - h, d0, ex)) / (2 * h) for K in Kfit],
                    [(B.a_pred(K, nu, d0 + h, ex) - B.a_pred(K, nu, d0 - h, ex)) / (2 * h) for K in Kfit],
                ],
                1,
            )
            sol, *_ = np.linalg.lstsq(M, -(p0 - mf), rcond=None)
            nu, d0 = nu + sol[0], d0 + sol[1]
        pairs[NG] = (nu, d0)
        pred[NG] = {K: B.a_pred(K, nu, d0, ex) for K in Klow + Kmid + Kout}
    RHO = {K: (pred[14][K] - pred[13][K]) / (pred[13][K] - pred[12][K]) for K in Klow}
    rb = float(np.median(list(RHO.values())))
    inf = {K: pred[14][K] + (pred[14][K] - pred[13][K]) * rb / (1 - rb) for K in Klow + Kmid + Kout}
    return pairs, rb, inf


pa, rba, ia = fit(Kmid)
pb, rbb, ib = fit(Klow + Kmid)
rms = lambda inf, Ks: float(np.sqrt(np.mean([(inf[K] - meas(K)) ** 2 for K in Ks])))
print(
    "pair NG = 14: fit on K_b 83..130 (%.13f, %.13f); fit on K_b 61..130 (%.13f, %.13f); move %+.3e, %+.3e"
    % (*pa[14], *pb[14], pb[14][0] - pa[14][0], pb[14][1] - pa[14][1])
)
print(
    "pair NG = 12: move %+.3e, %+.3e; NG = 13: move %+.3e, %+.3e; rho_bar %.4f and %.4f"
    % (pb[12][0] - pa[12][0], pb[12][1] - pa[12][1], pb[13][0] - pa[13][0], pb[13][1] - pa[13][1], rba, rbb)
)
print(
    "rms of a_inf - A: fit 83..130 over K_b 83..130 %.3e; fit 61..130 over all 70 events %.3e, over K_b 83..130 %.3e, over K_b 61..82 %.3e"
    % (rms(ia, Kmid), rms(ib, Klow + Kmid), rms(ib, Kmid), rms(ib, Klow))
)
print(
    "out of sample K_b 131..142 (1e-12), fit 83..130: "
    + ", ".join("%d %+.3f" % (K, (ia[K] - meas(K)) * 1e12) for K in Kout)
)
print(
    "out of sample K_b 131..142 (1e-12), fit 61..130: "
    + ", ".join("%d %+.3f" % (K, (ib[K] - meas(K)) * 1e12) for K in Kout)
)
lab = lambda q: "holds" if q else "does not hold"
p1 = abs(pb[14][0] - pa[14][0]) < 3e-12 and abs(pb[14][1] - pa[14][1]) < 3e-12
p2 = rms(ib, Klow + Kmid) < 6e-14
p3 = all(abs(ib[K] - meas(K)) < 0.1e-12 for K in Kout)
ctl = abs(rms(ia, Kmid) / 3.123e-14 - 1) < 0.01
print(
    "P1 pair moves by less than 3e-12: %s; P2 rms over 70 events below 6e-14: %s; P3 K_b 131..142 within 0.1e-12: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(ctl))
)
print("READING q1721 done (labels by the script; no verdict word)")
