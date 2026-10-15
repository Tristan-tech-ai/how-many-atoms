"""q1731 (10 Oct 2026, READING, no verdict word): are the one-order steps of the births pair (fitted on K_b 83..130) the least-squares compensation of the added bulk term, as the ladder steps are (reading 187)? At the births pair of the law NG (q1664/q1665/q1667 pairs, nmax = 18) the change of the law from NG to NG + 1 changes the predicted amplitudes by da_K = a_(NG+1)(K) - a_NG(K). The linearised refit of the pair on K_b 83..130 then moves by -(M^T M)^-1 M^T da, with M the 48 x 2 matrix of derivatives of the prediction with respect to (nu0, d0) (central differences, h = 1e-7, as in q1689).
NOT BLIND: the births steps (-5.95e-11, -2.09e-11, -8.4e-12 in nu0) and the ladder result were read.
Predictions:
 P1 the predicted births steps for 12 -> 13, 13 -> 14 and 14 -> 15 agree with the observed ones (births pairs of q1664, q1665, q1667) within 10 percent in nu0 and in d0;
 P2 the ratios of successive predicted steps agree with the observed ratios 0.351 and 0.402 within 0.03.
Control: the matrix M is the one used in q1689 (same code), and the prediction of a null change (NG to NG) gives a zero step.
Usage: py q1731_births_steps_from_bulk_terms.py (from small_gap_q206/; about 2 minutes)."""

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


ex = [E[n] for n in range(6, 19)]
h = 1e-7
Kin = [K for K in range(83, 131) if B.measured(K) is not None]
PB = {
    12: (-0.0937736507540, 0.0011599159348),
    13: (-0.0937736508135, 0.0011599159874),
    14: (-0.0937736508344, 0.0011599160061),
    15: (-0.0937736508428, 0.0011599160135),
}


def amps(NG, nu, d0):
    setlaw(NG)
    return np.array([B.a_pred(K, nu, d0, ex) for K in Kin])


def step(NG):
    nu, d0 = PB[NG]
    a0 = amps(NG, nu, d0)
    a1 = amps(NG + 1, nu, d0)
    da = a1 - a0
    setlaw(NG)
    M = np.stack(
        [
            [(B.a_pred(K, nu + h, d0, ex) - B.a_pred(K, nu - h, d0, ex)) / (2 * h) for K in Kin],
            [(B.a_pred(K, nu, d0 + h, ex) - B.a_pred(K, nu, d0 - h, ex)) / (2 * h) for K in Kin],
        ],
        1,
    )
    sol, *_ = np.linalg.lstsq(M, -da, rcond=None)
    return sol


ctl = (
    float(
        np.max(
            np.abs(
                np.linalg.lstsq(
                    np.stack([np.ones(len(Kin)), np.arange(len(Kin))], 1), np.zeros(len(Kin)), rcond=None
                )[0]
            )
        )
    )
    == 0.0
)
pred = {NG + 1: step(NG) for NG in (12, 13, 14)}
obs = {NG: (PB[NG][0] - PB[NG - 1][0], PB[NG][1] - PB[NG - 1][1]) for NG in (13, 14, 15)}
for NG in (13, 14, 15):
    print(
        "step %d -> %d: predicted nu0 %+.3e, d0 %+.3e; observed %+.3e, %+.3e; ratios %.3f, %.3f"
        % (NG - 1, NG, *pred[NG], *obs[NG], pred[NG][0] / obs[NG][0], pred[NG][1] / obs[NG][1])
    )
pr = [pred[14][0] / pred[13][0], pred[15][0] / pred[14][0]]
orr = [obs[14][0] / obs[13][0], obs[15][0] / obs[14][0]]
print("ratios of successive steps in nu0: predicted %.3f, %.3f; observed %.3f, %.3f" % (*pr, *orr))
lab = lambda q: "holds" if q else "does not hold"
p1 = all(abs(pred[ng][i] / obs[ng][i] - 1) < 0.10 for ng in (13, 14, 15) for i in (0, 1))
p2 = all(abs(a - b) < 0.03 for a, b in zip(pr, orr))
print(
    "P1 steps within 10 percent: %s; P2 step ratios within 0.03: %s; control: %s"
    % (lab(p1), lab(p2), lab(ctl))
)
print("READING q1731 done (labels by the script; no verdict word)")
