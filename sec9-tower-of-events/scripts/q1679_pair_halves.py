"""q1679 (10 Oct 2026, READING, no verdict word, predictions written at 10:03): is the wall pair constant in K? The pair is refitted separately on K_b 83..106 and on K_b 107..130 (four linearisation rounds) in the cells (NG, nmax) = (13, 14) and (14, 16), and the full range is refitted for comparison. (Reading 149 later showed that the best cells are zero crossings; the cells here were fixed before that and are kept.)
Predictions:
 P1 the two half-range pairs differ by less than 1e-10 in nu0 and in d0 for both cells;
 P2 the in-sample rms of each half (with its own pair) is below 2e-13;
 P3 the largest size over the lower births K_b 61..69 with the pair of the lower half is within a factor 2 of the one with the full-range pair.
Control: the full-range fits reproduce the in-sample rms of q1680 (1.740e-13 and 1.102e-13) within 2 percent.
Usage: py q1679_pair_halves.py (from small_gap_q206/; about 1 minute)."""

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


low = {K: meas(K) for K in range(61, 70)}
h = 1e-7


def fit(ks, ex):
    m = np.array([B.measured(K) for K in ks])
    nu, d0 = -0.0937736508347, 0.0011599160063
    for rnd in range(4):
        p0 = np.array([B.a_pred(K, nu, d0, ex) for K in ks])
        M = np.stack(
            [
                [(B.a_pred(K, nu + h, d0, ex) - B.a_pred(K, nu - h, d0, ex)) / (2 * h) for K in ks],
                [(B.a_pred(K, nu, d0 + h, ex) - B.a_pred(K, nu, d0 - h, ex)) / (2 * h) for K in ks],
            ],
            1,
        )
        sol, *_ = np.linalg.lstsq(M, -(p0 - m), rcond=None)
        nu, d0 = nu + sol[0], d0 + sol[1]
    rms = float(np.sqrt(((np.array([B.a_pred(K, nu, d0, ex) for K in ks]) - m) ** 2).mean()))
    return nu, d0, rms


K1 = [K for K in range(83, 107) if B.measured(K) is not None]
K2 = [K for K in range(107, 131) if B.measured(K) is not None]
K0 = K1 + K2
res = {}
for NG, nmax in ((13, 14), (14, 16)):
    setlaw(NG)
    ex = [E[n] for n in range(6, nmax + 1)]
    pf, p1, p2 = fit(K0, ex), fit(K1, ex), fit(K2, ex)
    sz = lambda p: max(abs(B.a_pred(K, p[0], p[1], ex) - low[K]) for K in low)
    res[(NG, nmax)] = (pf, p1, p2, sz(pf), sz(p1))
    print(
        "cell (%d, %d): full pair (%.13f, %.13f) rms %.3e; lower half (%.13f, %.13f) rms %.3e; upper half (%.13f, %.13f) rms %.3e"
        % (NG, nmax, *pf[:2], pf[2], *p1[:2], p1[2], *p2[:2], p2[2])
    )
    print(
        "   half difference: nu0 %+.3e, d0 %+.3e; lower-K size K_b 61..69: full pair %.3e, lower-half pair %.3e"
        % (p2[0] - p1[0], p2[1] - p1[1], res[(NG, nmax)][3], res[(NG, nmax)][4])
    )
lab = lambda q: "holds" if q else "does not hold"
p1_ = all(abs(res[c][2][0] - res[c][1][0]) < 1e-10 and abs(res[c][2][1] - res[c][1][1]) < 1e-10 for c in res)
p2_ = all(res[c][1][2] < 2e-13 and res[c][2][2] < 2e-13 for c in res)
p3_ = all(0.5 <= res[c][4] / res[c][3] <= 2 for c in res)
ctl = abs(res[(13, 14)][0][2] / 1.740e-13 - 1) < 0.02 and abs(res[(14, 16)][0][2] / 1.102e-13 - 1) < 0.02
print(
    "P1 half pairs within 1e-10: %s; P2 each half rms below 2e-13: %s; P3 lower-K size within a factor 2: %s; control: %s"
    % (lab(p1_), lab(p2_), lab(p3_), lab(ctl))
)
print("READING q1679 done (labels by the script; no verdict word)")
