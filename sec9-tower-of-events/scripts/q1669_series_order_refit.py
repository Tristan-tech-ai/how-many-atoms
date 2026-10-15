"""q1669 (10 Oct 2026, READING, no verdict word, predictions written at 09:44): the wall pair refitted to the 48 births K_b 83..130 with the bulk law through c28 (NG = 14) for the stored series truncated at nmax = 10, 12, 14, 18 and 20 (e_19 and e_20 from d22_out_20.json after a check that its e_2..e_18 equal those of d22_out_18.json).
Predictions:
 P1 the refitted pair at nmax = 14 differs from the one at nmax = 18 by less than 2e-11 in each parameter;
 P2 the in-sample rms at nmax = 14 is within 3 percent of the one at nmax = 18 (1.518e-13 in q1665);
 P3 adding e_19 and e_20 changes the in-sample rms by less than 1 percent.
Control: nmax = 18 reproduces the in-sample rms 1.518e-13 of q1665 to 3 digits, and the e_2..e_18 of the two files are equal to a relative 1e-30.
Usage: py q1669_series_order_refit.py (from small_gap_q206/; about 4 minutes)."""

import json, os, re
import numpy as np
import sympy as sp
import mpmath as mp
import births_law as B

mp.mp.dps = 60
H = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(H, "..", "centre_scripts")
S18 = json.load(open(os.path.join(C, "d22_out_18.json")))
S20 = json.load(open(os.path.join(C, "d22_out_20.json")))
same = max(abs(mp.mpf(S18["e%d" % n]) / mp.mpf(S20["e%d" % n]) - 1) for n in range(2, 19))
E = {n: float(S20["e%d" % n]) for n in range(2, 21)}


def parse(fn):
    txt = open(os.path.join(H, fn), encoding="utf-8").read()
    out = {}
    for n in range(26, 30, 2):
        m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, txt, re.M)
        if m:
            out[n] = float(
                mp.mpf(sp.N(sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi}), 30))
            )
    return out


f = parse("q1557_stdout_ORD28.txt")
FALL = (B.F + [-2.420924146814851e-10, -8.973800969707301e-11])[:13] + [f[26], f[28]]
B.F = FALL[:15]
B.NG = 14
B.g = [1.0]
for n in range(1, 15):
    B.g.append(-sum(B.F[j] * B.g[n - j] for j in range(1, n + 1)))
NU, D0 = -0.0937736508344, 0.0011599160061
Ks = [K for K in range(83, 131) if B.measured(K) is not None]
meas = np.array([B.measured(K) for K in Ks])
h = 1e-7
res = {}
for nmax in (10, 12, 14, 18, 20):
    ex = [E[n] for n in range(6, nmax + 1)]
    nu, d0 = NU, D0
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
    r = float(np.sqrt(((np.array([B.a_pred(K, nu, d0, ex) for K in Ks]) - meas) ** 2).mean()))
    res[nmax] = (nu, d0, r)
    print("nmax = %2d: pair (%.13f, %.13f), in-sample rms %.4e" % (nmax, nu, d0, r))
lab = lambda q: "holds" if q else "does not hold"
p1 = abs(res[14][0] - res[18][0]) < 2e-11 and abs(res[14][1] - res[18][1]) < 2e-11
p2 = abs(res[14][2] / res[18][2] - 1) < 0.03
p3 = abs(res[20][2] / res[18][2] - 1) < 0.01
ctl = abs(res[18][2] - 1.518e-13) < 5e-16 and same < 1e-30
print(
    "pair shifts nmax 14 to 18: (%+.3e, %+.3e); rms ratio 14/18 = %.4f; rms ratio 20/18 = %.5f; files agree to %.1e"
    % (
        res[18][0] - res[14][0],
        res[18][1] - res[14][1],
        res[14][2] / res[18][2],
        res[20][2] / res[18][2],
        float(same),
    )
)
print(
    "P1 pair within 2e-11: %s; P2 rms within 3 percent: %s; P3 e_19 and e_20 change the rms by less than 1 percent: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(ctl))
)
print("READING q1669 done (labels by the script; no verdict word)")
