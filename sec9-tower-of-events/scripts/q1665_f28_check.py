"""q1665 (10 Oct 2026, READING, no verdict word, second half): evaluation of F2 and F4 of q1643_prereg_f26_f28.txt (written 09:08:25, before the runs) and of F7 and F8 of its addendum (written 09:38:42, 17 seconds AFTER the line 'f28 = ' was printed in q1557_stdout_ORD28.txt at 09:38:25, and before the value of f28 was read; see the correction in that file) on the ORD = 28 output. The value of f28 itself was not read when this file was written.
Predictions (copied):
 F2 f28 is negative and f28/f26 lies between 0.39 and 0.47;
 F4 with the law through c28 and the refitted wall pair, the births differences move by less than 5e-11 relative to the c24 values (largest over K_b 83..130);
 F7 with the law through c28 (NG = 14, pair refitted from the c26 refit) the in-sample rms lies between 1.2e-13 and 2.4e-13 and the frozen out-of-sample rms over K_b 131..135, 137 between 2.4e-13 and 4.8e-13.
Control: f12..f26 of the ORD = 28 output equal the stored values and the ORD = 26 output (relative 1e-11), and NG = 13 reproduces the in-sample rms 3.743e-13 of q1644 to 3 digits.
Usage: py q1665_f28_check.py (from small_gap_q206/; about 5 minutes)."""

import json, os, re
import numpy as np
import sympy as sp
import mpmath as mp
import births_law as B

mp.mp.dps = 30
H = os.path.dirname(os.path.abspath(__file__))


def parse(fn):
    txt = open(os.path.join(H, fn), encoding="utf-8").read()
    out = {}
    for n in range(12, 30, 2):
        m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, txt, re.M)
        if m:
            out[n] = mp.mpf(
                sp.N(sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi}), 30)
            )
    return out


f28o, f26o = parse("q1557_stdout_ORD28.txt"), parse("q1557_stdout_ORD26.txt")
stored = {
    12: B.F[6],
    14: B.F[7],
    16: B.F[8],
    18: B.F[9],
    20: B.F[10],
    22: -2.420924146814851e-10,
    24: -8.973800969707301e-11,
}
rel = [abs(float(f28o[n]) / stored[n] - 1) for n in stored if n in f28o] + [abs(f28o[26] / f26o[26] - 1)]
print(
    "f28 = %s; f28/f26 = %s; control relative differences (7 stored plus f26 against the ORD 26 run): max %.1e over %d values"
    % (mp.nstr(f28o[28], 12), mp.nstr(f28o[28] / f28o[26], 8), float(max(rel)), len(rel))
)
lab = lambda q: "holds" if q else "does not hold"
f2 = f28o[28] < 0 and mp.mpf("0.39") <= f28o[28] / f28o[26] <= mp.mpf("0.47")
FALL = (B.F + [-2.420924146814851e-10, -8.973800969707301e-11])[:13] + [float(f28o[26]), float(f28o[28])]
S = json.load(open(os.path.join(H, "..", "centre_scripts", "d22_out_18.json")))
ex = [float(S["e%d" % n]) for n in range(6, 19)]
NU, D0 = -0.0937736507540, 0.0011599159348
Ks_in = [K for K in range(83, 131) if B.measured(K) is not None]
Ks_out = [K for K in (131, 132, 133, 134, 135, 137) if B.measured(K) is not None]


def setlaw(NG):
    B.F = FALL[: NG + 1]
    B.NG = NG
    B.g = [1.0]
    for n in range(1, NG + 1):
        B.g.append(-sum(B.F[j] * B.g[n - j] for j in range(1, n + 1)))


rms = lambda x: float(np.sqrt((np.asarray(x) ** 2).mean()))
res = {}
diffs = {}
meas = np.array([B.measured(K) for K in Ks_in])
h = 1e-7
for NG in (12, 13, 14):
    setlaw(NG)
    nu, d0 = NU, D0
    for rnd in range(2):
        p0 = np.array([B.a_pred(K, nu, d0, ex) for K in Ks_in])
        M = np.stack(
            [
                [(B.a_pred(K, nu + h, d0, ex) - B.a_pred(K, nu - h, d0, ex)) / (2 * h) for K in Ks_in],
                [(B.a_pred(K, nu, d0 + h, ex) - B.a_pred(K, nu, d0 - h, ex)) / (2 * h) for K in Ks_in],
            ],
            1,
        )
        sol, *_ = np.linalg.lstsq(M, -(p0 - meas), rcond=None)
        nu, d0 = nu + sol[0], d0 + sol[1]
    diffs[NG] = np.array([B.a_pred(K, nu, d0, ex) - B.measured(K) for K in Ks_in])
    preds = np.array([B.a_pred(K, nu, d0, ex) for K in Ks_in])
    out = [B.a_pred(K, nu, d0, ex) - B.measured(K) for K in Ks_out]
    res[NG] = (nu, d0, rms(diffs[NG]), rms(out), preds)
    print(
        "NG = %d: pair (%.13f, %.13f), in-sample rms %.3e, frozen out-of-sample rms %.3e" % (NG, *res[NG][:4])
    )
move = float(np.abs(res[14][4] - res[12][4]).max())
f4 = move < 5e-11
f7 = 1.2e-13 <= res[14][2] <= 2.4e-13 and 2.4e-13 <= res[14][3] <= 4.8e-13
ctl = max(rel) < 1e-11 and abs(res[13][2] - 3.743e-13) < 5e-16
print("largest move of a prediction between NG = 12 and 14: %.3e" % move)
print(
    "F2 f28 negative and f28/f26 in [0.39, 0.47]: %s; F4 predictions move by less than 5e-11: %s; F7 births rms in the ranges: %s; control: %s"
    % (lab(f2), lab(f4), lab(f7), lab(ctl))
)
print("READING q1665 done (labels by the script; no verdict word)")
