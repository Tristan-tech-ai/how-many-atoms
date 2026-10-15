"""q1667 (10 Oct 2026, READING, no verdict word): evaluation of the registered predictions F9, F10 and F11 of q1666_prereg_f30.txt (written 09:40:32 by code after a check that no f30 existed) on the ORD = 30 output of q1557_f12_symbolic.py. Run it when q1557_stdout_ORD30.txt contains the line 'f30 = '; before that it prints NOT EVALUATED.
Registered text, copied: F9 f30 is negative and f30/f28 lies between 0.43 and 0.52; F10 with the law through c30 (NG = 15, pair refitted from the c28 refit) the in-sample rms over K_b 83..130 lies between 4e-14 and 1.0e-13 and the frozen out-of-sample rms over K_b 131..135, 137 between 8e-14 and 2.0e-13; F11 the control: the ORD = 30 run returns f12..f28 equal to the stored and ORD = 28 values (relative 1e-11).
Control for the births part: NG = 14 reproduces the in-sample rms 1.518e-13 of q1665 to 3 digits.
Usage: py q1667_f30_check.py (from small_gap_q206/; about 6 minutes)."""

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
    for n in range(12, 32, 2):
        m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, txt, re.M)
        if m:
            out[n] = mp.mpf(
                sp.N(sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi}), 30)
            )
    return out


f30o, f28o = parse("q1557_stdout_ORD30.txt"), parse("q1557_stdout_ORD28.txt")
if 30 not in f30o:
    print("NOT EVALUATED: no line f30 = in q1557_stdout_ORD30.txt")
    raise SystemExit(0)
stored = {
    12: B.F[6],
    14: B.F[7],
    16: B.F[8],
    18: B.F[9],
    20: B.F[10],
    22: -2.420924146814851e-10,
    24: -8.973800969707301e-11,
}
rel = [abs(float(f30o[n]) / stored[n] - 1) for n in stored] + [
    abs(f30o[26] / f28o[26] - 1),
    abs(f30o[28] / f28o[28] - 1),
]
print(
    "f30 = %s; f30/f28 = %s; control relative differences (7 stored values, f26 and f28 against the ORD 28 run): max %.1e over %d values"
    % (mp.nstr(f30o[30], 12), mp.nstr(f30o[30] / f30o[28], 8), float(max(rel)), len(rel))
)
lab = lambda q: "holds" if q else "does not hold"
f9 = f30o[30] < 0 and mp.mpf("0.43") <= f30o[30] / f30o[28] <= mp.mpf("0.52")
FALL = (B.F + [-2.420924146814851e-10, -8.973800969707301e-11])[:13] + [
    float(f30o[26]),
    float(f30o[28]),
    float(f30o[30]),
]
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
meas = np.array([B.measured(K) for K in Ks_in])
h = 1e-7
for NG in (12, 13, 14, 15):
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
    out = [B.a_pred(K, nu, d0, ex) - B.measured(K) for K in Ks_out]
    res[NG] = (nu, d0, rms([B.a_pred(K, nu, d0, ex) - B.measured(K) for K in Ks_in]), rms(out))
    print("NG = %d: pair (%.13f, %.13f), in-sample rms %.3e, frozen out-of-sample rms %.3e" % (NG, *res[NG]))
f10 = 4e-14 <= res[15][2] <= 1.0e-13 and 8e-14 <= res[15][3] <= 2.0e-13
f11 = max(rel) < 1e-11 and abs(res[14][2] - 1.518e-13) < 5e-16
print(
    "rms ratios in sample c24..c30: %s"
    % ", ".join("%.3f" % (res[n + 1][2] / res[n][2]) for n in (12, 13, 14))
)
print(
    "F9 f30 negative and f30/f28 in [0.43, 0.52]: %s; F10 births rms in the ranges: %s; F11 control and births control: %s"
    % (lab(f9), lab(f10), lab(f11))
)
print("READING q1667 done (labels by the script; no verdict word)")
