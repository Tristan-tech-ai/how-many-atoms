"""q1687 (10 Oct 2026, READING, no verdict word, predictions written at 10:20): full-precision evaluation of the births model. a_pred is evaluated in 40-digit mpmath (as in q1670) and compared with the 18-digit A_event strings of the event files, not with their float64 conversion; the wall pair is refitted on the 55 events K_b 83..138 (four linearisation rounds) in the cells (NG, nmax) = (14, 14), (14, 16) and (14, 18). Only differences are printed (BLIND RULE).
Predictions:
 P1 the in-sample rms of the cell (14, 14) falls below 3e-14 (5.241e-14 with float64 amplitudes in q1684);
 P2 the rms of the cell (14, 18) changes by less than 20 percent of 1.621e-13 (q1684);
 P3 the largest single |a_pred - A| of the cell (14, 14) falls below 1e-13.
Control: with the float64 conversion of the same strings, the cell (14, 14) reproduces 5.241e-14 within 5 percent.
Usage: py q1687_mp_full_precision.py (from small_gap_q206/; about 5 minutes)."""

import json, os, re
import numpy as np
import sympy as sp
import mpmath as mp
import births_law as B

mp.mp.dps = 40
H = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(H, "..", "centre_scripts")
src = open(os.path.join(H, "births_law.py"), encoding="utf-8").read()
ns = {"PI": mp.pi}
for line in src.splitlines():
    if re.match(r"^(k1|k4|c6|c8|c10|f12|f14|f16|f18|f20) = ", line):
        exec(line, ns)
txt = open(os.path.join(H, "q1557_stdout_ORD28.txt"), encoding="utf-8").read()
fx = {}
for n in (22, 24, 26, 28):
    m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, txt, re.M)
    fx[n] = mp.mpf(sp.N(sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi}), 40))
FM = [
    mp.mpf(1),
    ns["k1"],
    ns["k4"],
    ns["c6"],
    ns["c8"],
    ns["c10"],
    ns["f12"],
    ns["f14"],
    ns["f16"],
    ns["f18"],
    ns["f20"],
    fx[22],
    fx[24],
    fx[26],
    fx[28],
]
NG = 14
G = [mp.mpf(1)]
for n in range(1, NG + 1):
    G.append(-sum(FM[j] * G[n - j] for j in range(1, n + 1)))
S20 = json.load(open(os.path.join(C, "d22_out_20.json")))
EM = {n: mp.mpf(S20["e%d" % n]) for n in range(2, 21)}
P16 = 16 * mp.pi**2


def dlaw(r):
    return P16 * (G[0] * r**3 / 3 + sum(G[n] * r ** (3 - 2 * n) / (3 - 2 * n) for n in range(1, NG + 1)))


def Nlaw(r):
    return P16 * (
        G[0] * r**4 / 4
        + G[1] * r**2 / 2
        + G[2] * mp.log(r)
        + sum(G[n] * r ** (4 - 2 * n) / (4 - 2 * n) for n in range(3, NG + 1))
    )


def a_pred_mp(K, nu, d0, nmax, Hs=mp.mpf("1e-3")):
    nu, d0 = mp.mpf(nu), mp.mpf(d0)
    Es = [EM[n] for n in range(2, nmax + 1)]
    rho = lambda a: mp.findroot(
        lambda r: dlaw(r) + d0 - a,
        mp.mpf("1.0") if a < 0 else (a - d0) ** (mp.mpf(1) / 3) / (P16 / 3) ** (mp.mpf(1) / 3),
        tol=mp.mpf(10) ** -35,
    )
    Nn = lambda a: Nlaw(rho(a)) + mp.mpf(1) / 2 - nu

    def dd(r):
        v = 4 * mp.pi * r
        return sum(E / v ** (4 + 2 * i) for i, E in enumerate(Es))

    f = lambda a: 2 * Nn(a) - K - mp.mpf(1) / 6 - dd((Nn(a + Hs) - Nn(a - Hs)) / (2 * Hs))
    return mp.findroot(f, mp.mpf("70") if K < 100 else mp.mpf("76"), tol=mp.mpf(10) ** -33)


Ks = [K for K in range(83, 139) if os.path.exists(os.path.join(H, "fw_prod", "event_Kb%d.json" % K))]
STR = {K: mp.mpf(json.load(open(os.path.join(H, "fw_prod", "event_Kb%d.json" % K)))["A_event"]) for K in Ks}


def fit(nmax, conv):
    meas = [mp.mpf(repr(float(STR[K]))) if conv else STR[K] for K in Ks]
    nu, d0 = mp.mpf("-0.0937736508488"), mp.mpf("0.0011599160189")
    h = mp.mpf("1e-7")
    for rnd in range(4):
        p0 = [a_pred_mp(K, nu, d0, nmax) for K in Ks]
        cn = [(a_pred_mp(K, nu + h, d0, nmax) - a_pred_mp(K, nu - h, d0, nmax)) / (2 * h) for K in Ks]
        cd = [(a_pred_mp(K, nu, d0 + h, nmax) - a_pred_mp(K, nu, d0 - h, nmax)) / (2 * h) for K in Ks]
        M = np.stack([[float(x) for x in cn], [float(x) for x in cd]], 1)
        rhs = -np.array([float(p - m) for p, m in zip(p0, meas)])
        sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        nu, d0 = nu + mp.mpf(float(sol[0])), d0 + mp.mpf(float(sol[1]))
    r = np.array([float(a_pred_mp(K, nu, d0, nmax) - m) for K, m in zip(Ks, meas)])
    return float(nu), float(d0), float(np.sqrt((r**2).mean())), float(np.abs(r).max())


R = {}
for nmax in (14, 16, 18):
    R[nmax] = fit(nmax, False)
    print(
        "cell (14, %d), full-precision strings: pair (%.13f, %.13f), in-sample rms over %d events %.4e, largest |a_pred - A| %.3e"
        % (nmax, *R[nmax][:2], len(Ks), R[nmax][2], R[nmax][3])
    )
conv14 = fit(14, True)
print(
    "cell (14, 14) with the float64 conversion of the same strings: rms %.4e, largest %.3e"
    % (conv14[2], conv14[3])
)
lab = lambda q: "holds" if q else "does not hold"
p1 = R[14][2] < 3e-14
p2 = abs(R[18][2] / 1.621e-13 - 1) < 0.20
p3 = R[14][3] < 1e-13
ctl = abs(conv14[2] / 5.241e-14 - 1) < 0.05
print(
    "P1 rms of (14, 14) below 3e-14: %s; P2 rms of (14, 18) within 20 percent of 1.621e-13: %s; P3 largest |a_pred - A| of (14, 14) below 1e-13: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(ctl))
)
print("READING q1687 done (labels by the script; no verdict word)")
