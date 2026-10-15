"""q1670 (10 Oct 2026, READING, no verdict word, predictions written at 09:49): is the 1e-13 floor of the births model an instrument floor? a_pred is evaluated in 40-digit mpmath (closed forms of the bulk coefficients up to f20 evaluated with mp.pi, f22..f28 from the ORD = 28 output, e_2..e_20 from d22_out_20.json as strings, root finding by mp.findroot to 1e-30, the same Hs = 1e-3 central difference as births_law.py), and the series-order ladder at the bulk law c28 of q1669 is repeated for nmax = 14, 18 and 20.
Predictions:
 P1 the in-sample rms at nmax = 18 changes by less than 10 percent when a_pred is evaluated in mpmath (1.520e-13 in q1669);
 P2 the rms sequence for nmax = 14, 18, 20 stays non-monotone: 6.0e-14, 1.5e-13, 1.6e-13 each within 10 percent;
 P3 the pair refitted in mpmath differs from the float64 pair of q1669 by less than 5e-12 in each parameter.
Control: the mp a_pred equals births_law.a_pred (float64) at K_b 83, 106 and 130 for the float64 inputs of q1669 (nmax = 18 pair) to within 5e-14.
Usage: py q1670_births_mp.py (from small_gap_q206/; a few minutes)."""

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


# control against the float64 module
B.F = [float(x) for x in FM]
B.NG = NG
B.g = [float(x) for x in G]
S18 = json.load(open(os.path.join(C, "d22_out_18.json")))
E64 = {n: float(S18["e%d" % n]) for n in range(2, 19)}
NU, D0 = -0.0937736508347, 0.0011599160063
ctl_d = []
for K in (83, 106, 130):
    am = a_pred_mp(K, NU, D0, 18)
    af = B.a_pred(K, NU, D0, [E64[n] for n in range(6, 19)])
    ctl_d.append(float(am - af))
print("control: a_pred(mp) - a_pred(float64) at K_b 83, 106, 130: %s" % ", ".join("%+.2e" % x for x in ctl_d))
Ks = [K for K in range(83, 131) if B.measured(K) is not None]
meas = [mp.mpf(repr(B.measured(K))) for K in Ks]
h = mp.mpf("1e-7")
res = {}
for nmax in (14, 18, 20):
    nu, d0 = mp.mpf("-0.0937736508344"), mp.mpf("0.0011599160061")
    for rnd in range(2):
        p0 = [a_pred_mp(K, nu, d0, nmax) for K in Ks]
        cn = [(a_pred_mp(K, nu + h, d0, nmax) - a_pred_mp(K, nu - h, d0, nmax)) / (2 * h) for K in Ks]
        cd = [(a_pred_mp(K, nu, d0 + h, nmax) - a_pred_mp(K, nu, d0 - h, nmax)) / (2 * h) for K in Ks]
        M = np.stack([[float(x) for x in cn], [float(x) for x in cd]], 1)
        rhs = -np.array([float(p - m) for p, m in zip(p0, meas)])
        sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        nu, d0 = nu + mp.mpf(float(sol[0])), d0 + mp.mpf(float(sol[1]))
    r = np.array([float(a_pred_mp(K, nu, d0, nmax) - m) for K, m in zip(Ks, meas)])
    res[nmax] = (float(nu), float(d0), float(np.sqrt((r**2).mean())))
    print("nmax = %2d (mp): pair (%.13f, %.13f), in-sample rms %.4e" % (nmax, *res[nmax]))
lab = lambda q: "holds" if q else "does not hold"
p1 = abs(res[18][2] / 1.520e-13 - 1) < 0.10
p2 = (
    all(abs(res[n][2] / v - 1) < 0.10 for n, v in ((14, 6.0e-14), (18, 1.5e-13), (20, 1.6e-13)))
    and res[14][2] < res[18][2] < res[20][2]
)
p3 = abs(res[18][0] - (-0.0937736508347)) < 5e-12 and abs(res[18][1] - 0.0011599160063) < 5e-12
ctl = max(abs(x) for x in ctl_d) < 5e-14
print(
    "P1 rms at nmax 18 within 10 percent of the float64 value: %s; P2 sequence non-monotone as in float64 within 10 percent: %s; P3 mp pair within 5e-12 of the float64 pair: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(ctl))
)
print("READING q1670 done (labels by the script; no verdict word)")
