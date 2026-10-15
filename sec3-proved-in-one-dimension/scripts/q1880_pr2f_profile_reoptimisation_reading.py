r"""q1880_pr2f_profile_reoptimisation_reading.py (14 Oct 2026, task PR2f, claim B3): second float64 search for a dual test profile, after q1879
showed that the q1878 optimum is negative between its grid points (K(-2.71875) = -6.5e-6) and fails lemma T at X0 = 16.
Changes against q1878 (decided from that FAIL, before this run):
 - constraint grid of step 0.002 on [-15, 0] (q1878: 0.05 on [-14, 0]);
 - margin m(x) = 1e-5 for x >= -6 and 1e-5 exp(-(x + 6)^2 / 4) below, so the bound follows the Gaussian decay of K in the bulk;
 - lemma T of q1879 at X0 = 15 as a constraint: log P(-15) - log N(-15) >= 1 (P, N as in q1879, with a2 = max(log Gmax, s_1^2 - log(nu_1/sqrt(2 pi)), 0)).
Start: the q1878 profile for n = 5; profiles written to q1880_profiles.json (key "5"), rounded later to 9 decimals by the certificate.
A reading with no verdict word; nothing is proved here; the certificate is q1879 run on q1880_profiles.json with --x0 15 (task PR2f).
Registered before the first run: P1 the optimiser ends with all grid constraints and the lemma constraint met; P2 the float check of the rounded
profile on a grid of step 0.0005 on [-15, 0] gives min (K - m/2) >= 0; prediction of PR2f: the dual value U is below 3.990.
Usage: py small_gap_q206/q1880_pr2f_profile_reoptimisation_reading.py   (from temporal_within/)"""

import json
import os

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
Z, W = np.polynomial.hermite.hermgauss(120)
UN = np.sqrt(2) * Z
UW = W / np.sqrt(np.pi)
X0 = 15.0


def margin(xs):
    return 1e-5 * np.where(xs >= -6, 1.0, np.exp(-((xs + 6) ** 2) / 4))


def unpack(p, n):
    g = p[0]
    return g, np.concatenate([p[1:n], [-g]]), p[n : 2 * n]


def Kvals(p, n, xs):
    g, s, nu = unpack(p, n)
    pts = xs[:, None] + UN[None, :]
    G = ndtr(-g - pts) + sum(
        nk * np.exp(-((pts - sk) ** 2) / 2) / np.sqrt(2 * np.pi) for sk, nk in zip(s, nu)
    )
    return np.log(np.maximum(G, 1e-300)) @ UW


def lemma(p, n):
    g, s, nu = unpack(p, n)
    nusum = nu.sum()
    gmax = 1 + nusum / np.sqrt(2 * np.pi)
    s1, nu1 = s[0], nu[0]
    nun = nu[-1]
    a2 = max(np.log(gmax), s1 * s1 - np.log(max(nu1, 1e-300) / np.sqrt(2 * np.pi)), 0.0)
    T = -g - 1 / nun
    Tp = -g - 2 / nun
    x = -X0
    D = T - x
    if D <= 0:
        return -50.0
    logP = (
        np.log(nun / (2 * gmax))
        - (x + g) ** 2 / 4
        - np.log(2 * np.sqrt(np.pi))
        + np.log(max(ndtr(np.sqrt(2) * (Tp - (x - g) / 2)), 1e-300))
    )
    logN = -D * D / 2 - 0.5 * np.log(2 * np.pi) + np.log((a2 + T * T) / D + 2 / D**3)
    return float(logP - logN)


def U(p, n):
    g, s, nu = unpack(p, n)
    return 2 * (nu.sum() - g)


n = 5
d = json.load(open(os.path.join(HERE, "q1878_profiles.json")))[str(n)]
p0 = np.array([d["g"]] + d["s"][:-1] + d["nu"])
XS = np.arange(-X0, 0.0001, 0.002)
M = margin(XS)
print(
    "start: U = %.6f; min(K - m) on the dense grid = %.3e; lemma(-15) = %.2f"
    % (U(p0, n), (Kvals(p0, n, XS) - M).min(), lemma(p0, n))
)
cons = [
    {"type": "ineq", "fun": lambda q: Kvals(q, n, XS) - M},
    {"type": "ineq", "fun": lambda q: lemma(q, n) - 1.0},
]
bounds = [(2.0, 9.0)] + [(-9.0, 0.5)] * (n - 1) + [(0.0, 10.0)] * n
r = minimize(
    U, p0, args=(n,), method="SLSQP", constraints=cons, bounds=bounds, options={"maxiter": 300, "ftol": 1e-12}
)
g, s, nu = unpack(r.x, n)
kmin = (Kvals(r.x, n, XS) - M).min()
lem = lemma(r.x, n)
rd = np.array([float("%.9f" % v) for v in r.x])
XF = np.arange(-X0, 0.0001, 0.0005)
fine = (Kvals(rd, n, XF) - margin(XF) / 2).min()
print(
    "optimised: U = %.6f; success %s (%s); min(K - m) = %.3e; lemma(-15) = %.2f"
    % (U(r.x, n), r.success, r.message, kmin, lem)
)
print("profile: g = %.9f; s = %s; nu = %s" % (g, np.round(s, 9).tolist(), np.round(nu, 9).tolist()))
print("float check of the 9-decimal profile on step 0.0005: min(K - m/2) = %.3e" % fine)
json.dump(
    {
        str(n): {
            "U": float(U(r.x, n)),
            "g": float(g),
            "s": [float(x) for x in s],
            "nu": [float(x) for x in nu],
            "lemma15": lem,
            "min_K_minus_margin": float(kmin),
            "fine_check": float(fine),
        }
    },
    open(os.path.join(HERE, "q1880_profiles.json"), "w"),
    indent=1,
)
p1 = bool(kmin >= -1e-12 and lem >= 1.0 - 1e-9)
p2 = bool(fine >= 0)
print(
    "P1 %s; P2 %s; prediction U < 3.990: %s"
    % (
        "held" if p1 else "did not hold",
        "held" if p2 else "did not hold",
        "held" if U(r.x, n) < 3.990 else "did not hold",
    )
)
print(
    "READING (no verdict): dense-grid float64 optimisation of the dual test profile; nothing is proved here; profile in q1880_profiles.json."
)
