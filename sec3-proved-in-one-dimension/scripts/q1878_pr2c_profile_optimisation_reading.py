r"""q1878_pr2c_profile_optimisation_reading.py (14 Oct 2026, task PR2c, claim B3): float64 search for a lower dual upper bound on beta.
For a test profile G(t) = Phi(-g - t) + sum_k nu_k phi(t - s_k) with s_n = -g (the innermost atom at the end of the Lebesgue part, as lemma T of q1876
needs) and K(x) = int phi(u) log G(x + u) du >= 0 for all x <= 0, the dual bound gives limsup (e^(C + h_1) - 2A) <= U(G) = 2(sum nu - g)
(q1876, q1877). This script minimises U over g, s_1..s_{n-1}, nu_1..nu_n by SLSQP, with K >= margin on the grid x in [-14, 0] step 0.05, K by
120-point Gauss-Hermite quadrature, starting from the certified 3-atom profile (U = 4.01569). It writes the optimised profiles to
q1878_profiles.json for a later ball-arithmetic certificate (the q1876 instrument). A reading with no verdict word; nothing here is proved.
Registered before the first run (prediction of task PR2c): with n = 5 atoms the minimised U is below 3.990.
Further registered readouts: P1 the start profile is feasible on the grid (min K >= 0); P2 for n = 3, 4, 5 the optimiser ends with min K >= margin
on the grid; P3 U(n = 5) < 3.990 (the prediction); P4 U decreases from n = 3 to n = 5.
Usage: py small_gap_q206/q1878_pr2c_profile_optimisation_reading.py   (from temporal_within/)"""

import json
import os

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
Z, W = np.polynomial.hermite.hermgauss(120)
U_NODES = np.sqrt(2) * Z
U_W = W / np.sqrt(np.pi)
XS = np.arange(-14.0, 0.0001, 0.05)
MARGIN = 1e-7


def phi(t):
    return np.exp(-t * t / 2) / np.sqrt(2 * np.pi)


def unpack(p, n):
    g = p[0]
    s = np.concatenate([p[1:n], [-g]])
    nu = p[n : 2 * n]
    return g, s, nu


def G(t, g, s, nu):
    return ndtr(-g - t) + sum(nk * phi(t - sk) for sk, nk in zip(s, nu))


def Kvals(p, n):
    g, s, nu = unpack(p, n)
    pts = XS[:, None] + U_NODES[None, :]
    v = G(pts, g, s, nu)
    return np.log(np.maximum(v, 1e-300)) @ U_W


def U(p, n):
    g, s, nu = unpack(p, n)
    return 2 * (nu.sum() - g)


def start(n):
    g = 4.361045
    s = [0.0, -2.68983] + list(np.linspace(-3.5, -4.2, max(0, n - 3)))
    nu = [3.49413, 2.25076] + [0.05] * max(0, n - 3) + [0.62400]
    return np.array([g] + s[: n - 1] + nu[:n])


res = {}
p0 = start(3)
p1 = bool(Kvals(p0, 3).min() >= 0)
print("start (n = 3): U = %.6f, min K on grid = %.3e" % (U(p0, 3), Kvals(p0, 3).min()))
for n in (3, 4, 5):
    p = start(n)
    cons = [{"type": "ineq", "fun": (lambda q, n=n: Kvals(q, n) - MARGIN)}]
    bounds = [(2.0, 8.0)] + [(-8.0, 0.5)] * (n - 1) + [(0.0, 10.0)] * n
    r = minimize(
        U,
        p,
        args=(n,),
        method="SLSQP",
        constraints=cons,
        bounds=bounds,
        options={"maxiter": 400, "ftol": 1e-12},
    )
    kmin = Kvals(r.x, n).min()
    g, s, nu = unpack(r.x, n)
    res[n] = {
        "U": float(U(r.x, n)),
        "minK": float(kmin),
        "success": bool(r.success),
        "g": float(g),
        "s": [float(x) for x in s],
        "nu": [float(x) for x in nu],
        "message": str(r.message),
    }
    print(
        "n = %d: U = %.6f, min K = %.3e, success %s; g = %.6f, s = %s, nu = %s"
        % (n, res[n]["U"], kmin, r.success, g, np.round(s, 5).tolist(), np.round(nu, 5).tolist())
    )
json.dump(res, open(os.path.join(HERE, "q1878_profiles.json"), "w"), indent=1)
p2 = all(res[n]["minK"] >= MARGIN * 0.5 for n in res)
p3 = res[5]["U"] < 3.990
p4 = res[3]["U"] >= res[4]["U"] - 1e-9 >= res[5]["U"] - 2e-9
print("P1 %s; P2 %s; P3 %s; P4 %s" % tuple("held" if x else "did not hold" for x in (p1, p2, p3, p4)))
print(
    "READING (no verdict): float64 optimisation of the dual test profile; nothing is proved here; profiles in q1878_profiles.json."
)
