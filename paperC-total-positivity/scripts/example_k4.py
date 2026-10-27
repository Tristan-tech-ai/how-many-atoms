"""Worked example of Section V of the paper: the capacity-achieving input with four mass points at one amplitude, in double
precision (checked numerically, without error bounds). Points +-u and +-A with masses w and w_e = (1 - 2w)/2. Solves the Kuhn-Tucker
system D'(u) = 0, D(u) = D(A), checks D <= D(A) on a grid of [0, A], computes the 2 x 2 matrix E'NE' of Theorem 1, the Gram part
S_G and the diagonal term D''(u), the reduced Jacobian S and the forcing b from finite-difference derivatives of the system, the Gram
part b_G of the forcing, and du/dA = -b/S, compared with a finite difference of the solution.
Usage: python example_k4.py [A]   (A = 3.5 by default); writes example_k4.json next to itself."""
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import fsolve

A0 = float(sys.argv[1]) if len(sys.argv) > 1 else 3.5
y = np.linspace(-A0 - 12.0, A0 + 12.0, 24001)        # one fixed grid for every amplitude used below
dy = y[1] - y[0]


def phi(t):
    return np.exp(-t * t / 2) / math.sqrt(2 * math.pi)


def integ(f):
    return float(np.sum(f) * dy)


def Q_of(u, w, A):
    we = (1 - 2 * w) / 2
    return w * (phi(y - u) + phi(y + u)) + we * (phi(y - A) + phi(y + A))


def D_of(x, Q):
    return integ(phi(y - x) * (np.log(phi(y - x)) - np.log(Q)))


def Dp_of(x, Q):                                      # D'(x) = -int phi(y - x) Q'/Q
    return -integ(phi(y - x) * np.gradient(Q, dy) / Q)


def F(u, w, A):
    Q = Q_of(u, w, A)
    return np.array([Dp_of(u, Q), D_of(u, Q) - D_of(A, Q)])


def solve(A, z0):
    z, info, ier, msg = fsolve(lambda z: F(z[0], z[1], A), z0, full_output=True, xtol=1e-13)
    return z, ier


z, ier = solve(A0, [A0 / 3, 0.15])
u, w = z
we = (1 - 2 * w) / 2
Q = Q_of(u, w, A0)
Cap = D_of(A0, Q)
dmax = max(D_of(x, Q) - Cap for x in np.linspace(0, A0, 801))

phu = phi(y - u) + phi(y + u)
phA = phi(y - A0) + phi(y + A0)
dphi = lambda c: (y - c) * phi(y - c) - (y + c) * phi(y + c)   # d/dc [phi(y - c) + phi(y + c)]
psi_u, psi_A, chi = w * dphi(u), we * dphi(A0), phu - phA
ip = lambda f, g: integ(f * g / Q)
G = np.array([[ip(psi_u, psi_u), ip(psi_u, psi_A)], [ip(psi_A, psi_u), ip(psi_A, psi_A)]])
gc = np.array([ip(psi_u, chi), ip(psi_A, chi)])
N = G - np.outer(gc, gc) / ip(chi, chi)
M = np.diag([-1.0, 1.0]) @ N @ np.diag([-1.0, 1.0])
c = -1 / (2 * w)
SG, bG = c * N[0, 0], c * N[0, 1]
po = lambda cc: (y - cc) * phi(y - cc) + (y + cc) * phi(y + cc)
Dpp = 0.5 * (w * ip(po(u), po(u)) + we * ip(po(u), po(A0)))
Dpp_fd = (Dp_of(u + 1e-4, Q) - Dp_of(u - 1e-4, Q)) / 2e-4

# Jacobian and forcing by central differences, then elimination of the mass (Schur complement)
e = 1e-5
Ju = (F(u + e, w, A0) - F(u - e, w, A0)) / (2 * e)
Jw = (F(u, w + e, A0) - F(u, w - e, A0)) / (2 * e)
FA = (F(u, w, A0 + e) - F(u, w, A0 - e)) / (2 * e)
S = Ju[0] - Jw[0] * Ju[1] / Jw[1]
b = FA[0] - Jw[0] * FA[1] / Jw[1]
Dp_A = Dp_of(A0, Q)
b_wall = -Jw[0] / Jw[1] * (-Dp_A)                    # the part of b that comes from the moving evaluation point A in row D(u) - D(A)
h = 1e-3
zp, _ = solve(A0 + h, z)
zm, _ = solve(A0 - h, z)
out = {"A": A0, "fsolve_ier": int(ier), "u": u, "w": w, "w_e": we, "C_nats": Cap, "max_D_minus_C_on_grid": dmax,
       "N": N.tolist(), "EpNEp": M.tolist(), "det_EpNEp": float(np.linalg.det(M)), "c": c,
       "S_G": SG, "Dpp_gram": Dpp, "Dpp_fd": Dpp_fd, "S_gram_plus_diag": SG + Dpp, "S_schur_fd": S,
       "b_G": bG, "b_wall": b_wall, "b_G_plus_wall": bG + b_wall, "b_schur_fd": b, "Dprime_A": Dp_A,
       "dudA_formula": -b / S, "dudA_fd": (zp[0] - zm[0]) / (2 * h), "grid_points": len(y), "grid_step": dy}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_k4.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
