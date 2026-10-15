"""tight_rho_patch (8 Oct 2026, amendment 2116): REPAIR of the density inversion used by every prediction a_med. rho_of(d, d0) of q1183_births_residual_halfline_depth_reading.py solves rho^3 - 3 rho / (8 pi^2) - 1 / (32 pi^2 rho) = 3 s0 (d + d0) with scipy brentq at its default xtol = 2e-12, which leaves errors in rho of up to 5e-13 (N error 4e-11, a error 5e-11, erratic in d). Usage: import the module chain that loads q1183 (for example q1321_hardwall_common_c1_blind, whose QQ.C.Q is q1183), then call apply(QQ.C.Q) BEFORE building any counting function N (the residual spline of a counting function built earlier keeps the old values). apply replaces q1183.rho_of by a brentq solve with xtol = 1e-16 followed by three Newton steps; the module-level N_of of q1183 looks rho_of up in the module globals at call time, so NOF = QQ.C.Q.N_of of q1304 uses it."""

import numpy as np
from scipy.optimize import brentq

pi = np.pi
s0 = 1 / (16 * pi**2)


def rho_tight(d, d0):
    f = lambda r: r**3 - 3 * r / (8 * pi**2) - 1 / (32 * pi**2 * r) - 3 * s0 * (d + d0)
    r = brentq(f, 0.02, 400, xtol=1e-16)
    for _ in range(3):
        r -= f(r) / (3 * r**2 - 3 / (8 * pi**2) + 1 / (32 * pi**2 * r**2))
    return float(r)


def apply(Q):
    Q.rho_of = rho_tight
