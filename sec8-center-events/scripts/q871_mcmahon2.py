"""q871: second-order McMahon model of centre-event shifts. With the event condition k R_d = j_{nu,s} (zeros of the Bessel core;
births nu = d/2 - 1, lift-offs nu = d/2; 1D: nu = -1/2, +1/2 with mu = 4 nu^2 = 1), McMahon gives
   shift_d = (d - 1) pi/(4k) - (mu - 1) w/k - (4/3)(mu - 1)(7 mu - 31) w^3/k,   w = 1/(8 k R).
Per event (k, w) are fixed by the 2D and 3D shifts (two equations); d = 4, 5, 8 (and any d) are then predicted with no parameter.
First order only (w^3 term dropped) is the quadratic form of 1579-1581. Functions only; used by the scoring scripts.
"""

import numpy as np
from scipy.optimize import fsolve


def mu(d, kind):
    nu = d / 2 - 1 if kind == "birth" else d / 2
    return 4 * nu * nu


def shift(d, kind, k, w, order=2):
    m = mu(d, kind)
    s = (d - 1) * np.pi / (4 * k) - (m - 1) * w / k
    if order >= 2:
        s -= (4 / 3) * (m - 1) * (7 * m - 31) * w**3 / k
    return s


def fit(s2, s3, kind, order=2):
    k0 = np.pi / (2 * s3) if s3 > 0 else 1.0
    w0 = 0.0

    def eqs(x):
        k, w = x
        return [shift(2, kind, k, w, order) - s2, shift(3, kind, k, w, order) - s3]

    sol, info, ier, msg = fsolve(eqs, [k0, w0], full_output=True, xtol=1e-14)
    return sol, ier == 1, float(np.max(np.abs(info["fvec"])))
