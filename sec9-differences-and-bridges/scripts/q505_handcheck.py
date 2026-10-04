"""Q505 hand check (verification gate): two change points recomputed by a DIFFERENT method. Symmetric reduced families, adaptive
quadrature (scipy.integrate.quad), root finding by brentq. Source U[-1, 1], kernel exp(-s (x - y)^2), L = sqrt(2 s).
 2 -> 3 (birth): the 2-atom family {-y, y}, weights 1/2 (c(+-y) = 1 by symmetry and normalisation), y from c'(y) = 0; birth when
 c(0) = 1.
 3 -> 4 (split): the 3-atom family {-y, 0, y}, weights (w, 1 - 2w, w), from c(y) = 1 and c'(y) = 0; split when c''(0) = 0.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, fsolve


def Z(x, s, ys, ws):
    return sum(w * np.exp(-s * (x - y) ** 2) for y, w in zip(ys, ws))


def c(t, s, ys, ws, der=0):
    def f(x):
        e = np.exp(-s * (x - t) ** 2) / Z(x, s, ys, ws) * 0.5
        d = x - t
        return e if der == 0 else (e * 2 * s * d if der == 1 else e * (4 * s * s * d * d - 2 * s))

    return quad(f, -1, 1, epsabs=1e-14, epsrel=1e-13, limit=400)[0]


def fam2(s):
    y = brentq(lambda y: c(y, s, [-y, y], [0.5, 0.5], 1), 0.05, 0.99)
    return y


def birth_signal(L):
    s = L * L / 2
    y = fam2(s)
    return c(0.0, s, [-y, y], [0.5, 0.5]) - 1


L23 = brentq(birth_signal, 2.8, 3.1, xtol=1e-12)
print(f"2 -> 3 birth: L = {L23:.9f}   (q505 2.9799389, NPMLE record 2.979950)")


def fam3(s, guess):
    def F(v):
        y, w = v
        ys, ws = [-y, 0.0, y], [w, 1 - 2 * w, w]
        return [c(y, s, ys, ws) - 1, c(y, s, ys, ws, 1)]

    return fsolve(F, guess, xtol=1e-13)


state = [0.6, 0.3]


def split_signal(L):
    global state
    s = L * L / 2
    y, w = fam3(s, state)
    state = [y, w]
    return c(0.0, s, [-y, 0.0, y], [w, 1 - 2 * w, w], 2)


L34 = brentq(split_signal, 3.9, 4.3, xtol=1e-12)
print(f"3 -> 4 split: L = {L34:.9f}   (q505 4.0999814, NPMLE record 4.100006)")
