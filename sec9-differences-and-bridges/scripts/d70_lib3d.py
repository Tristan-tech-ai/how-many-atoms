"""d70_lib3d (copy of d50_lib.py as text, kernel changed; Amendment 976): radial KKT solver for the 3D amplitude-constrained Gaussian
channel (input in the ball |x| <= A, noise N(0, I_3)).

Definitions only; nothing is computed or written at import time.

Radial problem. For |x| = r the output radius s = |Y| has density
    f(s|r) = (s/r) [phi(s - r) - phi(s + r)] = s E(s, r),  E = exp(-(s - r)^2/2) (1 - exp(-2 s r)) / (r sqrt(2 pi)),  E(s, 0) = 2 s phi(s).
With shells (radius r_j, mass w_j, uniform on the sphere) the 3D output density at |y| = s is p(s) = ptil(s)/(4 pi s), ptil = sum w_j E_j.
KKT function: i(r) = -int_0^inf f(s|r) log p(s) ds - (3/2) log(2 pi e). Support types as in d50_lib ('o' origin, 'r' shell, 'w' wall).
"""

import numpy as np

LOG2PIE = 1.5 * np.log(2 * np.pi * np.e)
SQ2PI = np.sqrt(2 * np.pi)


def quad_nodes(smax, h=0.5, n=24):
    """Gauss-Legendre panels of width about h on [0, smax]."""
    npan = int(np.ceil(smax / h))
    edges = np.linspace(0.0, smax, npan + 1)
    x, w = np.polynomial.legendre.leggauss(n)
    S, W = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        S.append(0.5 * (b - a) * x + 0.5 * (b + a))
        W.append(0.5 * (b - a) * w)
    return np.concatenate(S), np.concatenate(W)


def ratio_R(x):
    """R(x) = I1(x)/I0(x) and R'(x) = 1 - R/x - R^2, stable at small x."""
    x = np.asarray(x, dtype=float)
    R = i1e(x) / i0e(x)
    small = x < 1e-4
    Rox = np.where(small, 0.5 - x * x / 16.0, R / np.where(small, 1.0, x))
    Rp = 1.0 - Rox - R * R
    return R, Rp


class Grid:
    def __init__(self, A, pad=14.0, h=0.5, n=24):
        self.A = A
        self.s, self.wq = quad_nodes(A + pad, h, n)

    def E(self, r):
        """E[j, i] = exp(-(s_i - r_j)^2/2) (1 - exp(-2 s_i r_j)) / (r_j sqrt(2 pi)); f(s|r) = s E; E(s, 0) = 2 s phi(s)."""
        r = np.atleast_1d(np.asarray(r, dtype=float))
        s = self.s[None, :]
        rr = r[:, None]
        small = rr < 1e-8
        fac = np.where(small, 2 * s, -np.expm1(-2 * s * rr) / np.where(small, 1.0, rr))
        return np.exp(-0.5 * (s - rr) ** 2) * fac / SQ2PI

    def G(self, r):
        """g(s, r) = d log f / dr = s L(s r) - r, L(x) = coth x - 1/x (Langevin), and its r-derivative s^2 L'(s r) - 1."""
        r = np.atleast_1d(np.asarray(r, dtype=float))
        s = self.s[None, :]
        x = s * r[:, None]
        small = x < 1e-3
        xs = np.where(small, 1.0, x)
        Lx = np.where(small, x / 3 - x**3 / 45, 1 / np.tanh(xs) - 1 / xs)
        Lp = np.where(small, 1.0 / 3 - x**2 / 15, 1 / xs**2 - 1 / np.sinh(np.minimum(xs, 700)) ** 2)
        return -r[:, None] + s * Lx, -1.0 + s * s * Lp

    def ptil(self, r, w):
        return np.asarray(w) @ self.E(r)

    def kkt(self, rq, r, w, deriv=0):
        """i(rq) (and i', i'' if deriv >= 1, 2) for the input (r, w)."""
        pt = self.ptil(r, w)
        L = np.log(pt) - np.log(4 * np.pi * self.s)
        rq = np.atleast_1d(np.asarray(rq, dtype=float))
        out = []
        CH = 400
        vals = [np.empty(len(rq)) for _ in range(deriv + 1)]
        for a in range(0, len(rq), CH):
            rr = rq[a : a + CH]
            F = self.s[None, :] * self.E(rr) * self.wq[None, :]
            vals[0][a : a + CH] = -(F @ L) - LOG2PIE
            if deriv >= 1:
                g, gp = self.G(rr)
                vals[1][a : a + CH] = -((F * g) @ L)
                if deriv >= 2:
                    vals[2][a : a + CH] = -((F * (g * g + gp)) @ L)
        return vals if deriv else vals[0]

    def mutual_info(self, r, w):
        i = self.kkt(r, r, w)
        return float(np.dot(w, i))


def unpack(x, types, A):
    """Unknowns: interior radii (in order of 'r' entries), all masses, C."""
    nr = types.count("r")
    m = len(types)
    rint = x[:nr]
    w = x[nr : nr + m]
    C = x[nr + m]
    r = np.empty(m)
    k = 0
    for j, t in enumerate(types):
        if t == "o":
            r[j] = 0.0
        elif t == "w":
            r[j] = A
        else:
            r[j] = rint[k]
            k += 1
    return r, w, C


def pack(r, w, C, types):
    rint = [r[j] for j, t in enumerate(types) if t == "r"]
    return np.concatenate([np.asarray(rint, float), np.asarray(w, float), [C]])


def residual_jac(grid, x, types):
    A = grid.A
    r, w, C = unpack(x, types, A)
    m = len(types)
    nr = types.count("r")
    E = grid.E(r)  # m x ns
    g, gp = grid.G(r)
    s, wq = grid.s, grid.wq
    pt = w @ E
    L = np.log(pt) - np.log(4 * np.pi * grid.s)
    F = s[None, :] * E * wq[None, :]  # f(s|r_j) times quadrature weight
    i0 = -(F @ L) - LOG2PIE
    i1 = -((F * g) @ L)
    i2 = -((F * (g * g + gp)) @ L)
    B = E / pt[None, :]  # dL/dw_k = E_k / ptil
    Di_dw = -(F @ B.T)  # m x m
    Di1_dw = -((F * g) @ B.T)
    Bg = B * g  # dL/dr_k = w_k E_k g_k / ptil
    Di_dr = -(F @ Bg.T) * w[None, :] + np.diag(i1)
    Di1_dr = -((F * g) @ Bg.T) * w[None, :] + np.diag(i2)
    ridx = [j for j, t in enumerate(types) if t == "r"]
    neq = m + nr + 1
    res = np.empty(neq)
    J = np.zeros((neq, nr + m + 1))
    # i(r_j) - C for every support point
    res[:m] = i0 - C
    J[:m, :nr] = Di_dr[:, ridx]
    J[:m, nr : nr + m] = Di_dw
    J[:m, nr + m] = -1.0
    # i'(r_j) = 0 for interior rings
    for a, j in enumerate(ridx):
        res[m + a] = i1[j]
        J[m + a, :nr] = Di1_dr[j, ridx]
        J[m + a, nr : nr + m] = Di1_dw[j, :]
    res[m + nr] = w.sum() - 1.0
    J[m + nr, nr : nr + m] = 1.0
    return res, J, dict(r=r, w=w, C=C, i0=i0, i1=i1, i2=i2)


def newton(grid, x0, types, tol=1e-14, maxit=60, verbose=False):
    x = np.array(x0, float)
    res, J, info = residual_jac(grid, x, types)
    nrm = np.max(np.abs(res))
    for it in range(maxit):
        if nrm < tol:
            return x, nrm, it, True
        try:
            dx = np.linalg.solve(J, -res)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(J, -res, rcond=None)[0]
        lam = 1.0
        ok = False
        for _ in range(30):
            xn = x + lam * dx
            r, w, C = unpack(xn, types, grid.A)
            if np.all(w > -1e-300) and np.all((r >= 0) & (r <= grid.A)) and np.all(np.diff(np.sort(r)) > 0):
                try:
                    rn, Jn, infon = residual_jac(grid, xn, types)
                    nn = np.max(np.abs(rn))
                    if np.isfinite(nn) and nn < nrm * (1 - 1e-4 * lam) or nn < tol:
                        ok = True
                        break
                except FloatingPointError:
                    pass
            lam *= 0.5
        if not ok:
            return x, nrm, it, False
        x, res, J, nrm = xn, rn, Jn, nn
        if verbose:
            print(it, nrm, lam)
    return x, nrm, maxit, nrm < tol


def global_check(grid, r, w, C, ngrid=1500, refine=True):
    """max over [0, A] of i(r) - C away from the support, with the location of the largest local maximum.
    Returns (vmax, rmax, i(0)-C, i''(0))."""
    A = grid.A
    rq = np.linspace(0.0, A, ngrid)
    v = grid.kkt(rq, r, w) - C
    # local maxima of v on the grid, excluding those that sit at support points
    cand = []
    for k in range(1, ngrid - 1):
        if v[k] >= v[k - 1] and v[k] >= v[k + 1]:
            cand.append(rq[k])
    best = (-np.inf, None)
    for rc in cand:
        x = rc
        if refine:
            for _ in range(30):
                i0, i1, i2 = grid.kkt([x], r, w, deriv=2)
                if i2[0] >= 0:
                    break
                step = -i1[0] / i2[0]
                step = np.clip(step, -0.05, 0.05)
                x = min(max(x + step, 0.0), A)
                if abs(step) < 1e-13:
                    break
        if np.min(np.abs(np.asarray(r) - x)) < 1e-4:
            continue
        val = grid.kkt([x], r, w)[0] - C
        if val > best[0]:
            best = (val, x)
    i0v, _, i2v = grid.kkt([0.0], r, w, deriv=2)
    return best[0], best[1], i0v[0] - C, i2v[0]
