"""d85_annulus_lib (copy of d72_libnd.py as text; the input set is the shell ain <= |x| <= A instead of the d-ball). Changes: a support
type 'v' for the inner wall at r = ain (no stationarity equation, like the outer wall 'w'); quadrature on [max(0, ain - pad), A + pad];
the feasibility test and the global check run over [ain, A]. With ain = 0 and no 'v' it is d72_libnd.

Definitions only; nothing is computed or written at import time.

Radial problem (nu = d/2 - 1). For |x| = r the output radius s = |Y| has the noncentral chi density
    f(s|r) = s^(d/2) r^(-nu) exp(-(s^2 + r^2)/2) I_nu(s r) = s E(s, r),  E = (s/r)^nu exp(-(s - r)^2/2) Ie_nu(s r).
Output density at |y| = s: p(s) = ptil(s) / (S_{d-1} s^(d-2)), ptil = sum_j w_j E_j, S_{d-1} = 2 pi^(d/2)/Gamma(d/2).
KKT function: i(r) = -int f(s|r) log p(s) ds - (d/2) log(2 pi e). Score d log f/dr = -r + s R_nu(s r), R_nu = I_{nu+1}/I_nu.
"""

import numpy as np
from scipy.special import ive, gammaln


def quad_nodes(smin, smax, h=0.5, n=24):
    """Gauss-Legendre panels of width about h on [smin, smax]."""
    npan = int(np.ceil((smax - smin) / h))
    edges = np.linspace(smin, smax, npan + 1)
    x, w = np.polynomial.legendre.leggauss(n)
    S, W = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        S.append(0.5 * (b - a) * x + 0.5 * (b + a))
        W.append(0.5 * (b - a) * w)
    return np.concatenate(S), np.concatenate(W)


class Grid:
    def __init__(self, A, ain=0.0, pad=14.0, h=0.5, n=24, dim=2):
        self.A = A
        self.ain = ain
        self.dim = dim
        self.nu = dim / 2 - 1
        self.s, self.wq = quad_nodes(max(0.0, ain - pad), A + pad, h, n)
        self.logS = np.log(2) + (dim / 2) * np.log(np.pi) - gammaln(dim / 2)
        self.logc = (dim / 2) * np.log(2 * np.pi * np.e)
        self.Loff = self.logS + (dim - 2) * np.log(self.s)

    def E(self, r):
        r = np.atleast_1d(np.asarray(r, dtype=float))
        nu = self.nu
        s = self.s[None, :]
        rr = r[:, None]
        small = rr < 1e-8
        rs = np.where(small, 1.0, rr)
        reg = (s / rs) ** nu * np.exp(-0.5 * (s - rs) ** 2) * ive(nu, s * rs)
        lim = s ** (2 * nu) * np.exp(-0.5 * s * s - nu * np.log(2) - gammaln(nu + 1))
        return np.where(small, lim, reg)

    def G(self, r):
        r = np.atleast_1d(np.asarray(r, dtype=float))
        nu = self.nu
        s = self.s[None, :]
        x = s * r[:, None]
        small = x < 1e-6
        xs = np.where(small, 1.0, x)
        R = np.where(small, x / (2 * (nu + 1)), ive(nu + 1, xs) / ive(nu, xs))
        Rox = np.where(small, 1 / (2 * (nu + 1)), R / xs)
        Rp = 1.0 - (2 * nu + 1) * Rox - R * R
        return -r[:, None] + s * R, -1.0 + s * s * Rp

    def ptil(self, r, w):
        return np.asarray(w) @ self.E(r)

    def kkt(self, rq, r, w, deriv=0):
        """i(rq) (and i', i'' if deriv >= 1, 2) for the input (r, w)."""
        pt = self.ptil(r, w)
        L = np.log(pt) - self.Loff
        rq = np.atleast_1d(np.asarray(rq, dtype=float))
        CH = 400
        vals = [np.empty(len(rq)) for _ in range(deriv + 1)]
        for a in range(0, len(rq), CH):
            rr = rq[a : a + CH]
            F = self.s[None, :] * self.E(rr) * self.wq[None, :]
            vals[0][a : a + CH] = -(F @ L) - self.logc
            if deriv >= 1:
                g, gp = self.G(rr)
                vals[1][a : a + CH] = -((F * g) @ L)
                if deriv >= 2:
                    vals[2][a : a + CH] = -((F * (g * g + gp)) @ L)
        return vals if deriv else vals[0]

    def mutual_info(self, r, w):
        i = self.kkt(r, r, w)
        return float(np.dot(w, i))


AIN = [0.0]  # inner radius used by unpack when no grid is passed (set by the caller)


def unpack(x, types, A, ain=None):
    """Unknowns: interior radii (in order of 'r' entries), all masses, C."""
    ain = AIN[0] if ain is None else ain
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
        elif t == "v":
            r[j] = ain
        else:
            r[j] = rint[k]
            k += 1
    return r, w, C


def pack(r, w, C, types):
    rint = [r[j] for j, t in enumerate(types) if t == "r"]
    return np.concatenate([np.asarray(rint, float), np.asarray(w, float), [C]])


def residual_jac(grid, x, types):
    A = grid.A
    r, w, C = unpack(x, types, A, grid.ain)
    m = len(types)
    nr = types.count("r")
    E = grid.E(r)  # m x ns
    g, gp = grid.G(r)
    s, wq = grid.s, grid.wq
    pt = w @ E
    L = np.log(pt) - grid.Loff
    F = s[None, :] * E * wq[None, :]  # f(s|r_j) times quadrature weight
    i0 = -(F @ L) - grid.logc
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
            r, w, C = unpack(xn, types, grid.A, grid.ain)
            if (
                np.all(w > -1e-300)
                and np.all((r >= grid.ain) & (r <= grid.A))
                and np.all(np.diff(np.sort(r)) > 0)
            ):
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
    """max over [ain, A] of i(r) - C away from the support, with the location of the largest local maximum.
    Returns (vmax, rmax, i(ain)-C, i''(ain))."""
    A, a0 = grid.A, grid.ain
    rq = np.linspace(a0, A, ngrid)
    v = grid.kkt(rq, r, w) - C
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
                x = min(max(x + step, a0), A)
                if abs(step) < 1e-13:
                    break
        if np.min(np.abs(np.asarray(r) - x)) < 1e-4:
            continue
        val = grid.kkt([x], r, w)[0] - C
        if val > best[0]:
            best = (val, x)
    i0v, _, i2v = grid.kkt([a0], r, w, deriv=2)
    return best[0], best[1], i0v[0] - C, i2v[0]
