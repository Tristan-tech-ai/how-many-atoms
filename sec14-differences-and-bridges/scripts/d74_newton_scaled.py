"""d74_newton_scaled: d50_lib.newton copied as text with one change: the acceptance norm (and the linear solve) uses residual rows of
interior rings' stationarity equations divided by max(r_j, 1e-3), because i'(r) ~ r i''(0) near the origin makes the plain max-norm reject
full Newton steps next to a young ring (cond(J) ~ 5e11, 89 iterations at the 2D lift-off near A 26.44). Definitions only.
"""

import numpy as np


def newton_scaled(L, grid, x0, types, tol=1e-14, maxit=60):
    ridx = [j for j, t in enumerate(types) if t == "r"]
    m = len(types)

    def scaled(x):
        res, J, info = L.residual_jac(grid, x, types)
        r = info["r"]
        sc = np.ones(len(res))
        for a, j in enumerate(ridx):
            sc[m + a] = 1.0 / max(r[j], 1e-3)
        return res * sc, J * sc[:, None]

    x = np.array(x0, float)
    res, J = scaled(x)
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
            r, w, C = L.unpack(xn, types, grid.A)
            if np.all(w > -1e-300) and np.all((r >= 0) & (r <= grid.A)) and np.all(np.diff(np.sort(r)) > 0):
                try:
                    rn, Jn = scaled(xn)
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
    return x, nrm, maxit, nrm < tol


def newton_tsvd(L, grid, x0, types, tol=1e-14, maxit=60, rcond=1e-10):
    """Newton with a minimum-norm truncated-SVD step (singular values below rcond * smax dropped): next to a young ring the pair
    (radius, mass) is nearly undetermined (a small ring looks like a point mass), and the plain step wanders along that direction.
    """
    x = np.array(x0, float)
    res, J, info = L.residual_jac(grid, x, types)
    nrm = np.max(np.abs(res))
    best = (nrm, x.copy())
    for it in range(maxit):
        if nrm < tol:
            return x, nrm, it, True
        dx = np.linalg.lstsq(J, -res, rcond=rcond)[0]
        lam = 1.0
        ok = False
        for _ in range(30):
            xn = x + lam * dx
            r, w, C = L.unpack(xn, types, grid.A)
            if np.all(w > -1e-300) and np.all((r >= 0) & (r <= grid.A)) and np.all(np.diff(np.sort(r)) > 0):
                rn, Jn, infon = L.residual_jac(grid, xn, types)
                nn = np.max(np.abs(rn))
                if np.isfinite(nn) and nn < nrm * (1 - 1e-4 * lam) or nn < tol:
                    ok = True
                    break
            lam *= 0.5
        if not ok:
            return best[1], best[0], it, best[0] < tol
        x, res, J, nrm = xn, rn, Jn, nn
        if nrm < best[0]:
            best = (nrm, x.copy())
    return best[1], best[0], maxit, best[0] < tol
