"""q701b (copy of q701 as text with 6-degree bins: at the ring radius 1.15 a 2-degree bin spans 0.04, less than the grid spacing
0.05, so some bins held no grid point on the ridge; Amendment 1048). q701 (Amendment 1042 readout): for a q700 solution, per 2-degree bin of the polar angle over the whole turn, the maximum of D - max D
over interior grid points with gauge radius in [RLO, RHI] (the ring band) and over the wall points, in units of the bracket width.
Reports the worst bin of each and the number of bins below -1, -3 and -5 brackets. Usage: py q701_ring_wall.py FILE [FILE ...]
(env RLO, RHI; defaults 0.5 and 2.5)"""

import os, sys
import numpy as np

RLO = float(os.environ.get("RLO", 0.5))
RHI = float(os.environ.get("RHI", 2.5))
for f in sys.argv[1:]:
    z = np.load(f)
    x = z["x"]
    mask = z["mask"]
    Db = z["Db"]
    bx = z["bx"]
    by = z["by"]
    wg = z["wg"]
    P = float(z["P"])
    A = float(z["A"])
    B = float(z["B"]) if "B" in z.files else A
    gap = float(z["hi"] - z["lo"])
    X, Y = np.meshgrid(x, x, indexing="ij")
    # D on the grid is not saved; recompute it from the saved masses with the same kernel split as q696/q698/q700
    from scipy.fft import rfft2, irfft2
    from scipy import sparse

    h = float(z["h"])
    n = len(x)
    SA = 0.3
    M = len(bx)
    R = int(np.ceil(7 * SA / h))
    rows, cols, vals = [], [], []
    for m in range(M):
        i0 = int(round(bx[m] / h)) + n // 2
        j0 = int(round(by[m] / h)) + n // 2
        ii, jj = np.meshgrid(np.arange(i0 - R, i0 + R + 1), np.arange(j0 - R, j0 + R + 1), indexing="ij")
        vv = (
            np.exp(-0.5 * ((x[ii] - bx[m]) ** 2 + (x[jj] - by[m]) ** 2) / SA**2) / (2 * np.pi * SA**2) * h * h
        )
        rows.append((ii * n + jj).ravel())
        cols.append(np.full(ii.size, m))
        vals.append(vv.ravel())
    S = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, M)
    )
    d = np.minimum(np.arange(n), n - np.arange(n)) * h
    ker = lambda var: rfft2(
        np.exp(-0.5 * (d[:, None] ** 2 + d[None, :] ** 2) / var) / (2 * np.pi * var) * h * h
    )
    Ff, Fb = ker(1.0), ker(1.0 - SA**2)
    cf = lambda g: irfft2(rfft2(g) * Ff, s=(n, n))
    cb = lambda g: irfft2(rfft2(g) * Fb, s=(n, n))
    hY = np.log(2 * np.pi * np.e)
    p = np.maximum((cf(wg) + h * h * cb((S @ z["wb"]).reshape(n, n) / (h * h))) / (h * h), 1e-300)
    Dg = -cf(np.log(p)) - hY
    Dmax = max(Dg[mask].max(), Db.max())
    gauge = A * ((np.abs(X) / A) ** P + (np.abs(Y) / B) ** P) ** (1 / P)
    band = mask & (gauge >= RLO) & (gauge <= RHI)
    angg = np.degrees(np.arctan2(Y, X)) % 360
    angb = np.degrees(np.arctan2(by, bx)) % 360
    nb = 60
    e = np.linspace(0, 360, nb + 1)
    ring = np.array([(Dg[band & (angg >= e[k]) & (angg < e[k + 1])].max() - Dmax) / gap for k in range(nb)])
    wall = np.array([(Db[(angb >= e[k]) & (angb < e[k + 1])].max() - Dmax) / gap for k in range(nb)])
    cnt = lambda v: [int(np.sum(v < -t)) for t in (1, 3, 5)]
    print(
        f"{f}: p {P:g} A {A:g} B {B:g}; bracket {gap:.1e}; C in [{float(z['lo']):.6f}, {float(z['hi']):.6f}]"
    )
    print(
        f"   ring band [{RLO}, {RHI}]: worst bin {ring.min():.1f} brackets at {e[np.argmin(ring)]:.0f} deg; bins below -1/-3/-5: {cnt(ring)}"
    )
    print(
        f"   wall: worst bin {wall.min():.1f} brackets at {e[np.argmin(wall)]:.0f} deg; bins below -1/-3/-5: {cnt(wall)}"
    )
