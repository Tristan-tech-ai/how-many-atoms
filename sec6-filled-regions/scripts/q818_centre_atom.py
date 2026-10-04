"""q818 (03 Oct 2026): the formal density with a centre atom. Past the centre birth rm*(rp) the filled-ellipse optimum needs mass at
the centre. Model: input = (1 - m0) x [density f(t) = 1 + sum_{k=1..M} c_2k cos 2kt on the ellipse] + m0 x [atom at the origin]. Unknowns
c_2..c_2M and m0; equations: harmonics 2..2M of the KKT function on the curve vanish, and the centre value equals the curve mean
(D(0) = C). Prints m0 and C (minus log 2 pi e) at each rm. float64, polar quadrature as q816.
Usage: py q818_centre_atom.py RP "rm1,rm2,..." [M] [NR] [NA] [RMAX]"""

import sys, math
import numpy as np

RP = float(sys.argv[1])
RMS = [float(x) for x in sys.argv[2].split(",")]
M = int(sys.argv[3]) if len(sys.argv) > 3 else 10
NR = int(sys.argv[4]) if len(sys.argv) > 4 else 160
NA = int(sys.argv[5]) if len(sys.argv) > 5 else 192
RMAX = float(sys.argv[6]) if len(sys.argv) > 6 else 14.0
xr, wr = np.polynomial.legendre.leggauss(NR)
rho = (xr + 1) * RMAX / 2
wrho = wr * RMAX / 2
ang = 2 * np.pi * np.arange(NA) / NA
Y1 = (rho[:, None] * np.cos(ang)[None, :]).ravel()
Y2 = (rho[:, None] * np.sin(ang)[None, :]).ravel()
WY = (wrho[:, None] * rho[:, None] * np.full((1, NA), 2 * np.pi / NA)).ravel()
t = 2 * np.pi * np.arange(NA) / NA
cosk = np.array([np.cos(2 * k * t) for k in range(1, M + 1)])
phi0 = np.exp(-(Y1**2 + Y2**2) / 2) / (2 * np.pi)
L2PE = math.log(2 * math.pi) + 1


def model(u, Kc):
    c, m0 = u[:M], u[M]
    f = 1 + c @ cosk
    p = (1 - m0) * (Kc @ (f / NA)) + m0 * phi0
    L = np.log(p)
    r = -(Kc * (WY * L)[:, None]).sum(0)
    r0 = -(WY * phi0 * L).sum()
    h = np.array([2 * np.mean(r * cosk[k]) for k in range(M)])
    return np.concatenate([h, [r0 - r.mean()]]), r, r0


u = np.zeros(M + 1)
u[M] = 1e-3
for rm in RMS:
    z1 = RP * np.cos(t)
    z2 = rm * np.sin(t)
    Kc = np.exp(-((Y1[:, None] - z1[None, :]) ** 2 + (Y2[:, None] - z2[None, :]) ** 2) / 2) / (2 * np.pi)
    for it in range(60):
        F, r, r0 = model(u, Kc)
        if np.max(np.abs(F)) < 1e-13:
            break
        J = np.zeros((M + 1, M + 1))
        for j in range(M + 1):
            up = u.copy()
            up[j] += 1e-7
            J[:, j] = (model(up, Kc)[0] - F) / 1e-7
        step = np.linalg.solve(J, -F)
        lam = 1.0
        while lam > 1e-4:
            un = u + lam * step
            if 0 <= un[M] < 0.5:
                Fn = model(un, Kc)[0]
                if np.all(np.isfinite(Fn)) and np.max(np.abs(Fn)) < np.max(np.abs(F)):
                    u = un
                    break
            lam /= 2
        else:
            break
    F, r, r0 = model(u, Kc)
    print(
        f"rp {RP} rm {rm}: centre mass m0 = {u[M]:.10f}; C = {r.mean() - L2PE:.12f}; residual {np.max(np.abs(F)):.1e}; "
        f"min f {(1 + u[:M] @ cosk).min():.3f}; c {np.round(u[:M], 4).tolist()}",
        flush=True,
    )
