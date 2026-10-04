"""q819 (03 Oct 2026): when does the centre atom split? Model of q818 (formal density on the ellipse plus an atom at the centre with
D(0) = C); here also the Hessian of D at the centre by second differences of r(q) = -int phi_2(y - q) log p(y) dy (same Hessian as D).
A negative-definite Hessian keeps the centre atom; the first eigenvalue (along x1 or x2) reaching zero marks a split of the centre atom
into a pair on that axis. Prints m0, C, H11, H22 along a list of rm at fixed rp.
Usage: py q819_centre_split.py RP "rm1,rm2,..." [M] [NR] [NA] [RMAX]"""

import sys, math
import numpy as np

RP = float(sys.argv[1])
RMS = [float(x) for x in sys.argv[2].split(",")]
M = int(sys.argv[3]) if len(sys.argv) > 3 else 6
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
    return np.concatenate([h, [r0 - r.mean()]]), r, r0, L


def rq(q, L):
    ph = np.exp(-((Y1 - q[0]) ** 2 + (Y2 - q[1]) ** 2) / 2) / (2 * np.pi)
    return -(WY * ph * L).sum()


u = np.zeros(M + 1)
u[M] = 1e-3
for rm in RMS:
    z1 = RP * np.cos(t)
    z2 = rm * np.sin(t)
    Kc = np.exp(-((Y1[:, None] - z1[None, :]) ** 2 + (Y2[:, None] - z2[None, :]) ** 2) / 2) / (2 * np.pi)
    for it in range(80):
        F, r, r0, L = model(u, Kc)
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
            if 0 <= un[M] < 0.9:
                Fn = model(un, Kc)[0]
                if np.all(np.isfinite(Fn)) and np.max(np.abs(Fn)) < np.max(np.abs(F)):
                    u = un
                    break
            lam /= 2
        else:
            break
    F, r, r0, L = model(u, Kc)
    d = 1e-3
    c0 = rq((0, 0), L)
    H11 = (rq((d, 0), L) + rq((-d, 0), L) - 2 * c0) / d**2
    H22 = (rq((0, d), L) + rq((0, -d), L) - 2 * c0) / d**2
    print(
        f"rp {RP} rm {rm}: m0 {u[M]:.8f}; C {r.mean() - L2PE:.10f}; H11 {H11:+.6f}; H22 {H22:+.6f}; residual {np.max(np.abs(F)):.1e}; "
        f"min f {(1 + u[:M] @ cosk).min():.3f}",
        flush=True,
    )
