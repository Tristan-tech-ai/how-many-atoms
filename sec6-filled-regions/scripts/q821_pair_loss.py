"""q821 (03 Oct 2026, item 56 first step): TEXT COPY of q820 (pair model) with one addition after each solve: where the pair stops
being optimal. Scans D - C on the minor axis (0, b), b in [0, 1.5] (a birth there = the layout rule's Y pair, 2 -> 4), on the circle
of radius a around the origin (angle of the largest excess), and the Hessian of D at (a, 0) (H22 -> 0 = each pair atom splits
along x2). q820's docstring follows.
q820 (03 Oct 2026, BACKLOG 53 part b): after the centre atom's split, the formal-density model with a PAIR on the major axis.
Model = q819 (formal density f = 1 + sum c_2k cos 2kt on the ellipse, weight 1 - m0) with the centre atom replaced by two atoms of
mass m0/2 at (+-a, 0). Unknowns c_2..c_2M, m0, a; equations: harmonics 2..2M of the KKT function on the curve vanish, D(a, 0) = C
(C = mean of the KKT function on the curve), d/dx1 D at (a, 0) = 0. Started from the centre-atom solution (a small); reports m0, a, C,
the Hessian of D at (a, 0) (both diagonal entries; negative = the pair is a local maximum), D(0) - C (must stay < 0) and a check that the
pair solution beats the centre-atom one in C.
Usage: py q820_centre_pair.py RP "rm1,rm2,..." A_START [M] [NR] [NA] [RMAX]"""

import sys, math
import numpy as np

RP = float(sys.argv[1])
RMS = [float(x) for x in sys.argv[2].split(",")]
A0 = float(sys.argv[3])
M = int(sys.argv[4]) if len(sys.argv) > 4 else 6
NR = int(sys.argv[5]) if len(sys.argv) > 5 else 160
NA = int(sys.argv[6]) if len(sys.argv) > 6 else 192
RMAX = float(sys.argv[7]) if len(sys.argv) > 7 else 14.0
xr, wr = np.polynomial.legendre.leggauss(NR)
rho = (xr + 1) * RMAX / 2
wrho = wr * RMAX / 2
ang = 2 * np.pi * np.arange(NA) / NA
Y1 = (rho[:, None] * np.cos(ang)[None, :]).ravel()
Y2 = (rho[:, None] * np.sin(ang)[None, :]).ravel()
WY = (wrho[:, None] * rho[:, None] * np.full((1, NA), 2 * np.pi / NA)).ravel()
t = 2 * np.pi * np.arange(NA) / NA
cosk = np.array([np.cos(2 * k * t) for k in range(1, M + 1)])
L2PE = math.log(2 * math.pi) + 1


def ph(q1, q2):
    return np.exp(-((Y1 - q1) ** 2 + (Y2 - q2) ** 2) / 2) / (2 * np.pi)


def rq(q, L):
    return -(WY * ph(q[0], q[1]) * L).sum()


def model(u, Kc):
    c, m0, a = u[:M], u[M], u[M + 1]
    f = 1 + c @ cosk
    p = (1 - m0) * (Kc @ (f / NA)) + m0 * (ph(a, 0) + ph(-a, 0)) / 2
    L = np.log(p)
    r = -(Kc * (WY * L)[:, None]).sum(0)
    C = r.mean()
    h = np.array([2 * np.mean(r * cosk[k]) for k in range(M)])
    ra = rq((a, 0), L)
    d = 1e-4
    g = (rq((a + d, 0), L) - rq((a - d, 0), L)) / (2 * d)
    return np.concatenate([h, [ra - C, g]]), r, L


def centre_model(u, Kc):  # q819's model (a = 0, no slope equation), for the comparison of C
    c, m0 = u[:M], u[M]
    f = 1 + c @ cosk
    p = (1 - m0) * (Kc @ (f / NA)) + m0 * ph(0, 0)
    L = np.log(p)
    r = -(Kc * (WY * L)[:, None]).sum(0)
    h = np.array([2 * np.mean(r * cosk[k]) for k in range(M)])
    return np.concatenate([h, [rq((0, 0), L) - r.mean()]]), r, L


def newton(fun, u, Kc, n, ok):
    for it in range(80):
        F = fun(u, Kc)[0]
        if np.max(np.abs(F)) < 1e-12:
            break
        J = np.zeros((n, n))
        for j in range(n):
            up = u.copy()
            up[j] += 1e-7
            J[:, j] = (fun(up, Kc)[0] - F) / 1e-7
        step = np.linalg.lstsq(J, -F, rcond=None)[0]
        lam = 1.0
        while lam > 1e-5:
            un = u + lam * step
            if ok(un):
                Fn = fun(un, Kc)[0]
                if np.all(np.isfinite(Fn)) and np.max(np.abs(Fn)) < np.max(np.abs(F)):
                    u = un
                    break
            lam /= 2
        else:
            break
    return u, np.max(np.abs(fun(u, Kc)[0]))


u = np.zeros(M + 2)
u[M] = 0.088
u[M + 1] = A0
uc = np.zeros(M + 1)
uc[M] = 0.088
for rm in RMS:
    z1 = RP * np.cos(t)
    z2 = rm * np.sin(t)
    Kc = np.exp(-((Y1[:, None] - z1[None, :]) ** 2 + (Y2[:, None] - z2[None, :]) ** 2) / 2) / (2 * np.pi)
    uc, resc = newton(centre_model, uc, Kc, M + 1, lambda v: 0 <= v[M] < 0.9)
    Cc = centre_model(uc, Kc)[1].mean() - L2PE
    u, res = newton(model, u, Kc, M + 2, lambda v: 0 <= v[M] < 0.9 and v[M + 1] > 1e-6)
    F, r, L = model(u, Kc)
    C = r.mean() - L2PE
    a = u[M + 1]
    d = 1e-3
    c0 = rq((a, 0), L)
    H11 = (rq((a + d, 0), L) + rq((a - d, 0), L) - 2 * c0) / d**2
    H22 = (rq((a, d), L) + rq((a, -d), L) - 2 * c0) / d**2
    D0 = rq((0, 0), L) - r.mean()
    print(
        f"rp {RP} rm {rm}: pair a {a:.6f}, m0 {u[M]:.8f}, C {C:.12f} (centre-atom model C {Cc:.12f}, pair - centre {C - Cc:+.2e}); "
        f"Hessian at (a,0) {H11:+.6f} {H22:+.6f}; D(0) - C {D0:+.3e}; residual {res:.1e}; min f {(1 + u[:M] @ cosk).min():.3f}",
        flush=True,
    )
    bb = np.linspace(0, 1.5, 151)
    Db = np.array([rq((0, b), L) for b in bb]) - r.mean()
    th = np.linspace(0, np.pi / 2, 91)
    Dth = np.array([rq((a * np.cos(x), a * np.sin(x)), L) for x in th]) - r.mean()
    print(
        f"    minor axis: max D - C {Db.max():+.3e} at b {bb[np.argmax(Db)]:.2f}; circle radius a: max D - C {Dth.max():+.3e} at {np.degrees(th[np.argmax(Dth)]):.0f} deg, "
        f"min {Dth.min():+.3e}",
        flush=True,
    )
