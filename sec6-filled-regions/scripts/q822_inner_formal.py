"""q822 (03 Oct 2026, item 56 step ii): the INNER ring's own formal density. Model: the boundary formal density (q819's, weight 1 - mi)
plus an inner formal density of mass mi on a FREE inner curve rho(th) = a_0 + sum_{k=1}^{MI} a_k cos 2k th (polar angle th), density
g(th) = 1 + sum_{k=1}^{MI} d_k cos 2k th with weight dth/2pi. Unknowns c_2..c_2M, d_1..d_MI, a_0..a_MI, mi; equations: harmonics 2..2M
of the KKT function r on the boundary vanish; harmonics 0..2MI of r - C on the inner curve vanish (C = mean of r on the boundary);
harmonics 0..2MI of the radial derivative of r on the inner curve vanish. The copying law for a 2-atom ring on the major axis
(b_2 = 2) says the pair is lost where d_1 = 2 (second-level test, BACKLOG 56); the pair model (q821) loses the pair at rm 3.4097.
Continuation in rm from near the circle downward.
Usage: py q822_inner_formal.py RP "rm1,rm2,..." A0_START MI_START [M] [MI] [NI] [NR] [NA] [RMAX]"""

import sys, math
import numpy as np

RP = float(sys.argv[1])
RMS = [float(x) for x in sys.argv[2].split(",")]
A0 = float(sys.argv[3])
MI0 = float(sys.argv[4])
M = int(sys.argv[5]) if len(sys.argv) > 5 else 6
MI = int(sys.argv[6]) if len(sys.argv) > 6 else 4
NI = int(sys.argv[7]) if len(sys.argv) > 7 else 64
NR = int(sys.argv[8]) if len(sys.argv) > 8 else 160
NA = int(sys.argv[9]) if len(sys.argv) > 9 else 192
RMAX = float(sys.argv[10]) if len(sys.argv) > 10 else 14.0
xr, wr = np.polynomial.legendre.leggauss(NR)
rho = (xr + 1) * RMAX / 2
wrho = wr * RMAX / 2
ang = 2 * np.pi * np.arange(NA) / NA
Y1 = (rho[:, None] * np.cos(ang)[None, :]).ravel()
Y2 = (rho[:, None] * np.sin(ang)[None, :]).ravel()
WY = (wrho[:, None] * rho[:, None] * np.full((1, NA), 2 * np.pi / NA)).ravel()
t = 2 * np.pi * np.arange(NA) / NA
cosk = np.array([np.cos(2 * k * t) for k in range(1, M + 1)])
th = 2 * np.pi * np.arange(NI) / NI
cosi = np.array([np.cos(2 * k * th) for k in range(0, MI + 1)])  # row 0 = 1
L2PE = math.log(2 * math.pi) + 1


def unpack(u):
    return u[:M], u[M : M + MI], u[M + MI : M + 2 * MI + 1], u[M + 2 * MI + 1]


def model(u, Kc):
    c, d, a, mi = unpack(u)
    f = 1 + c @ cosk
    g = 1 + d @ cosi[1:]
    rr = a @ cosi
    q1 = rr * np.cos(th)
    q2 = rr * np.sin(th)
    Ki = np.exp(-((Y1[:, None] - q1[None, :]) ** 2 + (Y2[:, None] - q2[None, :]) ** 2) / 2) / (2 * np.pi)
    p = (1 - mi) * (Kc @ (f / NI if False else f / NA)) + mi * (Ki @ (g / NI))
    L = np.log(p)
    r = -(Kc * (WY * L)[:, None]).sum(0)
    C = r.mean()
    ri = -(Ki * (WY * L)[:, None]).sum(0)
    # radial derivative at the inner points: d/drho of -int phi(y - q) L dy = -int phi(y - q) ((y - q) . e_rho) L dy
    er1 = np.cos(th)
    er2 = np.sin(th)
    dri = -(
        Ki
        * ((Y1[:, None] - q1[None, :]) * er1[None, :] + (Y2[:, None] - q2[None, :]) * er2[None, :])
        * (WY * L)[:, None]
    ).sum(0)
    h = np.array([2 * np.mean(r * cosk[k]) for k in range(M)])
    hi = np.array([np.mean((ri - C) * cosi[k]) for k in range(MI + 1)])
    hd = np.array([np.mean(dri * cosi[k]) for k in range(MI + 1)])
    return np.concatenate([h, hi, hd]), C, g, rr


def newton(u, Kc):
    n = len(u)
    for it in range(100):
        F = model(u, Kc)[0]
        if np.max(np.abs(F)) < 1e-12:
            break
        J = np.zeros((n, n))
        for j in range(n):
            up = u.copy()
            up[j] += 1e-7
            J[:, j] = (model(up, Kc)[0] - F) / 1e-7
        step = np.linalg.lstsq(J, -F, rcond=None)[0]
        lam = 1.0
        while lam > 1e-6:
            un = u + lam * step
            if 0 < un[-1] < 0.5 and un[M + MI] > 0.02:
                Fn = model(un, Kc)[0]
                if np.all(np.isfinite(Fn)) and np.max(np.abs(Fn)) < np.max(np.abs(F)):
                    u = un
                    break
            lam /= 2
        else:
            break
    return u, np.max(np.abs(model(u, Kc)[0]))


u = np.zeros(M + 2 * MI + 2)
u[M + MI] = A0
u[-1] = MI0
for rm in RMS:
    z1 = RP * np.cos(t)
    z2 = rm * np.sin(t)
    Kc = np.exp(-((Y1[:, None] - z1[None, :]) ** 2 + (Y2[:, None] - z2[None, :]) ** 2) / 2) / (2 * np.pi)
    u, res = newton(u, Kc)
    F, C, g, rr = model(u, Kc)
    c, d, a, mi = unpack(u)
    print(
        f"rp {RP} rm {rm}: inner mass {mi:.8f}; C {C - L2PE:.12f}; inner radius a_0..: {np.round(a, 6).tolist()}; "
        f"inner d_1..: {np.round(d, 6).tolist()}; min g {g.min():+.3f}; residual {res:.1e}",
        flush=True,
    )
