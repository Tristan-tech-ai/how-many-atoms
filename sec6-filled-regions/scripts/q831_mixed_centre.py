"""q831 (03 Oct 2026, BACKLOG 64a): TEXT COPY of q816 on the MIXED curve of q790/q805 (env LAM): z(t) = R e^(it) + eta (e^(-it) +
LAM e^(3it)), R = rp - eps/2, eta = eps/(2 (1 + LAM)), eps = rp - rm (so the curve passes through (rp, 0) and (0, rm)); LAM 0 is the
ellipse and reproduces q816. The density is in the curve parameter t, as in q790. q816's docstring follows.
q816 (03 Oct 2026): where the filled-ellipse optimum first needs an interior atom, from the formal density alone.
For an input density f(t) = 1 + sum_{k=1..M} c_2k cos 2kt on the ellipse (rp cos t, rm sin t), solve the harmonics 2..2M of the KKT
function along the curve to zero (q790's formal density, here in double precision), then read D(0) - C, the KKT excess at the centre
(C = the mean of the KKT function on the curve). The predicted centre birth is the rm where D(0) - C = 0. Check case: the circle,
where f = 1 is exact and D(0) - C = 0 at R = 2.454 (Dytso-Al-Poor-Shamai, Table I, n = 2).
Usage: py q816_centre_birth.py RP RM_LO RM_HI [M] [NR] [NA] [RMAX]   (RM_LO = RM_HI: one evaluation; else bisection on the sign)
"""

import sys, math
import numpy as np

import os

LAM = float(os.environ.get("LAM", "0"))
RP = float(sys.argv[1])
LO = float(sys.argv[2])
HI = float(sys.argv[3])
M = int(sys.argv[4]) if len(sys.argv) > 4 else 10
NR = int(sys.argv[5]) if len(sys.argv) > 5 else 160
NA = int(sys.argv[6]) if len(sys.argv) > 6 else 256
RMAX = float(sys.argv[7]) if len(sys.argv) > 7 else 14.0
xr, wr = np.polynomial.legendre.leggauss(NR)
rho = (xr + 1) * RMAX / 2
wrho = wr * RMAX / 2
ang = 2 * np.pi * np.arange(NA) / NA
Y1 = (rho[:, None] * np.cos(ang)[None, :]).ravel()
Y2 = (rho[:, None] * np.sin(ang)[None, :]).ravel()
WY = (wrho[:, None] * rho[:, None] * np.full((1, NA), 2 * np.pi / NA)).ravel()  # polar area weights
t = 2 * np.pi * np.arange(NA) / NA
cosk = np.array([np.cos(2 * k * t) for k in range(1, M + 1)])


def setup(rm):
    eps = RP - rm
    R = RP - eps / 2
    et = eps / (2 * (1 + LAM))
    z1 = (R + et) * np.cos(t) + LAM * et * np.cos(3 * t)
    z2 = (R - et) * np.sin(t) + LAM * et * np.sin(3 * t)
    Kc = np.exp(-((Y1[:, None] - z1[None, :]) ** 2 + (Y2[:, None] - z2[None, :]) ** 2) / 2) / (
        2 * np.pi
    )  # phi_2(y - z_l)
    return Kc


def kkt(c, Kc):
    f = 1 + c @ cosk
    p = Kc @ (f / NA)
    L = np.log(p)
    r = -(Kc * (WY * L)[:, None]).sum(0)  # r(z_l) = -int phi_2(y - z_l) log p(y) dy
    return r, L


def harms(r):
    return np.array([2 * np.mean(r * cosk[k]) for k in range(M)])


def solve(rm, c0=None):
    Kc = setup(rm)
    c = np.zeros(M) if c0 is None else c0.copy()
    for it in range(40):
        r, L = kkt(c, Kc)
        h = harms(r)
        if np.max(np.abs(h)) < 1e-13:
            break
        J = np.zeros((M, M))
        d = 1e-6
        for j in range(M):
            cp = c.copy()
            cp[j] += d
            J[:, j] = (harms(kkt(cp, Kc)[0]) - h) / d
        step = np.linalg.solve(J, -h)
        lam = 1.0
        while lam > 1e-4:
            cn = c + lam * step
            fmin = (1 + cn @ cosk).min()
            hn = (
                harms(kkt(cn, Kc)[0]) if fmin > -1 else None
            )  # allow a non-positive formal density; p must stay positive
            if hn is not None and np.all(np.isfinite(hn)) and np.max(np.abs(hn)) < np.max(np.abs(h)):
                c = cn
                break
            lam /= 2
        else:
            break
    r, L = kkt(c, Kc)
    C = r.mean()
    r0 = -(WY * np.exp(-(Y1**2 + Y2**2) / 2) / (2 * np.pi) * L).sum()  # r(0)
    global LASTC, LASTR
    LASTC = C
    LASTR = r
    return r0 - C, c, np.max(np.abs(harms(r))), (1 + c @ cosk).min()


if LO == HI:
    v, c, res, fmin = solve(LO)
    print(
        f"rp {RP} rm {LO}: C_formal - log2pie = {LASTC - (math.log(2*math.pi) + 1):.13f}; sup_curve - log2pie = {LASTR.max() - (math.log(2*math.pi) + 1):.13f}; ripple {LASTR.max() - LASTR.min():.2e}; D(0) - C = {v:+.8e}; harmonic residual {res:.1e}; min f {fmin:.4f}; c_2..c_{2*M} = {np.round(c, 5).tolist()}"
    )
else:
    a, b = LO, HI
    va, ca, _, _ = solve(a)
    vb, cb, _, _ = solve(b, ca)
    print(
        f"rp {RP}: D(0) - C = {va:+.6e} at rm {a}, {vb:+.6e} at rm {b} (M {M}, NR {NR}, NA {NA}, RMAX {RMAX})",
        flush=True,
    )
    if va * vb > 0:
        print("no sign change")
        sys.exit()
    for it in range(40):
        m = 0.5 * (a + b)
        vm, cm, res, fmin = solve(m, ca)
        if vm * va > 0:
            a, va, ca = m, vm, cm
        else:
            b, vb = m, vm
        if b - a < 1e-7:
            break
    print(
        f"rp {RP}: centre birth predicted at rm = {0.5*(a + b):.7f} (eps {RP - 0.5*(a + b):.7f}); residual {res:.1e}; min f {fmin:.4f}"
    )
