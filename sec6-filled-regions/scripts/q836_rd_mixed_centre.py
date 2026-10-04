"""q836 (03 Oct 2026, problem x geometry cross): TEXT COPY of q829 (rate-distortion dual centre birth) on the mixed curve of
q790/q831 (env LAM; 0 = ellipse and reproduces q829). q829's docstring follows.
q829 (03 Oct 2026, BACKLOG 63, third bridge): TEXT COPY of q827 with the RATE-DISTORTION / NPMLE dual KKT function (q790 FUNC rd,
D_core2 _D_rd, as text): source N(0, s^2 I) (env SRC_S, default 2 as the record's rd family), mixing law pi = formal density f
on the ellipse; p = int phi(y - x) dpi; KKT function D(q) = int phi(y - q) P_src(y)/p(y) dy (= 1 on the support at the optimum);
centre excess D(0) - C, C = mean of D on the curve. q827's docstring follows (read 'D' for 'risk').
q827 (03 Oct 2026, BACKLOG 62, bridge test): the centre birth of the filled-ellipse problem for the LEAST FAVOURABLE PRIOR (bounded
normal mean in 2D, squared loss, unit noise), from the formal density alone. Same construction as q816 (capacity) with the LFP KKT
function of q790 (FUNC lfp, text): prior density f(t) = 1 + sum c_2k cos 2kt on the ellipse (rp cos t, rm sin t), weight dt/2pi;
posterior mean delta(y) = N(y)/p(y), N = int theta phi(y - theta) dpi, p = int phi(y - theta) dpi; risk
r(theta) = int phi(y - theta) |delta(y) - theta|^2 dy. Harmonics 2..2M of r on the curve are solved to zero (Newton, float64); the Bayes
risk B is the mean of r on the curve; the centre excess is D(0) - C. The LFP needs an interior atom once D(0) - C > 0.
Usage: py q827_lfp_centre.py RP RM_LO RM_HI [M] [NR] [NA] [RMAX]   (RM_LO = RM_HI: one evaluation; else bisection on the sign)
"""

import sys, math
import numpy as np

import os

LAM = float(os.environ.get("LAM", "0"))
RP = float(sys.argv[1])
LO = float(sys.argv[2])
HI = float(sys.argv[3])
M = int(sys.argv[4]) if len(sys.argv) > 4 else 8
NR = int(sys.argv[5]) if len(sys.argv) > 5 else 160
NA = int(sys.argv[6]) if len(sys.argv) > 6 else 256
RMAX = float(sys.argv[7]) if len(sys.argv) > 7 else 14.0
import os

S2 = float(os.environ.get("SRC_S", "2")) ** 2
xr, wr = np.polynomial.legendre.leggauss(NR)
rho = (xr + 1) * RMAX / 2
wrho = wr * RMAX / 2
ang = 2 * np.pi * np.arange(NA) / NA
Y1 = (rho[:, None] * np.cos(ang)[None, :]).ravel()
Y2 = (rho[:, None] * np.sin(ang)[None, :]).ravel()
WY = (wrho[:, None] * rho[:, None] * np.full((1, NA), 2 * np.pi / NA)).ravel()
t = 2 * np.pi * np.arange(NA) / NA
cosk = np.array([np.cos(2 * k * t) for k in range(1, M + 1)])


def setup(rm):
    eps = RP - rm
    R = RP - eps / 2
    et = eps / (2 * (1 + LAM))
    z1 = (R + et) * np.cos(t) + LAM * et * np.cos(3 * t)
    z2 = (R - et) * np.sin(t) + LAM * et * np.sin(3 * t)
    Kc = np.exp(-((Y1[:, None] - z1[None, :]) ** 2 + (Y2[:, None] - z2[None, :]) ** 2) / 2) / (2 * np.pi)
    return Kc, z1, z2


PSRC = np.exp(-(Y1**2 + Y2**2) / (2 * S2)) / (2 * np.pi * S2)


def risk_at(q1, q2, d1, d2):  # d1 = P_src/p (d2 unused): D(q) = int phi(y - q) P_src/p dy
    ph = np.exp(-((Y1 - q1) ** 2 + (Y2 - q2) ** 2) / 2) / (2 * np.pi)
    return (WY * ph * d1).sum()


def kkt(c, S):
    Kc, z1, z2 = S
    f = 1 + c @ cosk
    p = Kc @ (f / NA)
    d1 = PSRC / p
    r = Kc.T @ (WY * d1)
    return r, d1, None


def harms(r):
    return np.array([2 * np.mean(r * cosk[k]) for k in range(M)])


def solve(rm, c0=None):
    S = setup(rm)
    c = np.zeros(M) if c0 is None else c0.copy()
    for it in range(40):
        r, d1, d2 = kkt(c, S)
        h = harms(r)
        if np.max(np.abs(h)) < 1e-13:
            break
        J = np.zeros((M, M))
        dd = 1e-6
        for j in range(M):
            cp = c.copy()
            cp[j] += dd
            J[:, j] = (harms(kkt(cp, S)[0]) - h) / dd
        step = np.linalg.solve(J, -h)
        lam = 1.0
        while lam > 1e-4:
            cn = c + lam * step
            if (1 + cn @ cosk).min() > -1:
                hn = harms(kkt(cn, S)[0])
                if np.all(np.isfinite(hn)) and np.max(np.abs(hn)) < np.max(np.abs(h)):
                    c = cn
                    break
            lam /= 2
        else:
            break
    r, d1, d2 = kkt(c, S)
    B = r.mean()
    r0 = risk_at(0.0, 0.0, d1, d2)
    return r0 - B, c, np.max(np.abs(harms(r))), (1 + c @ cosk).min(), B, r


if LO == HI:
    v, c, res, fmin, B, r = solve(LO)
    print(
        f"RD s {S2**0.5} rp {RP} rm {LO}: C {B:.12f}; ripple {r.max() - r.min():.2e}; D(0) - C = {v:+.8e}; harmonic residual "
        f"{res:.1e}; min f {fmin:.4f}; c = {np.round(c, 5).tolist()}"
    )
else:
    a, b = LO, HI
    va, ca, _, _, _, _ = solve(a)
    vb, cb, _, _, _, _ = solve(b, ca)
    print(
        f"RD s {S2**0.5} rp {RP}: D(0) - C = {va:+.6e} at rm {a}, {vb:+.6e} at rm {b} (M {M}, NR {NR}, NA {NA}, RMAX {RMAX})",
        flush=True,
    )
    if va * vb > 0:
        print("no sign change")
        sys.exit()
    for it in range(40):
        m = 0.5 * (a + b)
        vm, cm, res, fmin, _, _ = solve(m, ca)
        if vm * va > 0:
            a, va, ca = m, vm, cm
        else:
            b, vb = m, vm
        if b - a < 1e-7:
            break
    print(
        f"RD s {S2**0.5} rp {RP}: centre birth predicted at rm = {0.5*(a + b):.7f} (eps {RP - 0.5*(a + b):.7f}); residual {res:.1e}; min f {fmin:.4f}"
    )
