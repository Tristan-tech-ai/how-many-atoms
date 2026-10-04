"""q823b (03 Oct 2026): TEXT COPY of q823 with more digits printed (Hessians %.4e) and D - C at 45 deg; model unchanged.
q823 (03 Oct 2026, BACKLOG 57): the inner QUARTET past the pair's loss. Model: q819's boundary formal density (weight 1 - mx - my)
plus an X pair (mass mx/2 at (+-a, 0)) and a Y pair (mass my/2 at (0, +-b)). Unknowns c_2..c_2M, mx, my, a, b; equations: harmonics
2..2M on the boundary vanish, D(a, 0) = C, d/dx1 D(a, 0) = 0, D(0, b) = C, d/dx2 D(0, b) = 0 (C = mean of the KKT function on the
boundary). After each solve: the Hessian of D at both pairs (tangential entries: H22 at (a, 0), H11 at (0, b); a zero = that pair
splits), and D - C on the ellipse through the four atoms (x = a cos s, y = b sin s), with the angle of its largest value off the atoms.
Functions ph, rq and the formal-density pieces are a text copy of q820.
Usage: py q823_quartet.py RP "rm1,rm2,..." A_START B_START MX_START MY_START [M] [NR] [NA] [RMAX]"""

import sys, math
import numpy as np

RP = float(sys.argv[1])
RMS = [float(x) for x in sys.argv[2].split(",")]
A0, B0, MX0, MY0 = (float(x) for x in sys.argv[3:7])
M = int(sys.argv[7]) if len(sys.argv) > 7 else 6
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
L2PE = math.log(2 * math.pi) + 1


def ph(q1, q2):
    return np.exp(-((Y1 - q1) ** 2 + (Y2 - q2) ** 2) / 2) / (2 * np.pi)


def rq(q, L):
    return -(WY * ph(q[0], q[1]) * L).sum()


def model(u, Kc):
    c = u[:M]
    mx, my, a, b = u[M : M + 4]
    f = 1 + c @ cosk
    p = (1 - mx - my) * (Kc @ (f / NA)) + mx * (ph(a, 0) + ph(-a, 0)) / 2 + my * (ph(0, b) + ph(0, -b)) / 2
    L = np.log(p)
    r = -(Kc * (WY * L)[:, None]).sum(0)
    C = r.mean()
    h = np.array([2 * np.mean(r * cosk[k]) for k in range(M)])
    d = 1e-4
    ra = rq((a, 0), L)
    ga = (rq((a + d, 0), L) - rq((a - d, 0), L)) / (2 * d)
    rb = rq((0, b), L)
    gb = (rq((0, b + d), L) - rq((0, b - d), L)) / (2 * d)
    return np.concatenate([h, [ra - C, ga, rb - C, gb]]), r, L


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
            if un[M] > 0 and un[M + 1] > 0 and un[M + 2] > 1e-3 and un[M + 3] > 1e-3:
                Fn = model(un, Kc)[0]
                if np.all(np.isfinite(Fn)) and np.max(np.abs(Fn)) < np.max(np.abs(F)):
                    u = un
                    break
            lam /= 2
        else:
            break
    return u, np.max(np.abs(model(u, Kc)[0]))


u = np.zeros(M + 4)
u[M : M + 4] = [MX0, MY0, A0, B0]
for rm in RMS:
    z1 = RP * np.cos(t)
    z2 = rm * np.sin(t)
    Kc = np.exp(-((Y1[:, None] - z1[None, :]) ** 2 + (Y2[:, None] - z2[None, :]) ** 2) / 2) / (2 * np.pi)
    u, res = newton(u, Kc)
    F, r, L = model(u, Kc)
    C = r.mean()
    mx, my, a, b = u[M : M + 4]
    dd = 1e-3
    ca = rq((a, 0), L)
    cb = rq((0, b), L)
    Ha22 = (rq((a, dd), L) + rq((a, -dd), L) - 2 * ca) / dd**2
    Hb11 = (rq((dd, b), L) + rq((-dd, b), L) - 2 * cb) / dd**2
    ss = np.linspace(0, np.pi / 2, 181)
    De = np.array([rq((a * np.cos(x), b * np.sin(x)), L) for x in ss]) - C
    off = (ss > 0.05) & (ss < np.pi / 2 - 0.05)
    k = np.argmax(np.where(off, De, -np.inf))
    D0 = rq((0, 0), L) - C
    print(
        f"rp {RP} rm {rm}: X pair a {a:.6f} mass {mx:.8f}; Y pair b {b:.6f} mass {my:.8f}; C {C - L2PE:.12f}; H22 at X {Ha22:+.4e}, "
        f"H11 at Y {Hb11:+.4e}; D45 - C {np.interp(np.pi/4, ss, De):+.3e}; inner-ellipse max D - C off atoms {De[k]:+.3e} at {math.degrees(ss[k]):.1f} deg; D(0) - C {D0:+.3e}; "
        f"residual {res:.1e}",
        flush=True,
    )
