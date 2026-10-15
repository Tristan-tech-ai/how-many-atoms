# q1703 (10 Oct 2026): copy of q1583_ball_rows.py with f22 to f30 and the order NG from the environment Q1703_NG (default 10 = the old law). Made by scratch/make_q1696.py.
"""q1583 (10 Oct 2026, READING, no verdict word; deep proof Q11 step 42, PROOF_TRACKS A step 59): ball-arithmetic enclosure of the first rows of the inverse wall Jacobian, and the sum-norm bound for the first depth.
Reading 85: the first rows of J^-1 (d_1, d_2, d_3, w_1, w_e) are uniform in the block size, the d_1 row is flat at about 25 over the D' rows, and the sum norm of the residual gives |dd_1| <= sum_k |row_k| |F_k| = 9.5e-09 for the residual of reading 77 (q1571), which is 1232 times the actual shift 7.7e-12. Those numbers came from midpoints. A bound for a proof needs enclosures. Here the arb Jacobian J of the block (law through c20, limit pair, 170 bits) is inverted in arb (J.inv()), and the midpoint and the radius of every stored entry of the rows of J^-1 for d_1, d_2, d_3, w_1 and w_e at the equation rows wall, D_k, D'_k with k <= 12 (and the whole d_1 row) are written to q1583_res_N<N>.json. The radius of the Jacobian itself is stored as well. This encloses the inverse of the matrix that was computed; it is not an enclosure of the solution of the nonlinear system (that would need the Newton-Kantorovich argument itself).
Predictions written before the run (taken from the registration text):
 P1 the largest relative radius (radius over |midpoint|) over the stored entries with |midpoint| above 1e-6 of the largest entry of their row is below 1e-25 for N = 10, 20 and 30;
 P2 the sum-norm bound from the whole d_1 row of the N = 30 block, sum over k = 11..30 of (|row_D,k| + radius) |F^D_k| + (|row_D',k| + radius) |F^D'_k|, with the residuals of q1569_res_N10.json, is within 1 percent of 9.53e-09 (q1571);
 P3 the largest radius of any entry of the stored rows is below 1e-20 for N = 30 (absolute);
 P4 the radius of the Jacobian itself is below 1e-40 for every N.
Usage: Q1583_NIN=10 (then 20, 30) py q1583_ball_rows.py run; then py q1583_check.py."""

import math
import json, os, time
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
PI = np.pi
SQ = np.sqrt(2 * PI)
HPHI = 0.5 * np.log(2 * PI * np.e)
phi = lambda t: np.exp(-(t**2) / 2) / SQ
k1 = 1 / (8 * PI**2)
k4 = 1 / (64 * PI**4) - 1 / (96 * PI**2)
c6 = 1 / (512 * PI**6) - 13 / (768 * PI**4)
c8 = 1 / (4096 * PI**8) - 17 / (1536 * PI**6) - 1 / (3072 * PI**4)
c10 = 1 / (32768 * PI**10) - 29 / (6144 * PI**8) - 109 / (368640 * PI**6)
f12 = -37 / (442368 * PI**6) - 617 / (393216 * PI**10) + 1 / (262144 * PI**12) + 229 / (327680 * PI**8)
f14 = (
    -33223 / (123863040 * PI**8)
    - 1393 / (3145728 * PI**12)
    + 1 / (2097152 * PI**14)
    + 3819 / (2621440 * PI**10)
)
f16 = (
    -16507 / (424673280 * PI**8)
    - 43261 / (123863040 * PI**10)
    - 175 / (1572864 * PI**14)
    + 1 / (16777216 * PI**16)
    + 42631 / (31457280 * PI**12)
)
f18 = (
    -54007 / (293601280 * PI**10)
    - 53549 / (165150720 * PI**12)
    - 215 / (8388608 * PI**16)
    + 1 / (134217728 * PI**18)
    + 71767 / (83886080 * PI**14)
)
f20 = (
    -64764727 / (190253629440 * PI**12)
    - 395711 / (13589544960 * PI**10)
    - 11659519 / (31708938240 * PI**14)
    - 2963 / (536870912 * PI**18)
    + 1 / (1073741824 * PI**20)
    + 848207 / (2013265920 * PI**16)
)
f22 = (
    -4717124861 / (25113479086080 * PI**12)
    - 67853183 / (217432719360 * PI**14)
    - 36965837 / (84557168640 * PI**16)
    - 14509 / (12884901888 * PI**20)
    + 1 / (8589934592 * PI**22)
    + 932899 / (5368709120 * PI**18)
)
f24 = (
    -80970037 / (166094438400 * PI**14)
    - 15501869 / (489223618560 * PI**12)
    - 1698780403 / (12176232284160 * PI**16)
    - 212791769 / (507343011840 * PI**18)
    - 5665 / (25769803776 * PI**22)
    + 1 / (68719476736 * PI**24)
    + 5382003 / (85899345920 * PI**20)
)
f26 = (
    -163831023637 / (621857577369600 * PI**14)
    - 442104764597 / (669692775628800 * PI**16)
    - 35634047 / (112742891520 * PI**20)
    - 533 / (12884901888 * PI**24)
    + 1 / (549755813888 * PI**26)
    + 41889419 / (2061584302080 * PI**22)
    + 388433609 / (13915694039040 * PI**18)
)
f28 = (
    -46880553743809 / (52236036499046400 * PI**16)
    - 3433570933 / (73057393704960 * PI**14)
    - 6814687675739 / (12858101292072960 * PI**18)
    - 50505172573 / (259759622062080 * PI**22)
    - 49777 / (6597069766656 * PI**26)
    + 1 / (4398046511104 * PI**28)
    + 299157287 / (49478023249920 * PI**24)
    + 110190843167 / (779278866186240 * PI**20)
)
f30 = (
    -86841758233067 / (179094982282444800 * PI**16)
    - 4097197486478951 / (2507329751954227200 * PI**18)
    - 3181330379107 / (11429423370731520 * PI**20)
    - 211030939429 / (2078076976496640 * PI**24)
    - 70721 / (52776558133248 * PI**28)
    + 1 / (35184372088832 * PI**30)
    + 73677617 / (43980465111040 * PI**26)
    + 1274740338953 / (6234230929489920 * PI**22)
)
F = [1.0, k1, k4, c6, c8, c10, f12, f14, f16, f18, f20, f22, f24, f26, f28, f30]
NG = int(os.environ.get("Q1703_NG", "10"))
g = [1.0]
for n in range(1, NG + 1):
    g.append(-sum(F[j] * g[n - j] for j in range(1, n + 1)))


def dlaw(r):
    return (
        16 * PI**2 * (g[0] * r**3 / 3 + sum(g[n] * r ** (3 - 2 * n) / (3 - 2 * n) for n in range(1, NG + 1)))
    )


def Nlaw(r):
    return (
        16
        * PI**2
        * (
            g[0] * r**4 / 4
            + g[1] * r**2 / 2
            + g[2] * np.log(r)
            + sum(g[n] * r ** (4 - 2 * n) / (4 - 2 * n) for n in range(3, NG + 1))
        )
    )


def dNlaw(r):
    return 16 * PI**2 * r * sum(g[n] * r ** (2 - 2 * n) for n in range(0, NG + 1))


def solve_rho(t):
    r = (np.maximum(t, 1.0) / (4 * PI**2)) ** 0.25 + 0.3
    for _ in range(100):
        r = r - (Nlaw(r) - t) / dNlaw(r)
    return r


NIN = int(os.environ.get("Q1583_NIN", "36"))
NOUT, KC = 70, 45


def fit_pair(dep, k0, k1_):
    k = np.arange(k0, k1_ + 1)

    def cost(nu):
        r = solve_rho(k + nu)
        res = dep[k - 1] - dlaw(r)
        d0 = res.mean()
        return np.sqrt((((res - d0) * r) ** 2).mean()), d0

    lo, hi = -0.3, 0.3
    gg = (np.sqrt(5) - 1) / 2
    a, b = hi - gg * (hi - lo), lo + gg * (hi - lo)
    for _ in range(100):
        if cost(a)[0] < cost(b)[0]:
            hi = b
        else:
            lo = a
        a, b = hi - gg * (hi - lo), lo + gg * (hi - lo)
    nu = 0.5 * (lo + hi)
    return nu, cost(nu)[1]


class Wall:
    def __init__(self, nu, d0, c):
        self.nu, self.d0, self.c = nu, d0, c
        kk = np.arange(NIN + 1, NOUT + 1)
        r = solve_rho(kk + nu)
        self.dout = dlaw(r) + d0
        self.wout = c / r
        self.kk = kk
        self.YS = np.arange(-(self.dout[-1] + 14.0), 15.0, 0.025)

    def trap(self, f):
        return 0.025 * (f.sum(axis=-1) - 0.5 * (f[..., 0] + f[..., -1]))

    def logQ(self, z):
        dd = np.concatenate([z[:NIN], self.dout])
        ww = np.concatenate([z[NIN : 2 * NIN], self.wout])
        Q = z[2 * NIN] * phi(self.YS) + (ww[None, :] * phi(self.YS[:, None] + dd[None, :])).sum(axis=1)
        return np.log(Q), dd

    def DD(self, x, lq):
        t = self.YS - x
        return -HPHI - self.trap(phi(t) * lq), -self.trap(t * phi(t) * lq)

    def resid(self, z):
        lq, dd = self.logQ(z)
        C = self.DD(-dd[KC - 1], lq)[0]
        r = np.zeros(2 * NIN + 1)
        r[0] = self.DD(0.0, lq)[0] - C
        for i in range(NIN):
            Dv, Dp = self.DD(-dd[i], lq)
            r[1 + i] = Dv - C
            r[1 + NIN + i] = Dp
        return r

    def jac(self, z):
        n = len(z)
        J = np.zeros((n, n))
        for j in range(n):
            h = 1e-6 * max(abs(z[j]), 1e-3)
            zp = z.copy()
            zm = z.copy()
            zp[j] += h
            zm[j] -= h
            J[:, j] = (self.resid(zp) - self.resid(zm)) / (2 * h)
        return J

    def solve(self, z0, maxit=30):
        z = z0.copy()
        r = self.resid(z)
        nr = np.abs(r).max()
        it = 0
        while nr > 1e-13 and it < maxit:
            step = np.linalg.solve(self.jac(z), r)
            lam = 1.0
            while lam > 1e-4:
                zn = z - lam * step
                if np.all(zn[NIN:] > 0) and np.all(np.diff(zn[:NIN]) > 0):
                    rn = self.resid(zn)
                    nn = np.abs(rn).max()
                    if nn < nr or lam < 2e-4:
                        break
                lam *= 0.5
            z, r, nr = zn, rn, nn
            it += 1
        return z, it, nr


def start(nu, d0, c):
    k = np.arange(1, NIN + 1)
    r = solve_rho(k + nu)
    return np.concatenate([dlaw(r) + d0, c / r, [3.5 * c]])


def run(c, nu=-0.09, d0=0.0, rounds=14, verbose=True):
    z = None
    hist = []
    for rd in range(1, rounds + 1):
        W = Wall(nu, d0, c)
        z0 = start(nu, d0, c) if z is None else z
        z, it, nr = W.solve(z0)
        nu_n, d0_n = fit_pair(z[:NIN], 6, 12)
        hist.append((rd, nu, d0, it, nr, nu_n - nu, d0_n - d0))
        if verbose:
            print(
                f"  round {rd}: pair in ({nu:+.9f}, {d0:+.9f}); Newton {it} iterations, residual {nr:.1e}; refit changes {nu_n - nu:+.2e}, {d0_n - d0:+.2e}",
                flush=True,
            )
        done = abs(nu_n - nu) < 1e-10 and abs(d0_n - d0) < 1e-10
        nu, d0 = nu_n, d0_n
        if done:
            break
    return z, nu, d0, hist


def delta_formula(wall_over_c, wk_over_c, nu, d0, n):
    k = np.arange(NIN + 1, n + 1)
    tail = (1 / solve_rho(k + nu)).sum()
    edge = dlaw(solve_rho(n + 0.5 + nu)) + d0
    return wall_over_c + wk_over_c[:NIN].sum() + tail - edge


from flint import arb, ctx

ctx.prec = 170
DELT = 1.990803973543
D1 = 0.286017905
PIA = arb.pi()
Hh = arb(1) / 40
k1a = 1 / (8 * PIA**2)
k4a = 1 / (64 * PIA**4) - 1 / (96 * PIA**2)
c6a = 1 / (512 * PIA**6) - 13 / (768 * PIA**4)
c8a = 1 / (4096 * PIA**8) - 17 / (1536 * PIA**6) - 1 / (3072 * PIA**4)
c10a = 1 / (32768 * PIA**10) - 29 / (6144 * PIA**8) - 109 / (368640 * PIA**6)
f12a = -37 / (442368 * PIA**6) - 617 / (393216 * PIA**10) + 1 / (262144 * PIA**12) + 229 / (327680 * PIA**8)
f14a = (
    -33223 / (123863040 * PIA**8)
    - 1393 / (3145728 * PIA**12)
    + 1 / (2097152 * PIA**14)
    + 3819 / (2621440 * PIA**10)
)
f16a = (
    -16507 / (424673280 * PIA**8)
    - 43261 / (123863040 * PIA**10)
    - 175 / (1572864 * PIA**14)
    + 1 / (16777216 * PIA**16)
    + 42631 / (31457280 * PIA**12)
)
f18a = (
    -54007 / (293601280 * PIA**10)
    - 53549 / (165150720 * PIA**12)
    - 215 / (8388608 * PIA**16)
    + 1 / (134217728 * PIA**18)
    + 71767 / (83886080 * PIA**14)
)
f20a = (
    -64764727 / (190253629440 * PIA**12)
    - 395711 / (13589544960 * PIA**10)
    - 11659519 / (31708938240 * PIA**14)
    - 2963 / (536870912 * PIA**18)
    + 1 / (1073741824 * PIA**20)
    + 848207 / (2013265920 * PIA**16)
)
f22a = (
    -4717124861 / (25113479086080 * PIA**12)
    - 67853183 / (217432719360 * PIA**14)
    - 36965837 / (84557168640 * PIA**16)
    - 14509 / (12884901888 * PIA**20)
    + 1 / (8589934592 * PIA**22)
    + 932899 / (5368709120 * PIA**18)
)
f24a = (
    -80970037 / (166094438400 * PIA**14)
    - 15501869 / (489223618560 * PIA**12)
    - 1698780403 / (12176232284160 * PIA**16)
    - 212791769 / (507343011840 * PIA**18)
    - 5665 / (25769803776 * PIA**22)
    + 1 / (68719476736 * PIA**24)
    + 5382003 / (85899345920 * PIA**20)
)
f26a = (
    -163831023637 / (621857577369600 * PIA**14)
    - 442104764597 / (669692775628800 * PIA**16)
    - 35634047 / (112742891520 * PIA**20)
    - 533 / (12884901888 * PIA**24)
    + 1 / (549755813888 * PIA**26)
    + 41889419 / (2061584302080 * PIA**22)
    + 388433609 / (13915694039040 * PIA**18)
)
f28a = (
    -46880553743809 / (52236036499046400 * PIA**16)
    - 3433570933 / (73057393704960 * PIA**14)
    - 6814687675739 / (12858101292072960 * PIA**18)
    - 50505172573 / (259759622062080 * PIA**22)
    - 49777 / (6597069766656 * PIA**26)
    + 1 / (4398046511104 * PIA**28)
    + 299157287 / (49478023249920 * PIA**24)
    + 110190843167 / (779278866186240 * PIA**20)
)
f30a = (
    -86841758233067 / (179094982282444800 * PIA**16)
    - 4097197486478951 / (2507329751954227200 * PIA**18)
    - 3181330379107 / (11429423370731520 * PIA**20)
    - 211030939429 / (2078076976496640 * PIA**24)
    - 70721 / (52776558133248 * PIA**28)
    + 1 / (35184372088832 * PIA**30)
    + 73677617 / (43980465111040 * PIA**26)
    + 1274740338953 / (6234230929489920 * PIA**22)
)
Fa = [arb(1), k1a, k4a, c6a, c8a, c10a, f12a, f14a, f16a, f18a, f20a, f22a, f24a, f26a, f28a, f30a]
ga = [arb(1)]
for n in range(1, NG + 1):
    ga.append(-sum((Fa[j] * ga[n - j] for j in range(1, n + 1)), arb(0)))


def dlaw_a(r):
    return (
        16
        * PIA**2
        * (ga[0] * r**3 / 3 + sum((ga[n] * r ** (3 - 2 * n) / (3 - 2 * n) for n in range(1, NG + 1)), arb(0)))
    )


def Nlaw_a(r):
    return (
        16
        * PIA**2
        * (
            ga[0] * r**4 / 4
            + ga[1] * r**2 / 2
            + ga[2] * r.log()
            + sum((ga[n] * r ** (4 - 2 * n) / (4 - 2 * n) for n in range(3, NG + 1)), arb(0))
        )
    )


def dNlaw_a(r):
    return 16 * PIA**2 * r * sum((ga[n] * r ** (2 - 2 * n) for n in range(0, NG + 1)), arb(0))


def solve_rho_a(target, rho0):
    r = arb(float(rho0))
    for _ in range(8):
        r = (r - (Nlaw_a(r) - target) / dNlaw_a(r)).mid()
    return r


HPHI_A = (2 * PIA * arb(1).exp()).log() / 2
SQ_A = (2 * PIA).sqrt()


def phi_a(t):
    return (-(t * t) / 2).exp() / SQ_A


class WallA:
    def __init__(self, nu, d0, c):
        self.nu, self.d0, self.c = nu, d0, c
        kk = np.arange(NIN + 1, NOUT + 1)
        r0 = solve_rho(kk + float(nu))
        nua = arb(repr(float(nu))) if not isinstance(nu, arb) else nu
        d0a = arb(repr(float(d0))) if not isinstance(d0, arb) else d0
        self.rho = [solve_rho_a(arb(int(k)) + nua, r0[i]) for i, k in enumerate(kk)]
        self.dout = [dlaw_a(r) + d0a for r in self.rho]
        self.wout = [arb(repr(float(c))) / r for r in self.rho]
        ymin = -(float(self.dout[-1]) + 14.0)
        n = int((15.0 - ymin) / 0.025) + 1
        self.y = [arb(repr(ymin)) + Hh * i for i in range(n)]
        self.n = n
        Qo = [arb(0)] * n
        for dk, wk in zip(self.dout, self.wout):
            x = -dk
            Qo = [q + wk * phi_a(yy - x) for q, yy in zip(Qo, self.y)]
        self.Qo = Qo
        xc = -self.dout[KC - NIN - 1]
        self.Pc = [phi_a(yy - xc) for yy in self.y]
        self.xc = xc

    def resid(self, z):
        N = NIN
        y = self.y
        n = self.n
        dd = z[:N]
        ww = z[N : 2 * N]
        we = z[2 * N]
        P = [[phi_a(yy + d) for yy in y] for d in dd]
        P0 = [phi_a(yy) for yy in y]
        Q = [self.Qo[i] + we * P0[i] + sum((ww[k] * P[k][i] for k in range(N)), arb(0)) for i in range(n)]
        lq = [q.log() for q in Q]

        def DD(Pk, x):
            s = sum((p * l for p, l in zip(Pk, lq)), arb(0))
            s2 = sum(((yy - x) * p * l for yy, p, l in zip(y, Pk, lq)), arb(0))
            return -HPHI_A - Hh * s, -Hh * s2

        C = DD(self.Pc, self.xc)[0]
        r = [DD(P0, arb(0))[0] - C]
        Dp = []
        for k in range(N):
            Dv, Dpv = DD(P[k], -dd[k])
            r.append(Dv - C)
            Dp.append(Dpv)
        return r + Dp, lq, C

    def Dprime_wall(self, z):
        N = NIN
        y = self.y
        n = self.n
        dd = z[:N]
        ww = z[N : 2 * N]
        we = z[2 * N]
        P = [[phi_a(yy + d) for yy in y] for d in dd]
        P0 = [phi_a(yy) for yy in y]
        Q = [self.Qo[i] + we * P0[i] + sum((ww[k] * P[k][i] for k in range(N)), arb(0)) for i in range(n)]
        lq = [q.log() for q in Q]
        return -Hh * sum((yy * p * l for yy, p, l in zip(y, P0, lq)), arb(0))


from flint import arb_mat


class WallB(WallA):
    def parts(self, z):
        N = NIN
        y = self.y
        n = self.n
        dd = z[:N]
        ww = z[N : 2 * N]
        we = z[2 * N]
        P = [[phi_a(yy + d) for yy in y] for d in dd]
        P0 = [phi_a(yy) for yy in y]
        Q = [self.Qo[i] + we * P0[i] + sum((ww[k] * P[k][i] for k in range(N)), arb(0)) for i in range(n)]
        return dd, ww, we, P, P0, Q

    def jac(self, z):
        N = NIN
        y = self.y
        n = self.n
        H = Hh
        dd, ww, we, P, P0, Q = self.parts(z)
        lq = [q.log() for q in Q]
        Qi = [1 / q for q in Q]
        cols = (
            [[-ww[k] * (y[i] + dd[k]) * P[k][i] for i in range(n)] for k in range(N)]
            + [P[k] for k in range(N)]
            + [P0]
        )
        S = arb_mat(n, 2 * N + 1, [cols[c][i] for i in range(n) for c in range(2 * N + 1)])
        RD = (
            [[P0[i] * Qi[i] for i in range(n)]]
            + [[P[k][i] * Qi[i] for i in range(n)] for k in range(N)]
            + [[self.Pc[i] * Qi[i] for i in range(n)]]
        )
        RDp = [[(y[i] + dd[k]) * P[k][i] * Qi[i] for i in range(n)] for k in range(N)]
        R = arb_mat(len(RD) + len(RDp), n, [v for row in RD + RDp for v in row])
        G = R * S
        Dp = []
        Dpp = []
        for k in range(N):
            Dp.append(-H * sum(((y[i] + dd[k]) * P[k][i] * lq[i] for i in range(n)), arb(0)))
            Dpp.append(-H * sum((((y[i] + dd[k]) ** 2 - 1) * P[k][i] * lq[i] for i in range(n)), arb(0)))
        nc = 2 * N + 1
        J = arb_mat(nc, nc)
        for c in range(nc):
            J[0, c] = -H * G[0, c] + H * G[N + 1, c]
            for k in range(N):
                J[1 + k, c] = -H * G[1 + k, c] + H * G[N + 1, c]
                J[N + 1 + k, c] = -H * G[N + 2 + k, c]
        for k in range(N):
            J[1 + k, k] = J[1 + k, k] - Dp[k]
            J[N + 1 + k, k] = J[N + 1 + k, k] - Dpp[k]
        return J


def newton_arb(WA, z0, tol=1e-40, maxit=8):
    z = [arb(repr(float(t))) for t in z0]
    n = len(z)
    hist = []
    for it in range(maxit):
        r, lq, C = WA.resid(z)
        rn = max(abs(float(t)) for t in r)
        hist.append(rn)
        if rn < tol:
            break
        J = WA.jac(z)
        dz = J.solve(arb_mat(n, 1, r))
        z = [z[i] - dz[i, 0].mid() for i in range(n)]
    return z, hist


DELT = 1.990803973543


def fprime(N):
    r = solve_rho(N)
    return -1.0 / (dNlaw(r) * r**2)


def f3(N, h=0.25):
    f = lambda x: 1.0 / solve_rho(x)
    return (f(N + 2 * h) - 2 * f(N + h) + 2 * f(N - h) - f(N - 2 * h)) / (2 * h**3)


def run_N():
    c = 0.0065
    nu, d0 = -0.093773586, 0.001159841
    Wd = Wall(nu, d0, c)
    zd, itd, nrd = Wd.solve(start(nu, d0, c))
    WA = WallB(nu, d0, c)
    z, hist = newton_arb(WA, zd)
    N = NIN
    out = dict(NIN=N, hist=hist[-1], depths=[z[k].str(30, radius=False) for k in range(min(N, 8))])
    dep = np.array([float(t) for t in z[:N]] + [float(t) for t in WA.dout])
    mas = np.array([float(t) for t in z[N : 2 * N]] + [float(t) for t in WA.wout])
    we = float(z[2 * N])
    y = np.array([float(t) for t in WA.y])
    H = 0.025
    Q = we * np.exp(-(y**2) / 2) / np.sqrt(2 * np.pi)
    for d_, w_ in zip(dep, mas):
        Q = Q + w_ * np.exp(-((y + d_) ** 2) / 2) / np.sqrt(2 * np.pi)
    lq = np.log(Q)
    hphi = 0.5 * np.log(2 * np.pi * np.e)

    def DD(x):
        t = y - x
        Pk = np.exp(-(t**2) / 2) / np.sqrt(2 * np.pi)
        return -hphi - H * (Pk * lq).sum(), -H * (t * Pk * lq).sum()

    C = DD(-dep[KC - 1])[0]
    FD = []
    FDp = []
    for k in range(1, 41):
        Dv, Dp = DD(-dep[k - 1])
        FD.append(Dv - C)
        FDp.append(Dp)
    out["FD"] = FD
    out["FDp"] = FDp
    if N >= 10:
        J = WA.jac(z)
        Ji = J.inv()
        out["row_D"] = [float(Ji[0, k]) for k in range(1, N + 1)]
        out["row_Dp"] = [float(Ji[0, N + k]) for k in range(1, N + 1)]
        nn = 2 * N + 1
        wk_ = [float(z[N + k]) for k in range(N)]
        out["JTu_flat"] = [float(sum(J[N + 1 + k, c] for k in range(N))) for c in range(nn)]
        out["JTu_w"] = [float(sum(wk_[k] * J[N + 1 + k, c] for k in range(N))) for c in range(nn)]
        out["Jcol_Dp_max"] = [float(max(abs(J[N + 1 + k, c]) for k in range(N))) for c in range(nn)]
        out["wk"] = wk_
        out["dk"] = [float(z[k]) for k in range(N)]
        out["we"] = float(z[2 * N])
        out["Jdiag_Dp"] = [float(J[N + 1 + k, k]) for k in range(N)]
        out["Jfull"] = [[J[a_, b_].str(40, radius=False) for b_ in range(nn)] for a_ in range(nn)]
        cols_eq = [0] + list(range(1, min(12, N) + 1)) + [N + k for k in range(1, min(12, N) + 1)]
        out["ball_eq"] = cols_eq
        out["ball_rows"] = {
            str(i_): [[Ji[i_, j_].mid().str(30, radius=False), float(Ji[i_, j_].rad())] for j_ in cols_eq]
            for i_ in (0, 1, 2, N, 2 * N)
        }
        out["ball_d1_all"] = [
            [Ji[0, j_].mid().str(30, radius=False), float(Ji[0, j_].rad())] for j_ in range(nn)
        ]
        out["Jrad_max"] = max(float(J[a_, b_].rad()) for a_ in range(nn) for b_ in range(nn))
    json.dump(out, open(f"q1583_res_N{N}.json", "w"), indent=1)
    print("done N", N, "final residual", hist[-1], flush=True)


def compare():
    from mpmath import mp, mpf

    mp.dps = 40
    a = json.load(open("q1583_res_N4.json"))
    b = json.load(open("q1583_res_N30.json"))
    FD = np.array(a["FD"])
    FDp = np.array(a["FDp"])
    ks = np.arange(1, 41)
    sel = (ks >= 5) & (ks <= 30)
    print("k, F^D_k, F^D'_k in the N = 4 configuration (every 3rd, k = 5..29):")
    for k in range(5, 30, 3):
        print(f"  {k}  {FD[k - 1]:+.3e}  {FDp[k - 1]:+.3e}")
    cf = np.polyfit(np.log(ks[sel].astype(float)), np.log(np.abs(FDp[sel])), 1)
    scatter = float(np.abs(np.log(np.abs(FDp[sel])) - np.polyval(cf, np.log(ks[sel].astype(float)))).max())
    sum_abs = float(np.abs(FDp[sel]).sum())
    print(
        f"max |F^D'| (k=5..30) = {np.abs(FDp[sel]).max():.3e}; max |F^D| = {np.abs(FD[sel]).max():.3e}; sum |F^D'| = {sum_abs:.3e}; 25 x sum = {25 * sum_abs:.3e}; power law k^{cf[0]:.2f}, largest log scatter {scatter:.2f}"
    )
    rD = np.array(b["row_D"])
    rDp = np.array(b["row_Dp"])
    pred = -sum(rD[k - 1] * FD[k - 1] + rDp[k - 1] * FDp[k - 1] for k in range(5, 31))
    actual = float(mpf(a["depths"][0]) - mpf(b["depths"][0]))
    print(
        f"actual d_1(N=4) - d_1(N=30) = {actual:+.4e}; predicted by the linear response = {pred:+.4e}; ratio {actual / pred if pred else float('nan'):.3f}"
    )
    p1 = np.abs(FDp[sel]).max() < 1e-5 and scatter < np.log(1.3)
    p2 = 25 * sum_abs < 1e-3
    p3 = bool(pred != 0 and abs(actual / pred - 1) < 0.30)
    p4 = np.abs(FDp[sel]).max() > 1e-9
    print(
        f"P1 |F^D'| below 1e-5 and a smooth power law: {'holds' if p1 else 'does not hold'}; P2 25 x sum below 1e-3: {'holds' if p2 else 'does not hold'}; P3 linear response within 30 percent: {'holds' if p3 else 'does not hold'}; P4 intermediate zone larger than 1e-9: {'holds' if p4 else 'does not hold'}"
    )
    print(
        f"for reference, q1549 (law through c10, instrument pair of reading 64): max |F^D'| 7.694e-08, max |F^D| 1.789e-07, actual d_1(N=4) - d_1(N=30) = -1.7824e-07"
    )
    p1n = np.abs(FDp[sel]).max() < 2.5e-8 and np.abs(FDp[(ks >= 17) & (ks <= 30)]).max() < 1.2e-11
    p2n = abs(actual) < 6e-8
    p3n = bool(pred != 0 and abs(abs(actual / pred) - 1) < 0.30)
    p4n = -14 <= cf[0] <= -9
    print(
        f"P1 max residual below 2.5e-8 and beyond k = 17 below 1.2e-11: {'holds' if p1n else 'does not hold'}; P2 first-depth shift below 6e-8: {'holds' if p2n else 'does not hold'}; P3 linear response within 30 percent in magnitude: {'holds' if p3n else 'does not hold'}; P4 power-law exponent between -14 and -9: {'holds' if p4n else 'does not hold'}"
    )
    print("READING q1569 done (labels by the script; no verdict word)")


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["compare"]:
        compare()
    else:
        run_N()
