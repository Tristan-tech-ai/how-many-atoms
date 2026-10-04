"""quadrature_bounds: rigorous ball-arithmetic evaluation of the capacity KKT function on the ellipse curve.

Channel Y = X + W, W ~ N(0, I_2); input = D2-symmetric ring on x(t) = (rp cos t, rm sin t) (orbits X = {0, pi}, Y = {pi/2, 3pi/2},
G = {a, -a, pi - a, pi + a}; unknowns v = [G angles] + [orbit masses X, Y, G...] + [C], the layout used throughout).
KKT function: D(q) = |q|^2/2 - I(q),  I(q) = int L(y) phi(y - q) dy,  L = log S,  S(y) = sum_k w_k exp(<y, x_k> - |x_k|^2/2)
(the relative entropy D(P_Y|X=x || P_Y) in nats).
Quadrature: global trapezoid grid y = (n h, l h), |n|, |l| <= N, shared by all query points:
  Q(q) = h^2 sum_{n,l} Lam_nl phi1(nh - q1) phi1(lh - q2),  Lam_nl = L(nh, lh).
Rigorous error bounds (see quad_err): discretisation (Poisson summation + strip of half-width a in each coordinate, valid while
a*rp < pi/2 so that S has no zero there) and truncation (Gaussian tails). Taylor coefficients in t of Q(x(t0 + s)) by contracting
the grid with power-series weights. No top-level computation; nothing is written to disk.
"""

import math
from flint import arb, arb_mat, arb_series, ctx

SQ2PI = None


def setprec(bits, cap=None, threads=None):
    global SQ2PI
    ctx.prec = bits
    if cap is not None:
        ctx.cap = cap
    if threads is not None:
        ctx.threads = threads
    SQ2PI = (2 * arb.pi()).sqrt()


def A(x):
    return x if isinstance(x, arb) else arb(x)


class Geo:
    """ellipse ring geometry (rp >= rm)."""

    def __init__(self, rp, rm, X, Y, nG):
        self.rp = A(rp)
        self.rm = A(rm)
        self.X = bool(X)
        self.Y = bool(Y)
        self.nG = nG
        self.nO = int(self.X) + int(self.Y) + nG

    def pos(self, t):
        return (self.rp * t.cos(), self.rm * t.sin())

    def dpos(self, t):
        return (-self.rp * t.sin(), self.rm * t.cos())

    def atoms(self, G, m):
        """list of (theta, weight, orbit index, dtheta/d(orbit angle) or 0)."""
        pi = arb.pi()
        out = []
        k = 0
        if self.X:
            out += [(arb(0), m[k], k, 0), (pi, m[k], k, 0)]
            k += 1
        if self.Y:
            out += [(pi / 2, m[k], k, 0), (3 * pi / 2, m[k], k, 0)]
            k += 1
        for i, a in enumerate(G):
            for th, sg in ((a, 1), (-a, -1), (pi - a, -1), (pi + a, 1)):
                out.append((th, m[k + i], k + i, sg))
        return out

    def reps(self, G):
        return ([arb(0)] if self.X else []) + ([arb.pi() / 2] if self.Y else []) + list(G)

    def split(self, v):
        nG, nO = self.nG, self.nO
        return list(v[:nG]), list(v[nG : nG + nO]), v[nG + nO]


class Grid:
    def __init__(self, h, N):
        self.h = A(h)
        self.N = N
        self.M = 2 * N + 1
        self.y = [n * self.h for n in range(-N, N + 1)]
        self.Y = N * self.h


# ---------------------------------------------------------------- grid functions
def grid_S(grid, geo, atoms):
    """Ex (M x K, with weights), Ey (K x M), and the raw factors for derivatives."""
    M = grid.M
    K = len(atoms)
    ys = grid.y
    xs = [geo.pos(th) for th, _, _, _ in atoms]
    w = [a[1] for a in atoms]
    c = [-(x1 * x1 + x2 * x2) / 2 for x1, x2 in xs]
    E1 = [[(ys[n] * xs[k][0] + c[k]).exp() for k in range(K)] for n in range(M)]  # without weights
    E2 = [[(ys[l] * xs[k][1]).exp() for l in range(M)] for k in range(K)]
    return xs, w, E1, E2


def lam_grid(grid, geo, atoms, keepS=False):
    M = grid.M
    K = len(atoms)
    xs, w, E1, E2 = grid_S(grid, geo, atoms)
    Ex = arb_mat(M, K, [E1[n][k] * w[k] for n in range(M) for k in range(K)])
    Ey = arb_mat(K, M, [E2[k][l] for k in range(K) for l in range(M)])
    S = (Ex * Ey).entries()
    Lam = arb_mat(M, M, [s.log() for s in S])
    if keepS:
        return Lam, (xs, w, E1, E2, S)
    return Lam


def deriv_grids(grid, geo, atoms, G, aux):
    """dL/dtheta_g (g = 0..nG-1) then dL/dm_o (o = 0..nO-1), as a generator of (name, arb_mat)."""
    xs, w, E1, E2, S = aux
    M = grid.M
    ys = grid.y
    nG, nO = geo.nG, geo.nO
    invS = [1 / s for s in S]
    for g in range(nG):
        ks = [k for k, at in enumerate(atoms) if at[3] != 0 and at[2] == (nO - nG) + g]
        dx = [geo.dpos(atoms[k][0]) for k in ks]
        dx = [(atoms[k][3] * d[0], atoms[k][3] * d[1]) for k, d in zip(ks, dx)]
        K4 = len(ks)
        Ea = arb_mat(
            M, K4, [E1[n][k] * w[k] * (ys[n] - xs[k][0]) * d[0] for n in range(M) for k, d in zip(ks, dx)]
        )
        Eb = arb_mat(K4, M, [E2[k][l] for k in ks for l in range(M)])
        Ec = arb_mat(M, K4, [E1[n][k] * w[k] for n in range(M) for k in ks])
        Ed = arb_mat(K4, M, [E2[k][l] * (ys[l] - xs[k][1]) * d[1] for k, d in zip(ks, dx) for l in range(M)])
        dS = (Ea * Eb + Ec * Ed).entries()
        yield ("th%d" % g, arb_mat(M, M, [a * b for a, b in zip(dS, invS)]))
    for o in range(nO):
        ks = [k for k, at in enumerate(atoms) if at[2] == o]
        K4 = len(ks)
        Ea = arb_mat(M, K4, [E1[n][k] for n in range(M) for k in ks])
        Eb = arb_mat(K4, M, [E2[k][l] for k in ks for l in range(M)])
        dS = (Ea * Eb).entries()
        yield ("m%d" % o, arb_mat(M, M, [a * b for a, b in zip(dS, invS)]))


# ---------------------------------------------------------------- Taylor contraction along the curve
def weights(grid, geo, t0, m):
    """A ((m+1) x M) and B (M x (m+1)): Taylor coefficients in s of phi1(y_n - rp cos(t0+s)), phi1(y_l - rm sin(t0+s))."""
    ctx.cap = m + 1
    s = arb_series([t0, arb(1)])
    cs = s.cos()
    sn = s.sin()
    u1 = cs * geo.rp
    u2 = sn * geo.rm
    ys = grid.y
    M = grid.M
    A_ = []
    B_ = []
    for n in range(M):
        d = u1 - ys[n]
        e = ((d * d) * arb(-0.5)).exp()
        cf = e.coeffs()
        cf = cf + [arb(0)] * (m + 1 - len(cf))
        A_.append(cf)
        d = u2 - ys[n]
        e = ((d * d) * arb(-0.5)).exp()
        cf = e.coeffs()
        cf = cf + [arb(0)] * (m + 1 - len(cf))
        B_.append(cf)
    Am = arb_mat(m + 1, M, [A_[n][i] for i in range(m + 1) for n in range(M)])
    Bm = arb_mat(M, m + 1, [B_[l][j] for l in range(M) for j in range(m + 1)])
    return Am, Bm


def contract(grid, G, Am, Bm, m):
    """Taylor coefficients c_0..c_m of h^2 sum G_nl phi1 phi1 along the curve (phi1 includes 1/sqrt(2 pi))."""
    P = Am * (G * Bm)
    f = grid.h * grid.h / (SQ2PI * SQ2PI)
    return [f * sum((P[i, k - i] for i in range(k + 1)), arb(0)) for k in range(m + 1)]


def contract_multi(grid, G, WB, m):
    """WB: list of (Am, Bm) for several expansion points; one product G * [B...]."""
    M = grid.M
    nb = len(WB)
    Bbig = arb_mat(
        M, nb * (m + 1), [WB[b][1][l, j] for l in range(M) for b in range(nb) for j in range(m + 1)]
    )
    GB = G * Bbig
    f = grid.h * grid.h / (SQ2PI * SQ2PI)
    out = []
    for b in range(nb):
        Am = WB[b][0]
        sub = arb_mat(M, m + 1, [GB[l, b * (m + 1) + j] for l in range(M) for j in range(m + 1)])
        P = Am * sub
        out.append([f * sum((P[i, k - i] for i in range(k + 1)), arb(0)) for k in range(m + 1)])
    return out


def half_norm2_series(geo, t0, m):
    ctx.cap = m + 1
    s = arb_series([t0, arb(1)])
    c = s.cos()
    sn = s.sin()
    e = (c * c * (geo.rp * geo.rp) + sn * sn * (geo.rm * geo.rm)) * arb(0.5)
    cf = e.coeffs()
    return cf + [arb(0)] * (m + 1 - len(cf))


# ---------------------------------------------------------------- rigorous error bounds
def quad_err(Aa, Bb, qr1, qr2, sig, a, h, Yc):
    """|I[G](q) - Q_N[G](q)| for complex q = qr + i qi with |qr_j| <= qr_j, |qi_j| <= sig, when the grid function G satisfies
    |G(y + i b e_j)| <= Aa + Bb(|y1| + |y2|) for real y, |b| < a (and on the real grid). Returns an upper bound (arb, mid only).
    """
    Aa, Bb, qr1, qr2, sig, a, h, Yc = [A(x) for x in (Aa, Bb, qr1, qr2, sig, a, h, Yc)]
    pi = arb.pi()
    eh = 3 * (-2 * pi * pi / (h * h)).exp()
    E = ((a + sig) ** 2 / 2 + sig * sig / 2).exp()
    disc = (
        4
        * E
        * (1 + eh)
        * (Aa + Bb * (qr1 + qr2 + arb("1.6") + arb("0.484") * h))
        / ((2 * pi * a / h).exp() - 1)
    )
    psi = lambda u: (-(u * u) / 2).exp() / (2 * pi).sqrt()

    def T1(c):
        return 2 * psi(Yc - c) * ((Aa + Bb * c) / (Yc - c) + Bb)

    def T0(c):
        return 2 * psi(Yc - c) / (Yc - c)

    assert float((Yc - qr1).mid()) >= 2 and float((Yc - qr2).mid()) >= 2
    full1 = lambda c: (1 + eh) * c + arb("0.7979") + arb("0.484") * h
    tr = (sig * sig).exp() * (
        T1(qr1) * (1 + eh) + T0(qr1) * Bb * full1(qr2) + T1(qr2) * (1 + eh) + T0(qr2) * Bb * full1(qr1)
    )
    return (disc + tr).upper()


def strip_consts(geo, a, msum_dev=0, mmin=None):
    """(A, B) strip bounds for L, dL/dtheta, dL/dm (derivation in the module docstring and in Section IV-A of the paper)."""
    rp = geo.rp
    a = A(a)
    beta = a * rp
    cb = beta.cos()
    assert float(beta.mid()) < math.pi / 2
    AL = rp * rp / 2 + abs(cb.log()) + beta + 2 * A(msum_dev)
    BL = rp
    Ath = rp * (a + rp) / cb
    Bth = rp / cb
    Am_ = 1 / (A(mmin) * cb) if mmin is not None else None
    return (AL, BL), (Ath, Bth), (Am_, arb(0))


def cauchy_coef_err(geo, AB, k, a, h, Yc, rho):
    """bound on the k-th Taylor coefficient (in t) of I[G](x(t)) - Q_N[G](x(t)) at real t: E_disc(rho)/rho^k (k >= 1),
    real-point bound for k = 0."""
    Aa, Bb = AB
    rho = A(rho)
    if k == 0:
        return quad_err(Aa, Bb, geo.rp, geo.rm, 0, a, h, Yc)
    ch = rho.cosh()
    sh = rho.sinh()
    E = quad_err(Aa, Bb, geo.rp * ch, geo.rp * ch, geo.rp * sh, a, h, Yc)
    return E / rho**k


def taylor_remainder_M(geo, AL, BL, rho, Yc):
    """sup over |tau - t0| <= rho of |q(tau).q(tau)/2| + |Q_N[L](x(tau))| (bounds the Taylor remainder of F = |x|^2/2 - Q)."""
    rho = A(rho)
    ch = rho.cosh()
    sh = rho.sinh()
    rp = geo.rp
    sig = rp * sh
    qr = rp * ch
    # |phi1(y - q)| = phi1(y - Re q) e^{(Im q)^2/2}; trapezoid sums of phi1 <= 1 + tiny, of phi1 |y| <= |c| + 0.7979 + 0.484 h <= |c| + 0.86 (h <= 1/8)
    Qb = (sig * sig).exp() * (AL + BL * (2 * qr + 2 * arb("0.86"))) * arb("1.0001")
    half = (rp * rp * (ch * ch + sh * sh) + rp * rp * (ch * ch + sh * sh)) / 2
    return (Qb + half).upper()
