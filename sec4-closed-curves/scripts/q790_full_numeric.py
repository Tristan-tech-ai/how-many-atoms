"""q790 (Amendment 1283): the full formal smooth density on the real ellipse (all orders in eps), by Newton on its Fourier
coefficients. Density f(t) = 1 + sum_{m=1}^{M} c_2m cos(2mt) on z(t) = (rp cos t, rm sin t), weight dt/2pi; KKT function of FUNC (cap,
lfp, rd as q784) evaluated on the same curve; unknowns c_2..c_2M make its harmonics 2..2M vanish. Plane quadrature as q784 (Gauss-Legendre
in rho x trapezoid in psi), t and s on one trapezoid grid of NA nodes. Jacobian by probes (exact up to the Newton nonlinearity).
Prints c_2m and, for the ring count K = 2m, the ratio c_K/2 (the comb rule's test quantity in its full form).
Usage: py q790_full_numeric.py FUNC RP EPS M [DEG] [NA] [RMAX] [PREC]"""

import sys, time, os
import mpmath as mp
from mpmath.calculus.quadrature import GaussLegendre
from flint import arb, arb_mat, ctx

FUNC = sys.argv[1]
M = int(sys.argv[4])
DEG = int(sys.argv[5]) if len(sys.argv) > 5 else 6
NA = int(sys.argv[6]) if len(sys.argv) > 6 else 96
RMAX = sys.argv[7] if len(sys.argv) > 7 else "20"
ctx.prec = int(sys.argv[8]) if len(sys.argv) > 8 else 200
mp.mp.dps = int(ctx.prec * 0.30103) + 10
DS = mp.mp.dps - 5
rp = arb(sys.argv[2])
eps = arb(sys.argv[3])
rm = rp - eps
pi = arb.pi()
T0 = time.time()
say = lambda *a: print(*a, flush=True)
nodes = GaussLegendre(mp.mp).calc_nodes(DEG, mp.mp.prec)
RMmp = mp.mpf(RMAX)
rho = [arb(mp.nstr((x + 1) * RMmp / 2, DS)) for x, w in nodes]
wr = [arb(mp.nstr(w * RMmp / 2, DS)) for x, w in nodes]
NR = len(rho)
ang = [2 * pi * j / NA for j in range(NA)]
ca = [a.cos() for a in ang]
sa = [a.sin() for a in ang]
LAM = arb(
    os.environ.get("LAM", "0")
)  # 1302: mixed curve z = R e^(it) + eta (e^(-it) + LAM e^(3it)); 0 = ellipse
R_ = rp - eps / 2
et_ = eps / (2 * (1 + LAM))
c3 = [(3 * a).cos() for a in ang]
s3 = [(3 * a).sin() for a in ang]
zx = [(R_ + et_) * ca[l] + LAM * et_ * c3[l] for l in range(NA)]
zy = [(R_ - et_) * sa[l] + LAM * et_ * s3[l] for l in range(NA)]
if (
    os.environ.get("SHAPE", "") == "polar"
):  # 1313: polar n-fold curve r(t) = rp - eps (1 - cos(NF t))/2, t the polar angle
    NFp = int(os.environ.get("NF", "2"))
    rr = [rp - eps * (1 - (NFp * a).cos()) / 2 for a in ang]
    zx = [rr[l] * ca[l] for l in range(NA)]
    zy = [rr[l] * sa[l] for l in range(NA)]
S2 = arb(os.environ.get("SRC_S", "1")) ** 2
rows = []
wx = []
Pd = []
X = []
for i in range(NR):
    for j in range(NA):
        x, y = rho[i] * ca[j], rho[i] * sa[j]
        X.append((x, y))
        rows += [(-((x - zx[l]) ** 2 + (y - zy[l]) ** 2) / 2).exp() / (2 * pi) for l in range(NA)]
        wx.append(wr[i] * rho[i] * 2 * pi / NA)
        Pd.append((-(rho[i] ** 2) / (2 * S2)).exp() / (2 * pi * S2))
NX = NR * NA
K = arb_mat(NX, NA, rows)
KT = K.transpose()
del rows
nrm = KT * arb_mat(NX, 1, wx)
say(
    f"q790 {FUNC} rp {rp.str(8, radius=False)} eps {eps.str(8, radius=False)} M {M}: {NR} x {NA} nodes, prec {ctx.prec}; normalisation error "
    f"{max(abs((nrm[l, 0] - 1).mid()) for l in range(NA)).str(3, radius=False)}  [{time.time()-T0:.0f}s]"
)
cosk = [[(2 * m * a).cos() for a in ang] for m in range(M + 1)]


def kkt(c):
    f = [1 + sum((c[m] * cosk[m][l] for m in range(1, M + 1)), arb(0)) for l in range(NA)]
    p = K * arb_mat(NA, 1, [f[l] / NA for l in range(NA)])
    if FUNC == "cap":
        v = KT * arb_mat(NX, 1, [wx[x] * p[x, 0].log() for x in range(NX)])
        return [-v[m, 0] for m in range(NA)]
    if FUNC == "rd":
        v = KT * arb_mat(NX, 1, [wx[x] * Pd[x] / p[x, 0] for x in range(NX)])
        return [v[m, 0] for m in range(NA)]
    Nx = K * arb_mat(NA, 1, [zx[l] * f[l] / NA for l in range(NA)])
    Ny = K * arb_mat(NA, 1, [zy[l] * f[l] / NA for l in range(NA)])
    u = KT * arb_mat(NX, 1, [wx[x] * (Nx[x, 0] ** 2 + Ny[x, 0] ** 2) / p[x, 0] ** 2 for x in range(NX)])
    vx = KT * arb_mat(NX, 1, [wx[x] * Nx[x, 0] / p[x, 0] for x in range(NX)])
    vy = KT * arb_mat(NX, 1, [wx[x] * Ny[x, 0] / p[x, 0] for x in range(NX)])
    return [
        u[m, 0] - 2 * (zx[m] * vx[m, 0] + zy[m] * vy[m, 0]) + (zx[m] ** 2 + zy[m] ** 2) * nrm[m, 0]
        for m in range(NA)
    ]


def harms(r):
    return [(2 * sum((r[l] * cosk[m][l] for l in range(NA)), arb(0)) / NA).mid() for m in range(1, M + 1)]


c = [arb(0)] * (M + 1)
probe = arb("1e-8")
for it in range(30):
    h = harms(kkt(c))
    nr = max(abs(x) for x in h)
    say(
        f"iteration {it}: max |KKT harmonic| {nr.str(3, radius=False)}; c = "
        + ", ".join(c[m].str(8, radius=False) for m in range(1, M + 1))
        + f"  [{time.time()-T0:.0f}s]"
    )
    if it > 0 and nr < arb("1e-40"):
        break
    J = arb_mat(M, M)
    for j in range(1, M + 1):
        cp = list(c)
        cp[j] = c[j] + probe
        hp = harms(kkt(cp))
        for i in range(M):
            J[i, j - 1] = ((hp[i] - h[i]) / probe).mid()
    dc = J.solve(arb_mat(M, 1, [-x for x in h]))
    c = [arb(0)] + [(c[m] + dc[m - 1, 0]).mid() for m in range(1, M + 1)]
for m in range(1, M + 1):
    say(f"  K {2*m:2d}: c_K {c[m].str(16, radius=False):>24}   c_K/2 {(c[m]/2).str(8, radius=False)}")
