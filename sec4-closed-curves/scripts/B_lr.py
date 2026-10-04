"""B_lr (2 Oct 2026): q790's full formal density, copied AS TEXT (q790 itself untouched), plus the linear-response matrix
L[i][j] = d(KKT cosine harmonic 2i)/d(density cosine coefficient c_2j) at the converged density, by a one-sided probe 1e-30 at
256 bits (q790's own Newton uses 1e-8). Writes a JSON file (env OUT) with c_2..c_2M (40 digits) and L (25 digits).
Same curve options as q790: env LAM (mixed curve), SHAPE=polar with NF, SRC_S (rd).
Usage: py B_lr.py FUNC RP EPS M [DEG] [NA] [RMAX] [PREC]"""

import sys, time, os, json
import mpmath as mp
from mpmath.calculus.quadrature import GaussLegendre
from flint import arb, arb_mat, ctx

FUNC = sys.argv[1]
M = int(sys.argv[4])
DEG = int(sys.argv[5]) if len(sys.argv) > 5 else 6
NA = int(sys.argv[6]) if len(sys.argv) > 6 else 96
RMAX = sys.argv[7] if len(sys.argv) > 7 else os.environ.get("BLR_RMAX", "20")
ctx.prec = int(sys.argv[8]) if len(sys.argv) > 8 else 256
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
LAM = arb(os.environ.get("LAM", "0"))
R_ = rp - eps / 2
et_ = eps / (2 * (1 + LAM))
c3 = [(3 * a).cos() for a in ang]
s3 = [(3 * a).sin() for a in ang]
zx = [(R_ + et_) * ca[l] + LAM * et_ * c3[l] for l in range(NA)]
zy = [(R_ - et_) * sa[l] + LAM * et_ * s3[l] for l in range(NA)]
if os.environ.get("SHAPE", "") == "polar":
    NFp = int(os.environ.get("NF", "2"))
    rr = [rp - eps * (1 - (NFp * a).cos()) / 2 for a in ang]
    zx = [rr[l] * ca[l] for l in range(NA)]
    zy = [rr[l] * sa[l] for l in range(NA)]
S2 = arb(os.environ.get("SRC_S", "1")) ** 2
rows = []
wx = []
Pd = []
for i in range(NR):
    for j in range(NA):
        x, y = rho[i] * ca[j], rho[i] * sa[j]
        rows += [(-((x - zx[l]) ** 2 + (y - zy[l]) ** 2) / 2).exp() / (2 * pi) for l in range(NA)]
        wx.append(wr[i] * rho[i] * 2 * pi / NA)
        Pd.append((-(rho[i] ** 2) / (2 * S2)).exp() / (2 * pi * S2))
NX = NR * NA
K = arb_mat(NX, NA, rows)
KT = K.transpose()
del rows
nrm = KT * arb_mat(NX, 1, wx)
nerr = max(abs((nrm[l, 0] - 1).mid()) for l in range(NA))
say(
    f"B_lr {FUNC} rp {rp.str(8, radius=False)} eps {eps.str(12, radius=False)} M {M}: {NR} x {NA} nodes, prec {ctx.prec}; normalisation error "
    f"{nerr.str(3, radius=False)}  [{time.time()-T0:.0f}s]"
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


def jac(c, probe):
    h = harms(kkt(c))
    J = arb_mat(M, M)
    for j in range(1, M + 1):
        cp = list(c)
        cp[j] = c[j] + probe
        hp = harms(kkt(cp))
        for i in range(M):
            J[i, j - 1] = ((hp[i] - h[i]) / probe).mid()
    return h, J


def solve(c, Mact):
    """Newton on harmonics 2..2Mact with c_(2m) = 0 for m > Mact; raises on nan / singular steps."""
    for it in range(30):
        h, J = jac(c, arb("1e-8"))
        h = h[:Mact]
        nr = max(abs(x) for x in h)
        if not nr.is_finite():
            raise ValueError("nan")
        say(
            f"  M {Mact} iteration {it}: max |KKT harmonic| {nr.str(3, radius=False)}  [{time.time()-T0:.0f}s]"
        )
        if it > 0 and nr < arb("1e-45"):
            return c
        Js = arb_mat(Mact, Mact, [J[i, j] for i in range(Mact) for j in range(Mact)])
        dc = Js.solve(arb_mat(Mact, 1, [-x for x in h]))
        c = [arb(0)] + [(c[m] + dc[m - 1, 0]).mid() if m <= Mact else arb(0) for m in range(1, M + 1)]
    return c


M0 = int(
    os.environ.get("M0", str(M))
)  # solve harmonics 2..2M0 only; c_(2m) = 0 above; L and residuals up to 2M
try:
    c = solve([arb(0)] * (M + 1), M0)
except (
    ValueError,
    ZeroDivisionError,
):  # direct start fails when the top coefficients are large: add them one by one
    say("direct Newton failed; progressive in M")
    c = [arb(0)] * (M + 1)
    for Ma in range(1, M0 + 1):
        c = solve(c, Ma)
h, L = jac(c, arb("1e-30"))
r0 = kkt(c)
C0 = (sum(r0, arb(0)) / NA).mid()
for m in range(1, M + 1):
    say(f"  K {2*m:2d}: c_K {c[m].str(20, radius=False):>28}   L_KK {L[m-1, m-1].str(12, radius=False)}")
out = {
    "func": FUNC,
    "rp": sys.argv[2],
    "eps": sys.argv[3],
    "M": M,
    "env": {k: os.environ.get(k, "") for k in ("LAM", "SHAPE", "NF", "SRC_S")},
    "nerr": nerr.str(3, radius=False),
    "resid": max(abs(x) for x in h[:M0]).str(3, radius=False),
    "C": C0.str(40, radius=False),
    "c": [c[m].str(40, radius=False) for m in range(1, M + 1)],
    "M0": M0,
    "r": [h[m].str(30, radius=False) for m in range(M)],
    "L": [[L[i, j].str(25, radius=False) for j in range(M)] for i in range(M)],
}
if os.environ.get("OUT"):
    with open(os.environ["OUT"], "w") as fh:
        json.dump(out, fh)
say(f"done  [{time.time()-T0:.0f}s]")
