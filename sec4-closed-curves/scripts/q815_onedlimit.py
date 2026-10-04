"""q815 (Amendment 1387): the K/2-convention copying loss for the two smallest rings on the pure ellipse, evaluated toward rm -> 0.
K = 2 (ring X): the copying ring supplies b_2 = 2, so its loss is where the formal density (one harmonic pair) has c_2 = 2; we fix
c_2 = 2 and find the eps where the KKT harmonic 2 vanishes. K = 4 (ring X + Y): the ring copies c_2 and supplies b_4 = 2, so its loss is
where the formal density (two pairs) has c_4 = 2; we fix c_4 = 2 and solve harmonics 2 and 4 for (c_2, eps).
Quadrature and KKT function: q790's (capacity, ellipse), copied as text; q790 is not imported (it runs at top level).
Usage: py q815_onedlimit.py K RP EPS_LO EPS_HI   (env PREC, NA, DEG, RMAX)"""

import sys, os, time
import mpmath as mp
from mpmath.calculus.quadrature import GaussLegendre
from flint import arb, arb_mat, ctx

K = int(sys.argv[1])
RP = sys.argv[2]
LO, HI = float(sys.argv[3]), float(sys.argv[4])
ctx.prec = int(os.environ.get("PREC", "128"))
mp.mp.dps = int(ctx.prec * 0.30103) + 10
DS = mp.mp.dps - 5
NA = int(os.environ.get("NA", "96"))
DEG = int(os.environ.get("DEG", "6"))
RMAX = os.environ.get("RMAX", "20")
nodes = GaussLegendre(mp.mp).calc_nodes(DEG, mp.mp.prec)
RMmp = mp.mpf(RMAX)
rho = [arb(mp.nstr((x + 1) * RMmp / 2, DS)) for x, w in nodes]
wr = [arb(mp.nstr(w * RMmp / 2, DS)) for x, w in nodes]
NR = len(rho)
pi = arb.pi()
ang = [2 * pi * j / NA for j in range(NA)]
ca = [a.cos() for a in ang]
sa = [a.sin() for a in ang]
T0 = time.time()
say = lambda *a: print(*a, flush=True)
M = K // 2
cosk = [[(2 * m * a).cos() for a in ang] for m in range(M + 1)]


def build(eps):
    rp = arb(RP)
    eps = arb(eps)
    R_ = rp - eps / 2
    et_ = eps / 2
    zx = [(R_ + et_) * ca[l] for l in range(NA)]
    zy = [(R_ - et_) * sa[l] for l in range(NA)]
    rows = []
    wx = []
    for i in range(NR):
        for j in range(NA):
            x, y = rho[i] * ca[j], rho[i] * sa[j]
            rows += [(-((x - zx[l]) ** 2 + (y - zy[l]) ** 2) / 2).exp() / (2 * pi) for l in range(NA)]
            wx.append(wr[i] * rho[i] * 2 * pi / NA)
    Km = arb_mat(NR * NA, NA, rows)
    KT = Km.transpose()

    def harms(c):  # q790's kkt (capacity) and its harmonics 2..2M
        f = [1 + sum((c[m] * cosk[m][l] for m in range(1, M + 1)), arb(0)) for l in range(NA)]
        p = Km * arb_mat(NA, 1, [f[l] / NA for l in range(NA)])
        if any(float(p[x, 0].mid()) <= 0 for x in range(NR * NA)):
            return None
        v = KT * arb_mat(NR * NA, 1, [wx[x] * p[x, 0].log() for x in range(NR * NA)])
        r = [-v[m, 0] for m in range(NA)]
        return [
            float((2 * sum((r[l] * cosk[m][l] for l in range(NA)), arb(0)) / NA).mid())
            for m in range(1, M + 1)
        ]

    return harms


def F(eps, c2_guess=0.5):
    h = build(eps)
    if K == 2:
        r = h([arb(0), arb(2)])
        return (r[0] if r else None), None
    c2 = c2_guess  # K = 4: c_4 = 2 fixed, solve harmonic 2 for c_2 (secant), report harmonic 4

    def r2(x):
        r = h([arb(0), arb(x), arb(2)])
        return r

    a, b = c2, c2 + 0.05
    ra, rb = r2(a), r2(b)
    for it in range(40):
        if ra is None or rb is None:
            return None, None
        if abs(rb[0]) < 1e-25 or b == a:
            break
        x = b - rb[0] * (b - a) / (rb[0] - ra[0])
        a, ra = b, rb
        b = x
        rb = r2(b)
    return (rb[1] if rb else None), b


say(f"q815 K {K} rp {RP}: scan eps in [{LO}, {HI}]  (prec {ctx.prec}, {NR} x {NA})")
pts = [LO + (HI - LO) * i / 8 for i in range(9)]
vals = []
g = 0.5
for e in pts:
    v, c2 = F(e, g)
    g = c2 if c2 is not None else g
    vals.append((e, v, c2))
    say(f"  eps {e:.6f} (rm {float(RP) - e:.6f}): residual {v} c_2 {c2}  [{time.time()-T0:.0f}s]")
for (e0, v0, _), (e1, v1, c21) in zip(vals[:-1], vals[1:]):
    if v0 is not None and v1 is not None and v0 * v1 < 0:
        a, b, fa, fb = e0, e1, v0, v1
        for it in range(40):
            m = (a + b) / 2
            fm, _ = F(m, c21 if c21 is not None else 0.5)
            if fm is None:
                break
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
            if b - a < 1e-9:
                break
        say(
            f"ROOT K {K} rp {RP}: eps* {0.5*(a + b):.9f}, rm* {float(RP) - 0.5*(a + b):.9f}  [{time.time()-T0:.0f}s]"
        )
