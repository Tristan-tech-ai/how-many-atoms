"""q817 (03 Oct 2026, BACKLOG 50): a rigorous upper bound on the capacity from the formal density, by duality.
For ANY output law q, C <= sup_{x in E} D(P_{Y|x} || q). Take q = the output of the D2-symmetric input that puts mass f(t_l)/N on N
equally spaced points t_l = 2 pi l / N of the ellipse, f the formal density of order M (float64, as q816); this q is a valid output law
whenever every f(t_l) > 0. D(.||q) is the KKT function of that discrete input, so quadrature_bounds / kkt_krawczyk bound it rigorously: the curve check
of kkt_krawczyk.curve_certify with C := U (no atom zones) proves D(.||q) < U on [0, pi/2] (closed at the exact pi/2), hence on the whole curve
by symmetry; the interior follows from convexity (Hess D >= (1 - max|x_l|^2) I >= 0 when rp <= 1). Result: C <= U.
Usage: py capacity_bracket.py RP RM M N MARGIN   (env PRECC 256, HDC 16, NNC 288, A 1.45, NT 32, MT 36, RHO 1, THREADS 3)
"""

import sys, os, math, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agents"))
from flint import arb, ctx
import quadrature_bounds as FC, kkt_krawczyk as FK

RP, RM, M, N, MARGIN = (
    float(sys.argv[1]),
    float(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
    float(sys.argv[5]),
)
assert N % 4 == 0
E = os.environ.get
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)

# ---- formal density in float64 (the q816 construction, copied as text)
NR, NA, RMAX = 160, 256, 12.0
xr, wr = np.polynomial.legendre.leggauss(NR)
rho = (xr + 1) * RMAX / 2
wrho = wr * RMAX / 2
ang = 2 * np.pi * np.arange(NA) / NA
Y1 = (rho[:, None] * np.cos(ang)[None, :]).ravel()
Y2 = (rho[:, None] * np.sin(ang)[None, :]).ravel()
WY = (wrho[:, None] * rho[:, None] * np.full((1, NA), 2 * np.pi / NA)).ravel()
t = 2 * np.pi * np.arange(NA) / NA
cosk = np.array([np.cos(2 * k * t) for k in range(1, M + 1)])
z1 = RP * np.cos(t)
z2 = RM * np.sin(t)
Kc = np.exp(-((Y1[:, None] - z1[None, :]) ** 2 + (Y2[:, None] - z2[None, :]) ** 2) / 2) / (2 * np.pi)


def kkt(c):
    f = 1 + c @ cosk
    p = Kc @ (f / NA)
    L = np.log(p)
    return -(Kc * (WY * L)[:, None]).sum(0)


def harms(r):
    return np.array([2 * np.mean(r * cosk[k]) for k in range(M)])


c = np.zeros(M)
for it in range(40):
    r = kkt(c)
    h = harms(r)
    if np.max(np.abs(h)) < 1e-14:
        break
    J = np.zeros((M, M))
    for j in range(M):
        cp = c.copy()
        cp[j] += 1e-6
        J[:, j] = (harms(kkt(cp)) - h) / 1e-6
    c = c + np.linalg.solve(J, -h)
log(f"formal density M {M}: harmonic residual {np.max(np.abs(harms(kkt(c)))):.1e}; c = {c.tolist()}")

# ---- the discrete input: N equally spaced points, weights f(t_l)/N (D2 orbits X, Y, G)
tl = 2 * np.pi * np.arange(N) / N
fl = 1 + np.array([sum(c[k - 1] * np.cos(2 * k * x) for k in range(1, M + 1)) for x in tl])
log(f"min f(t_l) = {fl.min():.6f} (must be > 0)")
assert fl.min() > 0
nG = N // 4 - 1
w_orbit = [fl[0], fl[N // 4]] + [fl[l] for l in range(1, N // 4)]  # X, Y, then G at t_l, l = 1..N/4-1
tot = 2 * w_orbit[0] + 2 * w_orbit[1] + 4 * sum(w_orbit[2:])
PRECC = int(E("PRECC", "256"))
FC.setprec(PRECC, threads=int(E("THREADS", "3")))
wB = [arb(float(x)) for x in w_orbit]
totB = 2 * wB[0] + 2 * wB[1] + 4 * sum(wB[2:], arb(0))
mB = [
    x / totB for x in wB
]  # normalised in ball arithmetic                        # exact rationals of the float values; sum exactly 1
G = [2 * arb.pi() * l / N for l in range(1, N // 4)]
geo = FC.Geo(arb(str(RP)), arb(str(RM)), 1, 1, nG)
dsum = FK.sum_w(geo, mB) - 1
print("sum of weights - 1:", dsum.str(5))
# q must be a probability law: if the enclosed sum is not exactly 1, the bound is raised by |log(sum)| (added to U below)
LOGSUM = abs((1 + dsum).log()).upper()

# ---- float estimate of sup D(.||q) on the curve, to choose U
xs = []
ws = []
for th, wv, _, _ in geo.atoms(G, mB):
    xs.append((float((geo.rp * th.cos()).mid()), float((geo.rm * th.sin()).mid())))
    ws.append(float(wv.mid()))
xs = np.array(xs)
ws = np.array(ws)
Kd = np.exp(-((Y1[:, None] - xs[None, :, 0]) ** 2 + (Y2[:, None] - xs[None, :, 1]) ** 2) / 2) / (2 * np.pi)
pd = Kd @ ws
Ld = np.log(pd)
tt = np.linspace(0, np.pi / 2, 2001)
Kt = np.exp(
    -((Y1[:, None] - RP * np.cos(tt)[None, :]) ** 2 + (Y2[:, None] - RM * np.sin(tt)[None, :]) ** 2) / 2
) / (2 * np.pi)
rt = -(Kt * (WY * Ld)[:, None]).sum(0)
L2PE = math.log(2 * math.pi) + 1
supf = rt.max() - L2PE
meanf = kkt(c).mean() - L2PE
log(f"float: formal C {meanf:.15f}; discrete-input curve sup {supf:.15f}; ripple {rt.max() - rt.min():.2e}")
U = arb(float(supf + MARGIN)) + LOGSUM
log(f"U = sup + {MARGIN:g} = {U.str(20)}")

# ---- rigorous: D(.||q) < U on [0, pi/2] (exact endpoint), Lam over the exact weights
gridc = FC.Grid(arb(1) / int(E("HDC", "16")), int(E("NNC", "288")))
a = arb(E("A", "1.45"))
mdev = abs(FK.sum_w(geo, mB) - 1).upper()
(ALb, BLb), _, _ = FC.strip_consts(geo, a, msum_dev=mdev)
atoms = geo.atoms(G, mB)
LamB = FC.lam_grid(gridc, geo, atoms)
log(f"Lam on the grid done ({len(atoms)} atoms, M {gridc.M})")
okc, rep = FK.curve_certify(
    geo,
    gridc,
    LamB,
    U,
    a,
    [],
    [],
    [],
    0.0,
    math.pi / 2,
    int(E("NT", "32")),
    int(E("MT", "36")),
    E("RHO", "1"),
    AL_BL=(ALb, BLb),
    log=log,
    t_hi_exact=arb.pi() / 2,
    atom_ts_arb=[],
)
log(
    f"curve check D(.||q) < U on [0, pi/2]: ok {okc}; max upper bound of D - U {rep['max_free']}; pieces {rep['pieces']}; failures {rep['worst'][:3]}"
)
maxx2 = max(float((geo.rp * th.cos()) ** 2 + (geo.rm * th.sin()) ** 2) for th, _, _, _ in atoms)
log(
    f"interior: max |x_l|^2 = {maxx2:.6f} (<= 1 gives Hess D >= 0, so the sup over the filled ellipse is on the boundary)"
)
if okc and maxx2 <= 1 + 1e-15:
    log(
        f"RESULT: C(rp {RP}, rm {RM}) <= {U.str(20)} (rigorous, by duality with the formal-density output law)"
    )

# ---- rigorous LOWER bound: the mutual information of the same N-point input, I = sum_l w_l D(x_l || q) <= C
WBl = [FC.weights(gridc, geo, th, 0) for th in geo.reps(G)]  # orbit representatives: X, Y, G...
c0s = []
for b0 in range(0, len(WBl), 6):
    c0s += [cc[0] for cc in FC.contract_multi(gridc, LamB, WBl[b0 : b0 + 6], 0)]
qerr = FC.quad_err(ALb, BLb, geo.rp, geo.rm, 0, a, gridc.h, gridc.Y)
mult = [2, 2] + [4] * nG
Ival = arb(0)
for th, c0v, mo, mm in zip(geo.reps(G), c0s, mult, mB):
    Dv = FK.ball(FC.half_norm2_series(geo, th, 0)[0] - c0v, qerr)
    Ival += mo * mm * Dv
log(f"LOWER bound: I(X;Y) of the N-point input = {Ival.str(20)}  (radius {float(Ival.rad()):.1e})")
log(
    f"SANDWICH (endpoints printed to 25 digits; round outward when quoting): {Ival.lower().str(25, radius=False)} <= C <= {U.upper().str(25, radius=False)}; width {float((U.upper() - Ival.lower()).mid()):.2e}"
)
