"""F_small: end-to-end certificate on a small record state.
Usage: py F_small.py STATE.json RP H_DEN N PREC A NT M RHO [DELTA_ATOM]
  grid h = 1/H_DEN, |n| <= N; strip half-width A (A*RP < pi/2); NT Taylor intervals of order M on [0, pi/2], Cauchy radius RHO.
Writes the polished state to F_work/<name>_polished.json and prints the certificate log."""

import sys, json, time, math

sys.path.insert(0, ".")
from flint import arb, arb_mat, ctx
import F_core as FC, F_kkt as FK

T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


st = json.load(open(sys.argv[1]))
RP = sys.argv[2]
HD = int(sys.argv[3])
N = int(sys.argv[4])
PREC = int(sys.argv[5])
a = arb(sys.argv[6])
NT = int(sys.argv[7])
M = int(sys.argv[8])
RHO = sys.argv[9]
DAT = float(sys.argv[10]) if len(sys.argv) > 10 else 0.02
FC.setprec(PREC, threads=2)
geo = FC.Geo(RP, st["rm"], st["X"], st["Y"], len(st["G"]))
grid = FC.Grid(arb(1) / HD, N)
log(
    f"state {sys.argv[1]}: rp {RP} rm {st['rm']} X {st['X']} Y {st['Y']} nG {len(st['G'])}; grid h 1/{HD}, Y {N/HD}, M {grid.M}; prec {PREC}; a {a}"
)
(AL, BL), _, _ = FC.strip_consts(geo, a)
eq = FC.quad_err(AL, BL, geo.rp, geo.rm, 0, a, grid.h, grid.Y)
log(f"quadrature bound for D at a real point: {float(eq):.3e}")
v = [arb(x) for x in st["G"]] + [arb(x) for x in st["m"]] + [arb(st["C"])]
v, F = FK.newton_point(geo, grid, v, a, steps=6, log=log, tol=float(eq) * 100)
res = max(float(abs(x).upper()) for x in F)
n = len(v)
# Krawczyk: r from |Y F| with growth
F0 = F
ok = False
fac = 10.0
_, J0, _ = FK.system(geo, grid, v, a, jac=True)
Jm = arb_mat(n, n, [arb(J0[i, j].mid()) for i in range(n) for j in range(n)])
Yi = Jm.inv()
YF = [float(abs((Yi * arb_mat(n, 1, F0))[i, 0]).upper()) for i in range(n)]
import numpy as np

sv = np.linalg.svd(np.array([[float(J0[i, j].mid()) for j in range(n)] for i in range(n)]), compute_uv=False)
log("singular values of J:", " ".join(f"{x:.3e}" for x in sv))
log("|Y F| componentwise:", " ".join(f"{x:.2e}" for x in YF))
for attempt in range(4):
    r = [max(fac * x, 1e-300) for x in YF]
    r = [max(x, fac * max(YF) * 1e-6) for x in r]
    ok, det = FK.krawczyk(geo, grid, v, r, a, F0=F0)
    log(
        f"Krawczyk attempt {attempt} (r = {fac:g} |YF|, max r {max(r):.2e}): ok {ok}; ||I - YJ(B)||_inf {det['normM']:.3e}; "
        f"|K - v^| / r: " + " ".join(f"{u/rr:.3f}" for u, rr in det["rows"])
    )
    if ok:
        break
    fac *= 10
if not ok:
    log("KRAWCZYK FAILED")
    sys.exit(1)
rbox = r
G, m, C = geo.split(v)
log(
    "exact KKT point enclosed: G deg "
    + ", ".join(f"{float(x.mid())*180/math.pi:.8f}" for x in G)
    + "; masses "
    + ", ".join(f"{float(x.mid()):.10f}" for x in m)
    + f"; C {C.str(25, radius=False)}; radii {max(rbox):.2e}"
)
mins = min(float(x.mid()) - rr for x, rr in zip(m, rbox[geo.nG : geo.nG + geo.nO]))
log(f"smallest mass lower bound {mins:.6e} (> 0 needed)")
# Lam over the box
B = [vh + arb(ri) * arb(0, 1) for vh, ri in zip(v, rbox)]
Gb, mb, Cb = geo.split(B)
atomsB = geo.atoms(Gb, mb)
sw = FK.sum_w(geo, mb)
mdev = abs(sw - 1).upper()
(ALb, BLb), _, _ = FC.strip_consts(geo, a, msum_dev=mdev)
LamB = FC.lam_grid(grid, geo, atomsB)
log("Lam over the box done")
# atoms in [0, pi/2]
ats = []
ars = []
k = 0
if geo.X:
    ats.append(0.0)
    ars.append(0.0)
    k += 1
if geo.Y:
    ats.append(math.pi / 2)
    ars.append(0.0)
    k += 1
for g in range(geo.nG):
    ats.append(float(v[g].mid()))
    ars.append(rbox[g])
ds = [DAT] * len(ats)
atsb = (
    ([arb(0)] if geo.X else []) + ([arb.pi() / 2] if geo.Y else []) + [arb(v[g].mid()) for g in range(geo.nG)]
)  # R2 gap 2
okc, rep = FK.curve_certify(
    geo,
    grid,
    LamB,
    Cb,
    a,
    ats,
    ars,
    ds,
    0.0,
    math.pi / 2,
    NT,
    M,
    RHO,
    AL_BL=(ALb, BLb),
    log=log,
    t_hi_exact=arb.pi() / 2,
    atom_ts_arb=atsb,
)  # R2 gap 1: closed at the exact pi/2
log(
    f"curve certificate ok {okc}: max upper bound of D - C off the atom zones {rep['max_free']}; max upper bound of (D - C)'' in "
    f"the atom zones {rep['max_d2']}; pieces {rep['pieces']}; zones {[(round(x,5), round(y,5)) for x, y in rep['Iks']]}; worst {rep['worst']}; "
    f"certified violations {rep['violations']} {rep['viol']}"
)
name = sys.argv[1].replace("\\", "/").split("/")[-1].replace(".json", "")
json.dump(
    {
        "rm": st["rm"],
        "X": st["X"],
        "Y": st["Y"],
        "rp": RP,
        "G": [x.mid().str(60, radius=False) for x in v[: geo.nG]],
        "m": [x.mid().str(60, radius=False) for x in v[geo.nG : geo.nG + geo.nO]],
        "C": v[-1].mid().str(60, radius=False),
        "box_r": rbox,
        "krawczyk_ok": ok,
        "curve_ok": okc,
    },
    open(f"F_work/{name}_polished.json", "w"),
    indent=1,
)
log("done")
