"""I_interior (03 Oct, referee R1 item T1): D - C inside the filled ellipse for a solved boundary ring (D_solver state file).
Grid q = (s rp cos t, s rm sin t), s in S_LIST, t in [0, 90] deg (D2 symmetry). Uses D_core2's KKT function through D_solver (FUNC from
env, here capfix), the same function as the ring solve. Prints the largest D - C over the grid and its place, and D - C at the centre.
Usage: py I_interior.py STATE.json [NT]   (env FUNC RP LAM PREC as for D_solver)"""

import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flint import arb
import D_solver as DS
from D_solver import Solver

d, v, rm0 = DS.load(sys.argv[1])
NT = int(sys.argv[2]) if len(sys.argv) > 2 else 19
S = Solver(d["X"], d["Y"], len(d["G"]))
P = S.P
rm = arb(rm0)
rp = arb(os.environ["RP"])
nG, nO = P.nG, P.nO
G = v[:nG]
m = v[nG : nG + nO]
C = v[nG + nO]
th, w = P.atoms(G, m)
xs = [P.pos(a, rm) for a in th]
D0 = P.D_grad((arb(0), arb(0)), xs, w, grad=False) - C
print(
    f"state {os.path.basename(sys.argv[1])}: rm {rm0}, rp {os.environ['RP']}, atoms {len(xs)}, max |x|^2 {max(float((x[0]**2 + x[1]**2).mid()) for x in xs):.5f}"
)
print(f"  D - C at the centre: {float(D0.mid()):+.6e}")
best = (-1e300, None)
pi = arb.pi()
for s in [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995]:
    row = []
    for j in range(NT):
        t = pi * j / (2 * (NT - 1))
        z = P.pos(t, rm)
        q = (arb(s) * z[0], arb(s) * z[1])
        val = float((P.D_grad(q, xs, w, grad=False) - C).mid())
        row.append(val)
        if val > best[0]:
            best = (val, (s, 90 * j / (NT - 1)))
    print(
        f"  s {s:5.3f}: max {max(row):+.4e} at t {90*row.index(max(row))/(NT - 1):5.1f} deg; min {min(row):+.4e}",
        flush=True,
    )
print(
    f"LARGEST D - C on the interior grid: {best[0]:+.6e} at s {best[1][0]}, t {best[1][1]:.1f} deg -> {'INTERIOR VIOLATION' if best[0] > 0 else 'no interior violation on this grid'}"
)
