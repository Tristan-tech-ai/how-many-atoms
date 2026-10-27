"""Values behind Fig. 2 of the paper: D(t) - C along the boundary point (cos t, r_m sin t), t in [0, pi/2], for the
certified ring, by the method of separate_evaluator.py (plain separable trapezoid rule with step 1/h_inverse on the square of
half-width Y, at P bits, without an error bound). Values are midpoints of balls; their radii are stored too. These values are checked
numerically only; the certified statements are the enclosures of Section IV. Writes margin_along_boundary.json next to itself.
Usage: python margin_along_boundary.py [n_uniform=181] [h_inverse=20] [Y=22] [bits=256]   (the paper uses n_uniform = 721)"""
import json
import math
import os
import sys
import time

from flint import arb, arb_mat, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
import sys  # noqa: E402
sys.path.insert(0, HERE)
from layout import find, state  # noqa: E402
nu = int(sys.argv[1]) if len(sys.argv) > 1 else 181
hinv = int(sys.argv[2]) if len(sys.argv) > 2 else 20
Yh = int(sys.argv[3]) if len(sys.argv) > 3 else 22
ctx.prec = int(sys.argv[4]) if len(sys.argv) > 4 else 256
t0 = time.time()
d = json.load(open(find("ring32_polished.json"), encoding="utf-8"))
rp, rm = arb(str(d.get("rp", "1.0"))), arb(str(d["rm"]))
th = [arb(str(g)) for g in d["G"]]
ms = [arb(str(x)) for x in d["m"]]
C = arb(str(d["C"]))
pi = arb.pi()
pts = []
for t, w in zip(th, ms):
    for s in (t, -t, pi - t, pi + t):
        pts.append((rp * s.cos(), rm * s.sin(), w))
h = arb(1) / hinv
N = Yh * hinv
ys = [h * n for n in range(-N, N + 1)]
M = len(ys)
K = len(pts)
E1 = arb_mat(M, K, [(y * x1 - (x1 * x1 + x2 * x2) / 2).exp() * w for y in ys for (x1, x2, w) in pts])
E2 = arb_mat(K, M, [(y * x2).exp() for (x1, x2, w) in pts for y in ys])
S = E1 * E2
L = arb_mat(M, M, [s.log() for s in S.entries()])
c = h / (2 * pi).sqrt()
print(f"setup {time.time() - t0:.1f} s, grid {M} per axis, {K} mass points, {ctx.prec} bits", flush=True)


def D(x1, x2):
    a = arb_mat(1, M, [c * (-(y - x1) ** 2 / 2).exp() for y in ys])
    b = arb_mat(M, 1, [c * (-(y - x2) ** 2 / 2).exp() for y in ys])
    return (x1 * x1 + x2 * x2) / 2 - (a * L * b)[0, 0]


tj = [float(g) for g in d["G"]]
ts = set(round(math.pi / 2 * i / (nu - 1), 12) for i in range(nu))
for g in tj:                        # denser sampling near each mass point, to draw the valleys
    for k in range(-12, 13):
        x = g + k * 0.0025
        if 0 <= x <= math.pi / 2:
            ts.add(round(x, 12))
ts = sorted(ts)
rows = []
t1 = time.time()
for i, t in enumerate(ts):
    ta = arb(repr(t))
    v = D(rp * ta.cos(), rm * ta.sin()) - C
    rows.append([t, float(v.mid()), float(v.rad())])
    if i == 4:
        print(f"about {(time.time() - t1) / 5:.2f} s per point, {len(ts)} points", flush=True)
at = [D(rp * g.cos(), rm * g.sin()) - C for g in th]
v0 = D(rp, arb(0)) - C
out = {"h": f"1/{hinv}", "Y": Yh, "bits": ctx.prec, "n_points": len(ts), "rows": rows,
       "mass_point_t": tj, "max_abs_at_mass_points": max(abs(float(x.mid())) for x in at), "vertex_t0": float(v0.mid()),
       "max_on_grid_away": max(r[1] for r in rows if min(abs(r[0] - g) for g in tj) > 0.004),
       "min_on_grid": min(r[1] for r in rows), "seconds": round(time.time() - t0, 1)}
json.dump(out, open(os.path.join(HERE, "margin_along_boundary.json"), "w"), indent=0)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
