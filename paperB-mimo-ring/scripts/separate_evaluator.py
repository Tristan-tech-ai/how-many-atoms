"""Separate evaluator of the optimality function (Observation 1 of the paper). It shares no code with the certificate:
it evaluates D(x) = |x|^2/2 - E log S(x + Z) by a plain separable trapezoid rule (step h, square of half-width Y) at P bits, without
an error bound, at the mass points of the certified center and at the two vertices (1, 0) and (0, r_m).
Input: the center file sec5-certificate/results/rerun_ring32_polished.json.
Usage: python separate_evaluator.py CENTER.json [h_inverse=20] [Y=22] [bits=256]; prints the values and writes
separate_evaluator_out.json next to itself."""
import json
import os
import sys
import time

from flint import arb, arb_mat, ctx

centre = sys.argv[1]
hinv = int(sys.argv[2]) if len(sys.argv) > 2 else 20
Yh = int(sys.argv[3]) if len(sys.argv) > 3 else 22
ctx.prec = int(sys.argv[4]) if len(sys.argv) > 4 else 256
t0 = time.time()
d = json.load(open(centre))
rp, rm = arb(d.get("rp", "1.0")), arb(d["rm"])
th = [arb(g) for g in d["G"]]
ms = [arb(x) for x in d["m"]]
C = arb(d["C"])
pi = arb.pi()
pts = []                                   # (x1, x2, mass): four points per orbit at t, -t, pi - t, pi + t
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


def D(x1, x2):
    a = arb_mat(1, M, [c * (-(y - x1) ** 2 / 2).exp() for y in ys])
    b = arb_mat(M, 1, [c * (-(y - x2) ** 2 / 2).exp() for y in ys])
    return (x1 * x1 + x2 * x2) / 2 - (a * L * b)[0, 0]


at = [D(rp * t.cos(), rm * t.sin()) - C for t in th]
v1 = D(rp, arb(0)) - C
v2 = D(arb(0), rm) - C
out = {"h": f"1/{hinv}", "Y": Yh, "bits": ctx.prec, "grid_points_per_axis": M,
       "max_abs_D_minus_C_at_mass_points": max(abs(float(x.mid())) for x in at),
       "D_minus_C_at_(1,0)": float(v1.mid()), "D_minus_C_at_(0,rm)": float(v2.mid()), "seconds": round(time.time() - t0, 1)}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "separate_evaluator_out.json"), "w"), indent=1)
