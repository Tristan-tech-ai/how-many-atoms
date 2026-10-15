import sys, glob, json
from flint import arb
from q1411_fast_walk import *

H = float(sys.argv[1])
out = sys.argv[2]
start = sys.argv[3]
Aend = float(sys.argv[4])
import os

W = Walker(192, out, H=H, pad=float(os.environ.get("Q1411_PAD", "9")))
F = W.F
A, u, w, we, wc = load(start, F)
u, w, we, wc, res, k = W.solve_at(A, u, w, we, wc)
say(f"H {H}: start re-solved residual {float(res):.1e} in {k} it")
r = W.run(A, (u, w, we, wc), Aend, hmax=0.01)
say("returned", r[0], float(r[1]))
