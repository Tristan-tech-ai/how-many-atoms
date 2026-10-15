"""q1725 (10 Oct 2026, READING, no verdict word): is the inner deviation delta_k of the solved N = 90 state (q1708_z_NG14_N90.json) from the formal law c28 the truncation of the bulk series? Delta_k = formal_k(c30) - formal_k(c28) at the same pair (the n = 15 term of the bulk law at the atom k), against delta_k = d_k - formal_k(c28), for k = 10..45. If delta_k is the series remainder, Delta_k carries its first term and delta_k / Delta_k is about 1 / (1 - r) with the order ratio r of the bulk terms (0.4 to 0.5), so Delta_k / delta_k is about 0.5 to 0.6.
NOT BLIND: delta_k (reading 170) and the observation that the rest rms on the window 20..40 fell from 4.05e-11 (c28) to 2.46e-11 (c30) were read; the check follows from the exponent: the next term of the bulk law goes as rho^(-27), which gives a ratio per atom exp(-27 / (4 k)), 0.71 at k = 20.
Predictions:
 P1 Delta_k has the sign of delta_k for every k = 12..30;
 P2 Delta_k / delta_k lies between 0.4 and 0.7 for every k = 16..26;
 P3 Delta_k / delta_k varies by less than 25 percent over k = 16..26 (largest over smallest below 1.25).
Control: formal_k(c28) from the NG = 14 module equals formal_k from the q1697 NG = 14 module used by q1709 (largest difference below 1e-50), and the NG = 15 module at the same pair returns finite values.
Usage: py q1725_inner_deviation_vs_next_order.py (seconds)."""

import os, sys, json, io, contextlib, importlib.util
import numpy as np

sys.dont_write_bytecode = True
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
D = json.load(open(os.path.join(here, "q1708_z_NG14_N90.json")))
N = D["N"]
nu, d0 = D["nu_used"], D["d0_used"]
from flint import arb


def formal(modfile, ng):
    os.environ["Q1696_NG"] = str(ng)
    os.environ["Q1703_NG"] = str(ng)
    sys.argv = ["x"]
    spec = importlib.util.spec_from_file_location(
        "m%d_%s" % (ng, os.path.basename(modfile)), os.path.join(here, modfile)
    )
    q = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(q)
    q.NU, q.D0 = nu, d0
    return q.formal_tail(0, N + 1)[0]


f14 = formal("q1697_krawczyk_ng.py", 14)
f14b = formal("q1704_krawczyk_ng15.py", 14)
f15 = formal("q1704_krawczyk_ng15.py", 15)
z = [arb(s) for s in D["z"]]
delta = np.array([float((z[k] - f14[k]).mid()) for k in range(N)])
Delta = np.array([float((f15[k] - f14[k]).mid()) for k in range(N)])
ctl = max(float((a - b).abs_upper()) for a, b in zip(f14, f14b)) < 1e-50 and bool(np.all(np.isfinite(Delta)))
print("k    delta_k        Delta_k (c30 minus c28)   Delta/delta")
for k in range(10, 46):
    print("%2d  %+.3e   %+.3e   %+.3f" % (k, delta[k - 1], Delta[k - 1], Delta[k - 1] / delta[k - 1]))
ks = np.arange(16, 27)
ra = Delta[ks - 1] / delta[ks - 1]
lab = lambda q_: "holds" if q_ else "does not hold"
p1 = bool(np.all(np.sign(Delta[11:30]) == np.sign(delta[11:30])))
p2 = bool(np.all((ra >= 0.4) & (ra <= 0.7)))
p3 = float(ra.max() / ra.min()) < 1.25
print(
    "Delta/delta over k = 16..26: %.3f to %.3f (largest over smallest %.3f)"
    % (ra.min(), ra.max(), ra.max() / ra.min())
)
print(
    "P1 sign equal for k = 12..30: %s; P2 ratio in [0.4, 0.7]: %s; P3 ratio varies by less than 25 percent: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(ctl))
)
print("READING q1725 done (labels by the script; no verdict word)")
