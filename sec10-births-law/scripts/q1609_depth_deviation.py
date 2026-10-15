"""q1609 (10 Oct 2026, READING, no verdict word; deep proof Q11 step 63): the deviation of the exact atoms of the clean 120-atom block from the formal bulk law, atom by atom.
The formal solution places the atom k at the depth d_k^formal = dlaw(rho_k) + d0 with the mass c / rho_k (the bulk law through c20, the limit pair nu0 = -0.093773586 and d0 = +0.001159841, c = 0.0065, rho_k from N(rho) = k + nu0). The solved clean block of 120 atoms (q1601_z_N120.json, 300 bits, residual 2.1e-71) is the exact solution of its system with the law atoms 121..200 frozen. The deviations delta_k = d_k - d_k^formal and eps_k = w_k - c / rho_k measure the remainder of the bulk series in the unknowns themselves (gap G1), not only in the residuals. Atoms near the block end (k > 90) feel the closure by the frozen atoms and are listed but not used for the fit.
Predictions written before the run:
 P1 |delta_k| falls monotonically from k = 12 to k = 60, apart from isolated sign changes of delta_k (counted by the script as increases of |delta_k| between neighbours);
 P2 the log-log slope of |delta_k| over k = 20..60 is between -30 and -15;
 P3 |delta_k| at k = 60 is below 1e-14;
 P4 the relative mass deviation |eps_k| rho_k / c at k = 60 is below 1e-12.
Usage: py q1609_depth_deviation.py (reads q1601_z_N120.json, a few seconds)."""

import os, sys, json, importlib.util
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
spec = importlib.util.spec_from_file_location("q1598m", os.path.join(here, "q1598_krawczyk_free90_120.py"))
q = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(q)
from flint import arb
from mpmath import mpf, nstr

N = 120
z = [arb(s) for s in json.load(open(os.path.join(here, "q1601_z_N120.json")))]
dt, wt = q.formal_tail(0, N)
dev_d = [float(z[k] - dt[k]) for k in range(N)]
rel_w = [float((z[N + k] - wt[k]) / wt[k]) for k in range(N)]
print("k, delta_k (depth), relative mass deviation:")
for k in (1, 2, 3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120):
    print("%4d %+.3e %+.3e" % (k, dev_d[k - 1], rel_w[k - 1]))
a = np.abs(np.array(dev_d))
inc = [k for k in range(12, 60) if a[k] > a[k - 1]]
ks = np.arange(20, 61)
slope = np.polyfit(np.log(ks.astype(float)), np.log(a[19:60]), 1)[0]
print(
    "increases of |delta_k| between k = 12 and 60 at k = %s; log-log slope of |delta_k| over k = 20..60: %.2f; |delta_60| %.3e; |relative mass deviation| at k = 60 %.3e"
    % (inc[:10], slope, a[59], abs(rel_w[59]))
)


def lab(x):
    return "holds" if x else "does not hold"


print(
    "P1 %s; P2 %s; P3 %s; P4 %s"
    % (lab(len(inc) <= 3), lab(-30 <= slope <= -15), lab(a[59] < 1e-14), lab(abs(rel_w[59]) < 1e-12))
)
print("READING q1609 done (labels by the script; no verdict word)")
