"""q1605 (10 Oct 2026, READING, no verdict word; deep proof Q11 step 61): the shift of the first depth when the clean block of 120 atoms is extended to 180 atoms, by Newton updates started from the solved 120-atom block plus the formal law atoms 121..180 (NOUT = 260, level atom 230, 300 bits).
Readings 102 and 103: the shifts of the first depth are +7.73310e-12, +1.81232e-18, +1.53487e-23 and +8.3076e-28 for the steps 10 to 30, 30 to 60, 60 to 90 and 90 to 120, the first Newton update reproduces the exact shift of a step to 0.2 percent, and the pattern of the ratios (4.3e+06, 1.2e+05, 1.8e+04) puts the remainder beyond 120 atoms near 1e-31. This script computes the step from 120 to 180 directly: two Newton updates from the continuation start (the second update fixes the value to eight digits in the earlier steps). No certificate is attempted for N = 180.
Predictions written before the run:
 P1 the shift after the first update is positive and lies between 1e-33 and 1e-29;
 P2 the shift after the second update differs from the one after the first by less than 1 percent (relative);
 P3 the residual before the first update is below 1e-17;
 P4 the shift 120 -> 180 is smaller than the shift 90 -> 120 (8.3076e-28) by a factor of at least 1e+03.
Usage: py q1605_update_120_180.py (about 15 minutes, one process, memory a few GB); the labels are printed by the same script.
"""

import os, sys, json, importlib.util

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
spec = importlib.util.spec_from_file_location("q1598m", os.path.join(here, "q1598_krawczyk_free90_120.py"))
q = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(q)
m = q.m
from flint import arb, arb_mat
from mpmath import mp, mpf, nstr

mp.dps = 60


def set_block_clean(N):
    m.NIN = N
    m.NOUT = N + 80
    m.KC = N + 50


q.set_block = set_block_clean
q.zfile = lambda N: os.path.join(here, "q1601_z_N%d.json" % N)
res = {}
for N0, N1 in ((120, 180),):
    zp = q.load_z(N0)
    set_block_clean(N1)
    WA = m.WallB(q.NU, q.D0, q.C0)
    dt, wt = q.formal_tail(N0, N1)
    z = zp[:N0] + dt + zp[N0 : 2 * N0] + wt + [zp[2 * N0]]
    n = 2 * N1 + 1
    d0 = z[0]
    print("N0 = %d, N1 = %d" % (N0, N1), flush=True)
    for it in range(1, 3):
        r, lq, C = WA.resid(z)
        J = WA.jac(z)
        dz = J.solve(arb_mat(n, 1, r))
        z = [z[i] - dz[i, 0].mid() for i in range(n)]
        sh = mpf(z[0].str(60, radius=False)) - mpf(d0.str(60, radius=False))
        res[it] = (sh, max(abs(float(t.mid())) for t in r))
        print(
            "   after update %d: d_1 shift so far %s (residual before %.2e)" % (it, nstr(sh, 8), res[it][1]),
            flush=True,
        )
p1 = res[1][0] > 0 and mpf("1e-33") < res[1][0] < mpf("1e-29")
p2 = abs(res[2][0] / res[1][0] - 1) < 0.01
p3 = res[1][1] < 1e-17
p4 = abs(res[2][0]) < mpf("8.3076e-28") / 1000


def lab(x):
    return "holds" if x else "does not hold"


print("P1 %s; P2 %s; P3 %s; P4 %s" % tuple(lab(x) for x in (p1, p2, p3, p4)))
print("READING q1605 done (labels by the script; no verdict word)")
