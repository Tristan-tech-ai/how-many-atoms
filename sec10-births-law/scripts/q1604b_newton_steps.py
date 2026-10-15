"""q1604b (10 Oct 2026, READING, post hoc, written after q1604 was read, no prediction registered, no verdict word): the first depth after each Newton update when the clean block N1 is started from the solved clean block N0 plus the formal law atoms (the continuation start of q1602), for (N0, N1) = (60, 90) and (90, 120). The first update is the linear response through the Jacobian at the start state, which differs from the Jacobian at the solved N1 state used in q1604. The exact shifts are those of q1602. Usage: py q1604b_newton_steps.py"""

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
for N0, N1 in ((60, 90), (90, 120)):
    zp = q.load_z(N0)
    set_block_clean(N1)
    WA = m.WallB(q.NU, q.D0, q.C0)
    dt, wt = q.formal_tail(N0, N1)
    z = zp[:N0] + dt + zp[N0 : 2 * N0] + wt + [zp[2 * N0]]
    n = 2 * N1 + 1
    d0 = z[0]
    zfin = q.load_z(N1)[0]
    print(
        "N0 = %d, N1 = %d: exact shift %s"
        % (N0, N1, nstr(mpf(zfin.str(60, radius=False)) - mpf(d0.str(60, radius=False)), 8)),
        flush=True,
    )
    for it in range(1, 5):
        r, lq, C = WA.resid(z)
        J = WA.jac(z)
        dz = J.solve(arb_mat(n, 1, r))
        z = [z[i] - dz[i, 0].mid() for i in range(n)]
        print(
            "   after update %d: d_1 shift so far %s (residual before %.2e)"
            % (
                it,
                nstr(mpf(z[0].str(60, radius=False)) - mpf(d0.str(60, radius=False)), 8),
                max(abs(float(t.mid())) for t in r),
            ),
            flush=True,
        )
print("READING q1604b done (post hoc; no verdict word)")
