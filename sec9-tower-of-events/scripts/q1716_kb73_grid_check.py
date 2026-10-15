"""q1716 (10 Oct 2026, READING, no verdict word): does the re-measured amplitude of K_b 73 (47.967376710994066746 at the quadrature spacing H = 0.05, reading 171) depend on the spacing H of the fast walker? The run starts from the same stored state (tower/K73 at A = 47.9631), re-solves it at 192 bits (tolerance 1e-54) and locates the event without after_split, for H = 0.04 and H = 0.07.
Predictions written before the run:
 P1 for H = 0.04 and H = 0.07 the located amplitude differs from the H = 0.05 value by less than 1e-14;
 P2 the control residual of the re-solved state is below 4.5e-55 for both spacings.
(The trapezoid error exp(-7.6/h) of reading 87 is below 1e-30 at h = 0.07, which is the reason for the expectation.)
Usage: py q1716_kb73_grid_check.py (about 2 minutes, one process)."""

import os, sys, json, time

sys.dont_write_bytecode = True
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from flint import arb, ctx
import q1411_fast_walk as QW

ctx.prec = 192
REF = arb("47.967376710994066746")
t0 = time.time()
fn = os.path.join(here, "..", "tower", "K73", "results_q220_gauss_K73_A47.9631.json")
res = {}
for H in (0.04, 0.07):
    W = QW.Walker(192, os.path.join(here, "fw_kb73_gridcheck_H%.2f" % H), H=H)
    A, u, w, we, wc = QW.load(fn, W.F)
    u, w, we, wc, rs, k = W.solve_at(A, u, w, we, wc, tol="1e-54")
    ctl = bool(rs < arb("4.5e-55")) and wc is not None and bool(wc > 0)
    r = W.walk_family(A, u, w, we, wc, 47.975, hmax=0.01)
    if r[0] != "event":
        print("H = %.2f: walk returned %s" % (H, r[0]))
        res[H] = None
        continue
    _, A_lo, st_lo, s_lo, A_hi, st_hi, s_hi = r
    out = W.locate((A_lo, st_lo, s_lo), (A_hi, st_hi, s_hi))
    A_ev = out[6]
    diff = float((A_ev - REF).mid())
    res[H] = (diff, ctl, float(rs))
    print(
        "H = %.2f: control residual %.1e (%s); event minus the H = 0.05 value: %+.3e [%.0fs]"
        % (H, float(rs), "ok" if ctl else "FAILED", diff, time.time() - t0),
        flush=True,
    )
lab = lambda x: "holds" if x else "does not hold"
p1 = all(v is not None and abs(v[0]) < 1e-14 for v in res.values())
p2 = all(v is not None and v[1] for v in res.values())
print("P1 %s; P2 %s" % (lab(p1), lab(p2)))
print("READING q1716 done (labels by the script; no verdict word)")
