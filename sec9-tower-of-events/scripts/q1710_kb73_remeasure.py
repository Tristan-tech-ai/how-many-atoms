"""q1710 (10 Oct 2026, READING, no verdict word, plan scratch/plan_kb73_remeasure.md): independent re-measurement of the birth K_b 73 with the fast arb walker (192 bits) from the stored tower state of the family K = 73 at A = 47.9631.
Outcomes registered before the run (written 12:10:04 by the clock, no output of this script exists):
 (a) the new event amplitude is within 1e-13 of the stored A_s = 47.96737671098957 (tower_K74_As.json): the stored measurement is sound;
 (b) the new amplitude is larger than the stored one by 11.4e-12, within 1e-12: the stored value was a measurement error of the double-precision tower;
 (c) any other difference: the tower and the arb walker disagree at this K.
Control (otherwise the run is VOID): the state re-solved at 192 bits at A = 47.9631 has a residual below 4.5e-55 (three evaluation floors of 1.5e-55) and a positive centre mass wc.
Usage: py q1710_kb73_remeasure.py (from small_gap_q206/; minutes; output folder fw_kb73_check)."""

import os, sys, json, glob, time

sys.dont_write_bytecode = True
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from flint import arb, ctx
import q1411_fast_walk as QW

ctx.prec = 192
t0 = time.time()
STORED = arb("47.96737671098957")
out = os.path.join(here, "fw_kb73_check")
W = QW.Walker(192, out, H=float(os.environ.get("Q1411_H", "0.05")))
fn = os.path.join(here, "..", "tower", "K73", "results_q220_gauss_K73_A47.9631.json")
A, u, w, we, wc = QW.load(fn, W.F)
u, w, we, wc, res, k = W.solve_at(A, u, w, we, wc)
ctl = (res < arb("4.5e-55")) and (wc is not None) and (wc > 0)
print(
    "control: re-solved state at A = %.4f: residual %.2e in %d updates, wc %s [%.0fs]"
    % (float(A), float(res), k, ("%.3e" % float(wc.mid())) if wc is not None else None, time.time() - t0),
    flush=True,
)
if not ctl:
    print("VOID: control failed")
    sys.exit(2)
r = W.run(A, (u, w, we, wc), 47.975, hmax=0.01)
ev = sorted(glob.glob(os.path.join(out, "event_Kb*.json")))
print(
    "walker returned %s; event files: %s [%.0fs]"
    % (r[0], [os.path.basename(f) for f in ev], time.time() - t0)
)
if not ev:
    print("VOID: no event file")
    sys.exit(2)
d = json.load(open(ev[0]))
Anew = arb(d["A_event"])
diff = float((Anew - STORED).mid())
print(
    "event file %s: new amplitude minus the stored one: %+.3e (A_lo, A_hi bracket width %.2e)"
    % (os.path.basename(ev[0]), diff, float((arb(d["A_hi"]) - arb(d["A_lo"])).mid()))
)
a = abs(diff) < 1e-13
b = abs(diff - 11.4e-12) < 1e-12
print(
    "outcome (a) %s; outcome (b) %s; outcome (c) %s"
    % (
        "holds" if a else "does not hold",
        "holds" if b else "does not hold",
        "holds" if not (a or b) else "does not hold",
    )
)
print("READING q1710 done (labels by the script; no verdict word)")
