"""mix_rm_walk_x0.py (14 Oct 2026, task MIX300/MIX301, reading, no verdict): continue the positive NPMLE ring on the mixed curve (rd, s 2,
rp 2.1, LAM 0.2, 200 nodes, 160 bits) past the death of its vertex pair. mix_rm_walk.py found the X pair mass going through zero between
rm 1.7398 (+2.95e-3) and 1.7392 (-5.97e-4) while every other orbit stayed positive; at a death the atom leaves the support, so below it
the ring is X0 Y1 G4 (18 atoms). Start: the rm 1.7398 state with the X entry removed, Newton at rm 1.7392 (tangent predictor), then the
secant walk through the band of amendment 1463. Check that removal is right: the stored scan of D - C off the atoms (which includes the
vertex, 0 deg) must stay <= 0. Output lines in the format of mix_rm_walk.py, read by small_gap_q206/q1888 (criterion unchanged).
Usage (from temporal_within): py mix_rm_walk_x0.py"""

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "D_work")
SRC = os.path.join(W, "D_newton_harm_rm1.7398_rdmxpos_K20.json")
d = json.load(open(SRC))
assert d["X"] == 1
d["X"] = 0
d["m"] = d["m"][1:]  # the X entry is the first mass (orbit order X, Y, G1..Gn)
START = os.path.join(W, "D_x0start_rm1.7398_rdmxpos.json")
json.dump(d, open(START, "w"), indent=1)
RMS = [
    "1.7392",
    "1.7386",
    "1.7380",
    "1.7375",
    "1.7370",
    "1.7368",
    "1.73665",
    "1.73655",
    "1.73645",
    "1.73635",
    "1.73625",
    "1.73615",
    "1.73605",
]
TAG = "_rdmxpos_K18x0"
env = dict(os.environ, FUNC="rd", RP="2.1", SRC_S="2", LAM="0.2000", PREC="160", NGH="200", TAG=TAG)
prevA, prevB = START, None
for rm in RMS:
    args = [
        sys.executable,
        os.path.join(HERE, "D_solver.py"),
        "newton",
        prevA,
        prevB or "-",
        rm,
        "harm",
        "30",
    ]
    r = subprocess.run(args, env=env, capture_output=True, text=True, cwd=HERE)
    out = os.path.join(W, "D_newton_harm_rm%s%s.json" % (rm, TAG))
    if not os.path.exists(out):
        print(
            "rm %s: no output; solver tail: %s" % (rm, (r.stdout + r.stderr)[-300:].replace("\n", " | ")),
            flush=True,
        )
        break
    s = json.load(open(out))
    mmin = min(float(x) for x in s["m"])
    res = float(s.get("residual", "nan"))
    c = subprocess.run(
        [sys.executable, os.path.join(HERE, "I_rd_centre.py"), out, "200", "160"],
        capture_output=True,
        text=True,
        cwd=HERE,
    )
    m = re.search(r"D\(0\) - C = ([-+0-9.e]+)", c.stdout)
    ex = m.group(1) if m else "none (%s)" % (c.stdout + c.stderr)[-200:].replace("\n", " | ")
    print(
        "rm %s: residual %.3e, smallest mass %+.6e, scan max %s, D(0) - C %s"
        % (rm, res, mmin, s.get("scanmax"), ex),
        flush=True,
    )
    if mmin <= 0 or not res < 1e-25:
        print("stop at rm %s: %s" % (rm, "a mass <= 0" if mmin <= 0 else "residual above 1e-25"), flush=True)
        break
    prevB, prevA = prevA, out
print("END mix_rm_walk_x0", flush=True)
