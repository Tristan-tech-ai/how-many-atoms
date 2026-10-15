"""mix_rm_walk.py (14 Oct 2026, task MIX301, reading, no verdict): walk the positive NPMLE ring on the mixed curve (rd, s 2, rp 2.1, LAM 0.2,
K 20, 200 Gauss-Hermite nodes, 160 bits) from rm 1.7416 down through the band [1.73615, 1.73655] registered in amendment 1463, with
D_solver.py newton (tangent predictor for the first step, secant predictor in harmonic coordinates after), and read the center excess
D(0) - C of every ring with I_rd_centre.py. Prints per rm: residual, smallest mass, scan maximum, D(0) - C. Stops at the first ring whose
smallest mass is <= 0 or whose residual is above 1e-25. The verdict of 1463 on these rings is read by a separate registered script.
Start: the last state of mix_lam_continuation2.py at LAM 0.2000 (D_work/D_newton_harm_rm1.7416_rdmxcont2_lam0.2000.json).
Usage (from temporal_within): py mix_rm_walk.py"""

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "D_work")
START = os.path.join(W, "D_newton_harm_rm1.7416_rdmxcont2_lam0.2000.json")
RMS = [
    "1.7410",
    "1.7404",
    "1.7398",
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
TAG = "_rdmxpos_K20"
env = dict(os.environ, FUNC="rd", RP="2.1", SRC_S="2", LAM="0.2000", PREC="160", NGH="200", TAG=TAG)
if not os.path.exists(START):
    print("no start state %s" % START)
    sys.exit(2)
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
    d = json.load(open(out))
    mmin = min(float(x) for x in d["m"])
    res = float(d.get("residual", "nan"))
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
        % (rm, res, mmin, d.get("scanmax"), ex),
        flush=True,
    )
    if mmin <= 0 or not res < 1e-25:
        print("stop at rm %s: %s" % (rm, "a mass <= 0" if mmin <= 0 else "residual above 1e-25"), flush=True)
        break
    prevB, prevA = prevA, out
print("END mix_rm_walk", flush=True)
