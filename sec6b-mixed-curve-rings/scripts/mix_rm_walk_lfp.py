"""mix_rm_walk_lfp.py (15 Oct 2026, task MIXL, reading, no verdict; text copy of mix_rm_walk.py): walk the positive LFP ring on the mixed
curve (lfp, rp 1.7, LAM 0.1, X1 Y1 G3, 120 Gauss-Hermite nodes, 128 bits) from rm 1.4036 down through the band [1.40137, 1.40177] of 1461, with
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
START = os.path.join(
    W, "D_newton_harm_rm1.4036_ngh120_p128_lfpmxX0_lam0.10000.json"
)  # 15 Oct: X0 Y1 G3 ring after the vertex pair died near LAM 0.0924
RMS = [
    "1.4032",
    "1.4028",
    "1.4024",
    "1.4020",
    "1.4018",
    "1.40177",
    "1.40167",
    "1.40157",
    "1.40147",
    "1.40137",
    "1.40127",
]
TAG = "_lfpmxpos"
env = dict(os.environ, FUNC="lfp", RP="1.7", SRC_S="1", LAM="0.1000", PREC="128", NGH="120", TAG=TAG)
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
    out = os.path.join(W, "D_newton_harm_rm%s_ngh120_p128%s.json" % (rm, TAG))
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
        [sys.executable, os.path.join(HERE, "I_lfp_centre.py"), out, "120", "128"],
        capture_output=True,
        text=True,
        cwd=HERE,
    )
    m = re.search(r"r\(0\) - C = ([-+0-9.e]+)", c.stdout)
    ex = m.group(1) if m else "none (%s)" % (c.stdout + c.stderr)[-200:].replace("\n", " | ")
    print(
        "rm %s: residual %.3e, smallest mass %+.6e, scan max %s, r(0) - C %s"
        % (rm, res, mmin, d.get("scanmax"), ex),
        flush=True,
    )
    if mmin <= 0 or not res < 1e-25:
        print("stop at rm %s: %s" % (rm, "a mass <= 0" if mmin <= 0 else "residual above 1e-25"), flush=True)
        break
    prevB, prevA = prevA, out
print("END mix_rm_walk_lfp", flush=True)
