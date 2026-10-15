"""mix_lam_continuation_lfp4.py (15 Oct 2026, task MIXL: after the vertex pair dies near LAM 0.0924, the X0 Y1 G3 ring from LAM 0.093 to 0.1; text copy of mix_lam_continuation2.py for FUNC lfp, rp 1.7, K 18, rm 1.4036, NGH 120, PREC 128, start = the 1443 ellipse state; reading, no verdict): continuation in the curve parameter LAM with a secant
predictor, from the NPMLE ellipse ring (rd, rp 2.1, K 20, rm 1.7416) towards the mixed curve LAM 0.2 at fixed rm.
Why v2: the first run (mix_lam_continuation.py) jumped LAM 0 -> 0.02 with the old state as the start and landed on a ring with smallest
mass -2286; the control at LAM 1e-12 (general-LAM code path, D_work/D_newton_harm_rm1.7416_rdmxcont_ctrl.json) reproduces the ellipse
state (residual 2.6e-40, smallest mass +0.0399), so the code path is sound and the start was the fault.
Step: the start of LAM_k is the linear extrapolation (40 digits, mpmath) of the converged states at LAM_{k-1} and LAM_{k-2}; Newton at
fixed rm by D_solver.py ('same rm' branch). Prints per step the residual, the smallest mass and the scan maximum; stops at the first step
whose smallest mass is <= 0 or whose residual is above 1e-25 (prediction written before the run).
Usage (from temporal_within): py mix_lam_continuation2.py [STEP] [LAM_END]"""

import json
import os
import subprocess
import sys

import mpmath as mp

mp.mp.dps = 50
HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "D_work")
step = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
lam_end = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
rm = "1.4036"
prev2 = (0.0925, os.path.join(W, "D_newton_harm_rm1.4036_ngh120_p128_lfpmxX0_lam0.09250.json"))
prev1 = (0.093, os.path.join(W, "D_newton_harm_rm1.4036_ngh120_p128_lfpmxX0_lam0.09300.json"))


def extrapolate(fa, la, fb, lb, lam):
    a, b = json.load(open(fa)), json.load(open(fb))
    t = mp.mpf(lam - la) / mp.mpf(la - lb)
    ext = lambda x, y: mp.nstr(mp.mpf(x) + (mp.mpf(x) - mp.mpf(y)) * t, 45)
    out = dict(a)
    out["G"] = [ext(x, y) for x, y in zip(a["G"], b["G"])]
    out["m"] = [ext(x, y) for x, y in zip(a["m"], b["m"])]
    out["C"] = ext(a["C"], b["C"])
    return out


k = 1
while 0.093 + k * step <= lam_end + 1e-12:
    lam = round(0.093 + k * step, 6)
    lams = "%.5f" % lam
    start = (
        json.load(open(prev1[1]))
        if prev2 is None
        else extrapolate(prev1[1], prev1[0], prev2[1], prev2[0], lam)
    )
    sp = os.path.join(W, "mixcontlfpX0_start_lam%s.json" % lams)
    json.dump(start, open(sp, "w"), indent=1)
    tag = "_lfpmxX0_lam" + lams
    env = dict(os.environ, FUNC="lfp", RP="1.7", SRC_S="1", LAM=lams, PREC="128", NGH="120", TAG=tag)
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "D_solver.py"), "newton", sp, "-", rm, "harm", "30"],
        env=env,
        capture_output=True,
        text=True,
        cwd=HERE,
    )
    out = os.path.join(W, "D_newton_harm_rm%s_ngh120_p128%s.json" % (rm, tag))
    if not os.path.exists(out):
        print(
            "LAM %s: no output; solver tail: %s" % (lams, (r.stdout + r.stderr)[-300:].replace("\n", " | ")),
            flush=True,
        )
        break
    d = json.load(open(out))
    mmin = min(float(x) for x in d["m"])
    res = float(d.get("residual", "nan"))
    print(
        "LAM %s: residual %.3e, smallest mass %+.6e, scan max %s" % (lams, res, mmin, d.get("scanmax")),
        flush=True,
    )
    os.remove(sp)
    if mmin <= 0 or not res < 1e-25:
        print(
            "stop at LAM %s: %s" % (lams, "a mass <= 0" if mmin <= 0 else "residual above 1e-25"), flush=True
        )
        break
    prev2, prev1 = prev1, (lam, out)
    k += 1
print("END mix_lam_continuation_lfp4", flush=True)
