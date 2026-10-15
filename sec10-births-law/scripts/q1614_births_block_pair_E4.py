"""q1614 (10 Oct 2026, READING, no verdict word; deep proof Q11 step 68): the births law with the pair of the exact wall block and the fifth term E_4 as the only free parameter.
Two facts. (1) q1567 fitted (nu0, d0, E_4) to the 48 measured births K_b 83..130 (law through c20, Delta with E_0..E_3 and a free E_4/v^12) and found the pair (-0.093773640, +0.001159907), E_4 = -8.94e+05, residual rms 9.1e-12, largest 2.5e-11. With the pair alone the residual rms was 8.9e-10 and the largest 2.35e-09. (2) q1610 found the self-consistent pair of the interior of the exact clean block of 90 atoms by the wall problem alone: (-0.093773648256, +0.001159913641) for the law through c20, and q1613 found (-0.093773650, +0.001159915) for the law through c24. This pair lies (-8.3e-09, +6.6e-09) from the q1567 pair. The wall route and the births route are independent: the first uses no capacity datum.
Test: fix the pair at the block pair and fit E_4 alone (linear least squares with explicit steps, as in q1567).
Predictions written before the run:
 P1 with the c20 block pair fixed and E_4 the only free parameter, the residual rms over the 48 births is below 3e-10 and the largest residual below 1e-09;
 P2 the fitted E_4 lies between -1.34e+06 and -4.5e+05 (within 50 percent of the q1567 value -8.94e+05);
 P3 positive control: the three-parameter fit of q1567 reproduced by this script's linear method has a residual rms between 4.5e-12 and 1.8e-11;
 P4 the one-parameter fit with the c20 block pair has a residual rms below that of the two-parameter pair-only fit (8.9e-10).
Usage: py q1614_births_block_pair_E4.py (about 3 minutes; executes q1567 silently for its functions)."""

import io, contextlib, os
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(H, "q1567_missing_E4.py"), "__name__": "q1567ns"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(H, "q1567_missing_E4.py"), encoding="utf-8").read(), ns)
offs, INS, Ks, E4f, rf = ns["offs"], ns["INS"], ns["Ks"], ns["E4f"], ns["rf"]
rm = lambda x: float(np.sqrt((x**2).mean()))
print(
    "q1567 three-parameter fit reproduced: E_4 = %.4g, residual rms %.2e, largest %.2e"
    % (E4f, rm(rf), np.abs(rf).max())
)


def fit_E4(nu, d0):
    base = offs(nu, d0, 0.0)
    h = 1e5
    col = (offs(nu, d0, h) - offs(nu, d0, -h)) / (2 * h)
    e4 = -float(col @ base) / float(col @ col)
    return e4, offs(nu, d0, e4), base


out = {}
for name, pair in (
    ("c20 block pair", (-0.093773648256, 0.001159913641)),
    ("c24 block pair", (-0.093773650, 0.001159915)),
    ("limit pair of q1564", (-0.093773592, 0.001159849)),
):
    e4, r, base = fit_E4(*pair)
    out[name] = (e4, r)
    print(
        "%s (%+.9f, %+.9f): E_4 = %.4g (%.3f x the q1567 value), residual rms %.2e, largest %.2e, mean %+.2e (without E_4: rms %.2e, mean %+.2e)"
        % (name, pair[0], pair[1], e4, e4 / E4f, rm(r), np.abs(r).max(), r.mean(), rm(base), base.mean())
    )
e4, r = out["c20 block pair"]
lab = lambda p: "holds" if p else "does not hold"
p1 = rm(r) < 3e-10 and np.abs(r).max() < 1e-9
p2 = -1.34e6 <= e4 <= -4.5e5
p3 = 4.5e-12 <= rm(rf) <= 1.8e-11
p4 = rm(r) < 8.9e-10
print(
    "P1 rms below 3e-10 and largest below 1e-09: %s; P2 E_4 in [-1.34e+06, -4.5e+05]: %s; P3 control reproduces the q1567 rms: %s; P4 rms below the pair-only 8.9e-10: %s"
    % (lab(p1), lab(p2), lab(p3), lab(p4))
)
print("READING q1614 done (labels by the script; no verdict word)")
