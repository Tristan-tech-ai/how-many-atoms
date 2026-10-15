"""q1615 (10 Oct 2026, READING, no verdict word; deep proof Q11 step 69): the shape of the births offset with the pair of the exact wall block.
q1614: with the c20 block pair (-0.093773648256, +0.001159913641) fixed, one free parameter E_4 reproduces the 48 measured births with residual rms 3.02e-10 and largest 6.37e-10, from an offset (E_4 = 0) of rms 1.15e-08 and mean +1.05e-08. The fitted E_4 = -8.006e+05 is 0.896 of the value of the three-parameter fit of q1567. The offset with this pair has a mean of +1.05e-08, a spread of 1.717e-08 and a largest value of 2.170e-08 at K_b 83 (q1611), so it may be of one sign. The term E_4 / v^12 with v = 4 pi rho has the shape rho^-12, and rho^4 is proportional to K at the centre, so the term falls like K^-3.
Test of the shape: take the 48 offsets of the births law with the block pair and E_4 = 0, and fit offset(K) = c K^-p.
Predictions written before the run:
 P1 all 48 offsets are positive for the c20 block pair;
 P2 the log-log slope p of the offsets against K_b (least squares on log offset) lies between 2.5 and 3.5 for the c20 block pair;
 P3 the same slope lies between 2.5 and 3.5 for the c24 block pair (-0.093773650, +0.001159915);
 P4 the power-law fit with c and p both free leaves an rms below 1.5e-10 for the c20 block pair.
Usage: py q1615_offset_shape.py (about 2 minutes; executes q1567 silently for its functions)."""

import io, contextlib, os
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(H, "q1567_missing_E4.py"), "__name__": "q1567ns"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(H, "q1567_missing_E4.py"), encoding="utf-8").read(), ns)
offs, Ks = ns["offs"], np.array(ns["Ks"], dtype=float)
rm = lambda x: float(np.sqrt((x**2).mean()))
res = {}
for name, pair in (
    ("c20 block pair", (-0.093773648256, 0.001159913641)),
    ("c24 block pair", (-0.093773650, 0.001159915)),
):
    r = offs(pair[0], pair[1], 0.0)
    pos = bool((r > 0).all())
    if pos:
        A = np.vstack([np.ones(len(Ks)), np.log(Ks)]).T
        coef, *_ = np.linalg.lstsq(A, np.log(r), rcond=None)
        p = -float(coef[1])
        c = float(np.exp(coef[0]))
        fit = c * Ks ** (-p)
        rr = r - fit
    else:
        p = float("nan")
        rr = r
    res[name] = dict(pos=pos, p=p, rms=rm(rr))
    print(
        "%s: offsets from %+.3e to %+.3e (K_b 83 first, K_b 130 last: %+.3e, %+.3e), all positive: %s; fitted power p = %.3f, fit rms %.2e"
        % (name, r.min(), r.max(), r[0], r[-1], pos, p, rm(rr))
    )
a = res["c20 block pair"]
b = res["c24 block pair"]
lab = lambda p: "holds" if p else "does not hold"
p1 = a["pos"]
p2 = a["pos"] and 2.5 <= a["p"] <= 3.5
p3 = b["pos"] and 2.5 <= b["p"] <= 3.5
p4 = a["rms"] < 1.5e-10
print(
    "P1 all offsets positive: %s; P2 slope in [2.5, 3.5] for c20: %s; P3 slope in [2.5, 3.5] for c24: %s; P4 power-law fit rms below 1.5e-10: %s"
    % (lab(p1), lab(p2), lab(p3), lab(p4))
)
print("READING q1615 done (labels by the script; no verdict word)")
