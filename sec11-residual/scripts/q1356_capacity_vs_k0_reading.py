"""q1356 (8 Oct 2026, EXPLORATORY reading, amendment 2123; no item, no verdict): is the capacity the k = 0 member of the hard-wall family on the converged instrument? Events: the degree-17 capacity events K_b 20-59 of q1352_events.json. Model: the frozen global three-term model of q1355 (coefficients in q1355_predictions.json, fitted on hard-wall exponents 0.02-0.75 only; the capacity is not in the fit). Printed per K_b: the measured R / Da, the model at k = 0, the difference, and the exponent k_eff for which the model equals the measurement at the event's v^2 (solved by bisection on k in [-0.3, 0.3]); then the mean and spread of the differences and of k_eff over K_b 26-59.
Control (first): the model at k = 0.02 reproduces the measured R / Da of the hard-wall k = 0.02 events (degree 17) of q1352_events.json with rms below 0.003 (in-sample); the solved k_eff of a synthetic event built from the model at k = 0.1 is 0.1 to 1e-9.
Usage: py q1356_capacity_vs_k0_reading.py  (from temporal_within/centre_scripts/; needs the Borel-Pade Da of q1321 via the import)
"""

import io, contextlib, functools, json, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
import numpy as np
from scipy.optimize import brentq

with contextlib.redirect_stdout(io.StringIO()):
    import q1321_hardwall_common_c1_blind as B
Da = functools.lru_cache(maxsize=None)(B.Da)
js = json.load(open("q1355_predictions.json"))
c = np.array(js["coef"])
degs = js["degs"]


def model(k, v):
    out = 0.0
    p = 0
    for n, d in enumerate(degs):
        out += sum(c[p + j] * k**j for j in range(d + 1)) / v**n
        p += d + 1
    return out


ev = json.load(open("q1352_events.json"))
cap = ev["capacity_deg17"]
hw = {float(k): v for k, v in ev["hard_wall_deg17"].items()}
r02 = [abs(model(0.02, v) - R / Da(v)) for K, R, v in hw[0.02]]
rms02 = float(np.sqrt(np.mean(np.square(r02))))
kt = brentq(lambda k: model(k, 100.0) - model(0.1, 100.0), -0.3, 0.3, xtol=1e-12)
ctl = rms02 < 0.003 and abs(kt - 0.1) < 1e-9
print(
    f"control (model at k = 0.02 on the k = 0.02 events: rms {rms02:.5f}; synthetic k_eff {kt:.10f}): {ctl}"
)
if not ctl:
    print("READING q1356: VOID (control)")
    sys.exit(2)
rows = []
for K, R, v in cap:
    y = R / Da(v)
    try:
        ke = brentq(lambda k: model(k, v) - y, -0.3, 0.3, xtol=1e-10)
    except ValueError:
        ke = float("nan")
    rows.append((K, v, y, model(0.0, v), y - model(0.0, v), ke))
print("K_b   v^2      R/Da    model(k=0)   difference   k_eff")
for K, v, y, m0, d, ke in rows:
    print(f"{K:3d}  {v:7.2f}  {y:7.4f}  {m0:9.4f}  {d:+10.4f}  {ke:+8.4f}")
sel = [r for r in rows if r[0] >= 26]
d = np.array([r[4] for r in sel])
ke = np.array([r[5] for r in sel])
print(
    f"K_b 26-59: mean difference {d.mean():+.4f}, rms {np.sqrt(np.mean(d ** 2)):.4f}, largest |difference| {np.abs(d).max():.4f}; k_eff mean {np.nanmean(ke):+.4f}, spread (sd) {np.nanstd(ke):.4f}"
)
print("READING q1356 (exploratory, no verdict)")
