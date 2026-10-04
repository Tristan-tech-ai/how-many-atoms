"""Q497: FROZEN forward predictions of the population-NPMLE change points (uniform data on [-a, a]) beyond a = 33.65,
written before any continuation goes past 33.65 (q496's arb run stopped there). Fits on the 29 change points with
a* >= 15 from q496 (results_q496_npmle_uniform.json, results_q496_npmle_uniform_mpsweep.json), y = K_after - 1/2:
  LAW3  y = c a^(4/3) + d a^(2/3) + b     (the form of the capacity and LFP count laws)
  ALOG  y = c a log a + d a + b
  A2    y = c a^2 + d a + b
Predicted a* for K_after = 47..70. The 45 -> 46 split at 33.6487 (q496, where the arb run stopped) is listed as observed,
not predicted. Writes npmle_forward_predictions.json once and refuses to overwrite it.
PRE-REGISTERED RULE (for whatever continuation later locates these changes): LAW3 is PREFERRED if its max |error| in a
over the located changes is below half of each alternative's; LAW3 FAILS if its max |error| exceeds 0.1."""

import json, os, time, numpy as np
from scipy.optimize import brentq

d = json.load(open("results_q496_npmle_uniform.json"))
mpv = json.load(open("results_q496_npmle_uniform_mpsweep.json"))
pts = {}
for src, ev in (("dbl", d["events"]), ("mp", mpv["events"])):
    for e in ev:
        if e.get("a_star") is None or e.get("K_after") is None:
            continue
        if e["K_after"] not in pts or src == "mp":
            pts[e["K_after"]] = float(e["a_star"])
A = np.array(sorted((a_, K) for K, a_ in pts.items()))
s = A[:, 0] >= 15
x, y = A[s, 0], A[s, 1] - 0.5
forms = {
    "LAW3": lambda a: [a ** (4 / 3), a ** (2 / 3), 1.0],
    "ALOG": lambda a: [a * np.log(a), a, 1.0],
    "A2": lambda a: [a**2, a, 1.0],
}
coef = {}
for k, f in forms.items():
    X = np.array([f(t) for t in x])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    coef[k] = c
    print(f"{k}: coefficients {np.round(c, 6)}, rms {np.sqrt(np.mean((X @ c - y)**2)):.4f}")
pred = []
for K in range(47, 71):
    row = dict(K_after=K)
    for k, f in forms.items():
        row[k] = brentq(lambda a: np.dot(coef[k], f(a)) - (K - 0.5), 20, 120)
    pred.append(row)
    print(f"   K -> {K}: LAW3 {row['LAW3']:8.4f}   ALOG {row['ALOG']:8.4f}   A2 {row['A2']:8.4f}")
OUT = "npmle_forward_predictions.json"
assert not os.path.exists(OUT), "refusing to overwrite"
json.dump(
    dict(
        written=time.strftime("%Y-%m-%d %H:%M:%S"),
        fit_window="a* >= 15, 29 change points to 33.076",
        coefficients={k: list(map(float, v)) for k, v in coef.items()},
        observed_not_predicted={"46": 33.6487},
        predictions=pred,
    ),
    open(OUT, "x"),
    indent=1,
)
print("written", OUT)
