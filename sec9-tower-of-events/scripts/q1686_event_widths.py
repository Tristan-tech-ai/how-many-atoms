"""q1686 (10 Oct 2026, READING, no verdict word, predictions written at 10:19): bracket widths W = A_hi - A_lo of the stored capacity events K_b 83..138 (fw_prod), computed from the high-precision strings of the event files with mpmath; only widths are printed, never an amplitude (BLIND RULE). The widths are compared with the in-sample rms floor 5.241e-14 of the cell (NG 14, nmax 14) found in q1684.
Predictions:
 P1 the median width over the events is below 1e-12;
 P2 the largest width is below 1e-10;
 P3 the in-sample rms 5.241e-14 lies between 1/3 and 3 times the median half-width W/2.
Control: a synthetic event file with A_lo and A_hi 1e-13 apart gives the width 1e-13 through the same code.
Usage: py q1686_event_widths.py (from small_gap_q206/; seconds)."""

import json, os, tempfile
import numpy as np
import mpmath as mp

mp.mp.dps = 60
H = os.path.dirname(os.path.abspath(__file__))


def width(fn):
    d = json.load(open(fn))
    return float(mp.mpf(d["A_hi"]) - mp.mpf(d["A_lo"]))


W = {}
for K in range(83, 139):
    fn = os.path.join(H, "fw_prod", "event_Kb%d.json" % K)
    if os.path.exists(fn):
        W[K] = width(fn)
v = np.array(list(W.values()))
med = float(np.median(v))
print(
    "events %d; width: median %.3e, mean %.3e, smallest %.3e, largest %.3e (K_b %d); widths (1e-14) at K_b 83, 100, 130, 138: %s"
    % (
        len(v),
        med,
        v.mean(),
        v.min(),
        v.max(),
        max(W, key=W.get),
        ", ".join("%d %.1f" % (K, W[K] * 1e14) for K in (83, 100, 130, 138) if K in W),
    )
)
with tempfile.TemporaryDirectory() as t:
    fn = os.path.join(t, "event_Kb999.json")
    json.dump(
        {"A_event": "70.5", "A_lo": "70.5", "A_hi": str(mp.mpf("70.5") + mp.mpf("1e-13"))}, open(fn, "w")
    )
    cw = width(fn)
lab = lambda q: "holds" if q else "does not hold"
p1 = med < 1e-12
p2 = v.max() < 1e-10
p3 = (med / 2) / 3 <= 5.241e-14 <= (med / 2) * 3
ctl = abs(cw / 1e-13 - 1) < 1e-6
print("ratio of the rms floor 5.241e-14 to the median half-width: %.3f" % (5.241e-14 / (med / 2)))
print(
    "P1 median width below 1e-12: %s; P2 largest width below 1e-10: %s; P3 rms floor within a factor 3 of the median half-width: %s; control (synthetic width %.3e): %s"
    % (lab(p1), lab(p2), lab(p3), cw, lab(ctl))
)
print("READING q1686 done (labels by the script; no verdict word)")
