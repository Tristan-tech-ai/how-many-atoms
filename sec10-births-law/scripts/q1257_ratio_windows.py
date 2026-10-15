"""q1257 (8 Oct 2026, a reading, no item): large-order ratio model r_n = e_(n+1)/e_n = a n + b + c/n fitted on windows of n, with e_2 .. e_20 (d22_out_20.json), and the increments r_(n+1) - r_n. Draft reading 4 finding 11.
Control (first): the window n = 10..17 reproduces a = 4.4450, b = 6.513, c = 38.83 of Section 6.1 to the printed digits (4.4450 within 0.0005, 6.513 within 0.0005 and 38.83 within 0.005).
Usage: py q1257_ratio_windows.py  (from temporal_within/centre_scripts/)"""

import json, re, sys
import numpy as np

E = {
    int(k[1:]): float(v)
    for k, v in json.load(open("d22_out_20.json")).items()
    if re.fullmatch(r"e\d+", k) and k != "e1"
}
r = {n: E[n + 1] / E[n] for n in range(2, 20)}


def fit(lo, hi):
    n = np.arange(lo, hi + 1)
    M = np.vstack([n, np.ones_like(n, dtype=float), 1.0 / n]).T
    return np.linalg.lstsq(M, np.array([r[k] for k in n]), rcond=None)[0]


a, b, c = fit(10, 17)
ok = abs(a - 4.4450) < 5e-4 and abs(b - 6.513) < 5e-4 and abs(c - 38.83) < 5e-3
print(f"control: window 10-17 gives a {a:.4f}, b {b:.3f}, c {c:.2f}: {ok}")
if not ok:
    print("READING q1257: VOID (control)")
    sys.exit(2)
for lo, hi in ((10, 17), (10, 19), (12, 19), (14, 19)):
    a, b, c = fit(lo, hi)
    print(f"window n = {lo}..{hi}: a {a:.4f}, b {b:.3f}, c {c:.2f}")
print(
    "increments r_(n+1) - r_n for n = 10..18: "
    + ", ".join(f"{n}: {r[n + 1] - r[n]:.4f}" for n in range(10, 19))
)
print("READING q1257 (no verdict)")
