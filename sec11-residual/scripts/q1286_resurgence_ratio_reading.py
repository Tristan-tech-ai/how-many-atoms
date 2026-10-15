"""q1286 (8 Oct 2026, EXPLORATORY reading, no item): the ratio sigma_k(K_b) = R_k / Da of the measured residual R_k (1e-9) to the Stokes-type ambiguity of the median sum, Da(v^2) = |Im S_+(1/v^2)| / (2 rho) (1e-9; S_+ the lateral Borel sum at theta = 0.3 of the diagonal Pade [8/8] of the series e_2 .. e_18, as q1285), at the v^2 of each event, for the NPMLE (k = 1), immune (k = 2) and k = 3 series (events of q1282 with the same inclusion rule).
Control (first): the same evaluation as q1285 reproduces Da(115) = 0.21082e-9 within 1e-4 relative, and Da is monotone decreasing over the grid used.
Usage: py q1286_resurgence_ratio_reading.py  (from temporal_within/centre_scripts/)"""

import io, contextlib, sys
import numpy as np
import mpmath as mp

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import q1282_extended_decay_reading as D
    import q1220_borel_pade_births_law as T
mp.mp.dps = 30
Bf, p, q = T.machine(T.coef_series(), 8, 8)


def Da(v2):
    x = mp.mpf(1) / v2
    return abs(float(mp.im(T.ray(Bf, x, 0.3)))) / (2 * np.sqrt(v2) / (4 * np.pi)) * 1e9


ok = abs(Da(115.0) - 0.21082) < 1e-4 * 0.21082 and Da(100.0) > Da(110.0) > Da(120.0)
print(f"control: Da(115) = {Da(115.0):.5f}e-9: {ok}")
if not ok:
    print("READING q1286: VOID (control)")
    sys.exit(2)
for k in (1, 2, 3):
    rows = D.data.get(k, [])
    print(f"k = {k}: sigma = R / Da, events {len(rows)}")
    line = []
    for K, r, v in rows:
        if K in (14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 44, 48, 50, 52):
            line.append(f"K_b {K} (v2 {v:.0f}): {r / Da(v):.3f}")
    print("   " + "; ".join(line))
print("READING q1286 (exploratory, no verdict)")
