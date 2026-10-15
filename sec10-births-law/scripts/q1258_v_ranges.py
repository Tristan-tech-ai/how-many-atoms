"""q1258 (8 Oct 2026, a reading, no item): v = 4 pi rho at the order-16 root of the births law for the 20 capacity events K_b 61 to 80 and the 13 NPMLE transitions K_b 54 to 66 (cubic spline of N_half as q1209). Draft reading 4 finding 10(a).
Control (first): the capacity K_b 80 root a_16 equals 51.484240309367 within 1e-9 less (the q1211 control value, 1e-10 allowed) and v at K_b 80 is between 12.4 and 12.8.
Usage: py q1258_v_ranges.py  (from temporal_within/centre_scripts/)"""

import os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
import q1209_births_law_derived_series as G


def v_at(N, K):
    a = G.pred_series(N, K, 16)
    rho = (N(a + G.H) - N(a - G.H)) / (2 * G.H)
    return a, 4 * np.pi * rho


a80, v80 = v_at(G.Ncap, 80)
ok = abs(a80 - 51.484240309367) < 1e-10 and 12.4 < v80 < 12.8
print(f"control: capacity K_b 80 a_16 {a80:.12f}, v {v80:.3f}: {ok}")
if not ok:
    print("READING q1258: VOID (control)")
    sys.exit(2)
vc = [v_at(G.Ncap, K)[1] for K in range(61, 81)]
vn = [v_at(G.Nnp, K)[1] for K in range(54, 67)]
print(
    f"capacity K_b 61-80: v {min(vc):.3f} to {max(vc):.3f}; NPMLE K_b 54-66: v {min(vn):.3f} to {max(vn):.3f}"
)
print("READING q1258 (no verdict)")
