"""q1288 (8 Oct 2026, EXPLORATORY reading, no item): the ratio sigma_k = R_k / Da of q1286 with the ambiguity Da computed from the diagonal Pade [9/9] of the series to e_20 (q1223) instead of [8/8] to e_18, theta 0.3, for k = 1, 2, 3 (events of q1282). Is the variation of sigma with v^2 smaller with the longer series?
Control (first): the median sum Re (S_+ + S_-)/2 of the [9/9] approximant at x = 1/115 reproduces the table q1223_table.json (key 0.3) within 1e-10 relative.
Usage: py q1288_sigma_e20_reading.py  (from temporal_within/centre_scripts/)"""

import io, contextlib, json, sys
import numpy as np
import mpmath as mp

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import q1282_extended_decay_reading as D
    import q1220_borel_pade_births_law as T18
    import q1223_borel_pade_e20 as T20
mp.mp.dps = 30
B18 = T18.machine(T18.coef_series(), 8, 8)[0]
B20 = T20.machine(T20.coef_series(), 9, 9)[0]
tab = json.load(open("q1223_table.json"))
tab.pop("sha256")
T3 = np.polynomial.chebyshev.Chebyshev(tab["0.3"], domain=tab["domain"])
x0 = mp.mpf(1) / 115
med = float(mp.re((T20.ray(B20, x0, 0.3) + T20.ray(B20, x0, -0.3)) / 2))
ok = abs(med - float(T3(1 / 115.0))) < 1e-10 * abs(med)
print(f"control: [9/9] median sum at x = 1/115 {med:.6e} against the table {float(T3(1 / 115.0)):.6e}: {ok}")
if not ok:
    print("READING q1288: VOID (control)")
    sys.exit(2)


def Da(Bf, ray, v2):
    x = mp.mpf(1) / v2
    return abs(float(mp.im(ray(Bf, x, 0.3)))) / (2 * np.sqrt(v2) / (4 * np.pi)) * 1e9


print(
    "Da at v^2 = 70, 85, 100, 115, 130: [8/8] "
    + ", ".join(f"{Da(B18, T18.ray, v):.4g}" for v in (70, 85, 100, 115, 130))
    + "; [9/9] "
    + ", ".join(f"{Da(B20, T20.ray, v):.4g}" for v in (70, 85, 100, 115, 130))
)
for k in (1, 2, 3):
    rows = [(K, r, v) for K, r, v in D.data.get(k, []) if K in (14, 18, 22, 26, 30, 34, 36, 40, 44)]
    s18 = [r / Da(B18, T18.ray, v) for K, r, v in rows]
    s20 = [r / Da(B20, T20.ray, v) for K, r, v in rows]
    print(f"k = {k}: K_b {[K for K, _, _ in rows]}")
    print(f"   sigma [8/8]: " + ", ".join(f"{x:.3f}" for x in s18) + f"; range {max(s18) / min(s18) - 1:.3f}")
    print(f"   sigma [9/9]: " + ", ".join(f"{x:.3f}" for x in s20) + f"; range {max(s20) / min(s20) - 1:.3f}")
print("READING q1288 (exploratory, no verdict)")
