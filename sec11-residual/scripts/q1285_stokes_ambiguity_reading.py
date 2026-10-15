"""q1285 (8 Oct 2026, a reading, no item): the size of the Stokes-type ambiguity of the median-sum law. For x = 1/v^2 at v^2 = 100, 105, ..., 130 (K_b 29 to 52) the imaginary part of the lateral Borel sum S_+(x) (theta = 0.3, diagonal Pade [8/8], coefficients e_2 .. e_18 as q1220) is computed; S_med = Re (S_+ + S_-)/2 is the median sum, and |Im S_+| measures the ambiguity of the resummation. Converted to the amplitude by Delta a = |Im S_+| / (2 rho) with rho = v / (4 pi) (the law is 2 N(a) = K_b + 1/6 + delta, derivative 2 N' = 2 rho), in units 1e-9, to compare with the common offset of q1283 (0.006e-9, and the values 0.0115, 0.009, 0.0037e-9 inferred at K_b 40, 44, 48 in amendment 2066).
Control (first): the Re part reproduces the table of q1220 at x = 1/115 within 1e-10 (relative to its size), and the imaginary part of the lateral sum of the synthetic branch-point series of q1220 (ctl_exact) at x = 1/115 is the same on the rays theta = 0.3 and 0.5 to 1e-5 relative (analytic deformation) and nonzero (the ambiguity is of order exp(-v^2 / alpha) for every x; a first version of this control, zero below the singularity, was conceptually wrong and failed before any value of the Pade series was read).
Usage: py q1285_stokes_ambiguity_reading.py  (from temporal_within/centre_scripts/)"""

import io, contextlib, json, sys
import numpy as np
import mpmath as mp

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import q1220_borel_pade_births_law as T
mp.mp.dps = 30
Bf, p, q = T.machine(T.coef_series(), 8, 8)
tab = json.load(open("q1220_table.json"))
tab.pop("sha256")
T3 = np.polynomial.chebyshev.Chebyshev(tab["0.3"], domain=tab["domain"])
x0 = mp.mpf(1) / 115
sp = T.ray(Bf, x0, 0.3)
sm = T.ray(Bf, x0, -0.3)
re_med = float(mp.re((sp + sm) / 2))
ok1 = abs(re_med - float(T3(1 / 115.0))) < 1e-10 * abs(re_med)
xs_c = mp.mpf(1) / 115
i3 = mp.im(T.ray(T.ctl_exact, xs_c, 0.3))
i5 = mp.im(T.ray(T.ctl_exact, xs_c, 0.5))
below = abs(i3 - i5) < 1e-5 * abs(i3)
above = abs(i3) > 0
print(
    f"control: median sum at x = 1/115 {re_med:.6e} vs table {float(T3(1 / 115.0)):.6e}: {ok1}; control branch-point series: Im S_+ at theta 0.3 {float(i3):.6e}, at 0.5 {float(i5):.6e} (invariant to 1e-5 relative): {below} (nonzero: {above})"
)
if not (ok1 and below and above):
    print("READING q1285: VOID (control)")
    sys.exit(2)
print(
    "v^2    K_b(approx)   median sum delta      Im S_+          Delta a = |Im|/(2 rho) in 1e-9      exp(-v^2/4.445)"
)
for v2 in range(100, 131, 5):
    x = mp.mpf(1) / v2
    sp = T.ray(Bf, x, 0.3)
    rho = np.sqrt(v2) / (4 * np.pi)
    d = abs(float(mp.im(sp))) / (2 * rho) * 1e9
    print(
        f"{v2:4d}   {float(mp.re(sp)):+.4e}  {float(mp.im(sp)):+.4e}   {d:.5f}    {np.exp(-v2 / 4.445):.3e}"
    )
print("READING q1285 (no verdict)")
