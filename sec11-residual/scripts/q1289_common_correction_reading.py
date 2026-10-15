"""q1289 (8 Oct 2026, EXPLORATORY reading, no item): is the variation of sigma_k = R_k / Da with v^2 a common factor (1 + g / v^2) (the first correction in 1/v^2 of the transseries prefactor)? Fit ln(R_k / Da) = ln sigma_k_inf + ln(1 + g / v^2) with g common to k = 1, 2, 3 and sigma_k_inf free, on the events with K_b <= 40 (v^2 <= 115, where the common offset of amendment 2066 is negligible), using the ambiguity Da of the [8/8] approximant (q1286; the [9/9] one agrees to 0.1 percent, q1288). Also the fit with g free per k.
Control (first): the routine recovers g = -13 and sigma_inf = (2.9, 5.4, 7.7) from noiseless synthetic data at the real v^2 values to 1e-8.
Usage: py q1289_common_correction_reading.py  (from temporal_within/centre_scripts/)"""

import io, contextlib, sys
import numpy as np
import mpmath as mp
from scipy.optimize import least_squares

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import q1282_extended_decay_reading as D
    import q1220_borel_pade_births_law as T
mp.mp.dps = 30
Bf = T.machine(T.coef_series(), 8, 8)[0]


def Da(v2):
    x = mp.mpf(1) / v2
    return abs(float(mp.im(T.ray(Bf, x, 0.3)))) / (2 * np.sqrt(v2) / (4 * np.pi)) * 1e9


pts = {k: [(K, r, v) for K, r, v in D.data.get(k, []) if K <= 40 and K != 38] for k in (1, 2, 3)}
dd = {k: np.array([Da(v) for _, _, v in pts[k]]) for k in pts}
vv = {k: np.array([v for _, _, v in pts[k]]) for k in pts}
ly = {k: np.log(np.array([r for _, r, _ in pts[k]]) / dd[k]) for k in pts}


def common(lys, g0=-10.0):
    f = lambda q: np.concatenate(
        [q[1 + i] + np.log(1 + q[0] / vv[k]) - lys[k] for i, k in enumerate((1, 2, 3))]
    )
    r = least_squares(f, [g0, 1.0, 1.6, 1.9])
    return r.x, np.sqrt(np.mean(r.fun**2))


syn = {k: np.log(s) + np.log(1 - 13.0 / vv[k]) for k, s in zip((1, 2, 3), (2.9, 5.4, 7.7))}
xs, _ = common(syn)
ok = abs(xs[0] + 13) < 1e-8 and np.allclose(np.exp(xs[1:]), [2.9, 5.4, 7.7], atol=1e-7)
print(f"control: synthetic recovery g = {xs[0]:.6f}, sigma_inf = {np.round(np.exp(xs[1:]), 5)}: {ok}")
if not ok:
    print("READING q1289: VOID (control)")
    sys.exit(2)
x, rms = common(ly)
print(
    f"common g: g = {x[0]:.3f}, sigma_inf = "
    + ", ".join(f"{np.exp(v):.4f}" for v in x[1:])
    + f", rms of ln residual {rms:.4f}; events "
    + ", ".join(f"k {k}: {len(vv[k])}" for k in vv)
)
for k in (1, 2, 3):
    f = lambda q: q[1] + np.log(1 + q[0] / vv[k]) - ly[k]
    r = least_squares(f, [-10.0, 1.0])
    print(
        f"  k = {k} alone: g = {r.x[0]:.3f}, sigma_inf = {np.exp(r.x[1]):.4f}, rms {np.sqrt(np.mean(r.fun ** 2)):.4f}"
    )
print("amplitudes sigma_inf relative to k = 1: " + ", ".join(f"{np.exp(v - x[1]):.4f}" for v in x[1:]))
print(
    "max |ln residual| per k (common g): "
    + ", ".join(
        f"k {k}: {np.abs(x[1 + i] + np.log(1 + x[0] / vv[k]) - ly[k]).max():.4f}"
        for i, k in enumerate((1, 2, 3))
    )
)
print("READING q1289 (exploratory, no verdict)")
