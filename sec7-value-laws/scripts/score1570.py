"""1570: fit R_eff of the bulk shape of sqrt(p) to the ground state of a disc (J0) or ball (sinc), rho in [0, m - 1.5]."""

import json, os, math
import numpy as np
from scipy.special import i0e, j0
from scipy.optimize import minimize_scalar

os.chdir(r"centre_scripts")
j01 = 2.404825557695773
d = 3.34776


def S0(x):
    x = np.asarray(x, float)
    xs = np.where(x < 1e-4, 1.0, x)
    return np.where(x < 1e-4, 1 - x + 2 * x * x / 3, (1 - np.exp(-2 * xs)) / (2 * xs))


cases = [
    ("q861_runs/q861_state_m4.5.json", 2),
    ("q861b_runs/q861b_state_m3.5.json", 2),
    ("q861b_runs/q861b_state_m4.0.json", 2),
    ("q865_runs/q865_state_m3.5.json", 3),
    ("q865_runs/q865_state_m4.5.json", 3),
]
ok = True
for f, dim in cases:
    s = json.load(open(f))
    m = s["m"]
    r = np.array(s["r"])
    w = np.array(s["w"])
    rho = np.linspace(0, m - 1.5, 400)
    E = np.exp(-((rho[:, None] - r[None, :]) ** 2) / 2)
    z = rho[:, None] * r[None, :]
    p = (E * (i0e(z) if dim == 2 else S0(z))) @ w
    g = np.sqrt(p / p[0])
    shape = (
        (lambda R: j0(j01 * rho / R)) if dim == 2 else (lambda R: np.sinc(rho / R))
    )  # np.sinc(x) = sin(pi x)/(pi x)
    res = minimize_scalar(lambda R: np.sum((g - shape(R)) ** 2), bounds=(m + 0.1, m + 10), method="bounded")
    R = res.x
    inb = abs(R - (m + d)) <= 0.3
    ok = ok and inb
    print(
        f"{f} (d {dim}, m {m}): R_eff fit {R:.4f}; m + delta {m + d:.4f}; diff {R - m - d:+.4f} ({'in' if inb else 'OUT'}); "
        f"rms residual {math.sqrt(res.fun/len(rho)):.2e}; alternatives m {m:.2f}, m + delta/2 {m + d/2:.2f}"
    )
print("1570:", "PASS" if ok else "FAIL")
