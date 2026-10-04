"""q837 (03 Oct 2026, used in 1470): eccentric fourth harmonic of the inner formal density from converged rows only (residual <= 1e-9),
merging the two capacity mixed-curve inner2 logs; harm() is a text copy of q825/r3_param. Saved from the scratch reader of 1470.
"""

import re, numpy as np


def harm(a, d, K):
    th = np.linspace(0, 2 * np.pi, 200001)[:-1]
    rho = sum(ak * np.cos(2 * k * th) for k, ak in enumerate(a))
    g = 1 + sum(dk * np.cos(2 * (k + 1) * th) for k, dk in enumerate(d))
    x, y = rho * np.cos(th), rho * np.sin(th)
    A, B = rho[0], rho[len(th) // 4]
    s = np.unwrap(np.arctan2(y / B, x / A))
    return 2 * np.mean(np.cos(K * s) * g), (A - B) / (A + B)


pat = re.compile(
    r"rm ([0-9.]+): .*inner radius a_0..: (\[[^\]]*\]); inner d_1..: (\[[^\]]*\]).*residual ([0-9.e+-]+)"
)
rows = {}
for f in ["q833_capmx37_inner2.log", "q833_capmx37_inner2b.log"]:
    for l in open(f):
        m = pat.search(l)
        if m and float(m.group(4)) < 1e-9:
            rows[float(m.group(1))] = (eval(m.group(2)), eval(m.group(3)), float(m.group(4)))
for rm in sorted(rows):
    if 3.435 < rm < 3.45:
        e, ep = harm(rows[rm][0], rows[rm][1], 4)
        print(rm, round(e, 6), "eps", round(ep, 4), "res", rows[rm][2])
