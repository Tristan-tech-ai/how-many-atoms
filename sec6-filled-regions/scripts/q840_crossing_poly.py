"""q840 (03 Oct 2026): interpolation error of q825's eccentric crossing on a coarse scan. harm() and the log regex are TEXT COPIES of
q825_eccentric_loss.py. Prints the crossing of the K-th eccentric harmonic with 2 from linear (2 rows), quadratic (3 rows) and cubic
(4 rows) fits around the sign change, so a coarse scan's crossing carries a stated interpolation error.
Usage: py q840_crossing_poly.py pair|quartet LOG"""

import sys, re
import numpy as np

mode, log = sys.argv[1], sys.argv[2]
K = 2 if mode == "pair" else 4
pat = re.compile(
    r"rm ([0-9.]+): .*inner radius a_0..: (\[[^\]]*\]); inner d_1..: (\[[^\]]*\]).*residual ([0-9.e+-]+)"
)


def harm(a, d, K):
    th = np.linspace(0, 2 * np.pi, 200001)[:-1]
    rho = sum(ak * np.cos(2 * k * th) for k, ak in enumerate(a))
    g = 1 + sum(dk * np.cos(2 * (k + 1) * th) for k, dk in enumerate(d))
    x, y = rho * np.cos(th), rho * np.sin(th)
    A, B = rho[0], rho[len(th) // 4]
    s = np.unwrap(np.arctan2(y / B, x / A))
    return 2 * np.mean(np.cos(K * s) * g), 2 * np.mean(np.cos(K * th) * g), (A - B) / (A + B)


rows = {}
for line in open(log):
    m = pat.search(line)
    if m:
        rm = float(m.group(1))
        rows[rm] = harm(eval(m.group(2)), eval(m.group(3)), K)[0] - 2
xs = np.array(sorted(rows))
fs = np.array([rows[x] for x in xs])
i = [j for j in range(len(xs) - 1) if fs[j] * fs[j + 1] < 0][0]
for name, idx in (
    ("linear", [i, i + 1]),
    ("quadratic lo", [i - 1, i, i + 1]),
    ("quadratic hi", [i, i + 1, i + 2]),
    ("cubic", [i - 1, i, i + 1, i + 2]),
):
    idx = [j for j in idx if 0 <= j < len(xs)]
    c = np.polyfit(xs[idx] - xs[i], fs[idx], len(idx) - 1)
    rts = [r.real + xs[i] for r in np.roots(c) if abs(r.imag) < 1e-12 and 0 <= r.real <= xs[i + 1] - xs[i]]
    print(f"{mode} {log}: {name:12s} crossing rm {rts[0]:.6f}" if rts else f"{name}: no root in the bracket")
