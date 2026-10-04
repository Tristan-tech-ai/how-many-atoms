"""q824 (03 Oct 2026, BACKLOG 59): the nested-copying offset from planar moments. A formal density g(th) = 1 + d_1 cos 2th + d_2 cos 4th
on the inner curve rho(th) = a_0 + a_1 cos 2th has, to first order in eps = a_1/a_0,
  quadrupole / second moment  = d_1/2 + eps (1 - d_1^2/2)  ->  the pair's value 1 is reached at d_1 = 2 (1 + eps),
  hexadecapole / fourth moment = d_2/2 + eps d_1 (1 - d_2)  ->  the quartet's value 1 is reached at d_2 = 2 (1 + eps d_1)
(a pair at radius a has quadrupole/second moment 1; an X + Y quartet has hexadecapole/fourth moment 1 whatever its radii and masses).
This script reads q822 logs, evaluates the moment conditions on the logged states, and prints where they cross zero (linear
interpolation between logged rm): the corrected copying loss points. Mode 'pair' uses d_1 and eps from an MI 1 log; 'quartet' uses d_1,
d_2 and eps from an MI 2 log. It also prints the exact moment ratios (no first-order expansion) computed on the curve by quadrature.
Usage: py q824_moment_offset.py pair|quartet LOG"""

import sys, re
import numpy as np

mode, log = sys.argv[1], sys.argv[2]
rows = []
for l in open(log):
    m = re.search(
        r"rm ([0-9.]+):.*inner radius a_0..: \[([^]]*)\]; inner d_1..: \[([^]]*)\].*residual (\S+)", l
    )
    if m:
        a = [float(x) for x in m.group(2).split(",")]
        d = [float(x) for x in m.group(3).split(",")]
        rows.append((float(m.group(1)), a, d, float(m.group(4))))
th = np.linspace(0, 2 * np.pi, 4096, endpoint=False)


def exact(a, d):
    rho = sum(ak * np.cos(2 * k * th) for k, ak in enumerate(a))
    g = 1 + sum(dk * np.cos(2 * (k + 1) * th) for k, dk in enumerate(d))
    if mode == "pair":
        return np.mean(rho**2 * np.cos(2 * th) * g) / np.mean(rho**2 * g) - 1  # pair: Q/R = 1
    return np.mean(rho**4 * np.cos(4 * th) * g) / np.mean(rho**4 * g) - 1  # quartet: H/R4 = 1


out = []
for rm, a, d, res in rows:
    eps = a[1] / a[0]
    if mode == "pair":
        lin = d[0] - 2 * (1 + eps)
    else:
        lin = d[1] - 2 * (1 + eps * d[0])
    out.append((rm, lin, exact(a, d), eps, d, res))
out.sort()
for i in range(len(out) - 1):
    for j, name in ((1, "first-order"), (2, "exact moment")):
        f0, f1 = out[i][j], out[i + 1][j]
        if f0 * f1 < 0:
            r = out[i][0] + (out[i + 1][0] - out[i][0]) * f0 / (f0 - f1)
            print(
                f"{mode}: {name} condition crosses zero at rm {r:.6f} (between {out[i][0]} and {out[i+1][0]}); eps {out[i][3]:.5f}, "
                f"d {out[i][4]}",
                flush=True,
            )
