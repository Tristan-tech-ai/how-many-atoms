"""q825 (03 Oct 2026): the inner rings' copying loss in the ECCENTRIC parameter of the inner curve (the boundary convention of A2
Section III), from q822 logs. The function harm() is a TEXT COPY of the referee's round-3 check (scratchpad r3_param.py, A2 R3,
finding 1; arc-length branch dropped): the logged density g(th) dth/2pi on rho(th) = sum a_k cos 2k th is re-expressed in s, where
x = A cos s, y = B sin s with A = rho(0), B = rho(pi/2), and its K-th harmonic 2 <g cos K s> is compared with the ring's b_K = 2
(pair: K = 2 from an MI 1 log; quartet: K = 4 from an MI 2 log). Prints that harmonic at each logged rm and the rm where it crosses 2
(linear interpolation between logged rows).
Usage: py q825_eccentric_loss.py pair|quartet LOG"""

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


rows = []
for line in open(log):
    m = pat.search(line)
    if m:
        rm = float(m.group(1))
        a = eval(m.group(2))
        d = eval(m.group(3))
        e, p, ep = harm(a, d, K)
        rows.append((rm, e, p, ep, float(m.group(4))))
rows.sort()
for i in range(len(rows) - 1):
    f0, f1 = rows[i][1] - 2, rows[i + 1][1] - 2
    if f0 * f1 < 0:
        r = rows[i][0] + (rows[i + 1][0] - rows[i][0]) * f0 / (f0 - f1)
        print(
            f"{mode}: eccentric-parameter harmonic K={K} crosses 2 at rm {r:.6f} (rows {rows[i][0]}, {rows[i+1][0]}: "
            f"{rows[i][1]:.6f}, {rows[i+1][1]:.6f}; polar {rows[i][2]:.6f}, {rows[i+1][2]:.6f}; eps {rows[i][3]:.4f}; "
            f"residuals {rows[i][4]:.0e}, {rows[i+1][4]:.0e})"
        )
