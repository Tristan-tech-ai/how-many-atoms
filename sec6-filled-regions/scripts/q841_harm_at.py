"""q841 (03 Oct 2026): the K-th eccentric harmonic of the inner formal density at given rm values (cubic fit through the four logged
rows around each rm), to state a pair or quartet loss as a miss in the harmonic (E - 1 = harmonic/2 - 1). harm() and the regex are TEXT
COPIES of q825_eccentric_loss.py.
Usage: py q841_harm_at.py pair|quartet LOG rm1,rm2,..."""

import sys, re
import numpy as np

mode, log = sys.argv[1], sys.argv[2]
K = 2 if mode == "pair" else 4
RMS = [float(x) for x in sys.argv[3].split(",")]
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
        rows[float(m.group(1))] = harm(eval(m.group(2)), eval(m.group(3)), K)
xs = np.array(sorted(rows))
for rm in RMS:
    i = int(np.searchsorted(xs, rm))
    idx = [j for j in range(i - 2, i + 2) if 0 <= j < len(xs)]
    h = np.polyval(np.polyfit(xs[idx] - rm, [rows[xs[j]][0] for j in idx], len(idx) - 1), 0.0)
    ep = np.polyval(np.polyfit(xs[idx] - rm, [rows[xs[j]][2] for j in idx], len(idx) - 1), 0.0)
    print(f"{mode} {log}: rm {rm:.6f}: eccentric harmonic {h:.6f}, E - 1 = {h/2 - 1:+.3e}, eps {ep:.4f}")
