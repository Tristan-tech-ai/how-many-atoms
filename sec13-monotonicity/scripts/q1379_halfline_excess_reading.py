"""q1379 (9 Oct 2026, EXPLORATORY reading, no item, no verdict; py q1379_halfline_excess_reading.py from temporal_within/centre_scripts/): can delta = 1.990803974 of the scaling law (amendment 2132) be read as the limit of the half-line mass excess? For the middle state of the K = 82 family: c_j = 2 (A + delta) p_j (scaled masses, edge atom j = 0, inward), cum_j = sum_(i<=j) c_i, depth d_j = A - x_j, excess g_j = cum_j - d_j (the cumulative mass over the bulk density mu = 1 / (2 (A + delta)), minus the depth; independent of A for the edge-indexed atoms). Check first that the sum of c over the right half is (A + delta) / 2 (the normalisation). Then fit g(d) = g_inf + a d^(-gamma) for 3 < d < 38. Expected before the run: unknown. If g_inf came out equal to delta, delta would be a pure half-line quantity; if not, delta involves the centre family."""

import glob, json, os, re
import numpy as np
from scipy.optimize import curve_fit

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def numstr(s):
    return re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", str(s).replace("np.float64(", "")).group(0)


dl = 1.990803974
L = []
for f in glob.glob(os.path.join(ROOT, "**", "results_q2*_gauss_K82_A*.json"), recursive=True):
    try:
        j = json.load(open(f))
        res = float(numstr(j["residual"]))
    except Exception:
        continue
    if j.get("verdict") == "OPTIMAL" and res < 1e-20:
        L.append((float(numstr(j["A"])), j))
L.sort(key=lambda t: t[0])
A, j = L[len(L) // 2]
u = np.array([float(numstr(x)) for x in j["u"]])
w = np.array([float(numstr(x)) for x in j["w"]])
we = float(numstr(j["we"]))
x = np.concatenate([[A], u[::-1]])
p = np.concatenate([[we], w[::-1]])
d = A - x
c = 2 * (A + dl) * p
cum = np.cumsum(c)
g = cum - d
print(
    f"K 82, A {A:.4f}: sum c over the right half {c.sum() / 2:.9f}, (A + delta)/2 = {(A + dl) / 2:.9f}",
    flush=True,
)
print(
    "excess g = cum - depth at selected atoms: "
    + ", ".join(f"j {k}: d {d[k]:.2f} g {g[k]:.4f}" for k in (0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36)),
    flush=True,
)
m = (d > 3) & (d < 38)
f = lambda dd, ginf, a, gam: ginf + a * dd ** (-gam)
for gam0 in (0.33, 0.5, 1.0):
    popt, _ = curve_fit(f, d[m], g[m], p0=[2.0, 1.0, gam0], maxfev=20000)
    print(
        f"fit g_inf + a d^-gamma (start gamma {gam0}): g_inf {popt[0]:.4f}, a {popt[1]:.4f}, gamma {popt[2]:.4f}, rms {np.sqrt(np.mean((f(d[m], *popt) - g[m]) ** 2)):.1e}",
        flush=True,
    )
g2 = np.polyfit(np.log(d[m]), g[m], 1)
print(
    f"logarithmic fit g = {g2[1]:.4f} + {g2[0]:.4f} ln d, rms {np.sqrt(np.mean((np.polyval(g2, np.log(d[m])) - g[m]) ** 2)):.1e}",
    flush=True,
)
print("READING q1379 (exploratory, no verdict)")
