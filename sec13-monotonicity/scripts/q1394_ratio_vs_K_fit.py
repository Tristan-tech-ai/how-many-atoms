"""q1394 reading (9 Oct 2026, EXPLORATORY, no verdict; py q1394_ratio_vs_K_fit.py from temporal_within/small_gap_q206/): the ratio R = P/N of the innermost flow (basis D1 of q1389) at the fixed family stage f0.97 for K = 62..82, from q1389_out_fixed.txt (21 states, even and odd K separately). Fits R(K) = L + a K^-p with p in {1, 4/3, 2, free} by least squares (exact linear fit for fixed p, Nelder-Mead for free p), prints L, a, p, rms and the spread of L over the models. Expected before the run: R decreases with K for both parities; the fitted limit L stays above 1 for the one-over-K models, and the spread of L over the models is larger than the rms, so L is not determined. Control: synthetic data R = 1.03 + 2.0/K^1.3 recovered by the free fit to L within 2e-3 (0.001 noise-free) and p within 0.1; a synthetic sequence R = 0.98 + 3/K gives L < 1."""

import re, sys
import numpy as np
from scipy.optimize import minimize

rows = []
cur = None
for line in open("q1389_out_fixed.txt", encoding="utf-8"):
    m = re.match(r"K (\d+) (\S+) A (\S+) \((odd|even), \S+\), stored-system flow (\S+), J", line)
    if m:
        cur = dict(K=int(m.group(1)), A=float(m.group(3)), par=m.group(4), flow=float(m.group(5)))
        rows.append(cur)
        continue
    m = re.match(r"\s+D1\s+: .*ratio (\S+);", line)
    if m and cur is not None and "R" not in cur:
        cur["R"] = float(m.group(1))
assert len(rows) == 21 and all("R" in r for r in rows), "parse"


def fit_fixed(K, R, p):
    X = np.c_[np.ones_like(K), K ** (-p)]
    c, *_ = np.linalg.lstsq(X, R, rcond=None)
    res = R - X @ c
    return c[0], c[1], p, float(np.sqrt(np.mean(res**2)))


def fit_free(K, R):
    best = None
    for p0 in (0.5, 1.0, 2.0, 4.0):
        f = lambda q: fit_fixed(K, R, abs(q[0]))[3]
        o = minimize(f, [p0], method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-14, maxiter=4000))
        r = fit_fixed(K, R, abs(o.x[0]))
        if best is None or r[3] < best[3]:
            best = r
    return best


def models(K, R):
    return [fit_fixed(K, R, p) for p in (1.0, 4 / 3, 2.0)] + [fit_free(K, R)]


Kc = np.arange(62, 83, dtype=float)
Rs = 1.03 + 2.0 / Kc**1.3
mm = models(Kc, Rs)[-1]
c1 = abs(mm[0] - 1.03) < 2e-3 and abs(mm[2] - 1.3) < 0.1
Rn = 0.98 + 3.0 / Kc
c2 = models(Kc, Rn)[0][0] < 1
print(
    "control synthetic recovery:",
    "OK" if c1 else "FAILED",
    f"(L {mm[0]:.5f}, p {mm[2]:.3f}); control below-one limit:",
    "OK" if c2 else "FAILED",
)
for par in ("even", "odd"):
    sel = sorted([r for r in rows if r["par"] == par], key=lambda r: r["K"])
    K = np.array([r["K"] for r in sel], float)
    R = np.array([r["R"] for r in sel])
    print(
        f"{par}: {len(sel)} states, K {int(K[0])}..{int(K[-1])}, R from {R[0]:.5f} to {R[-1]:.5f}, decrements per 2 in K: first {R[0] - R[1]:.5f}, last {R[-2] - R[-1]:.5f}; monotone decreasing {bool(np.all(np.diff(R) < 0))}"
    )
    Ls = []
    for L, a, p, rms in models(K, R):
        Ls.append(L)
        print(f"    L + a K^-p: p {p:.4f}, L {L:.5f}, a {a:.4f}, rms {rms:.2e}")
    print(
        f"    spread of L over the models: {min(Ls):.5f} to {max(Ls):.5f}; all above 1: {all(t > 1 for t in Ls)}"
    )
print("READING q1394 done")
