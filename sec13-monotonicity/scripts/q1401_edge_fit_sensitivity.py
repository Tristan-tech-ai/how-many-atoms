"""q1401 reading (9 Oct 2026, EXPLORATORY, no verdict; py q1401_edge_fit_sensitivity.py from temporal_within/small_gap_q206/): sensitivity of the extrapolated limit of the edge-only ratio R_e and of rho = -c_1/E to the power p in L + a K^-p, for the stage-f0.97 states K = 62..82 (data from q1400_out.txt is not stored per state, so the quantities are recomputed from the dumps by the same code as q1400). Expected before the run: for R_e the limit L is above 1.04 for p in {2/3, 1, 4/3, 2} and the two parities differ by less than 0.01 at p = 1; rho stays below 0.4 for every p. Control: synthetic R = 1.07 + 1.1/K^1.3 gives L within 2e-3 at the true p and a limit that differs from the truth for a wrong p (so the sensitivity is visible)."""

import glob, json
import numpy as np
from mpmath import mp, mpf, matrix, lu_solve

mp.dps = 40


def quantities(fn):
    d = json.load(open(fn))
    m = d["m"]
    odd = d["odd"]
    n = 2 * m + 1 + (1 if odd else 0)
    idx = (2 * m + 1) if odd else 0
    J = matrix(n, n)
    for a in range(n):
        for c in range(n):
            J[a, c] = mpf(d["J"][a][c])
    b = [mpf(t) for t in d["b"]]
    ref = lu_solve(J, matrix([-t for t in b]))[idx]
    T = matrix(n, n)
    for a in range(n):
        T[a, a] = 1
    for a in range(1, m):
        T[a, 0] = -1
    if odd:
        T[2 * m + 1, 0] = -1
    Jd = T * J
    bd = T * matrix(b)
    e = matrix(n, 1)
    e[idx] = 1
    x = lu_solve(Jd.T, e)
    c = [-x[k] * bd[k] for k in range(n)]
    assert abs(sum(c) - ref) / abs(ref) < mpf("1e-24"), "control"
    six = [c[j] for j in (m - 3, m - 2, m - 1)] + [c[m + j] for j in (m - 3, m - 2, m - 1)]
    return (
        d["K"],
        float(sum(t for t in six if t > 0) / -sum(t for t in six if t < 0)),
        float(-c[0] / (ref - c[0])),
        odd,
    )


def fit(K, Y, p):
    X = np.c_[np.ones_like(K), K ** (-p)]
    cf, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return cf[0], float(np.sqrt(np.mean((Y - X @ cf) ** 2)))


Kc = np.arange(62, 83, 2, float)
Rs = 1.07 + 1.1 / Kc**1.3
Lt = fit(Kc, Rs, 1.3)[0]
Lw = fit(Kc, Rs, 1.0)[0]
ctl = abs(Lt - 1.07) < 2e-3 and abs(Lw - 1.07) > 1e-3
print("control: true-p limit", round(Lt, 5), "wrong-p limit", round(Lw, 5), "->", "OK" if ctl else "FAILED")
data = [quantities(f"q1388_jb_K{K}_f0.97.json") for K in range(62, 83)]
for odd in (False, True):
    sel = [t for t in data if t[3] == odd]
    K = np.array([t[0] for t in sel], float)
    for j, nm in ((1, "R_e"), (2, "rho")):
        Y = np.array([t[j] for t in sel])
        print(
            ("odd" if odd else "even"),
            nm,
            ", ".join(
                f"p={p:.3f}: L {fit(K, Y, p)[0]:.5f} (rms {fit(K, Y, p)[1]:.1e})"
                for p in (2 / 3, 1.0, 4 / 3, 2.0)
            ),
        )
print("READING q1401 done")
