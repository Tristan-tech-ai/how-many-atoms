"""q1400 reading (9 Oct 2026, EXPLORATORY, no verdict; py q1400_edge_ratio_vs_K.py from temporal_within/small_gap_q206/): for every dump of q1388 (27 gap states K = 19..24, the K = 61/62 states, the 21 stage-f0.97 states K = 62..82, and K = 70, 80 states) the basis-D1 contributions of the rows D and D' of the three outermost atoms give the edge-only ratio R_e = (sum of positive terms)/(sum of negative terms of those six rows); also rho = -c_1/E, E/flow and P/N. Printed sorted by K with the within-K range; then a fit R_e = L + a/K over the stage f0.97 states (K = 62..82, even and odd separately) and over all states of each parity. Expected before the run: R_e is a smooth decreasing function of K (1.09 near K = 20 to 1.08 near K = 80) with a within-K range far smaller than the change between K = 20 and K = 80; the 1/K fit of the stage states gives L near 1.065. Control: the sum of all contributions equals the stored flow to 1e-24 relative (per state)."""

import glob, json, sys
import numpy as np
from mpmath import mp, mpf, matrix, lu_solve

mp.dps = 40
files = sorted(glob.glob("q1388_jb_K*.json") + glob.glob("gapjb/q1388_jb_K*.json"))
rows = []
for fn in files:
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
    ok = abs(sum(c) - ref) / abs(ref) < mpf("1e-24")
    top = [m - 3, m - 2, m - 1]
    six = [c[j] for j in top] + [c[m + j] for j in top]
    Pe = sum(t for t in six if t > 0)
    Ne = -sum(t for t in six if t < 0)
    P = sum(t for t in c if t > 0)
    N = -sum(t for t in c if t < 0)
    E = ref - c[0]
    rows.append(
        dict(
            K=d["K"],
            A=float(mpf(d["A"])),
            par="odd" if odd else "even",
            label=d["label"],
            Re=float(Pe / Ne),
            rho=float(-c[0] / E),
            ef=float(E / ref),
            R=float(P / N),
            ok=bool(ok),
            file=fn,
        )
    )
assert rows and all(r["ok"] for r in rows), "control failed or no rows"
rows.sort(key=lambda r: (r["K"], r["A"]))
print("states", len(rows), "controls all OK")
byK = {}
for r in rows:
    byK.setdefault(r["K"], []).append(r)
for K in sorted(byK):
    g = byK[K]
    print(
        f"K {K} ({g[0]['par']}): {len(g)} states, A {min(r['A'] for r in g):.3f}-{max(r['A'] for r in g):.3f}: R_e {min(r['Re'] for r in g):.5f}-{max(r['Re'] for r in g):.5f}; rho {min(r['rho'] for r in g):.5f}-{max(r['rho'] for r in g):.5f}; P/N {min(r['R'] for r in g):.5f}-{max(r['R'] for r in g):.5f}"
    )


def fit(K, Y):
    X = np.c_[np.ones_like(K), 1 / K]
    cf, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return cf[0], cf[1], float(np.sqrt(np.mean((Y - X @ cf) ** 2)))


for par in ("even", "odd"):
    st = [r for r in rows if r["par"] == par and "f0.97" in r["file"] and r["K"] >= 62]
    K = np.array([r["K"] for r in st], float)
    for nm in ("Re", "rho"):
        L, a, rms = fit(K, np.array([r[nm] for r in st]))
        print(f"{par} stage f0.97, {len(st)} states: {nm} = L + a/K: L {L:.5f}, a {a:.4f}, rms {rms:.2e}")
print("READING q1400 done")
