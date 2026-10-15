"""q1496b (9 Oct 2026, READING, no verdict word; follow-up of q1496, written after its output was read): the limits of the margins P/N and R_e per parity on all rows K = 83..130 for series with 3, 4, 5 and 6 terms in K^(-1/2) (exponents 0.5, 1, 1.5, 2, 2.5, 3), fitted by the normal equations at 50 digits (mpmath; the plain least squares is ill conditioned at 5 and 6 terms).
Predictions written before the run (after q1496 showed that the even-odd gap of the limit falls from 2.4e-4 with 3 terms to 6e-6 with 4 terms): P1 with 5 terms the even-odd gap of both limits stays below 3e-5; P2 the common limit (mean of the even and odd limits) of P/N changes by less than 1e-4 between 4 and 5 terms; P3 the same for R_e. Labels by the script; nothing is registered; the data have about 16 digits, so the fits beyond 5 terms are printed with the warning that noise may dominate.
Usage: py q1496b_limit_vs_terms.py (from small_gap_q206/)."""

import json
from mpmath import mp, mpf, matrix, lu_solve

mp.dps = 50
rows = [r for r in json.load(open("q1416_rows.json")) if 83 <= r["K"] <= 130 and r.get("control_ok") is True]
assert sorted(r["K"] for r in rows) == list(range(83, 131))


def limit(par, q, nterm):
    s = sorted((r["K"], r[q]) for r in rows if r["parity"] == par)
    n = len(s)
    exps = [mpf(k) / 2 for k in range(1, nterm)]
    X = matrix(n, nterm)
    y = matrix(n, 1)
    for i, (K, v) in enumerate(s):
        X[i, 0] = 1
        for j, e in enumerate(exps):
            X[i, j + 1] = mpf(K) ** (-e)
        y[i] = mpf(v)
    c = lu_solve(X.T * X, X.T * y)
    return float(c[0])


tab = {}
for q in ("PN", "Re"):
    for nt in (3, 4, 5, 6):
        ev, od = limit("even", q, nt), limit("odd", q, nt)
        tab[(q, nt)] = (ev, od)
        print(
            f"{q}: {nt} terms: even {ev:.7f}, odd {od:.7f}, gap odd - even {od - ev:+.2e}, mean {(ev + od) / 2:.7f}"
            + ("   (noise may dominate)" if nt >= 6 else "")
        )
p1 = all(abs(tab[(q, 5)][1] - tab[(q, 5)][0]) < 3e-5 for q in ("PN", "Re"))
mean = lambda q, nt: (tab[(q, nt)][0] + tab[(q, nt)][1]) / 2
p2 = abs(mean("PN", 5) - mean("PN", 4)) < 1e-4
p3 = abs(mean("Re", 5) - mean("Re", 4)) < 1e-4
print(f"P1 5 terms: even-odd gap of both limits below 3e-5: {'holds' if p1 else 'does not hold'}")
print(
    f"P2 common limit of P/N changes by less than 1e-4 from 4 to 5 terms: {'holds' if p2 else 'does not hold'} ({mean('PN', 5) - mean('PN', 4):+.2e})"
)
print(
    f"P3 common limit of R_e changes by less than 1e-4 from 4 to 5 terms: {'holds' if p3 else 'does not hold'} ({mean('Re', 5) - mean('Re', 4):+.2e})"
)
print("READING q1496b done (labels by the script; no verdict word)")
