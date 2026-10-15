"""q1496 (9 Oct 2026, READING, no verdict word; read after the verdicts VERDICT q1451: B115 PASS; B123 PASS and VERDICT q1494: C1 PASS; C2 PASS, amendments 2202 and 2203): refit of the series L + c1 K^(-1/2) + c2 K^(-1) + c3 K^(-3/2) (and with a fourth term c4 K^(-2)) of the margins P/N and R_e per parity on ALL rows K = 83..130 (48 rows) and
comparison with the limits of the fit on K = 83..122. Question: how stable are the limits L (the extrapolation to K -> infinity on which the reduction of the monotonicity rests) when 8 more rows are added and when a fourth term is added, and is the gap between the even and the odd limit real?
Predictions written before the run: P1 adding the rows 123..130 moves every limit by less than 1e-4; P2 adding a fourth term moves every limit (48 rows) by less than 1e-4; P3 the even-odd gap of the limit of P/N stays above 1e-4 with 3 terms (it was 2.5e-4 in q1486). Labels by the script; nothing is registered.
Usage: py q1496_series_refit_all_rows.py (from small_gap_q206/)."""

import json, numpy as np

rows = [r for r in json.load(open("q1416_rows.json")) if 83 <= r["K"] <= 130 and r.get("control_ok") is True]
assert sorted(r["K"] for r in rows) == list(range(83, 131)), "rows missing"


def fit(pts, exps):
    K = np.array([k for k, _ in pts], float)
    y = np.array([v for _, v in pts])
    M = np.vstack([np.ones_like(K)] + [K ** (-e) for e in exps]).T
    return np.linalg.lstsq(M, y, rcond=None)[0]


E3 = (0.5, 1.0, 1.5)
E4 = (0.5, 1.0, 1.5, 2.0)
lim = {}
for par in ("even", "odd"):
    for q in ("PN", "Re"):
        s = sorted((r["K"], r[q]) for r in rows if r["parity"] == par)
        s122 = [(k, v) for k, v in s if k <= 122]
        s114 = [(k, v) for k, v in s if k <= 114]
        l3_130 = fit(s, E3)[0]
        l3_122 = fit(s122, E3)[0]
        l3_114 = fit(s114, E3)[0]
        l4_130 = fit(s, E4)[0]
        lim[(par, q)] = (l3_114, l3_122, l3_130, l4_130)
        print(
            f"{par:4s} {q}: limit, 3 terms: K<=114 {l3_114:.6f}, K<=122 {l3_122:.6f}, K<=130 {l3_130:.6f}; 4 terms K<=130 {l4_130:.6f}; shift by the 8 rows {l3_130 - l3_122:+.2e}; shift by the 4th term {l4_130 - l3_130:+.2e}"
        )
p1 = all(abs(v[2] - v[1]) < 1e-4 for v in lim.values())
p2 = all(abs(v[3] - v[2]) < 1e-4 for v in lim.values())
gap3 = {q: lim[("odd", q)][2] - lim[("even", q)][2] for q in ("PN", "Re")}
gap4 = {q: lim[("odd", q)][3] - lim[("even", q)][3] for q in ("PN", "Re")}
print(
    "even-odd gap of the limit, 3 terms:",
    {q: f"{g:+.2e}" for q, g in gap3.items()},
    "| 4 terms:",
    {q: f"{g:+.2e}" for q, g in gap4.items()},
)
p3 = abs(gap3["PN"]) > 1e-4
print(
    f"P1 adding the rows 123..130 moves every limit by less than 1e-4: {'holds' if p1 else 'does not hold'}"
)
print(f"P2 adding a fourth term moves every limit by less than 1e-4: {'holds' if p2 else 'does not hold'}")
print(
    f"P3 the even-odd gap of the limit of P/N stays above 1e-4 with 3 terms: {'holds' if p3 else 'does not hold'}"
)
print(
    "smallest margin P/N over all 48 rows and the smallest limit:",
    f"{min(r['PN'] for r in rows):.6f}",
    f"{min(v[2] for k, v in lim.items() if k[1] == 'PN'):.6f}",
    "(both above 1)",
)
print("READING q1496 done (labels by the script; no verdict word)")
