"""q1464 (9 Oct 2026, READING, no verdict): series structure of the six edge fractions, the response entries and the margins, rows K = 83..114 (q1463_terms.json; parities apart; K <= 114 only). Two variables: K^(-1/2) with powers (1/2, 1, 3/2) and w = 1/v_c^2 with powers (1, 2, 3), v_c = 4 pi rho_c, rho_c from 4 pi^2 rho^4 - rho^2 + ln(rho)/6 + 0.5482 = (K + 1/6)/2. Quantities: the five free normalised response entries x_j (q1448 normalisation), the six edge fractions f_j = t_j / sum of the mass-type t (t_j = x_j bd_j), R_e = sum mass-type t over minus the sum of position-type t, P/N is not recomputed here (see q1461). For each: rms of the three-term fit in K^(-1/2), rms in w, the limit L and the coefficients a1, a2, a3 of the w-series. Usage: py q1464_series_structure.py (from small_gap_q206/)."""

import json, numpy as np
from scipy.optimize import brentq

pi = np.pi


def rho_c(K):
    return brentq(lambda r: 4 * pi**2 * r**4 - r**2 + np.log(r) / 6 + 0.5482 - (K + 1 / 6) / 2, 0.05, 20)


d = json.load(open("q1463_terms.json"))
ks = sorted(int(k) for k in d)
assert max(ks) <= 114


def fitq(K, y, var, ex):
    M = np.vstack([np.ones_like(K)] + [var**e for e in ex]).T
    c = np.linalg.lstsq(M, y, rcond=None)[0]
    return c, np.sqrt(np.mean((M @ c - y) ** 2))


for par in ("even", "odd"):
    sel = [k for k in ks if d[str(k)]["parity"] == par]
    K = np.array(sel, float)
    w = np.array([1 / (4 * pi * rho_c(k)) ** 2 for k in sel])
    kh = K**-0.5
    X = np.array([d[str(k)]["x_six"] for k in sel])
    BD = np.array([d[str(k)]["bd_six"] for k in sel])
    T = X * BD
    F = T / T[:, 3:].sum(axis=1)[:, None]
    Re = T[:, 3:].sum(axis=1) / -T[:, :3].sum(axis=1)
    print(f"== {par}: K {sel[0]}..{sel[-1]}, n = {len(sel)}")
    print(
        "   quantity            rms(K^-1/2: 0.5,1,1.5)   rms(w: 1,2,3)   L(K^-1/2)   L(w)      a1(w)      a2(w)      a3(w)"
    )
    items = (
        [(f"x entry {j}", np.abs(X[:, j])) for j in (0, 1, 3, 4, 5)]
        + [(f"fraction {j}", F[:, j]) for j in range(6)]
        + [("R_e (six terms)", Re)]
    )
    for name, y in items:
        c1, r1 = fitq(kh, y, kh, (1, 2, 3))
        c2, r2 = fitq(w, y, w, (1, 2, 3))
        print(
            f"   {name:18s}  {r1:10.1e}          {r2:10.1e}   {c1[0]:+10.5f}  {c2[0]:+10.5f}  {c2[1]:+9.3f}  {c2[2]:+9.1f}  {c2[3]:+9.0f}"
        )
