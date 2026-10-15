"""q1378 (9 Oct 2026, EXPLORATORY reading, no item, no verdict; py q1378_depth_monotone_reading.py from temporal_within/centre_scripts/): do all atoms of the capacity-achieving input move outward at least as fast as the boundary, dx_j/dA >= 1, i.e. does the depth d_j = A - x_j of every interior atom (centre atom excluded) fail to increase with A? Heuristic behind the test (stated before the run): with a flat mass density mu = 1/(2 (A + delta)) the mass of the m innermost atoms is T_m = mu b_m with b_m the cell boundary (x_m + x_(m+1)) / 2, so dT_m/dA = mu [b_m' - b_m / (A + delta)]; if every x_j' >= 1 then b_m' >= 1 > b_m / (A + delta) because b_m < A, which gives dT_m/dA > 0 for every m. The flat-density law holds only to 0.1 percent, so this is a heuristic, not a proof.
Test: all consecutive same-family pairs of the stored capacity states (K >= 60, 1e-6 < A2 - A1 < 0.2): x_j' from finite differences for the interior atoms (the edge atom has x' = 1 by definition; the centre atom, fixed at 0, is excluded). Printed: the smallest x_j' over all pairs and atoms, its location (atom index from the edge, K, phase), the number of (pair, atom) cases with x_j' < 1 - 1e-9 and the largest shortfall, the largest x_j', and the count of cases with x_j' < 1 by atom index from the centre. Expected before the run: unknown (the sampled values in q1370 were all >= 1.0000).
"""

import glob, json, os, re
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def numstr(s):
    return re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", str(s).replace("np.float64(", "")).group(0)


states = []
for f in glob.glob(os.path.join(ROOT, "**", "results_q2*_gauss_K*_A*.json"), recursive=True):
    try:
        j = json.load(open(f))
        res = float(numstr(j["residual"]))
        K = int(j["K"])
        A = float(numstr(j["A"]))
    except Exception:
        continue
    if j.get("verdict") == "OPTIMAL" and res < 1e-20 and K >= 60:
        states.append((K, j.get("wc") not in (None, "None"), A, np.array([float(numstr(x)) for x in j["u"]])))
states.sort(key=lambda t: (t[0], t[1], t[2]))
worst = (9.0, None)
short = 0
total = 0
big = 0.0
byidx = {}
for s1, s2 in zip(states[:-1], states[1:]):
    if s1[0] == s2[0] and s1[1] == s2[1] and 1e-6 < s2[2] - s1[2] < 0.2:
        dA = s2[2] - s1[2]
        xp = (s2[3] - s1[3]) / dA
        total += len(xp)
        big = max(big, xp.max())
        i = int(np.argmin(xp))
        if xp[i] < worst[0]:
            worst = (float(xp[i]), (s1[0], s1[1], round(s1[2], 4), i))
        for k in np.where(xp < 1 - 1e-9)[0]:
            short += 1
            byidx[int(k)] = byidx.get(int(k), 0) + 1
print(
    f"pairs analysed; (pair, interior atom) cases {total}; smallest dx/dA {worst[0]:.6f} at (K, centre, A, index from the centre) {worst[1]}; cases with dx/dA < 1 - 1e-9: {short}; largest dx/dA {big:.4f}",
    flush=True,
)
print("cases with dx/dA < 1 by index from the centre (first 12):", sorted(byidx.items())[:12], flush=True)
print("READING q1378 (exploratory, no verdict)")
