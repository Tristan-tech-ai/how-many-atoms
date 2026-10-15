"""q1391 (9 Oct 2026, control for q1387/q1389, no verdict; py q1391_fd_richardson_control.py from temporal_within/small_gap_q206/): finite-difference check of the implicit-function flow of u_1 at the K = 62 states with the IFT flow printed by q1389 (start, f0.1, f0.25), using the stored neighbours: central difference with step h = 0.005 (D1) and 0.01 (D2) and the Richardson combination (4 D1 - D2) / 3 (needs the states at A +- 0.005, +- 0.01; the start state A = 41.74 has only one stored neighbour on the left of step 0.005, so it uses the forward Richardson-free values only and is reported as such). Expected before the run: relative difference between the IFT flow and the Richardson value below 2e-3 at f0.1 and f0.25."""

import glob, json, re, os
from mpmath import mp, mpf

mp.dps = 40
fs = {}
for f in glob.glob(os.path.join("..", "capacity_even62_family", "results_q220_gauss_K62_A*.json")):
    j = json.load(open(f))
    if j["verdict"] == "OPTIMAL":
        fs[float(j["A"])] = j
As = sorted(fs)


def u1(A):
    return mpf(fs[A]["u"][0])


ift = {41.74: mpf("11.0283031511"), 41.78375: mpf("2.24158555128"), 41.843175: mpf("0.863245006733")}
for A0, val in ift.items():
    A = min(As, key=lambda t: abs(t - A0))
    i = As.index(A)
    out = f"A {A}: "
    lo, hi = As[i - 1], As[i + 1]
    h1 = (hi - lo) / 2
    sym1 = abs((A - lo) - (hi - A)) < 1e-9
    D1 = (u1(hi) - u1(lo)) / mpf(hi - lo)
    out += f"central step {h1:.4f} (symmetric {sym1}) {mp.nstr(D1, 10)} (rel diff to IFT {mp.nstr(abs(D1 - val) / val, 3)})"
    if i >= 2 and i + 2 < len(As):
        lo2, hi2 = As[i - 2], As[i + 2]
        sym2 = abs((A - lo2) - (hi2 - A)) < 1e-9
        D2 = (u1(hi2) - u1(lo2)) / mpf(hi2 - lo2)
        ratio = (hi2 - lo2) / (hi - lo)
        out += f"; second central step {(hi2 - lo2) / 2:.4f} (symmetric {sym2}) {mp.nstr(D2, 10)}"
        if sym1 and sym2 and abs(ratio - 2) < 1e-6:
            R = (4 * D1 - D2) / 3
            out += f"; Richardson {mp.nstr(R, 10)} (rel diff to IFT {mp.nstr(abs(R - val) / val, 3)})"
    print(out)
