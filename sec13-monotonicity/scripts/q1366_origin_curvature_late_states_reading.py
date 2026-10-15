"""q1366 (9 Oct 2026, EXPLORATORY reading, no item, no verdict; py q1366_origin_curvature_late_states_reading.py from temporal_within/centre_scripts/): is the origin still a local minimum of the capacity KKT function in the last stored OPTIMAL states of the even-K families? Barletta and Dytso (2609.10039, Cor. 1) show that at a birth at the origin G_A''(0) < 0 (inactive contact of multiplicity two), so just before a birth the origin must be a local maximum with two extra minima, and the count of sign changes of D' on (0, A) is K - 1 instead of K - 2. The last stored states may lie before that thin window.
Reading: for K = 60, 62, 64, 66, 80, 82 the last 8 OPTIMAL states by A: the sign of D' at the first grid point x = 0.01 (ball arithmetic as q1363b), the first sign change position, the count. Expected before the run: D'(0.01) > 0 and the count K - 2 in all, unless the window is reached. A state with D'(0.01) < 0 has the origin as a local maximum.
Controls: those of q1363 (imported)."""

import os, sys

sys.argv = ["x", "62"]
src = open("q1363b_capacity_state_choice.py").read()
src = src[: src.index("Ks = [int(a)")]
exec(compile(src, "q1363", "exec"))
for K in (60, 62, 64, 66, 80, 82):
    L = sorted(ST[K], key=lambda x: x[0])[-8:]
    for A, f, j in L:
        A_s, u_s, w_s, we_s, wc_s = state(j)
        ch, und, mx, mn, rad, pat = analyse(A_s, u_s, w_s, we_s, wc_s)
        xs, ds = dprime_ball(A_s, u_s, w_s, we_s, wc_s)
        print(
            f"K {K} A {A:.6f} centre_signal {j.get('centre_signal')} u1 {float(u_s[0]):.4f} D'(0.01) sign {'+' if ds[0] > 0 else '-' if ds[0] < 0 else '?'} first change {ch[0]:.3f} count {len(ch)} (K-2 = {K-2}) undecided {und}",
            flush=True,
        )
print("READING q1366 (exploratory, no verdict)")
