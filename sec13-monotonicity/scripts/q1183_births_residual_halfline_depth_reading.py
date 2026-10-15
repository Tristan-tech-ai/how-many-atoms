"""q1183 (8 Oct 2026, registered in amendment 1923): READING (no verdict) on the source of the growth of the births-law residual r_K = measured - predicted amplitude (T116: ratios 1.1555 ... 1.2015).
The prediction of q1085 (copied as text) uses the half-line data of centre_scripts/c50_cap_D100_norefit.json: stage D = 92, positions d_j up to D - 20 as spline knots, law fitted on 4 <= d <= 30.
Candidates (amendment 1923):
 C1 far-end truncation of the half-line stage: predictions from the converged stages D = 74, 80, 86, 92 (the stage D = 98 has conv False and is not used); r_K(D) for K_b = 61 to 79.
 C2 the spline knot cut: D - 20 (baseline), D - 30, D - 40 at the stage D = 92.   C3 the law-fit window: 4 to 30 (baseline), 6 to 40, 10 to 50 at the stage D = 92.
Measured amplitudes: A_cross of tower_K<K_b+1>_birth_out.json (even K_b), A_s of tower_K<K_b+1>_As.json (odd K_b), K_b = 61 to 79.
Registered expectations printed at the end (Reading, not a verdict): E1 r(80) > r(86) > r(92) > 0 at every K_b 70 to 79; E2 ln[r_86 / r_92] in [1.2, 3.0] at K_b 74 to 79 (C1: 6 kappa with kappa about 0.3); the rival C1-null: |ln[r_86 / r_92]| < 0.5 at K_b 74 to 79;
E3 |ln r_86(a_K) - ln r_92(a_K + 6)| < 0.5 for K_b 61 to 66 (translation in depth; ln r_92 interpolated linearly in a); C2, C3: shift of a_pred at K_b 79 below 1e-9 (not a source) or above 1e-8 (can matter).
Positive control (run first, and alone with --registration-run): the baseline (D = 92, cut 20, window 4 to 30) reproduces a_pred of q635_capacity_K74_79_predictions.json for K_b 74 to 79 within 1e-10.
Usage: py q1183_births_residual_halfline_depth_reading.py [--registration-run]   (run from temporal_within/)
"""

import json, os, sys
import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.interpolate import CubicSpline

pi = np.pi
s0 = 1 / (16 * pi**2)
E = [
    4 / 3 + 8 * pi**2 / 45,
    176 / 15 + 32 * pi**2 / 9 + 64 * pi**4 / 1575,
    23792 / 315 + 8416 * pi**2 / 135 - 4544 * pi**4 / 2025 + 2176 * pi**6 / 33075,
    248768 / 567
    + 2083072 * pi**2 / 2835
    - 112448 * pi**4 / 1215
    - 16524544 * pi**6 / 2679075
    + 345088 * pi**8 / 1091475,
]
A = "scratch"


def dder(r, n=4):
    v = 4 * pi * r
    return sum(E[i] / v ** (4 + 2 * i) for i in range(n))


def rho_of(d, d0):
    return brentq(lambda r: r**3 - 3 * r / (8 * pi**2) - 1 / (32 * pi**2 * r) - 3 * s0 * (d + d0), 0.02, 400)


def N_of(d, d0, nu0):
    r = rho_of(d, d0)
    return 4 * pi**2 * r**4 - r**2 + np.log(r) / 6 + nu0


def spline(depths, law):
    x = np.asarray(depths)
    cs = CubicSpline(x, np.arange(1, len(x) + 1) - 0.5 - np.array([N_of(t, *law) for t in x]))
    return (lambda a: N_of(a, *law) + float(cs(a))), float(x[-1])


STAGES = json.load(open("centre_scripts/c50_cap_D100_norefit.json"))["stages"]


def predict(D, cut=20.0, win=(4.0, 30.0), Ks=range(61, 80)):
    sc = [s for s in STAGES if s["D"] == D][0]
    if not sc["conv"]:
        raise SystemExit(f"stage D = {D} is not converged")
    dc = np.array(sc["d"][: 1 + sc["nfree"]])
    jj = np.arange(1, len(dc) + 1) - 0.5
    mm = (dc >= win[0]) & (dc <= win[1])
    capl = tuple(
        least_squares(lambda q: np.array([N_of(x, q[0], q[1]) for x in dc[mm]]) - jj[mm], [0.0558, 0.5482]).x
    )
    dc = dc[dc <= D - cut]
    N, end = spline(dc, capl)
    out = {}
    for K in Ks:
        out[K] = brentq(
            lambda a: 2 * N(a) - K - 1 / 6 - dder(rho_of(a, capl[0]), 4), 30, min(70.0, end), xtol=1e-13
        )
    return out, end


def measured(Kb):
    if Kb % 2 == 0:
        return json.load(open(f"{A}/tower_K{Kb + 1}_birth_out.json"))["A_cross"]
    return json.load(open(f"{A}/tower_K{Kb + 1}_As.json"))["A_s"]


def control():
    ref = json.load(open(f"{A}/q635_capacity_K74_79_predictions.json"))["capacity"]["pred"]
    base, _ = predict(92.0, 20.0, (4.0, 30.0), range(74, 80))
    dev = max(abs(base[K] - ref[str(K)]["a_pred"]) for K in range(74, 80))
    print(
        f"positive control: baseline D = 92, cut 20, window 4 to 30 against the stored a_pred of K_b 74 to 79: largest difference {dev:.2e} -> {dev < 1e-10}"
    )
    return dev < 1e-10


def main():
    ok = control()
    if not ok:
        print("READING q1183: VOID (the baseline does not reproduce the stored predictions)")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("READING q1183: controls only")
        sys.exit(0)
    Ks = list(range(61, 80))
    meas = {K: measured(K) for K in Ks}
    R = {}
    ends = {}
    for D in (74.0, 80.0, 86.0, 92.0):
        p, e = predict(D, 20.0, (4.0, 30.0), Ks)
        R[D] = {K: meas[K] - p[K] for K in Ks}
        ends[D] = e
    print("knot ends:", {D: round(e, 3) for D, e in ends.items()})
    print("residual r_K(D) = measured - predicted, in units of 1e-9 (stage D = 74, 80, 86, 92):")
    for K in Ks:
        print(f"  K_b {K}: a {meas[K]:.4f}  " + "  ".join(f"D{int(D)} {R[D][K] * 1e9:+12.3f}" for D in R))
    print("ln ratios:")
    for K in Ks:
        try:
            print(
                f"  K_b {K}: ln r86/r92 {np.log(R[86.0][K] / R[92.0][K]):+.3f}  ln r80/r86 {np.log(R[80.0][K] / R[86.0][K]):+.3f}  ln r74/r80 {np.log(R[74.0][K] / R[80.0][K]):+.3f}"
            )
        except (ValueError, FloatingPointError):
            print(f"  K_b {K}: a residual changes sign, ratio undefined")
    e1 = all(R[80.0][K] > R[86.0][K] > R[92.0][K] > 0 for K in range(70, 80))
    lr = {K: np.log(R[86.0][K] / R[92.0][K]) for K in range(74, 80) if R[86.0][K] > 0 and R[92.0][K] > 0}
    e2 = len(lr) == 6 and all(1.2 <= v <= 3.0 for v in lr.values())
    null = len(lr) == 6 and all(abs(v) < 0.5 for v in lr.values())
    a_s = np.array([meas[K] for K in Ks])
    lr92 = np.array([np.log(abs(R[92.0][K])) for K in Ks])
    d3 = {}
    for K in range(61, 67):
        at = meas[K] + 6.0
        if at <= a_s[-1]:
            d3[K] = np.log(abs(R[86.0][K])) - np.interp(at, a_s, lr92)
    e3 = len(d3) == 6 and all(abs(v) < 0.5 for v in d3.values())
    print("E3 differences ln r_86(a_K) - ln r_92(a_K + 6):", {K: round(float(v), 3) for K, v in d3.items()})
    print(
        f"E1 monotone in D at K_b 70 to 79: {e1}; E2 ln r86/r92 in [1.2, 3.0] at K_b 74 to 79: {e2}; C1-null (|ln| < 0.5) : {null}; E3 translation within 0.5 at K_b 61 to 66: {e3}"
    )
    base, _ = predict(92.0, 20.0, (4.0, 30.0), range(74, 80))
    print("C2 knot cut (stage 92), shift of a_pred relative to the baseline (cut 20):")
    for cut in (30.0, 40.0):
        p, e = predict(92.0, cut, (4.0, 30.0), range(74, 80))
        print(
            f"  cut {cut:.0f} (knot end {e:.2f}): "
            + "  ".join(f"K_b {K} {p[K] - base[K]:+.2e}" for K in range(74, 80))
        )
    print("C3 law-fit window (stage 92), shift of a_pred relative to the baseline (window 4 to 30):")
    for win in ((6.0, 40.0), (10.0, 50.0)):
        p, e = predict(92.0, 20.0, win, range(74, 80))
        print(f"  window {win}: " + "  ".join(f"K_b {K} {p[K] - base[K]:+.2e}" for K in range(74, 80)))
    print("END READING q1183 (no verdict)")


if __name__ == "__main__":
    main()
