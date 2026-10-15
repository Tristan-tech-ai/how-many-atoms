"""q1240 (8 Oct 2026, registered in amendment 2017): sensitivity of the NPMLE prediction (series to order 16) to the interpolation of the residuals of the half-line atoms.
N(a) = N4(a) + S(a), N4 the O(rho^-4) law with the NPMLE edge numbers (-0.22676, 0.14770), S an interpolant through (p_j, (j - 1/2) - N4(p_j)) of the last converged stage of c33_npmc_D110.json (atoms with depth <= 90), as q1205/q1209. Variants: A cubic spline not-a-knot, B quintic interpolating B-spline, D cubic spline through every second atom.
Controls (first): A reproduces pred_series(Nnp, K, 16, 20, 80) of q1209 within 1e-12 at K_b 54, 60, 66; D differs from A by more than 1e-12 at K_b 60.
Item I3: the largest |a_B - a_A| over the thirteen NPMLE transitions K_b 54 to 66 is below 0.005e-9. Readings: per-event differences and the deviations A_meas - a_pred for A and B.
Usage: py q1240_npmle_interpolation.py  (from temporal_within/centre_scripts/)"""

import json, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.optimize import brentq
import q1209_births_law_derived_series as G

T = G.T
sn = [s for s in json.load(open("c33_npmc_D110.json"))["stages"] if s["conv"]][-1]
pn = np.array(sn["p"][: sn["nfree"]])
pn = pn[pn <= 90]
LAW = (-0.22676, 0.14770)
pi = np.pi
s0 = 1 / (16 * pi**2)


def rho4(d):
    return brentq(
        lambda r: r**3 - 3 * r / (8 * pi**2) - 1 / (32 * pi**2 * r) - 3 * s0 * (d + LAW[0]), 0.02, 400
    )


def N4(d):
    r = rho4(d)
    return 4 * pi**2 * r**4 - r**2 + np.log(r) / 6 + LAW[1]


R = np.arange(1, len(pn) + 1) - 0.5 - np.array([N4(d) for d in pn])
H = G.H


def make(v):
    if v == "A":
        s = CubicSpline(pn, R)
    elif v == "B":
        s = make_interp_spline(pn, R, k=5)
    else:
        s = CubicSpline(pn[::2], R[::2])
    return lambda a: N4(a) + float(s(a))


def pred(N, K):
    f = lambda a: 2 * N(a) - K - 1 / 6 - G.delta((N(a + H) - N(a - H)) / (2 * H), 16)
    return brentq(f, 20.0, 80.0, xtol=1e-14)


def main():
    NA = make("A")
    c1 = all(abs(pred(NA, K) - G.pred_series(G.Nnp, K, 16, 20.0, 80.0)) < 1e-12 for K in (54, 60, 66))
    ND = make("D")
    c2 = abs(pred(ND, 60) - pred(NA, 60)) > 1e-12
    print(
        f"control: A reproduces q1209 at K_b 54, 60, 66 within 1e-12: {c1}; D differs from A by {abs(pred(ND, 60) - pred(NA, 60)):.2e} at K_b 60 (> 1e-12): {c2}"
    )
    if not (c1 and c2):
        print("READING q1240: VOID (control)")
        sys.exit(2)
    NB = make("B")
    meas = {
        K: json.load(open(f"q607_u1205_K{K}.json" if K != 62 else "q607_u1205r_K62.json"))["a_t"]
        for K in range(54, 67)
    }
    dAB = {K: (pred(NB, K) - pred(NA, K)) * 1e9 for K in meas}
    dA = {K: (meas[K] - pred(NA, K)) * 1e9 for K in meas}
    dB = {K: (meas[K] - pred(NB, K)) * 1e9 for K in meas}
    print("B - A (1e-9): " + ", ".join(f"{K}: {v:+.4f}" for K, v in dAB.items()))
    print("deviation with A (1e-9): " + ", ".join(f"{K}: {v:+.4f}" for K, v in dA.items()))
    print("deviation with B (1e-9): " + ", ".join(f"{K}: {v:+.4f}" for K, v in dB.items()))
    i3 = max(abs(v) for v in dAB.values()) < 0.005
    print(f"item I3 (largest |B - A| {max(abs(v) for v in dAB.values()):.4f}e-9 < 0.005e-9): {i3}")
    print("READING q1240: I3 " + str(i3) + " (no verdict)")


if __name__ == "__main__":
    main()
