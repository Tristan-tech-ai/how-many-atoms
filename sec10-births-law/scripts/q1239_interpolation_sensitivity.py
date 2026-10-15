"""q1239 (8 Oct 2026, registered in amendment 2014): sensitivity of the prediction a_pred (order 16) to the interpolation of the residuals of the half-line atoms.
Counting function N(a) = N_law(a) + S(a), S an interpolant through the points (d_j, (j - 1/2) - N_law(d_j)) of the capacity half-line c50c D = 110, knots to depth 90 as q1209 (depth <= 90). Variants: (A) cubic spline, not-a-knot (the one used in q1209); (B) quintic interpolating B-spline (scipy make_interp_spline, k = 5); (C) cubic spline with natural end conditions; (D) cubic spline through every second atom (odd j) only, evaluated everywhere.
Events: K_b 61 to 80 (the law with the series to order 16). Reported: a_pred of each variant minus (A), in 1e-9.
Items: I1 the largest |a_B - a_A| over the ten births K_b 62 to 80 is below 0.005e-9; I2 the same over the ten splits K_b 61 to 79 is below 0.005e-9. Reading without item: variants C and D.
Controls (first): variant A reproduces pred_series of q1209 for K_b 61, 70, 80 within 1e-12; the variant D differs from A by more than 1e-12 at K_b 70 (the comparison can detect an interpolation change).
Usage: py q1239_interpolation_sensitivity.py  (from temporal_within/centre_scripts/)"""

import json, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.optimize import brentq, least_squares
import q1209_births_law_derived_series as G

Q = G.Q
sc = [s for s in json.load(open("c50_capc_D110.json"))["stages"] if s["D"] == 110.0][0]
dc = np.array(sc["d"][: 1 + sc["nfree"]])
jj = np.arange(1, len(dc) + 1) - 0.5
mm = (dc >= 4) & (dc <= 30)
CAPL = tuple(
    least_squares(lambda q: np.array([Q.N_of(x, q[0], q[1]) for x in dc[mm]]) - jj[mm], [0.0558, 0.5482]).x
)
sel = dc <= 90.0
X = dc[sel]
R = jj[sel] - np.array([Q.N_of(x, *CAPL) for x in X])
H = G.H


def make(variant):
    if variant == "A":
        s = CubicSpline(X, R)
    elif variant == "B":
        s = make_interp_spline(X, R, k=5)
    elif variant == "C":
        s = CubicSpline(X, R, bc_type="natural")
    else:
        s = CubicSpline(X[::2], R[::2])
    end = float(X[-1])
    return lambda a: Q.N_of(a, *CAPL) + float(s(a)), end


def pred(N, K):
    f = lambda a: 2 * N(a) - K - 1 / 6 - G.delta((N(a + H) - N(a - H)) / (2 * H), 16)
    return brentq(f, 30.0, 70.0, xtol=1e-14)


def main():
    NA, _ = make("A")
    c1 = all(abs(pred(NA, K) - G.pred_series(G.Ncap, K, 16)) < 1e-12 for K in (61, 70, 80))
    ND, _ = make("D")
    c2 = abs(pred(ND, 70) - pred(NA, 70)) > 1e-12
    print(
        f"control: variant A reproduces q1209 at K_b 61, 70, 80 within 1e-12: {c1}; variant D differs from A by {abs(pred(ND, 70) - pred(NA, 70)):.2e} at K_b 70 (> 1e-12): {c2}"
    )
    if not (c1 and c2):
        print("READING q1239: VOID (control)")
        sys.exit(2)
    res = {}
    for v in "BCD":
        N, _ = make(v)
        res[v] = {K: (pred(N, K) - pred(NA, K)) * 1e9 for K in range(61, 81)}
        b = max(abs(res[v][K]) for K in range(62, 81, 2))
        s = max(abs(res[v][K]) for K in range(61, 80, 2))
        print(
            f"variant {v} - A (1e-9): max births {b:.4f}, max splits {s:.4f}; births "
            + ", ".join(f"{K}: {res[v][K]:+.4f}" for K in range(62, 81, 2))
            + "; splits "
            + ", ".join(f"{K}: {res[v][K]:+.4f}" for K in range(61, 80, 2))
        )
    i1 = max(abs(res["B"][K]) for K in range(62, 81, 2)) < 0.005
    i2 = max(abs(res["B"][K]) for K in range(61, 80, 2)) < 0.005
    print(f"item I1 (births, quintic against cubic < 0.005e-9): {i1}; item I2 (splits): {i2}")
    print("READING q1239: I1 " + str(i1) + ", I2 " + str(i2) + " (no verdict)")


if __name__ == "__main__":
    main()
