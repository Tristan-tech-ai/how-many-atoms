"""q635: predictions for Amendment 938. Fully derived births law, 2 N_half^F(a_t) = K_b + 1/6 + delta_der(rho), delta_der = e2/v^4 +
e3/v^6 + e4/v^8 + e5/v^10 (CENTRE_E2_DERIVATION closed forms), v = 4 pi rho, rho the law density at d = a_t with the functional's standard
edge number (as in every earlier comparison). N_half from each functional's half-line (law + residual spline, knots to D - 20):
immune = imm_D132 run stage D = 100; NPMLE = npm_D92; capacity = cap_D100_norefit run stage D = 92 (law refit 4 <= d <= 30).
No transition enters. Also printed: the same with delta through e4, the nulls c = 1/3 and delta = 3.2/v^4. Writes q635_predictions.json.
"""

import json, numpy as np
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


si = [s for s in json.load(open("c33_imm_D132.json"))["stages"] if s["D"] == 100.0][0]
pi_ = np.array(si["p"][: si["nfree"]])
pi_ = pi_[pi_ <= si["D"] - 20]
sn = json.load(open("c33_npm_D92.json"))["stages"][-1]
pn = np.array(sn["p"][: sn["nfree"]])
pn = pn[pn <= sn["D"] - 20]
sc = [s for s in json.load(open("c50_cap_D100_norefit.json"))["stages"] if s["D"] == 92.0][0]
dc = np.array(sc["d"][: 1 + sc["nfree"]])
jj = np.arange(1, len(dc) + 1) - 0.5
mm = (dc >= 4) & (dc <= 30)
capl = tuple(
    least_squares(lambda q: np.array([N_of(x, q[0], q[1]) for x in dc[mm]]) - jj[mm], [0.0558, 0.5482]).x
)
dc = dc[dc <= sc["D"] - 20]
F = {"capacity": (spline(dc, capl), capl[0], [67, 68, 69, 70, 71, 72], (30, 70))}
out = {}
for name, ((N, end), d0, Ks, (lo, hi)) in F.items():
    out[name] = {"knot_end": end, "pred": {}}
    for K in Ks:
        g = lambda dl: brentq(lambda a: 2 * N(a) - K - 1 / 6 - dl(a), lo, hi, xtol=1e-13)
        a5 = g(lambda a: dder(rho_of(a, d0), 4))
        a4 = g(lambda a: dder(rho_of(a, d0), 3))
        a0 = g(lambda a: 0.0)
        a32 = g(lambda a: 3.2 / (4 * pi * rho_of(a, d0)) ** 4)
        out[name]["pred"][K] = {
            "a_pred": a5,
            "a_pred_e4": a4,
            "null_c_third": a0,
            "null_3.2": a32,
            "rho": rho_of(a5, d0),
            "inside": end - a5,
        }
        print(
            f"{name:8s} K_b {K}: a_pred = {a5:.9f} (thru e4 {a4 - a5:+.1e})  rho {rho_of(a5, d0):.4f}  knot end - a = {end - a5:.1f}  "
            f"nulls: 1/3 {a0 - a5:+.2e}, 3.2/v^4 {a32 - a5:+.2e}"
        )
json.dump(out, open("../scratch/q635_capacity_K67_72_predictions.json", "w"), indent=1)
