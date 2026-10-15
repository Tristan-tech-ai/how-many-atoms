"""q1211 (8 Oct 2026): sensitivity of the births law with the derived series (order 16, rho = dN_half/da) to the half-line stage and to the choices of the spline: capacity K_b 61, 70, 79, 80 with the c50c stages D = 86, 92, 98, 104, 110, 116, 122 (files c50_capc_D110.json and c50_capc_D140.json), the cut of the spline knots (D - 20, D - 30, D - 40) and the window of the law fit that serves as the base of the spline (4 to 30, 6 to 40, 10 to 50). Reading, no verdict.
Usage: E_FILE=d22_out_from_log.json py q1211_derived_sensitivity.py  (from temporal_within/centre_scripts/)"""

import json, os, re, sys
import numpy as np
from scipy.optimize import brentq, least_squares

sys.argv = ["x"]
EF = os.environ.get("E_FILE", "d22_out_from_log.json")
E = {int(k[1:]): float(v) for k, v in json.load(open(EF)).items() if re.fullmatch(r"e\d+", k) and k != "e1"}
os.chdir("..")
sys.path.insert(0, ".")
import q1183_births_residual_halfline_depth_reading as Q

os.chdir("centre_scripts")
pi = np.pi
H = 1e-3


def delta(rho):
    v = 4 * pi * rho
    return sum(E[n] / v ** (2 * n) for n in range(2, 17))


stages = {}
for fn in ("c50_capc_D110.json", "c50_capc_D140.json"):
    for s in json.load(open(fn))["stages"]:
        if s["conv"] and s["D"] not in stages:
            stages[s["D"]] = s


def pred(D, K, cut=20.0, win=(4.0, 30.0)):
    sc = stages[D]
    dc = np.array(sc["d"][: 1 + sc["nfree"]])
    jj = np.arange(1, len(dc) + 1) - 0.5
    mm = (dc >= win[0]) & (dc <= win[1])
    capl = tuple(
        least_squares(
            lambda q: np.array([Q.N_of(x, q[0], q[1]) for x in dc[mm]]) - jj[mm], [0.0558, 0.5482]
        ).x
    )
    N, end = Q.spline(dc[dc <= D - cut], capl)
    return brentq(
        lambda a: 2 * N(a) - K - 1 / 6 - delta((N(a + H) - N(a - H)) / (2 * H)),
        30.0,
        min(end - 0.1, 70.0),
        xtol=1e-14,
    )


Ks = (61, 70, 79, 80)
ref = {K: pred(110.0, K) for K in Ks}
print("reference (D = 110, cut 20, window 4-30): a_pred", {K: round(ref[K], 12) for K in Ks})
for D in (86.0, 92.0, 98.0, 104.0, 110.0, 116.0, 122.0):
    if D not in stages:
        continue
    print(
        f"stage D = {D:.0f}: change of a_pred (1e-9)", {K: f"{(pred(D, K) - ref[K]) * 1e9:+.4f}" for K in Ks}
    )
for cut in (20.0, 30.0, 40.0):
    print(
        f"D = 110, knot cut D - {cut:.0f}: change (1e-9)",
        {K: f"{(pred(110.0, K, cut=cut) - ref[K]) * 1e9:+.4f}" for K in Ks},
    )
for win in ((4.0, 30.0), (6.0, 40.0), (10.0, 50.0)):
    print(
        f"D = 110, law-fit window {win}: change (1e-9)",
        {K: f"{(pred(110.0, K, win=win) - ref[K]) * 1e9:+.4f}" for K in Ks},
    )
print("END READING q1211")
