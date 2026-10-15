"""POST HOC (8 Oct 2026): the defect of the births law with the TRUE local density (data density N_half' of the c50c stage D = 110, equal to the derived-law density to 1e-9 in the residual) fitted by a series in 1/v, v = 4 pi rho_data. defect(K_b) = 2 N(A_meas) - K_b - 1/6 - delta(rho_data(A_meas)) with delta through e5; fits defect = sum_n c_n v^-n over n in several sets. No criterion registered. Run with py from temporal_within/."""

import json, sys
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, ".")
import q1183_births_residual_halfline_depth_reading as Q

sc = [s for s in json.load(open("centre_scripts/c50_capc_D110.json"))["stages"] if s["D"] == 110.0][0]
Q.STAGES = [sc]
D = 110.0
dc = np.array(sc["d"][: 1 + sc["nfree"]])
jj = np.arange(1, len(dc) + 1) - 0.5
mm = (dc >= 4) & (dc <= 30)
capl = tuple(
    least_squares(lambda q: np.array([Q.N_of(x, q[0], q[1]) for x in dc[mm]]) - jj[mm], [0.0558, 0.5482]).x
)
N, end = Q.spline(dc[dc <= D - 20.0], capl)
H = 1e-3
KS = list(range(61, 80))
v = []
de = []
vl = []
dl = []
for K in KS:
    A = Q.measured(K)
    rho = (N(A + H) - N(A - H)) / (2 * H)
    v.append(4 * np.pi * rho)
    de.append(2 * float(N(A)) - K - 1 / 6 - Q.dder(rho, 4))
    rl = Q.rho_of(A, capl[0])
    vl.append(4 * np.pi * rl)
    dl.append(2 * float(N(A)) - K - 1 / 6 - Q.dder(rl, 4))
v = np.array(v)
de = np.array(de)
vl = np.array(vl)
dl = np.array(dl)
print("v (data density) K_b 61, 70, 79:", v[0], v[9], v[-1])
print("defect with the data density (1e-9):", {k: round(float(x) * 1e9, 2) for k, x in zip(KS, de)})
print("defect with the fitted-law density (1e-9):", {k: round(float(x) * 1e9, 2) for k, x in zip(KS, dl)})


def fit(ns, y=de, x=v, label=""):
    M = np.array([[xx ** (-n) for n in ns] for xx in x])
    c, *_ = np.linalg.lstsq(M / np.abs(y).max(), y / np.abs(y).max(), rcond=None)
    c = c
    res = y - M @ c
    print(
        f"basis v^-n, n = {ns}{label}: coefficients {['%.4g' % t for t in c]}; rms {np.sqrt(np.mean(res ** 2)):.3e}; max {np.abs(res).max():.3e}"
    )
    return c


for ns in (
    (8,),
    (10,),
    (6,),
    (4,),
    (8, 10),
    (6, 8),
    (6, 8, 10),
    (6, 8, 10, 12),
    (4, 6, 8, 10),
    (8, 10, 12),
    (3, 4, 5, 6),
):
    fit(ns)
print("for comparison e2..e5 = 3.088, 50.78, 535.5, -4254.5")
print("END POST HOC")
