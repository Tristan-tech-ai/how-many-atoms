"""q1293 (8 Oct 2026, EXPLORATORY reading, no item, no verdict): per-k fits of ln(R_k / Da) = ln sigma_inf(k) + ln(1 + g(k) / v^2) for k = 1, 1.5, 2, 2.5, 3, 4, 5 on the located events with K_b 14 to 40 (K_b 38 excluded; the common offset of amendment 2066 is negligible there), the inclusion rule |a_quintic - a_septic| <= 0.02 R, and the tables of sigma_inf(k), g(k) and the rms of ln R; also the fit with g shared by all k.
Data: k = 1 NPMLE (stored 20 to 44, refreshed 34 to 48), k = 2 immune (q624 24 to 28, c19 14 to 23, 29 to 37, 36 to 48), k = 3 (q607_u1287 K_b 14 to 23, q607_u1271 28 to 37, q607_u1279_k3 38 to 52), k = 4 and 5 (u1292 14 to 28, u1273 / u1276 29 to 36, u1279 38 to 52), k = 1.5 and 2.5 (u1292 14 to 28 and 37 to 44, u1276 29 to 36). Da = |Im S_+| / (2 rho) of the [8/8] approximant (q1285).
Control (first): the routine recovers (sigma_inf, g) = (3.1, -13) and (7.9, -19) from noiseless synthetic data at the real v^2 values; and the event counts per k are those of the registered runs (k = 1.5: 8 + 15 + 8, k = 2.5: 8 + 15 + 8 before the inclusion rule).
Usage: py q1293_sigma_g_of_k_reading.py  (from temporal_within/centre_scripts/)"""

import io, contextlib, json, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
import numpy as np
import mpmath as mp
from scipy.optimize import least_squares

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import q1250_quintic_residual as Q
    import q1271_linear_in_k_blind as Z
    import q1273_k4_curvature_blind as W
    import q1220_borel_pade_births_law as T
    import q1282_extended_decay_reading as D
mp.mp.dps = 30
Bf = T.machine(T.coef_series(), 8, 8)[0]


def Da(v2):
    x = mp.mpf(1) / v2
    return abs(float(mp.im(T.ray(Bf, x, 0.3)))) / (2 * np.sqrt(v2) / (4 * np.pi)) * 1e9


def locr(f):
    try:
        j = json.load(open(f + ".json"))
        hl = [x for x in j["hist"] if "res" in x and "a" in x]
        a = j.get("a_t")
        return a if (a is not None and hl and hl[-1]["res"] < 1e-18 and abs(a - hl[-1]["a"]) < 1e-9) else None
    except Exception:
        return None


law = {
    1.5: json.load(open("q1276_predictions.json"))["1.5"]["law"],
    2.5: json.load(open("q1276_predictions.json"))["2.5"]["law"],
    4.0: json.load(open("q1273_predictions.json"))["law"],
    5.0: json.load(open("q1276_predictions.json"))["5.0"]["law"],
}
stage = {1.5: "c33_k15c.json", 2.5: "c33_k25c.json", 4.0: "c33_k4c.json", 5.0: "c33_k5c.json"}
N5 = {k: W.nhalf(stage[k], 92.0, 5, tuple(law[k])) for k in stage}
N7 = {k: W.nhalf(stage[k], 92.0, 7, tuple(law[k])) for k in stage}
raw = {}
raw[1.5] = {K: locr(f"q607_u1292_k15_K{K}") for K in list(range(14, 29)) + list(range(37, 45))}
for K in range(29, 37):
    raw[1.5][K] = locr(f"q607_u1276_k15_K{K}")
raw[2.5] = {K: locr(f"q607_u1292_k25_K{K}") for K in list(range(14, 29)) + list(range(37, 45))}
for K in range(29, 37):
    raw[2.5][K] = locr(f"q607_u1276_k25_K{K}")
raw[4.0] = {K: locr(f"q607_u1292_k4_K{K}") for K in range(14, 29)}
for K in range(29, 37):
    raw[4.0][K] = locr(f"q607_u1273_K{K}")
for K in range(38, 45):
    raw[4.0][K] = locr(f"q607_u1279_k4_K{K}")
raw[5.0] = {K: locr(f"q607_u1292_k5_K{K}") for K in range(14, 29)}
for K in range(29, 37):
    raw[5.0][K] = locr(f"q607_u1276_k5_K{K}")
for K in range(38, 45):
    raw[5.0][K] = locr(f"q607_u1279_k5_K{K}")
cnt = {k: sum(1 for a in v.values() if a is not None) for k, v in raw.items()}
data = {}
for k in (1.5, 2.5, 4.0, 5.0):
    for K, a in sorted(raw[k].items()):
        if a is None or K == 38 or K > 40:
            continue
        am5, am7 = Q.root(N5[k], K), Q.root(N7[k], K)
        r = (a - am5) * 1e9
        if r > 0 and abs(am5 - am7) * 1e9 <= 0.02 * r:
            data.setdefault(k, []).append((K, r, W.v2f(N5[k], a)))
for k in (1, 2, 3):
    data[float(k)] = [(K, r, v) for K, r, v in D.data.get(k, []) if K <= 40 and K != 38]
# k = 3 events at K_b 14 to 23 (q1287) are not in q1282's data: add them
N3 = Z.nhalf(3, "c33_k3c.json", 92.0, 5)
N37 = Z.nhalf(3, "c33_k3c.json", 92.0, 7)
for K in range(14, 24):
    a = locr(f"q607_u1287_K{K}")
    if a is not None:
        am5, am7 = Q.root(N3, K), Q.root(N37, K)
        r = (a - am5) * 1e9
        if r > 0 and abs(am5 - am7) * 1e9 <= 0.02 * r:
            data[3.0].append((K, r, W.v2f(N3, a)))
data[3.0].sort()
ok_cnt = cnt[1.5] == 31 and cnt[2.5] == 31
xs = np.array([v for _, _, v in data[2.0]])


def fit(k, g0=-12.0):
    d = np.array(data[k])
    y = np.log(d[:, 1] / np.array([Da(v) for v in d[:, 2]]))
    v = d[:, 2]
    f = lambda q: q[1] + np.log(1 + q[0] / v) - y
    r = least_squares(f, [g0, 1.0])
    return r.x[0], float(np.exp(r.x[1])), float(np.sqrt(np.mean(r.fun**2))), len(d)


vs = np.array([69.2, 80, 90, 100, 110.0])
syn = lambda s, g: np.log(s) + np.log(1 + g / vs)
r1 = least_squares(lambda q: q[1] + np.log(1 + q[0] / vs) - syn(3.1, -13), [-10, 1.0])
r2 = least_squares(lambda q: q[1] + np.log(1 + q[0] / vs) - syn(7.9, -19), [-10, 1.0])
ok_fit = (
    abs(r1.x[0] + 13) < 1e-8
    and abs(np.exp(r1.x[1]) - 3.1) < 1e-8
    and abs(r2.x[0] + 19) < 1e-8
    and abs(np.exp(r2.x[1]) - 7.9) < 1e-8
)
print(f"control: fit recovery {ok_fit}; event counts of the runs: {cnt} (k = 1.5, 2.5 expect 31): {ok_cnt}")
if not (ok_fit and ok_cnt):
    print("READING q1293: VOID (control)")
    sys.exit(2)
print("k      events  K_b range     v^2 range     sigma_inf    g        rms(ln R)")
res = {}
for k in sorted(data):
    g, s, rms, n = fit(k)
    d = np.array(data[k])
    res[k] = (s, g)
    print(
        f"{k:4.1f}   {n:4d}   {int(d[0, 0])}-{int(d[-1, 0])}   {d[0, 2]:6.1f}-{d[-1, 2]:6.1f}   {s:8.4f}  {g:8.3f}   {rms:.4f}"
    )
allk = sorted(data)
idx = {k: i for i, k in enumerate(allk)}
rows = [(k, np.log(r / Da(v)), v) for k in allk for K, r, v in data[k]]


def resid(q):
    return np.array([q[1 + idx[k]] + np.log(1 + q[0] / v) - ly for k, ly, v in rows])


q0 = [-14.0] + [1.0] * len(allk)
rr = least_squares(resid, q0)
print(
    f"shared g = {rr.x[0]:.3f}; rms of ln R {np.sqrt(np.mean(rr.fun ** 2)):.4f} over {len(rows)} events; sigma_inf: "
    + ", ".join(f"{k}: {np.exp(rr.x[1 + idx[k]]):.4f}" for k in allk)
)
print("READING q1293 (exploratory, no verdict)")
