"""q1282 (8 Oct 2026, EXPLORATORY reading, no item, no verdict): the residuals R_k(K_b) (1e-9, quintic median-sum law) of the functionals k = 1 (NPMLE), 2 (immune), 3, 4, 5 with their v^2, using the data of run1279a/b/c.sh (K_b 38 to 52) and the earlier runs below; (a) per-k fits ln R = a + q ln v^2 - b v^2 with standard errors and corr(b, q); (b) the common fit with b and q shared by the five series and a_k free; (c) c_k = R_k / R_NPMLE at equal v^2 against K_b for k = 2 to 5.
Inclusion rule (from the purpose: the half-line interpolation must not change R by more than 2 percent): an event enters if R > 0 and |a_med(quintic) - a_med(septic)| <= 0.02 R; located events only (last residual below 1e-18 and |a_t - last a| < 1e-9; events of the older tables are taken as stored).
Control (first): the fit routine recovers q = 2.5, b = 0.23 and a_k from a noiseless synthetic series at the real v^2 values to 1e-8, and the NPMLE event K_b 36 has R = 2.2643e-9 (q1251 listing, stored amplitude) within 0.002e-9 (the refreshed amplitude used here is lower by 0.001e-9, q1269).
Usage: py q1282_extended_decay_reading.py  (from temporal_within/centre_scripts/)"""

import json, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
import numpy as np
from scipy.optimize import least_squares
import q1250_quintic_residual as Q
import q1271_linear_in_k_blind as Z
import q1273_k4_curvature_blind as W

C, N, m = Q.C, Q.N, Z.m
KS = (1, 2, 3, 4, 5)


def loc(f):
    try:
        j = json.load(open(f + ".json"))
        hl = [x for x in j["hist"] if "res" in x and "a" in x]
        a = j.get("a_t")
        return a if (a is not None and hl and hl[-1]["res"] < 1e-18 and abs(a - hl[-1]["a"]) < 1e-9) else None
    except Exception:
        return None


def get(f):
    return (lambda j: j.get("a_t") or j.get("A_t"))(json.load(open(f + ".json")))


sn = [s for s in json.load(open("c33_immc_D146.json"))["stages"] if s["conv"] and s["D"] == 92.0][0]
pi_ = np.array(sn["p"][: sn["nfree"]])
pi_ = pi_[pi_ <= 72]
law4 = tuple(json.load(open("q1273_predictions.json"))["law"])
law5 = tuple(json.load(open("q1276_predictions.json"))["5.0"]["law"])


def nh(deg):
    return {
        1: Q.make(lambda a: Q.N4(a, N.LAW), N.pn, deg),
        2: Q.make(lambda a: Q.N4(a, (0.22573, 0.06806)), pi_, deg),
        3: Z.nhalf(3, "c33_k3c.json", 92.0, deg),
        4: W.nhalf("c33_k4c.json", 92.0, deg, law4),
        5: W.nhalf("c33_k5c.json", 92.0, deg, law5),
    }


N5, N7 = nh(5), nh(7)
ev = {k: {} for k in KS}
for K, f in m.NPM_FILES.items():
    if 20 <= K <= 44 and K != 38:
        ev[1][K] = get(f)
for K in range(34, 49):
    ev[1][K] = json.load(open(f"q607_u1268_K{K}.json"))["a_t"]
for K in range(38, 53):
    a = loc(f"q607_u1279_K{K}")
    if a is not None:
        ev[1][K] = a
for e in json.load(open("q624_immune_table.json")):
    if 24 <= int(e[1]) <= 28:
        ev[2][int(e[1])] = float(e[0])
for K in range(14, 24):
    ev[2][K] = json.load(open(f"c19_u1263_K{K}.json"))["a_t"]
for K in range(29, 38):
    ev[2][K] = json.load(open(f"c19_u1231_K{K}.json"))["a_t"]
for K in range(36, 49):
    ev[2][K] = json.load(open(f"c19_u1267_K{K}.json"))["a_t"]
for K in range(38, 53):
    a = loc(f"c19_u1279_K{K}")
    if a is not None:
        ev[2][K] = a
for K in range(28, 38):
    a = loc(f"q607_u1271_K{K}")
    ev[3][K] = a if a is not None else None
for K in range(29, 37):
    a = loc(f"q607_u1273_K{K}")
    ev[4][K] = a if a is not None else None
for K in range(29, 37):
    a = loc(f"q607_u1276_k5_K{K}")
    ev[5][K] = a if a is not None else None
for k in (3, 4, 5):
    for K in range(38, 53):
        a = loc(f"q607_u1279_k{k}_K{K}")
        if a is not None:
            ev[k][K] = a
data = {}
for k in KS:
    for K, a in sorted(ev[k].items()):
        if a is None:
            continue
        am5, am7 = Q.root(N5[k], K), Q.root(N7[k], K)
        r = (a - am5) * 1e9
        if r > 0 and abs(am5 - am7) * 1e9 <= 0.02 * r:
            data.setdefault(k, []).append((K, r, W.v2f(N5[k], a)))
ok1 = any(K == 36 and abs(r - 2.2643) < 0.002 for K, r, v in data.get(1, []))
xs = np.array([v for _, _, v in data[2]])
ys = 0.5 + 2.5 * np.log(xs) - 0.23 * xs


def fit1(x, y, w=None):
    A = np.vstack([np.ones_like(x), np.log(x), -x]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ beta
    s2 = r @ r / max(len(y) - 3, 1)
    cov = s2 * np.linalg.inv(A.T @ A)
    return beta, np.sqrt(np.diag(cov)), cov[1, 2] / np.sqrt(cov[1, 1] * cov[2, 2]), np.sqrt(s2)


bt, *_ = fit1(xs, ys)
ok2 = abs(bt[1] - 2.5) < 1e-8 and abs(bt[2] - 0.23) < 1e-8 and abs(bt[0] - 0.5) < 1e-8
print(
    f"control: synthetic recovery {ok2}; NPMLE K_b 36 residual {ok1}; events per k: "
    + ", ".join(f"{k}: {len(data[k])} (K_b {data[k][0][0]}-{data[k][-1][0]})" for k in KS if k in data)
)
if not (ok1 and ok2):
    print("READING q1282: VOID (control)")
    sys.exit(2)
print("(a) per-k fits ln R = a + q ln v^2 - b v^2:")
for k in KS:
    if k not in data or len(data[k]) < 6:
        continue
    d = np.array(data[k])
    beta, se, rho, rms = fit1(d[:, 2], np.log(d[:, 1]))
    print(
        f"  k = {k}: {len(d)} events, v^2 {d[0, 2]:.1f}-{d[-1, 2]:.1f}, b = {beta[2]:.4f} +- {se[2]:.4f}, q = {beta[1]:.2f} +- {se[1]:.2f}, corr {rho:+.3f}, rms {rms:.4f}; 1/b = {1 / beta[2]:.3f}"
    )
rows = [(k, np.log(r), v) for k in KS if k in data for K, r, v in data[k]]
idx = {k: i for i, k in enumerate([k for k in KS if k in data])}
ks = list(idx)
X = np.zeros((len(rows), 2 + len(ks)))
Y = np.zeros(len(rows))
for i, (k, ly, v) in enumerate(rows):
    X[i, 0] = np.log(v)
    X[i, 1] = -v
    X[i, 2 + idx[k]] = 1
    Y[i] = ly
beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
res = Y - X @ beta
s2 = res @ res / (len(Y) - len(beta))
cov = s2 * np.linalg.inv(X.T @ X)
print(
    f"(b) common fit over {len(Y)} events: q = {beta[0]:.2f} +- {np.sqrt(cov[0, 0]):.2f}, b = {beta[1]:.4f} +- {np.sqrt(cov[1, 1]):.4f}, corr {cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]):+.3f}, rms {np.sqrt(s2):.4f}; 1/b = {1 / beta[1]:.3f}; a_k = "
    + ", ".join(f"{k}: {beta[2 + idx[k]]:.3f}" for k in ks)
)
print("(c) c_k = R_k / R_NPMLE at equal v^2 (NPMLE interpolated in ln R against v^2):")
dn = np.array(data[1])
o = np.argsort(dn[:, 2])
interp = lambda x: (
    float(np.exp(np.interp(x, dn[o, 2], np.log(dn[o, 1])))) if dn[o, 2].min() <= x <= dn[o, 2].max() else None
)
for k in (2, 3, 4, 5):
    if k not in data:
        continue
    line = []
    for K, r, v in data[k]:
        rn = interp(v)
        if rn is not None and K in (29, 32, 36, 38, 40, 42, 44, 46, 48, 50, 52):
            line.append(f"{K}: {r / rn:.4f}")
    print(f"  k = {k}: " + ", ".join(line))
print("READING q1282 (exploratory, no verdict)")
