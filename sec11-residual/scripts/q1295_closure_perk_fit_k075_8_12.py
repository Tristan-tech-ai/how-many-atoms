"""q1295 (8 Oct 2026, EXPLORATORY reading, no item, no verdict): per-k fits ln(R / Da) = ln sigma + ln(1 + g / v^2) for the exponents k = 0.75, 8, 12 on all located events K_b 14 to 40 (K_b 38 excluded as in q1293; inclusion rule |a_quintic - a_septic| <= 0.02 R), where the events K_b 14-28 and 37-40 are those of the failed test q1294 and K_b 29-36 those of q1278 and q1281; then the table of sigma(k) and g(k) over k = 0.75 to 12 with the values of q1293 (parsed from its stored output), and a three-parameter fit (extra term h / v^4) as a Reading of whether the two-parameter form is enough at large k.
Control (first): the routine recovers (sigma, g) = (3.1, -13) and (7.9, -19) from noiseless synthetic data at the real v^2 values; the fit of the k = 2 events of q1282 reproduces the q1293 values (5.5832, -15.019) within 1e-3; the number of events per exponent equals the number of files that pass the located rule (printed).
Usage: py q1295_closure_perk_fit_k075_8_12.py  (from temporal_within/centre_scripts/)"""

import io, contextlib, json, os, re, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
from scipy.optimize import least_squares

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import q1294_closure_linear_k_blind as M
Q, Z, W, D, Da, locr, nh = M.Q, M.Z, M.W, M.D, M.Da, M.locr, M.nh
TAG = {0.75: ("k075", "q607_u1278_k075_K"), 8.0: ("k8", "q607_u1281_k8_K"), 12.0: ("k12", "q607_u1281_k12_K")}
data = {}
count = {}
for k, (tag, old) in TAG.items():
    N5, _ = nh(k, 5)
    N7, _ = nh(k, 7)
    rows = []
    n = 0
    for K in list(range(14, 29)) + list(range(29, 37)) + list(range(37, 41)):
        a = locr(f"q607_u1294_{tag}_K{K}") if K not in range(29, 37) else locr(f"{old}{K}")
        if a is None:
            continue
        n += 1
        if K == 38:
            continue
        am5, am7 = Q.root(N5, K), Q.root(N7, K)
        r = (a - am5) * 1e9
        if r > 0 and abs(am5 - am7) * 1e9 <= 0.02 * r:
            rows.append((K, r, W.v2f(N5, a)))
    data[k] = rows
    count[k] = n
data[2.0] = [(K, r, v) for K, r, v in D.data.get(2, []) if K <= 40 and K != 38]


def fit(k, g0=-12.0, three=False):
    d = np.array(data[k])
    v = d[:, 2]
    y = np.log(d[:, 1] / np.array([Da(x) for x in v]))
    if three:
        f = lambda q: q[1] + np.log(1 + q[0] / v + q[2] / v**2) - y
        r = least_squares(f, [g0, 1.0, 0.0])
    else:
        f = lambda q: q[1] + np.log(1 + q[0] / v) - y
        r = least_squares(f, [g0, 1.0])
    return r.x, float(np.sqrt(np.mean(r.fun**2))), len(d)


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
x2, _, _ = fit(2.0)
ok_k2 = abs(np.exp(x2[1]) - 5.5832) < 1e-3 and abs(x2[0] + 15.019) < 1e-3
ok_cnt = all(count[k] >= 19 for k in count)
print(
    f"control: synthetic recovery {ok_fit}; k = 2 reproduces q1293 (sigma {np.exp(x2[1]):.4f}, g {x2[0]:.3f}) {ok_k2}; events located (K_b 14-40 incl. 38) {count} each at least 19: {ok_cnt}"
)
if not (ok_fit and ok_k2 and ok_cnt):
    print("READING q1295: VOID (control)")
    sys.exit(2)
prev = {}
for line in open("../scratch/q1293_out.txt", encoding="utf-8"):
    m = re.match(r"\s*(\d+\.\d)\s+(\d+)\s+\d+-\d+\s+[\d.]+-\s*[\d.]+\s+([\d.]+)\s+(-[\d.]+)\s+([\d.]+)", line)
    if m:
        prev[float(m.group(1))] = (float(m.group(3)), float(m.group(4)), float(m.group(5)), int(m.group(2)))
if len(prev) != 7:
    print(f"READING q1295: VOID (parsed {len(prev)} rows of q1293_out.txt, expected 7)")
    sys.exit(2)
res = dict(prev)
for k in (0.75, 8.0, 12.0):
    x, rms, n = fit(k)
    v = np.array([e[2] for e in data[k]])
    res[k] = (float(np.exp(x[1])), float(x[0]), rms, n)
    x3, rms3, _ = fit(k, three=True)
    print(
        f"k {k}: events {n} (K_b 14-40, v^2 {v.min():.1f}-{v.max():.1f}); two-parameter fit sigma {np.exp(x[1]):.4f}, g {x[0]:.3f}, rms ln R {rms:.4f}; three-parameter (h / v^4): sigma {np.exp(x3[1]):.4f}, g {x3[0]:.3f}, h {x3[2]:.1f}, rms {rms3:.4f}"
    )
print("k      sigma      g        rms     sigma/(k+0.141)   sigma/sigma(1)   linear-law sigma  linear-law g")
for k in sorted(res):
    s, g, rms, n = res[k]
    print(
        f"{k:5.2f}  {s:8.4f}  {g:8.3f}  {rms:.4f}  {s / (k + 0.140737):8.4f}        {s / res[1.0][0]:7.4f}        {2.616046 * (k + 0.140737):8.4f}      {-10.1349 - 2.4524 * k:8.3f}"
    )
print("READING q1295 (exploratory, no verdict)")
