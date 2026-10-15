"""q1297 (8 Oct 2026, EXPLORATORY reading, no item, no verdict): joint fits over all ten exponents k = 0.75 to 12 (events K_b 14-40, K_b 38 excluded; the events of q1296) of ln(R / Da) = ln sigma(k) + ln F(k, v^2) with sigma(k) = A (k + k0) and
 M1: F = 1 + (a0 + a1 k) / v^2                                   (4 parameters)
 M2: F = 1 + (a0 + a1 k + a2 k^2) / v^2                          (5)
 M3: F = 1 + (a0 + a1 k) / v^2 + (b0 + b1 k + b2 k^2) / v^4      (7)
 M4: F = 1 + (a0 + a1 k) / v^2 + b2 k^2 / v^4                    (6)
 M5: F = exp((a0 + a1 k) / v^2)                                  (4)
 M6: F = 1 / (1 - (a0 + a1 k) / v^2)                             (4)
Printed: rms of ln R over all events, the per-k rms, and for each model the same fit on k <= 5 only with the rms of the events k = 0.75, 8, 12 predicted out of that fit (these events are seen; this only compares the forms and chooses nothing).
Control (first): the form functions equal the q1296 forms at a fixed (k, v^2) (here v is the density v^2, a first version squared it again and the synthetic control could not see it); synthetic recovery of M3 (7 parameters) from noiseless data at the real (k, v^2) grid to 1e-5 relative; the rms of M1 on k = 1..5 within 3 times the mean per-k rms of q1296 F1 (0.0042 over all k; here the number is printed and compared with 0.012).
Usage: py q1297_joint_closure_models.py  (from temporal_within/centre_scripts/)"""

import io, contextlib, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
from scipy.optimize import least_squares

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import q1296_closure_forms_all_k as C
data, Da = C.data, C.Da
ev = [(k, K, r, v) for k in sorted(data) for K, r, v in data[k]]
kk = np.array([e[0] for e in ev])
vv = np.array([e[3] for e in ev])
dav = np.array([Da(x) for x in vv])
yy = np.log(np.array([e[2] for e in ev]) / dav)


def lnF(m, p, k, v):
    x = 1.0 / v
    if m == "M1":
        return np.log(1 + (p[2] + p[3] * k) * x)
    if m == "M2":
        return np.log(1 + (p[2] + p[3] * k + p[4] * k**2) * x)
    if m == "M3":
        return np.log(1 + (p[2] + p[3] * k) * x + (p[4] + p[5] * k + p[6] * k**2) * x**2)
    if m == "M4":
        return np.log(1 + (p[2] + p[3] * k) * x + p[4] * k**2 * x**2)
    if m == "M5":
        return (p[2] + p[3] * k) * x
    if m == "M6":
        return -np.log(1 - (p[2] + p[3] * k) * x)


NP = {"M1": 4, "M2": 5, "M3": 7, "M4": 5, "M5": 4, "M6": 4}


def model(m, p, k, v):
    return np.log(p[0] * (k + p[1])) + lnF(m, p, k, v)


def fitm(m, mask, p0=None):
    best = None
    starts = (
        [p0]
        if p0 is not None
        else [[2.6, 0.14, -10.0, -2.0] + [0.0] * (NP[m] - 4), [2.6, 0.14, -15.0, -1.0] + [50.0] * (NP[m] - 4)]
    )
    for s in starts:
        try:
            r = least_squares(
                lambda p: model(m, p, kk[mask], vv[mask]) - yy[mask],
                s,
                xtol=1e-14,
                ftol=1e-14,
                gtol=1e-14,
                max_nfev=4000,
            )
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


# control: synthetic M3 recovery
ptrue = np.array([2.6, 0.14, -12.0, -2.0, 300.0, 40.0, 10.0])
ysyn = model("M3", ptrue, kk, vv)
rs = least_squares(
    lambda p: model("M3", p, kk, vv) - ysyn,
    [2.5, 0.2, -10.0, -1.0, 100.0, 10.0, 5.0],
    xtol=1e-14,
    ftol=1e-14,
    gtol=1e-14,
    max_nfev=20000,
)
ok_syn = bool(np.all(np.abs(rs.x / ptrue - 1) < 1e-5))
allm = np.ones(len(kk), bool)
lowm = kk <= 5.0
r1 = fitm("M1", lowm)
rms1 = float(np.sqrt(np.mean(r1.fun**2)))
ok_m1 = rms1 < 0.012
pf = [0, 0, -15.019, 0.0]
ok_form = (
    abs(lnF("M1", pf, 2.0, 90.0) - np.log(1 - 15.019 / 90.0)) < 1e-12
    and abs(lnF("M5", pf, 2.0, 90.0) + 15.019 / 90.0) < 1e-12
    and abs(lnF("M6", pf, 2.0, 90.0) + np.log(1 + 15.019 / 90.0)) < 1e-12
    and abs(
        lnF("M3", [0, 0, -15.019, 0.0, 300.0, 0.0, 0.0], 2.0, 90.0)
        - np.log(1 - 15.019 / 90.0 + 300.0 / 90.0**2)
    )
    < 1e-12
)
print(
    f"events {len(kk)}; control: forms use 1/v^2 with v^2 the density, as in q1296 {ok_form}; synthetic M3 recovery {ok_syn} (max relative error {np.abs(rs.x / ptrue - 1).max():.2e}); M1 on k <= 5 rms {rms1:.4f} below 0.012: {ok_m1}"
)
if not (ok_syn and ok_m1 and ok_form):
    print("READING q1297: VOID (control)")
    sys.exit(2)
for m in NP:
    r = fitm(m, allm)
    f = r.fun
    rms = float(np.sqrt(np.mean(f**2)))
    perk = ", ".join(f"{k:g}: {np.sqrt(np.mean(f[kk == k] ** 2)):.4f}" for k in sorted(set(kk)))
    rl = fitm(m, lowm)
    pred = model(m, rl.x, kk[~lowm], vv[~lowm]) - yy[~lowm]
    out = {
        k: (float(np.sqrt(np.mean(pred[kk[~lowm] == k] ** 2))), float(np.mean(pred[kk[~lowm] == k])))
        for k in (8.0, 12.0)
    }
    pk = {k: float(np.mean(model(m, rl.x, kk[kk == k], vv[kk == k]) - yy[kk == k])) for k in (0.75,)}
    print(
        f"{m} ({NP[m]} parameters): all-k rms {rms:.4f}; parameters {np.array2string(r.x, precision=8, suppress_small=True)}\n   per-k rms: {perk}\n   fit on k <= 5 (rms {np.sqrt(np.mean(rl.fun ** 2)):.4f}): predicted k = 0.75 mean offset of ln R {pk[0.75]:+.4f}; k = 8: rms {out[8.0][0]:.4f} mean {out[8.0][1]:+.4f}; k = 12: rms {out[12.0][0]:.4f} mean {out[12.0][1]:+.4f}"
    )
print("READING q1297 (exploratory, no verdict)")
