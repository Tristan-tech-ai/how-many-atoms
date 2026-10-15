"""q1299 (8 Oct 2026, EXPLORATORY reading, no item, no verdict): does a second term beyond sigma Da (1 + g / v^2) describe the capacity residual (quintic spline, stored amplitudes of the c60 families, K_b 20 to 45 except 38 and 39, the 24 events of q1290)?
Models for the relative misfit (R_model / R - 1) by least squares:
 C1: R = sigma Da (1 + g / v^2)                       (2 parameters)
 C2: R = sigma Da (1 + g / v^2) + tau Da^2            (3)   second-order Stokes-type term
 C3: R = sigma Da (1 + g / v^2) + tau Da^2 + chi Da^3 (4)
 C4: R = sigma Da (1 + g / v^2 + h / v^4)             (3)   the form of q1295 with an extra series term, for comparison at equal parameter count
Printed: parameters, rms of the relative misfit, the misfit per event for C2 and C4; also the values at k = 0 of the closure law M2 of amendment 2076 (sigma(0) = A k0, g(0) = a0) for comparison with the fitted sigma and g of C2.
Control (first): Da(115) = 0.21082e-9 within 1e-4 relative; the capacity residual at K_b 36 is 0.5456e-9 within 0.001e-9 (q1290); synthetic recovery of C2 and C4 from noiseless data at the real v^2 of the 22 events to 1e-6 relative; C2 on synthetic data with tau = 0 returns tau = 0 within 1e-8 (the term is not forced).
Usage: py q1299_capacity_second_term_reading.py  (from temporal_within/centre_scripts/)"""

import io, contextlib, json, sys
import numpy as np
import mpmath as mp
from scipy.optimize import least_squares

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import q1250_quintic_residual as Q
    import q1271_linear_in_k_blind as Z
    import q1220_borel_pade_births_law as T
mp.mp.dps = 30
Bf = T.machine(T.coef_series(), 8, 8)[0]


def Da(v2):
    x = mp.mpf(1) / v2
    return abs(float(mp.im(T.ray(Bf, x, 0.3)))) / (2 * np.sqrt(v2) / (4 * np.pi)) * 1e9


C, m = Z.C, Z.m
capN = Q.make(lambda a: C.Q.N_of(a, *C.CAPL), C.X, 5)
gv = lambda f: (lambda j: j.get("a_t") or j.get("A_t"))(json.load(open(f + ".json")))
rows = []
for K, f in sorted(m.CAP_FILES.items()):
    if 20 <= K <= 45 and K not in (38, 39):
        a = gv(f)
        r = (a - Q.root(capN, K)) * 1e9
        v2 = Z.v2f(capN, a)
        rows.append((K, r, v2))
K_ = np.array([x[0] for x in rows])
R_ = np.array([x[1] for x in rows])
V_ = np.array([x[2] for x in rows])
D_ = np.array([Da(v) for v in V_])


def mod(name, p, d, v):
    if name == "C1":
        return p[0] * d * (1 + p[1] / v)
    if name == "C2":
        return p[0] * d * (1 + p[1] / v) + p[2] * d**2
    if name == "C3":
        return p[0] * d * (1 + p[1] / v) + p[2] * d**2 + p[3] * d**3
    if name == "C4":
        return p[0] * d * (1 + p[1] / v + p[2] / v**2)


P0 = {"C1": [0.5, -10.0], "C2": [0.5, -10.0, -1e-3], "C3": [0.5, -10.0, -1e-3, 0.0], "C4": [0.5, -10.0, 50.0]}


def fit(name, R, v, d, starts=None):
    best = None
    for s in starts or [P0[name], [1.0, -30.0] + P0[name][2:], [2.0, -60.0] + P0[name][2:]]:
        try:
            r = least_squares(
                lambda p: mod(name, p, d, v) / R - 1, s, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=4000
            )
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


ok0 = abs(Da(115.0) - 0.21082) < 1e-4 * 0.21082 and abs(R_[K_ == 36][0] - 0.5456) < 0.001
pt = {"C2": [0.5, -12.0, -5e-4], "C4": [0.5, -12.0, 60.0]}
oks = True
for nm, p in pt.items():
    ysyn = mod(nm, p, D_, V_)
    r = fit(nm, ysyn, V_, D_, [[0.4, -8.0] + P0[nm][2:]])
    oks = oks and bool(np.all(np.abs(r.x / np.array(p) - 1) < 1e-6))
rz = fit("C2", mod("C2", [0.5, -12.0, 0.0], D_, V_), V_, D_, [[0.4, -8.0, -1e-4]])
oks = oks and abs(rz.x[2]) < 1e-8
print(f"events {len(K_)}; control: Da(115) and R(36) {ok0}; synthetic recovery and no forced term {oks}")
if not (ok0 and oks):
    print("READING q1299: VOID (control)")
    sys.exit(2)
res = {}
for nm in ("C1", "C2", "C3", "C4"):
    r = fit(nm, R_, V_, D_)
    rms = float(np.sqrt(np.mean(r.fun**2)))
    res[nm] = r
    print(
        f"{nm}: parameters {np.array2string(r.x, precision=5)}; rms of the relative misfit {rms:.4f}; largest {np.abs(r.fun).max():.4f}"
    )
print("K_b   v^2     R(1e-9)   C2 misfit   C4 misfit")
for i in range(len(K_)):
    print(f"{K_[i]:3d} {V_[i]:7.2f} {R_[i]:10.4f} {res['C2'].fun[i]:+10.4f} {res['C4'].fun[i]:+10.4f}")
A, k0, a0 = 2.60436356, 0.17144331, -11.10486513
print(f"closure law M2 at k = 0: sigma {A * k0:.4f}, g {a0:.3f}")
print("READING q1299 (exploratory, no verdict)")
