"""q1296 (8 Oct 2026, EXPLORATORY reading, no item, no verdict): which two-parameter correction form F(g / v^2) of the closure R_k = sigma_k Da(v^2) F fits all ten exponents k = 0.75 to 12 (events K_b 14-40, K_b 38 excluded, inclusion rule of q1293/q1295): F1 = 1 + g / v^2 (the form of amendment 2072), F2 = exp(g / v^2), F3 = 1 / (1 - g / v^2). Also the three-parameter reference of q1295. Output: per k sigma, g, rms of ln R for each form; then linear fits of sigma(k) and g(k) over all ten exponents for each form.
Control (first): synthetic recovery for each form from noiseless data at the real v^2; F1 reproduces the (sigma, g) of q1293 / q1295 for k = 2 and k = 12 within 1e-3.
Seen events are used for the choice among forms; any form chosen here needs its own blind test with parameters fixed in advance.
Usage: py q1296_closure_forms_all_k.py  (from temporal_within/centre_scripts/)"""

import io, contextlib, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
from scipy.optimize import least_squares

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import q1293_sigma_g_of_k_reading as A
    import q1295_closure_perk_fit_k075_8_12 as B
Da = A.Da
data = {k: list(v) for k, v in A.data.items()}
for k in (0.75, 8.0, 12.0):
    data[k] = list(B.data[k])
FORMS = {
    "F1 1+g/v2": lambda g, v: np.log(1 + g / v),
    "F2 exp(g/v2)": lambda g, v: g / v,
    "F3 1/(1-g/v2)": lambda g, v: -np.log(1 - g / v),
}


def fit(k, form):
    d = np.array(data[k])
    v = d[:, 2]
    y = np.log(d[:, 1] / np.array([Da(x) for x in v]))
    L = FORMS[form]
    best = None
    for g0 in (-5.0, -15.0, -30.0, -60.0):
        try:
            r = least_squares(lambda q: q[1] + L(q[0], v) - y, [g0, 1.0], xtol=1e-14, ftol=1e-14, gtol=1e-14)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return float(best.x[0]), float(np.exp(best.x[1])), float(np.sqrt(np.mean(best.fun**2)))


vs = np.array([69.2, 80, 90, 100, 110.0])
ok = True
for f, L in FORMS.items():
    yy = np.log(5.0) + L(-20.0, vs)
    r = least_squares(lambda q: q[1] + L(q[0], vs) - yy, [-10.0, 1.0], xtol=1e-14, ftol=1e-14, gtol=1e-14)
    ok = ok and abs(r.x[0] + 20) < 1e-7 and abs(np.exp(r.x[1]) - 5) < 1e-7
g2, s2, _ = fit(2.0, "F1 1+g/v2")
g12, s12, _ = fit(12.0, "F1 1+g/v2")
ok1 = (
    abs(g2 + 15.019) < 1e-3
    and abs(s2 - 5.5832) < 1e-3
    and abs(g12 + 28.381) < 1e-3
    and abs(s12 - 32.5716) < 1e-3
)
print(f"control: synthetic recovery of all forms {ok}; F1 reproduces q1293 / q1295 at k = 2 and 12 {ok1}")
if not (ok and ok1):
    print("READING q1296: VOID (control)")
    sys.exit(2)
ks = sorted(data)
out = {f: [] for f in FORMS}
print("k      " + "   ".join(f"{f:>26s}" for f in FORMS))
print("       " + "   ".join("  sigma      g     rms  " for f in FORMS))
for k in ks:
    cells = []
    for f in FORMS:
        g, s, rms = fit(k, f)
        out[f].append((k, s, g, rms))
        cells.append(f"{s:8.3f} {g:8.3f} {rms:.4f}")
    print(f"{k:5.2f}  " + "   ".join(cells))
print()
for f in FORMS:
    a = np.array(out[f])
    k, s, g, rms = a.T
    cs = np.polyfit(k, s, 1)
    cg = np.polyfit(k, g, 1)
    rs = s / np.polyval(cs, k) - 1
    rg = g - np.polyval(cg, k)
    print(
        f"{f}: mean rms ln R {rms.mean():.4f}, max {rms.max():.4f}; sigma(k) = {cs[0]:.4f} k + {cs[1]:.4f} (intercept at k = {-cs[1] / cs[0]:.3f}), largest relative residual {np.abs(rs).max():.4f}; g(k) = {cg[0]:.4f} k + {cg[1]:.4f}, largest residual {np.abs(rg).max():.3f}"
    )
print("READING q1296 (exploratory, no verdict)")
