"""q1355a (8 Oct 2026, EXPLORATORY reading before the registration of q1355; amendment 2122): a global fit R / Da = sum_{n=0}^{2} P_n(k) / v^(2n) with polynomials P_n of degree d_n in k, on the degree-17 hard-wall events of q1352_events.json (exponents 0.02, 0.05, 0.052, 0.054, 0.056, 0.058, 0.06, 0.075, 0.09, 0.1, 0.25, 0.35, 0.5, 0.75; only R > 0 events), by linear least squares on the relative misfit. Leave-one-exponent-out cross-validation on k = 0.1, 0.25, 0.35, 0.5: the held-out exponent is predicted from the others; printed: the largest relative misfit and the largest absolute misfit in R / Da over the held-out events, per model (d0, d1, d2). The exponents of the planned blind test (0.15, 0.4, 0.6) are not in the data.
Control (first): synthetic recovery of a known global model (P0 = 1.9 + 2 k + 0.7 k^2, P1 = -160 - 70 k, P2 = 800 + 3000 k) from noiseless data at the real v^2 to 1e-7.
Usage: py tight_run.py q1355a_global_fit_cv_reading.py  (from temporal_within/centre_scripts/)"""

import io, contextlib, json, os, sys, itertools

os.environ["E_FILE"] = "d22_out_from_log.json"
import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    import q1321_hardwall_common_c1_blind as B
import functools

Da = functools.lru_cache(maxsize=None)(B.Da)  # Da(v) is expensive (Borel-Pade ray): cache by value
hw = {float(k): v for k, v in json.load(open("q1352_events.json"))["hard_wall_deg17"].items()}


def design(ev, degs):
    rows, rhs, meta = [], [], []
    for k, evs in ev.items():
        for K, R, v in evs:
            y = R / Da(v)
            x = 1.0 / v
            row = []
            for n, d in enumerate(degs):
                for j in range(d + 1):
                    row.append(k**j * x**n / y)
            rows.append(row)
            rhs.append(1.0)
            meta.append((k, K, v, y))
    return np.array(rows), np.array(rhs), meta


def fit(ev, degs):
    A, b, meta = design(ev, degs)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return c


def predict(c, degs, k, v):
    out = 0.0
    p = 0
    for n, d in enumerate(degs):
        out += sum(c[p + j] * k**j for j in range(d + 1)) / v**n
        p += d + 1
    return out


# control
syn = {
    k: [
        (K, (1.9 + 2 * k + 0.7 * k * k + (-160 - 70 * k) / v + (800 + 3000 * k) / v**2) * Da(v), v)
        for K, R, v in e
    ]
    for k, e in hw.items()
}
cs = fit(syn, (2, 1, 1))
ctl = bool(np.allclose(cs, [1.9, 2, 0.7, -160, -70, 800, 3000], rtol=1e-7, atol=1e-7))
print(f"control (synthetic global recovery): {ctl}")
if not ctl:
    print("READING q1355a: VOID (control)")
    sys.exit(2)
ev = {k: [e for e in evs if e[1] > 0] for k, evs in hw.items() if sum(1 for e in evs if e[1] > 0) >= 1}
print("exponents and events used:", {k: len(v) for k, v in ev.items()})
print(
    "model (d0, d1, d2): all-data rms of relative misfit | leave-one-out over k = 0.1, 0.25, 0.35, 0.5: largest relative misfit, largest |dR/Da|"
)
best = []
for degs in itertools.product((1, 2, 3, 4), (1, 2, 3), (0, 1, 2)):
    c = fit(ev, degs)
    A, b, meta = design(ev, degs)
    rms = float(np.sqrt(np.mean((A @ c - 1) ** 2)))
    worst_rel = 0.0
    worst_abs = 0.0
    for kh in (0.1, 0.25, 0.35, 0.5):
        tr = {k: v for k, v in ev.items() if k != kh}
        ch = fit(tr, degs)
        for K, R, v in ev[kh]:
            y = R / Da(v)
            pr = predict(ch, degs, kh, v)
            worst_rel = max(worst_rel, abs(pr / y - 1))
            worst_abs = max(worst_abs, abs(pr - y))
    best.append((worst_abs, degs, rms, worst_rel))
best.sort()
for wa, degs, rms, wr in best[:12]:
    print(f"  {degs}: rms {rms:.5f} | largest relative {wr:.4f}, largest absolute {wa:.4f}")
print("READING q1355a (exploratory, no verdict)")
