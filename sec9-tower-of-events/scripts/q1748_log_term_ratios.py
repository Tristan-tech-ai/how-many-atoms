"""q1748 (10 Oct 2026, READING, no verdict word): is there a logarithm in the large-order law of the centre series? Fit r_n = alpha n + beta + gamma ln n + delta / n (model L) and r_n = alpha n + beta + delta / n (model 0) to the stored ratios r_n = e_n / e_(n-1) (centre_scripts/d22_out_20.json, mpmath 40 digits) on n = 10..17, and predict r_18, r_19, r_20.
NOT BLIND: the ratios, their increments and second differences were seen in readings 197 and 199; these fits were not run before the predictions were written. Predictions (written at 13:34):
 P1 model L fitted on n = 10..17 predicts r_18, r_19 and r_20 to within 0.01 absolute;
 P2 the fitted alpha of model L lies in 4.3..4.9;
 P3 model 0 predicts r_18, r_19, r_20 worse than model L by at least a factor 3 (compared by the largest absolute error).
Controls (otherwise VOID): C1 synthetic ratios 4.79 n + 5 + 0.3 ln n + 2/n: model L recovers them (predictions of n = 18..20 to 1e-8); C2 synthetic ratios without a log term (4.79 n + 5 + 2/n): model L finds gamma = 0 within 1e-6.
Sensitivity (printed, no label): the same fits on n = 11..18 predicting r_19, r_20, and model Q = alpha n + beta + delta / n + eps / n^2 on n = 10..17.
Usage: py q1748_log_term_ratios.py (from small_gap_q206/; seconds)."""

import json, os, sys
import numpy as np
import mpmath as mp

mp.mp.dps = 40
H = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(H, "..", "centre_scripts", "d22_out_20.json")))
E = {n: mp.mpf(S["e%d" % n]) for n in range(2, 21)}
r = {n: float(E[n] / E[n - 1]) for n in range(3, 21)}


def basis(model, n):
    n = float(n)
    return {"L": [n, 1.0, np.log(n), 1.0 / n], "0": [n, 1.0, 1.0 / n], "Q": [n, 1.0, 1.0 / n, 1.0 / n**2]}[
        model
    ]


def fit(model, rr, ns):
    X = np.array([basis(model, n) for n in ns])
    y = np.array([rr[n] for n in ns])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c


def pred(model, c, n):
    return float(np.dot(basis(model, n), c))


syn1 = {n: 4.79 * n + 5 + 0.3 * np.log(n) + 2.0 / n for n in range(3, 21)}
syn2 = {n: 4.79 * n + 5 + 2.0 / n for n in range(3, 21)}
W = list(range(10, 18))
cs1 = fit("L", syn1, W)
c1 = max(abs(pred("L", cs1, n) - syn1[n]) for n in (18, 19, 20))
cs2 = fit("L", syn2, W)
print(
    "C1 synthetic with log: largest prediction error %.2e; C2 synthetic without log: fitted gamma %.2e"
    % (c1, cs2[2])
)
if not (c1 < 1e-8 and abs(cs2[2]) < 1e-6):
    print("VOID: control failed")
    sys.exit(2)
out = {}
for model in ("L", "0", "Q"):
    c = fit(model, r, W)
    errs = {n: pred(model, c, n) - r[n] for n in (18, 19, 20)}
    out[model] = (c, errs)
    print(
        "model %s on n = 10..17: coefficients %s; errors at n = 18, 19, 20: %s; largest %.3e"
        % (
            model,
            np.array2string(c, precision=5),
            " ".join("%+.3e" % errs[n] for n in (18, 19, 20)),
            max(abs(v) for v in errs.values()),
        )
    )
W2 = list(range(11, 19))
for model in ("L", "0"):
    c = fit(model, r, W2)
    errs = {n: pred(model, c, n) - r[n] for n in (19, 20)}
    print(
        "sensitivity model %s on n = 11..18: errors at n = 19, 20: %s"
        % (model, " ".join("%+.3e" % errs[n] for n in (19, 20)))
    )
lab = lambda q: "holds" if q else "does not hold"
eL = max(abs(v) for v in out["L"][1].values())
e0 = max(abs(v) for v in out["0"][1].values())
al = out["L"][0][0]
print(
    "P1 model L predicts r_18..r_20 within 0.01: %s (largest %.3e); P2 alpha of model L in 4.3..4.9: %s (alpha = %.4f); P3 model 0 worse by at least a factor 3: %s (ratio %.3g)"
    % (lab(eL < 0.01), eL, lab(4.3 <= al <= 4.9), al, lab(e0 >= 3 * eL), e0 / eL)
)
print("READING q1748 done (labels by the script; no verdict word)")
