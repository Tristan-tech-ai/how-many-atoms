"""q1740b (10 Oct 2026, READING, no verdict word): do the increments d_n = r_n - r_(n-1) of the ratios r_n = e_n / e_(n-1) of the stored centre series tend to 16 pi^2 / A, the value for the same Borel singularity as the bulk series (A = 32.957 gives 4.7915, A = 32.307 gives 4.888)? Hypothesis H_S: the centre series has the same singularity as the bulk series.
NOT BLIND: the increments 4.241 (n = 15) to 4.304 (n = 20) were seen in reading 197; the fits below were not run before the predictions were written. Stored e_2..e_20 (centre_scripts/d22_out_20.json), mpmath at 40 digits.
Predictions (written at 13:22):
 P1 the three-parameter fit d_n = d_inf - a/n + b/n^2 on n = 15..20 gives d_inf between 4.5 and 5.1;
 P2 the largest residual of that fit is below 1e-4;
 P3 the same fit on n = 14..19 gives d_inf within 0.1 of the fit on n = 15..20.
Controls (otherwise VOID): C1 a synthetic increment sequence d_n = 4.79 - 3/n + 5/n^2 is recovered by the same code on both windows to 1e-8; C2 the same sequence with an extra 40/n^3 term shows the bias of the three-parameter fit on n = 15..20 (printed, no bar): it is the honest size of the extrapolation uncertainty.
Usage: py q1740b_ratio_increments.py (from small_gap_q206/; seconds)."""

import json, os, sys
import numpy as np
import mpmath as mp

mp.mp.dps = 40
H = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(H, "..", "centre_scripts", "d22_out_20.json")))
E = {n: mp.mpf(S["e%d" % n]) for n in range(2, 21)}
r = {n: E[n] / E[n - 1] for n in range(3, 21)}
d = {n: float(r[n] - r[n - 1]) for n in range(4, 21)}


def fit(dd, ns, three=True):
    X = np.array([[1.0, -1.0 / n] + ([1.0 / n**2] if three else []) for n in ns])
    y = np.array([dd[n] for n in ns])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c, y - X @ c


syn = lambda n, extra=0.0: 4.79 - 3.0 / n + 5.0 / n**2 + extra / n**3
W1, W2 = list(range(15, 21)), list(range(14, 20))
s1 = {n: syn(n) for n in range(10, 21)}
c1 = max(abs(fit(s1, W1)[0][0] - 4.79), abs(fit(s1, W2)[0][0] - 4.79))
print("C1 synthetic d_inf recovered on both windows: largest error %.2e" % c1)
s2 = {n: syn(n, 40.0) for n in range(10, 21)}
print(
    "C2 synthetic with +40/n^3: three-parameter fit on n = 15..20 gives d_inf %.4f (true 4.79), on n = 14..19 %.4f"
    % (fit(s2, W1)[0][0], fit(s2, W2)[0][0])
)
if c1 > 1e-8:
    print("VOID: control C1 failed")
    sys.exit(2)
print("increments d_n (n = 14..20): " + " ".join("%d %.5f" % (n, d[n]) for n in range(14, 21)))
res = {}
for name, W in (("15..20", W1), ("14..19", W2)):
    for three in (True, False):
        c, rs = fit(d, W, three)
        res[(name, three)] = (c, rs)
        print(
            "window %s, %s: d_inf = %.4f, a = %.4f%s; largest residual %.2e"
            % (
                name,
                "three parameters" if three else "two parameters",
                c[0],
                c[1],
                (", b = %.4f" % c[2]) if three else "",
                abs(rs).max(),
            )
        )
print(
    "Stokes values 16 pi^2 / A: A = 32.957 gives %.4f, A = 32.307 gives %.4f"
    % (16 * np.pi**2 / 32.957, 16 * np.pi**2 / 32.307)
)
lab = lambda q: "holds" if q else "does not hold"
cA, rA = res[("15..20", True)]
cB, _ = res[("14..19", True)]
print(
    "P1 d_inf in [4.5, 5.1]: %s (d_inf = %.4f); P2 largest residual below 1e-4: %s (%.2e); P3 windows agree within 0.1: %s (difference %.4f)"
    % (
        lab(4.5 <= cA[0] <= 5.1),
        cA[0],
        lab(abs(rA).max() < 1e-4),
        abs(rA).max(),
        lab(abs(cA[0] - cB[0]) < 0.1),
        abs(cA[0] - cB[0]),
    )
)
print("READING q1740b done (labels by the script; no verdict word)")
