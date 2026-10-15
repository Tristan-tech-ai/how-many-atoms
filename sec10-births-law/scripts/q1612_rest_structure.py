"""q1612 (10 Oct 2026, READING, no verdict word; deep proof Q11 step 66): the structure of the rest of 5.5e-11 that remains in the interior of the exact blocks after the edge pair is removed.
q1610 fitted a change of the pair on k = 30..60 of the clean block N = 90 and left a rest with rms 5.46e-11 and largest value 1.65e-10 that does not shrink when the pair is iterated. q1609b found the same size (6.0e-11) for N = 120 on k = 30..70. This script takes the stored clean solutions q1601_z_N90.json and q1601_z_N120.json (formal pair), forms the depth deviations from the formal law, fits delta_k = a + b s_k on k = 30..60 (s_k = central spacing of the formal depths), and studies the rest r_k.
Hypotheses with cheap tests (three candidates, rule 3): (H1) the rest is a property of the bulk law and not of the frozen boundary of the block; (H2) the rest is the first-harmonic ripple, which alternates in sign from atom to atom; (H3) the rest is a smooth function of k, the remainder of the bulk series beyond the order used.
Predictions written before the run:
 P1 the rests of N = 90 and N = 120 on k = 30..60 have a correlation coefficient above 0.9 and an rms ratio between 0.8 and 1.25 (H1);
 P2 the fraction of consecutive atoms k, k+1 in 30..59 with r_k r_(k+1) < 0 is below 0.3 for both blocks (against H2);
 P3 after a fit of r_k by a polynomial of degree 3 in k the rms of the remainder is below 0.2 of the rms of r_k for both blocks (H3);
 P4 for N = 120 the pair fitted on k = 60..90 differs from the pair fitted on k = 30..60 by more than 1e-9 in d0 (the smooth rest leaks into the pair, as the 1.8e-9 spread of q1609b suggests).
Usage: py q1612_rest_structure.py (about 1 minute)."""

import os, sys, json, importlib.util
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
spec = importlib.util.spec_from_file_location("q1598m", os.path.join(here, "q1598_krawczyk_free90_120.py"))
q = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(q)
from flint import arb


def load(N):
    z = [arb(s) for s in json.load(open(os.path.join(here, "q1601_z_N%d.json" % N)))]
    dt, wt = q.formal_tail(0, N + 1)
    d = np.array([float(z[k]) for k in range(N)])
    f = np.array([float(t) for t in dt])
    s = np.array([(f[k + 1] - f[k - 1]) / 2 if k > 0 else f[1] - f[0] for k in range(N)])
    return d - f[:N], s


def pairfit(delta, s, lo, hi):
    idx = np.arange(lo - 1, hi)
    A = np.vstack([np.ones(len(idx)), s[idx]]).T
    coef, *_ = np.linalg.lstsq(A, delta[idx], rcond=None)
    return coef, delta[idx] - A @ coef, idx + 1


rest = {}
for N in (90, 120):
    delta, s = load(N)
    coef, res, ks = pairfit(delta, s, 30, 60)
    rest[N] = res
    rms = float(np.sqrt((res**2).mean()))
    alt = float(np.mean(res[:-1] * res[1:] < 0))
    kk = ks.astype(float)
    pc = np.polyfit((kk - 45) / 15, res, 3)
    rem = res - np.polyval(pc, (kk - 45) / 15)
    remrms = float(np.sqrt((rem**2).mean()))
    print(
        "N = %d: pair change on 30..60: delta d0 %+.4e, delta nu0 %+.4e; rest rms %.3e, largest %.3e; sign changes between consecutive atoms %.3f; cubic remainder rms %.3e (ratio %.3f)"
        % (N, coef[0], coef[1], rms, np.abs(res).max(), alt, remrms, remrms / rms)
    )
    print(
        "   rest r_k in units of 1e-11 at k = 30, 33, ..., 60:",
        " ".join("%+.1f" % (res[i] * 1e11) for i in range(0, 31, 3)),
    )
    if N == 120:
        c2, r2, _ = pairfit(delta, s, 60, 90)
        c3, r3, _ = pairfit(delta, s, 45, 75)
        print(
            "   N = 120 pair fitted on 60..90: delta d0 %+.4e, delta nu0 %+.4e, rest rms %.3e; on 45..75: delta d0 %+.4e, delta nu0 %+.4e; spread of delta d0 over the three windows %.2e"
            % (
                c2[0],
                c2[1],
                float(np.sqrt((r2**2).mean())),
                c3[0],
                c3[1],
                max(coef[0], c2[0], c3[0]) - min(coef[0], c2[0], c3[0]),
            )
        )
        d0_30, d0_60 = coef[0], c2[0]
        rms60 = float(np.sqrt((r2**2).mean()))
        rms30 = rms
    if N == 90:
        alt90, ratio90 = alt, remrms / rms
    if N == 120:
        alt120, ratio120 = alt, remrms / rms
corr = float(np.corrcoef(rest[90], rest[120])[0, 1])
rr = float(np.sqrt((rest[120] ** 2).mean()) / np.sqrt((rest[90] ** 2).mean()))
print(
    "correlation of the rests of N = 90 and N = 120 on k = 30..60: %.4f; rms ratio (120 over 90) %.3f"
    % (corr, rr)
)
lab = lambda p: "holds" if p else "does not hold"
p1 = corr > 0.9 and 0.8 <= rr <= 1.25
p2 = alt90 < 0.3 and alt120 < 0.3
p3 = ratio90 < 0.2 and ratio120 < 0.2
p4 = abs(d0_60 - d0_30) > 1e-9
print(
    "P1 correlation above 0.9 and rms ratio in [0.8, 1.25]: %s; P2 sign changes below 0.3: %s; P3 cubic remainder below 0.2 of the rest: %s; P4 pair on 60..90 differs by more than 1e-9 in d0: %s"
    % (lab(p1), lab(p2), lab(p3), lab(p4))
)
print("READING q1612 done (labels by the script; no verdict word)")
