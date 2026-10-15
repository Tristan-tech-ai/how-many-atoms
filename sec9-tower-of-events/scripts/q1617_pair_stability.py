"""q1617 (10 Oct 2026, READING, no verdict word; deep proof Q11 step 71): is the edge pair of the interior of the exact blocks stable under the block size and the fit window once the bulk law runs through c24?
Readings 107 to 109: the pair of the exact block (fit window k = 30..60, bulk law through c20) is (-0.093773648256, +0.001159913641) for N = 90. q1609b/q1612 found that the pair fitted on k = 60..90 of the N = 120 block differs by 1.3e-09 in d0 from the one on k = 30..60 and that the three windows 30..60, 45..75 and 60..90 spread by 3.2e-09, because the smooth rest leaks into the pair. Reading 108: the law through c24 reduces the rest by a factor of 3 to 6.
This script takes the stored clean solutions q1601_z_N60.json, q1601_z_N90.json and q1601_z_N120.json (formal pair, solved with law atoms on the c20 law beyond the block), forms the deviations from the formal law through c20 and through c24 (float depths as in q1613), fits the pair change delta d0 + delta nu0 s_k on several windows, and compares.
Predictions written before the run:
 P1 with the law through c24 the spread of delta d0 over the windows 30..60, 45..75 and 60..90 of the N = 120 block is below 1e-09 (it was 3.2e-09 with c20);
 P2 with the law through c24 the pair fitted on the window 20..40 agrees between the blocks N = 60, 90 and 120 to 1e-09 in d0 and in nu0 (largest difference over the three blocks);
 P3 with the law through c24 the pair fitted on 30..60 agrees between N = 90 and N = 120 to 3e-09 in d0 and in nu0;
 P4 with the law through c24 the spread of the combination d0 + 0.95 nu0 over the three windows of the N = 120 block is below 5e-10.
Usage: py q1617_pair_stability.py (about 1 minute)."""

import os, sys, json, importlib.util
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
spec = importlib.util.spec_from_file_location("q1613m", os.path.join(here, "q1613_rest_vs_next_orders.py"))
src = open(os.path.join(here, "q1613_rest_vs_next_orders.py"), encoding="utf-8").read()
src = src[: src.index("LIM = ")]
ns = {"__file__": os.path.join(here, "q1613_rest_vs_next_orders.py"), "__name__": "q1613ns"}
import io, contextlib

with contextlib.redirect_stdout(io.StringIO()):
    exec(src, ns)
dep, farb, NU, D0 = ns["dep"], ns["farb"], ns["NU"], ns["D0"]
pairfit = ns["pairfit"]
from flint import arb


def deltas(N):
    z = [arb(s) for s in json.load(open(os.path.join(here, "q1601_z_N%d.json" % N)))]
    d = np.array([float(z[k]) for k in range(N)])
    s = np.array([(farb[k + 1] - farb[k - 1]) / 2 if k > 0 else farb[1] - farb[0] for k in range(N)])
    return d - dep[10][:N], d - dep[12][:N], s


res = {}
for N in (60, 90, 120):
    d20, d24, s = deltas(N)
    for lo, hi in ((20, 40), (30, 60), (45, 75), (60, 90)):
        if hi > N - 15:
            continue
        for name, dl in (("c20", d20), ("c24", d24)):
            c, r = pairfit(dl, s, lo, hi)
            res[(N, lo, hi, name)] = (c[0], c[1], float(np.sqrt((r**2).mean())))
for key in sorted(res):
    N, lo, hi, name = key
    a, b, rms = res[key]
    print(
        "N = %3d window %2d..%2d law %s: delta d0 %+.5e, delta nu0 %+.5e, combination d0 + 0.95 nu0 %+.5e, rest rms %.2e"
        % (N, lo, hi, name, a, b, a + 0.95 * b, rms)
    )
w120 = [(30, 60), (45, 75), (60, 90)]
sp = lambda name, f: max(f(res[(120, lo, hi, name)]) for lo, hi in w120) - min(
    f(res[(120, lo, hi, name)]) for lo, hi in w120
)
d0s20, d0s24 = sp("c20", lambda t: t[0]), sp("c24", lambda t: t[0])
cs24 = sp("c24", lambda t: t[0] + 0.95 * t[1])
print(
    "N = 120 spread over the windows 30..60, 45..75, 60..90: delta d0 %.2e (c20), %.2e (c24); combination %.2e (c24)"
    % (d0s20, d0s24, cs24)
)
a20 = [res[(N, 20, 40, "c24")] for N in (60, 90, 120)]
p2d = max(t[0] for t in a20) - min(t[0] for t in a20)
p2n = max(t[1] for t in a20) - min(t[1] for t in a20)
print(
    "window 20..40, law c24: largest difference between N = 60, 90, 120: delta d0 %.2e, delta nu0 %.2e"
    % (p2d, p2n)
)
t90, t120 = res[(90, 30, 60, "c24")], res[(120, 30, 60, "c24")]
p3d, p3n = abs(t90[0] - t120[0]), abs(t90[1] - t120[1])
print("window 30..60, law c24: N = 90 minus N = 120: delta d0 %.2e, delta nu0 %.2e" % (p3d, p3n))
lab = lambda p: "holds" if p else "does not hold"
print(
    "P1 c24 spread of delta d0 below 1e-09: %s; P2 window 20..40 agreement below 1e-09: %s; P3 N = 90 against 120 below 3e-09: %s; P4 combination spread below 5e-10: %s"
    % (lab(d0s24 < 1e-9), lab(p2d < 1e-9 and p2n < 1e-9), lab(p3d < 3e-9 and p3n < 3e-9), lab(cs24 < 5e-10))
)
print("READING q1617 done (labels by the script; no verdict word)")
