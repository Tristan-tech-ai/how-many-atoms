"""q1618 (10 Oct 2026, READING, no verdict word; deep proof Q11 step 72): out-of-sample test of the births law with the pair of the exact wall block.
Reading 109 fitted E_4 and E_5 on the 48 measured births K_b 83..130 with the pair fixed at the c20 wall pair (-0.093773648256, +0.001159913641) of q1610: E_4 = -4.251e+05, E_5 = -6.538e+07 (q1616), in-sample rms 2.23e-11, largest 4.92e-11. The one-term fit is E_4 = -8.006e+05 (q1614). The offset without E_4 and E_5 is one-signed and follows c K^-3.49 (q1615), +4.529e-09 at K_b 130.
This script FREEZES that model (no refit) and compares it with the capacity events K_b >= 131 that exist in small_gap_q206/fw_prod. These are the blind events of the registered block B131 (gate q1527, INCOMPLETE). The model was fixed before any amplitude of those events was looked at; the amplitude A_event is loaded inside the script and only the difference predicted minus measured is printed, never an amplitude. The gate q1527 is not touched and this reading is not a verdict.
Predictions written before the run (the events present are expected to be K_b 131, 132, 133, 134, 137; a different set makes the labels void):
 P1 with the frozen two-term model (wall pair, E_4 and E_5 above) all differences predicted minus measured are below 3e-10 in absolute value;
 P2 with the wall pair and E_4 = E_5 = 0 all differences are positive and between 2e-09 and 8e-09 (the K^-3.49 power law continued from +4.5e-09 at K_b 130);
 P3 the two-term model is closer to the measured amplitude than the pair-only Aitken limit pair of q1564 (-0.093773592, +0.001159849) with E = 0, at every event;
 P4 the spread (largest minus smallest) of the differences of the two-term model is below 2e-10.
Usage: py q1618_births_out_of_sample.py (about 1 minute)."""

import io, contextlib, os, json
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "q1616_E4_E5.py"), encoding="utf-8").read()
src = src[: src.index("out = {}")]
ns = {"__file__": os.path.join(H, "q1616_E4_E5.py"), "__name__": "q1616ns"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(src, ns)
a_pred = ns["a_pred"]
WALL = (-0.093773648256, 0.001159913641)
LIMP = (-0.093773592, 0.001159849)
E45 = [-4.251e05, -6.538e07]
E4 = [-8.006e05]
have = {}
for K in range(131, 151):
    f = os.path.join(H, "fw_prod", f"event_Kb{K}.json")
    if os.path.exists(f):
        have[K] = float(json.load(open(f))["A_event"])
Kl = sorted(have)
expected = [131, 132, 133, 134, 137]
print("events K_b >= 131 present in fw_prod:", Kl, "; expected set:", expected)


def diff(pair, extra):
    return np.array([a_pred(K, pair[0], pair[1], extra) - have[K] for K in Kl])


d_two = diff(WALL, E45)
d_one = diff(WALL, E4)
d_zero = diff(WALL, [])
d_lim = diff(LIMP, [])
fmt = lambda a: " ".join("%+.2e" % x for x in a)
print("K_b:                                  " + " ".join("%9d" % K for K in Kl))
print("wall pair, E_4 = E_5 = 0:             " + fmt(d_zero))
print("wall pair, E_4 alone (-8.006e+05):    " + fmt(d_one))
print("wall pair, E_4 and E_5 (frozen):      " + fmt(d_two))
print("Aitken limit pair, E = 0:             " + fmt(d_lim))
print(
    "largest |difference|: two-term %.2e, one-term %.2e, zero-term %.2e, limit pair %.2e; spread of the two-term differences %.2e"
    % (
        np.abs(d_two).max(),
        np.abs(d_one).max(),
        np.abs(d_zero).max(),
        np.abs(d_lim).max(),
        d_two.max() - d_two.min(),
    )
)
lab = lambda p: "holds" if p else "does not hold"
same = Kl == expected
if not same:
    print("SET DIFFERS from the expected set: labels are void")
p1 = np.abs(d_two).max() < 3e-10
p2 = bool((d_zero > 0).all() and (d_zero >= 2e-9).all() and (d_zero <= 8e-9).all())
p3 = bool((np.abs(d_two) < np.abs(d_lim)).all())
p4 = (d_two.max() - d_two.min()) < 2e-10
if same:
    print(
        "P1 two-term differences below 3e-10: %s; P2 zero-term differences positive in [2e-09, 8e-09]: %s; P3 two-term closer than the limit pair at every event: %s; P4 spread below 2e-10: %s"
        % (lab(p1), lab(p2), lab(p3), lab(p4))
    )
print("READING q1618 done (labels by the script; no verdict word)")
