"""q1755 (10 Oct 2026, READING, no verdict word): evaluation of the registration q1754 (centre_scripts/q1754_prereg_d1_ratio.txt, band [12.72222, 12.96222] for s_10 = d_1(11)/d_1(10), where d_1(n) is the coefficient of pi^(2n-4) in e_n). Reads e_11 from centre_scripts/q1634_out_11.json and e_10 from q1634_out_10.json.
Label: holds when s_10 lies inside the band, else does not hold. Selftest (--selftest): 12.85 gives holds, 12.3 gives does not hold, the coefficient extraction on e_10 returns the stored d_1(10) (a nonzero rational), and a missing file makes the real run print VOID and exit with code 2.
Usage: py q1755_eval_d1_ratio_11.py [--selftest] (from small_gap_q206/; seconds)."""

import json, os, sys
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
x = sp.Symbol("x")
BAND = (12.72222, 12.96222)


def d1_of(s, n):
    ex = sp.expand(sp.sympify(s.replace("pi", "PI"), locals={"PI": x}))
    return sp.Rational(sp.Poly(ex, x).coeff_monomial(x ** (2 * n - 4)))


d10 = d1_of(json.load(open(os.path.join(H, "..", "centre_scripts", "q1634_out_10.json")))["e10"], 10)
inband = lambda v: BAND[0] <= float(v) <= BAND[1]
if "--selftest" in sys.argv:
    a = inband(12.85)
    b = not inband(12.3)
    c = d10 != 0
    print("SELFTEST q1755: 12.85 inside %s; 12.3 outside %s; d_1(10) = %s nonzero %s" % (a, b, d10, c))
    ok = a and b and c
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)
f = os.path.join(H, "..", "centre_scripts", "q1634_out_11.json")
if not os.path.exists(f):
    print("VOID: output file missing (%s)" % f)
    sys.exit(2)
d11 = d1_of(json.load(open(f))["e11"], 11)
s10 = float(d11 / d10)
print("d_1(10) = %s, d_1(11) = %s, s_10 = %.6f, band [%.5f, %.5f]" % (d10, d11, s10, BAND[0], BAND[1]))
print("registered prediction (s_10 inside the band): %s" % ("holds" if inband(s10) else "does not hold"))
print("READING q1755 done (labels by the script; no verdict word)")
