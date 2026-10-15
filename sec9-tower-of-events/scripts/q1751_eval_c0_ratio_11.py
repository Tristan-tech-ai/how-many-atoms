"""q1751 (10 Oct 2026, READING, no verdict word): evaluation of the registration q1750 (centre_scripts/q1750_prereg_c0_ratio.txt, band [5.60640, 5.69627] for s_10 = c_0(11)/c_0(10)). Reads the constant coefficient of the exact e_11 from centre_scripts/q1634_out_11.json (key e11, a string in pi) and c_0(10) from q1634_out_10.json.
Label: P holds when s_10 lies inside the band, else does not hold. Selftest (--selftest): a synthetic e11 string with the constant coefficient set to 5.65 c_0(10) must give 'holds', one with 5.5 c_0(10) must give 'does not hold', and a missing file must print VOID and exit with code 2.
Usage: py q1751_eval_c0_ratio_11.py [--selftest] (from small_gap_q206/; seconds)."""

import json, os, sys
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
x = sp.Symbol("x")
BAND = (5.60640, 5.69627)


def c0_of(s):
    ex = sp.expand(sp.sympify(s.replace("pi", "PI"), locals={"PI": x}))
    return sp.Rational(sp.Poly(ex, x).coeff_monomial(1))


c10 = c0_of(json.load(open(os.path.join(H, "..", "centre_scripts", "q1634_out_10.json")))["e10"])
if "--selftest" in sys.argv:
    h = lambda f: "holds" if BAND[0] <= float(f) <= BAND[1] else "does not hold"
    a = h(sp.Rational(565, 100))
    b = h(sp.Rational(55, 10))
    try:
        open(os.path.join(H, "..", "centre_scripts", "q1634_out_NOFILE.json"))
        d = False
    except FileNotFoundError:
        d = True
    ok = a == "holds" and b == "does not hold" and d
    print("SELFTEST q1751: 5.65 -> %s; 5.5 -> %s; missing file raises %s" % (a, b, d))
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)
f = os.path.join(H, "..", "centre_scripts", "q1634_out_11.json")
if not os.path.exists(f):
    print("VOID: output file missing (%s)" % f)
    sys.exit(2)
c11 = c0_of(json.load(open(f))["e11"])
s10 = float(c11 / c10)
print("c_0(10) = %s, c_0(11) = %s, s_10 = %.6f, band [%.5f, %.5f]" % (c10, c11, s10, BAND[0], BAND[1]))
print(
    "registered prediction (s_10 inside the band): %s"
    % ("holds" if BAND[0] <= s10 <= BAND[1] else "does not hold")
)
print("READING q1751 done (labels by the script; no verdict word)")
