"""q1776 (10 Oct 2026, READING, no verdict word): evaluation of the registration q1775 (q1775_prereg_f32_bottom.txt, 14:32:03): the coefficient of pi^-16 in the exact f32 = F_16 (q1557_stdout_ORD32.txt) times 8^16 is negative and its absolute value lies in [2.480617e+10, 2.638799e+10]. Label: holds or does not hold.
Selftest (--selftest): the same extraction on the f30 line of q1557_stdout_ORD30.txt with 8^15 gives exactly -86841758233067/179094982282444800 times 8^15 (the stored b(15)), a value moved by 10 percent is outside the band test, and a missing f32 line makes the real run print VOID and exit with code 2.
Usage: py q1776_eval_f32_bottom.py [--selftest] (from small_gap_q206/; seconds)."""

import os, re, sys
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
x = sp.Symbol("x")
LO, HI = 2.480617e10, 2.638799e10


def line_of(fn, n):
    p = os.path.join(H, fn)
    if not os.path.exists(p):
        return None
    m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, open(p, encoding="utf-8").read(), re.M)
    return m.group(1).strip() if m else None


def coef16(s, scale):
    e = sp.sympify(s.replace("pi", "PI"), locals={"PI": sp.pi})
    cf = {m[0]: sp.Rational(c) for m, c in sp.Poly(sp.expand(e.subs(sp.pi, 1 / x)), x).terms()}
    return cf.get(16, sp.Integer(0)) * scale


inband = lambda v: v < 0 and LO <= abs(float(v)) <= HI
if "--selftest" in sys.argv:
    v = coef16(line_of("q1557_stdout_ORD30.txt", 30), sp.Integer(8) ** 15)
    ok1 = v == sp.Rational(-86841758233067, 179094982282444800) * sp.Integer(8) ** 15
    ok2 = (
        inband(sp.Rational(-25000000000))
        and not inband(sp.Rational(-27500000000))
        and not inband(sp.Rational(25000000000))
    )
    ok3 = line_of("q1557_stdout_ORD30.txt", 32) is None
    print(
        "SELFTEST q1776: f30 extraction equals the stored b(15) times 8^15 %s; band test cases %s; no f32 line in the ORD30 file %s"
        % (ok1, ok2, ok3)
    )
    ok = ok1 and ok2 and ok3
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)
s = line_of("q1557_stdout_ORD32.txt", 32)
if s is None:
    print("VOID: no f32 line in q1557_stdout_ORD32.txt")
    sys.exit(2)
v = coef16(s, sp.Integer(8) ** 16)
print(
    "coefficient of pi^-16 times 8^16 = %s (%.6e); registered band for the absolute value [%.6e, %.6e], negative"
    % (v, float(v), LO, HI)
)
print("registered prediction: %s" % ("holds" if inband(v) else "does not hold"))
print("READING q1776 done (labels by the script; no verdict word)")
