"""q1766 (10 Oct 2026, READING, no verdict word): evaluation of the registration q1765 (small_gap_q206/q1765_prereg_f32.txt, 14:00:02) on the exact f32 = F_16 from the run py q1557_f12_symbolic.py 32 (stdout q1557_stdout_ORD32.txt, started about 14:00:30).
Labels: R1 the coefficient of pi^-32 equals 1/8^16; R2 the coefficient of pi^-30 times 8^16 equals c_1(16) = -196352/3; R3 the coefficient of pi^-28 times 8^16 equals c_2(16) = 1845151984/15; R4 the lowest power of 1/pi is 16 and no lower power occurs. R5 (the logarithm coefficient L_16, u^15 coefficient times 16 x 8^16 equal to -878560/3) follows from R1 to R3 by the recursion and is not evaluated separately.
Selftest (--selftest): the same code applied to the f30 line of q1557_stdout_ORD30.txt with j = 15 (top, level 1 and level 2 formulas, lowest power u^8) gives true; a copy of the f32 expression with one digit changed in the pi^-30 coefficient gives R2 false; a missing f32 line makes the real run print VOID and exit with code 2.
Usage: py q1766_eval_f32.py [--selftest] (from small_gap_q206/; seconds).
"""

import os, re, sys
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
x = sp.Symbol("x")
c1 = lambda j: -sp.Integer(j) * (j - 1) * (6 * j**3 + j**2 - 19 * j + 16) / 90
c2 = (
    lambda j: sp.Integer(j)
    * (j - 1)
    * (j - 2)
    * (j - 3)
    * (118 * j**6 - 1092 * j**5 + 1135 * j**4 + 14790 * j**3 - 46973 * j**2 + 27054 * j + 22212)
    / 340200
)


def line_of(fn, n):
    t = open(os.path.join(H, fn), encoding="utf-8").read()
    m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, t, re.M)
    return m.group(1).strip() if m else None


def coeffs(s):
    e = sp.sympify(s.replace("pi", "PI"), locals={"PI": sp.pi})
    return {m[0]: sp.Rational(c) for m, c in sp.Poly(sp.expand(e.subs(sp.pi, 1 / x)), x).terms()}


def labels(cf, j):
    return (
        cf.get(2 * j, 0) == sp.Rational(1, 8**j),
        cf.get(2 * j - 2, 0) * 8**j == c1(j),
        cf.get(2 * j - 4, 0) * 8**j == c2(j),
        min(cf) == 2 * (-(-j // 2)),
    )


if "--selftest" in sys.argv:
    s30 = line_of("q1557_stdout_ORD30.txt", 30)
    cf = coeffs(s30)
    ok1 = all(labels(cf, 15))
    bad = dict(cf)
    bad[28] = bad[28] * (1 + sp.Rational(1, 10**6))
    ok2 = labels(bad, 15)[1] is False
    ok3 = line_of("q1557_stdout_ORD30.txt", 32) is None
    print(
        "SELFTEST q1766: f30 with j = 15 gives all four true %s; a changed coefficient makes R2 false %s; no f32 line in the ORD30 file %s"
        % (ok1, ok2, ok3)
    )
    ok = ok1 and ok2 and ok3
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)
s32 = (
    line_of("q1557_stdout_ORD32.txt", 32)
    if os.path.exists(os.path.join(H, "q1557_stdout_ORD32.txt"))
    else None
)
if s32 is None:
    print("VOID: no f32 line in q1557_stdout_ORD32.txt")
    sys.exit(2)
cf = coeffs(s32)
r1, r2, r3, r4 = labels(cf, 16)
lab = lambda q: "holds" if q else "does not hold"
print("powers of 1/pi in f32: %s" % sorted(cf))
print(
    "coefficient of pi^-32 times 8^16: %s; of pi^-30 times 8^16: %s (prediction -196352/3); of pi^-28 times 8^16: %s (prediction 1845151984/15)"
    % (cf.get(32, 0) * 8**16, cf.get(30, 0) * 8**16, cf.get(28, 0) * 8**16)
)
print(
    "R1 top coefficient 1/8^16: %s; R2 level 1 equals c_1(16): %s; R3 level 2 equals c_2(16): %s; R4 lowest power pi^-16: %s"
    % (lab(r1), lab(r2), lab(r3), lab(r4))
)
print("READING q1766 done (labels by the script; no verdict word)")
