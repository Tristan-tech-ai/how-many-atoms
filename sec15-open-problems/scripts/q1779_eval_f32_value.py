"""q1779 (10 Oct 2026, READING, no verdict word): evaluation of the registration q1778 (q1778_prereg_f32_value.txt, 14:33:23): the numeric value of f32 = F_16 lies in [-3.969084e-12, -3.381072e-12]. The value is computed from the exact expression of the f32 line (mpmath, 40 digits) and, as a cross-check, read from the decimal printed after the expression. The ratio-model value -3.587489e-12 is printed for comparison.
Selftest (--selftest): the same code on the f30 line of q1557_stdout_ORD30.txt returns -7.25424082052e-12 to 11 digits from both routes; the band test returns true inside and false outside; a missing f32 line makes the real run print VOID and exit with code 2.
Usage: py q1779_eval_f32_value.py [--selftest] (from small_gap_q206/; seconds)."""

import os, re, sys
import sympy as sp
import mpmath as mp

mp.mp.dps = 40
H = os.path.dirname(os.path.abspath(__file__))
LO, HI = -3.969084e-12, -3.381072e-12
RM = -3.587489e-12


def parse(fn, n):
    p = os.path.join(H, fn)
    if not os.path.exists(p):
        return None
    m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= (\S+))?\s*$" % n, open(p, encoding="utf-8").read(), re.M)
    if not m:
        return None
    e = sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi})
    return float(mp.mpf(sp.N(e, 40))), (float(m.group(2).replace("E", "e")) if m.group(2) else None)


inband = lambda v: LO <= v <= HI
if "--selftest" in sys.argv:
    a = parse("q1557_stdout_ORD30.txt", 30)
    ok1 = (
        a is not None
        and abs(a[0] / -7.25424082052e-12 - 1) < 1e-10
        and a[1] is not None
        and abs(a[1] / a[0] - 1) < 1e-10
    )
    ok2 = inband(-3.6e-12) and not inband(-3.3e-12) and not inband(-4.1e-12) and not inband(3.6e-12)
    ok3 = parse("q1557_stdout_ORD30.txt", 32) is None
    print(
        "SELFTEST q1779: f30 value from the expression and from the printed decimal agree with -7.25424082052e-12 %s; band test %s; no f32 line in the ORD30 file %s"
        % (ok1, ok2, ok3)
    )
    ok = ok1 and ok2 and ok3
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)
a = parse("q1557_stdout_ORD32.txt", 32)
if a is None:
    print("VOID: no f32 line in q1557_stdout_ORD32.txt")
    sys.exit(2)
print(
    "f32 from the exact expression: %.6e; from the printed decimal: %s; band [%.6e, %.6e]; ratio-model value %.6e; ratio f32 / ratio model = %.4f"
    % (a[0], ("%.6e" % a[1]) if a[1] is not None else "none", LO, HI, RM, a[0] / RM)
)
print("registered prediction: %s" % ("holds" if inband(a[0]) else "does not hold"))
print("READING q1779 done (labels by the script; no verdict word)")
