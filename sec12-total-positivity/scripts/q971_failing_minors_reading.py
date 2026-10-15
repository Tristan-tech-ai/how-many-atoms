"""q971 (7 Oct 2026): READING (no verdict) that lists the initial minors of X = [T | v] (sigma = +1 for odd m, the orientation of q969) that are certainly negative at 800 bits, for one state:
type (rows 0..k-1 with columns j..j+k-1, or rows i..i+k-1 with columns 0..k-1), order k, the position, and the base-10 logarithm of the absolute value of the minor and of the minimal positive minor of the same order.
Usage: py q971_failing_minors_reading.py STATE.json (from temporal_within/)"""

import sys, os
from flint import arb, ctx
import q969_extended_tp_test as Q
import q970_extended_tp_diagnosis as D

path = sys.argv[1]
A_str, m, odd, Tm, vv = D.build(path)
ctx.prec = 800
X = [[arb(x) for x in row] + [arb(vv[p])] for p, row in enumerate(Tm)]
n = m + 1
print(f"state {os.path.basename(path)}: A {A_str}, m {m}", flush=True)
neg = []
lg = lambda d: float((abs(d).log() / arb(10).log()).mid().str(8, radius=False))
for k in range(1, m + 1):
    for j in range(0, n - k + 1):
        d = Q.minor(X, range(0, k), range(j, j + k))
        if d < 0:
            neg.append(("rows 0..k-1, columns from j", k, j, lg(d)))
    for i in range(1, m - k + 1):
        d = Q.minor(X, range(i, i + k), range(0, k))
        if d < 0:
            neg.append(("rows from i, columns 0..k-1", k, i, lg(d)))
for t in neg:
    print(f"   certainly negative: {t[0]} = {t[2]}, order {t[1]}, log10|minor| {t[3]:.2f}")
print(f"number of certainly negative initial minors {len(neg)} of {m * n}")
# the T part alone
negT = 0
for k in range(1, m + 1):
    for j in range(0, m - k + 1):
        if Q.minor(X, range(0, k), range(j, j + k)) <= 0:
            negT += 1
    for i in range(1, m - k + 1):
        if Q.minor(X, range(i, i + k), range(0, k)) <= 0:
            negT += 1
print(
    f"initial minors of T alone (columns 0..m-1) that are not certainly positive at 800 bits: {negT} of {m * m}"
)
print("END q971 READING (no verdict)")
