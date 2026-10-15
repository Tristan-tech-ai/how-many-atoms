"""q1604 (10 Oct 2026, READING, no verdict word; deep proof Q11 step 60): the signed first-order response of the first depth to freeing the frozen atoms beyond a solved clean block, against the exact shifts of q1602.
Reading 101 found that the absolute-value bound for freeing the atoms 61..90 is 8.3e-16 against the exact shift 1.5e-23, because the residuals of the frozen atoms beyond a solved block have a boundary layer of about five atoms and cancel in their effect. The signed sum d_1(N1) - d_1(N0) = - sum over k = N0 + 1..N1 of (row_D,k F^D_k + row_D',k F^D'_k) uses the first row of the inverse Jacobian at the solved clean N1 state (q1603_res_N<N1>.json, 300 bits) and the residuals F of the frozen atoms at the solved clean N0 state (the lists FD, FDp of q1603_res_N<N0>.json, level atom N0 + 50, k = N0 + 1..N0 + 40). The exact shifts are the differences of the first depths in q1601_z_N<N>.json.
Predictions written before the run:
 P1 for (N0, N1) = (60, 90) the signed response is within 20 percent of the exact shift;
 P2 for (N0, N1) = (90, 120) the signed response is within 20 percent of the exact shift;
 P3 for (N0, N1) = (30, 60) the signed response with the rows of the solved clean N = 60 state is within 10 percent of the exact shift (the hybrid rows of reading 100 gave a ratio of 1.029);
 P4 at most 8 atoms carry 90 percent of the sum of the absolute values of the terms in the cases (60, 90) and (90, 120).
Usage: py q1604_signed_response.py, then the labels are printed by the same script (it only reads stored files).
"""

import json, os
import numpy as np
from mpmath import mp, mpf, nstr

mp.dps = 60


def rows(N):
    R = json.load(open("q1603_res_N%d.json" % N))
    return [mpf(s) for s, _ in R["row0"]]


def res(N):
    R = json.load(open("q1603_res_N%d.json" % N))
    return [mpf(x) for x in R["FD"]], [mpf(x) for x in R["FDp"]]


z = {N: mpf(json.load(open("q1601_z_N%d.json" % N))[0]) for N in (30, 60, 90, 120)}
out = {}
for N0, N1 in ((30, 60), (60, 90), (90, 120)):
    r = rows(N1)
    FD, FDp = res(N0)
    terms = []
    for k in range(N0 + 1, N1 + 1):
        j = k - N0 - 1
        terms.append(-(r[k] * FD[j] + r[N1 + k] * FDp[j]))
    lin = sum(terms)
    exact = z[N1] - z[N0]
    ab = sorted((abs(t) for t in terms), reverse=True)
    tot = sum(ab)
    cum = 0
    n90 = 0
    for t in ab:
        cum += t
        n90 += 1
        if cum >= 0.9 * tot:
            break
    out[(N0, N1)] = dict(lin=lin, exact=exact, ratio=lin / exact, n90=n90, absum=tot, terms=terms)
    print(
        "(%d, %d): exact %s, signed response %s, ratio %s; sum of absolute terms %s; atoms carrying 90 percent of it: %d; largest terms at k = %s"
        % (
            N0,
            N1,
            nstr(exact, 8),
            nstr(lin, 8),
            nstr(lin / exact, 6),
            nstr(tot, 4),
            n90,
            [N0 + 1 + i for i in sorted(range(len(terms)), key=lambda i: -abs(terms[i]))[:4]],
        )
    )
p1 = abs(out[(60, 90)]["ratio"] - 1) < 0.2
p2 = abs(out[(90, 120)]["ratio"] - 1) < 0.2
p3 = abs(out[(30, 60)]["ratio"] - 1) < 0.1
p4 = out[(60, 90)]["n90"] <= 8 and out[(90, 120)]["n90"] <= 8


def lab(x):
    return "holds" if x else "does not hold"


print("P1 %s; P2 %s; P3 %s; P4 %s" % tuple(lab(x) for x in (p1, p2, p3, p4)))
print("READING q1604 done (labels by the script; no verdict word)")
