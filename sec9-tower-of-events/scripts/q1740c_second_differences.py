"""q1740c (10 Oct 2026, FACT TABLE after q1740b, no predictions, no verdict word): the second differences D_n = d_n - d_(n-1) = r_n - 2 r_(n-1) + r_(n-2) of the ratios r_n = e_n / e_(n-1) of the stored centre series (centre_scripts/d22_out_20.json), their ratios D_n / D_(n-1), and the effective power p_n in D_n / D_(n-1) = ((n-1)/n)^p_n, from mpmath at 40 digits. Written because q1740b found that the increments do not follow a 1/n expansion on n = 14..20; the table is a fact, not a test.
Usage: py q1740c_second_differences.py (from small_gap_q206/; seconds)."""

import json, os
import mpmath as mp

mp.mp.dps = 40
H = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(H, "..", "centre_scripts", "d22_out_20.json")))
E = {n: mp.mpf(S["e%d" % n]) for n in range(2, 21)}
r = {n: E[n] / E[n - 1] for n in range(3, 21)}
d = {n: r[n] - r[n - 1] for n in range(4, 21)}
D = {n: d[n] - d[n - 1] for n in range(5, 21)}
for n in range(9, 21):
    q = D[n] / D[n - 1]
    p = mp.log(q) / mp.log(mp.mpf(n - 1) / n)
    print(
        "n = %2d: r_n = %s, d_n = %s, D_n = %s, D_n / D_(n-1) = %s, effective power p_n = %s"
        % (n, mp.nstr(r[n], 8), mp.nstr(d[n], 7), mp.nstr(D[n], 5), mp.nstr(q, 5), mp.nstr(p, 4))
    )
print("TABLE q1740c done (facts; no labels)")
