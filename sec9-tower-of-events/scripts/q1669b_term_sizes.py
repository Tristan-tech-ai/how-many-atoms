"""q1669b (10 Oct 2026, READING, post hoc, no predictions, written after q1669 was read): the change of the predicted amplitude a_pred when the stored coefficient e_n is added one at a time (pair and law fixed: bulk law through c28, the refitted pair of nmax = 18 from q1669), for n = 12..20 at K_b 83, 106 and 130.
Usage: py q1669b_term_sizes.py (from small_gap_q206/)."""

import json, os, re
import sympy as sp
import mpmath as mp
import births_law as B

mp.mp.dps = 40
H = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(H, "..", "centre_scripts")
S20 = json.load(open(os.path.join(C, "d22_out_20.json")))
E = {n: float(S20["e%d" % n]) for n in range(2, 21)}
txt = open(os.path.join(H, "q1557_stdout_ORD28.txt"), encoding="utf-8").read()
f = {}
for n in (26, 28):
    m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, txt, re.M)
    f[n] = float(mp.mpf(sp.N(sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi}), 30)))
B.F = ((B.F + [-2.420924146814851e-10, -8.973800969707301e-11])[:13] + [f[26], f[28]])[:15]
B.NG = 14
B.g = [1.0]
for n in range(1, 15):
    B.g.append(-sum(B.F[j] * B.g[n - j] for j in range(1, n + 1)))
NU, D0 = -0.0937736508347, 0.0011599160063
for K in (83, 106, 130):
    prev = B.a_pred(K, NU, D0, [E[n] for n in range(6, 12)])
    line = []
    for n in range(12, 21):
        cur = B.a_pred(K, NU, D0, [E[m] for m in range(6, n + 1)])
        line.append("e_%d %+.2e" % (n, cur - prev))
        prev = cur
    print("K_b %d: change of a_pred when e_n is added: " % K + ", ".join(line))
print("READING q1669b done (post hoc, no labels)")
