"""Numbers of Appendix C and Section VI of the paper, written to numbers_A2.json next to this script:
(1) the parts of J(1.1648) in ball arithmetic (python-flint, 256 bits): R, j_a, j_b, j_c, the bracket and 2 Phibar(R);
(2) the differences K - 0.4002 (A + beta/2)^(4/3) over all stored optima, with the ranges of A and K.
Usage: python numbers_A2.py"""
import json
import math
import os
from flint import arb, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
import sys  # noqa: E402
sys.path.insert(0, HERE)
from layout import find, state  # noqa: E402
ctx.prec = 256
SQ2 = arb(2).sqrt(); PI = arb.pi()
phi = lambda x: (-(x * x) / 2).exp() / (2 * PI).sqrt()
Phi = lambda x: (1 + (x / SQ2).erf()) / 2
Phibar = lambda x: (1 - (x / SQ2).erf()) / 2
lam = arb("1.1648")
R = (2 * (1 + lam)).sqrt()
ja = Phi(R) - arb(1) / 2; jb = phi(arb(0)) - phi(R); jc = ja - R * phi(R)
br = -lam * ja + R * jb - jc / 2
tail = 2 * Phibar(R)
J = 2 * br + tail
mid = lambda x: float(x.mid())
out = {"R": mid(R), "ja": mid(ja), "jb": mid(jb), "jc": mid(jc), "minus_lam_ja": mid(-lam * ja), "R_jb": mid(R * jb),
       "minus_half_jc": mid(-jc / 2), "bracket": mid(br), "two_Phibar_R": mid(tail), "J": mid(J), "J_negative": bool(J < 0)}
D = json.load(open(find("count_law_data.json"), encoding="utf-8"))
B2 = D["beta_half"]; C_LAW = 0.4002
diffs = [(s[0], s[1], s[1] - C_LAW * (s[0] + B2) ** (4 / 3)) for s in D["states"]]
small = [d for d in diffs if d[0] < 29.85]
import flint
out.update({"flint_version": flint.__version__, "A_min": min(s[0] for s in D["states"]), "A_max": max(s[0] for s in D["states"]),
            "K_min": min(s[1] for s in D["states"]), "K_max": max(s[1] for s in D["states"])})
out.update({"all_n": len(diffs), "all_min": min(d[2] for d in diffs), "all_max": max(d[2] for d in diffs),
            "small": [[a, k, x] for a, k, x in small]})
json.dump(out, open(os.path.join(HERE, "numbers_A2.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(out))
