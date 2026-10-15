"""q693 (exploratory): q661's centre readout (definitions copied as text) for a fractional-d ball run of q684; K_eff = 2 x rings +
origin; c = d - 1 in the bulk law. Usage: py q693_centre_dfrac.py D FILE [FILE ...]"""

import sys, json
import numpy as np

src = open("q661_centre_const.py", encoding="utf-8").read()
src = src[: src.rindex('if sys.argv[1] == "control":')]
_argv = sys.argv
sys.argv = ["q661", "none"]
ns = {}
exec(src, ns)
sys.argv = _argv
d = float(sys.argv[1])
for f in sys.argv[2:]:
    for e_ in json.load(open(f))["events"]:
        A = e_["A_lo"]
        r = np.array(e_["r"])
        depths = A - r[r > 1e-12] if "o" in e_["types"] else A - r
        K = sum(1 if t == "o" else 2 for t in e_["types"])
        if A < 12:
            continue
        e, rho, rms, n = ns["centre_e"](A, depths, d - 1, K)
        print(
            f"d {d:g} {e_['kind']:8s} K_eff {K:2d}->{K + 1:2d} A {A:9.5f}: e = {e:+.5f}  rho_c {rho:.4f}  (fit rms {rms:.1e} on {n}); "
            f"(3d - 2)/6 = {(3*d - 2)/6:.4f}, 1/6 + 0.93 (d - 1)/2 = {1/6 + 0.465*(d - 1):.4f}"
        )
