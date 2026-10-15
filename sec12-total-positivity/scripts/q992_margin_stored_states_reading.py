"""q992 (7 Oct 2026): READING (no verdict): the critical factor t* of q988 at stored capacity states of the families K 50, K 60 and K 70 (output q988_K50_60_70_out.txt) next to the last states of the families K 62, K 64, K 66 (q988_all_out.txt): K (t* - 1) for each.
Usage: py q992_margin_stored_states_reading.py"""

import re

A = "scratch/"


def parse(f):
    out = []
    for l in open(A + f, encoding="utf-8").read().splitlines():
        m = re.match(
            r"results_q220_gauss_K(\d+)_A([0-9.]+)\.json: A [0-9.]+, m (\d+); N1\(1\) positive (\w+); .*crossing ([0-9.]+)",
            l,
        )
        if m:
            out.append((int(m.group(1)), float(m.group(2)), int(m.group(3)), m.group(4), float(m.group(5))))
    return out


print(
    "READING q992: stored states of the families K 50, K 60, K 70 (the last state of the family lies at the birth K_b = K)"
)
for K, a, mm, pos, t in parse("q988_K50_60_70_out.txt"):
    print(f"K {K}, A {a}, m {mm}: N1(1) positive {pos}; t* {t:.4f}; K (t* - 1) = {K * (t - 1):.3f}")
allr = parse("q988_all_out.txt")
print(
    "last states of the families (q988_all_out.txt): "
    + ", ".join(
        f"K {K} A {a}: t* {t:.4f} K(t* - 1) {K * (t - 1):.3f}"
        for (K, a, mm, pos, t) in allr
        if K in (62, 64, 66) and a > {62: 42.2, 64: 43.3, 66: 44.3}[K]
    )
)
print("END q992 READING (no verdict)")
