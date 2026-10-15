"""q991 (7 Oct 2026): READING (no verdict): the dependence on K of the critical factor t* of q988 at the first, middle and last state of each family K 61 to K 66 (parsed from scratch/q988_all_out.txt): t* - 1, K (t* - 1) and the exponent p of t* - 1 ~ K^-p between K 61 and K 66 for each position in the family.
Usage: py q991_margin_scaling_reading.py"""

import re, math, sys

A = "scratch/"
rows = []
for l in open(A + "q988_all_out.txt", encoding="utf-8").read().splitlines():
    m = re.match(
        r"results_q220_gauss_K(\d+)_A([0-9.]+)\.json: A [0-9.]+, m (\d+); N1\(1\) positive (\w+); .*crossing ([0-9.]+)",
        l,
    )
    if m:
        rows.append((int(m.group(1)), float(m.group(2)), int(m.group(3)), m.group(4), float(m.group(5))))
if not rows:
    print("VOID q991: no rows")
    sys.exit(2)
by = {}
for K, a, mm, pos, t in rows:
    by.setdefault(K, []).append((a, t))
print("READING q991: critical factor t* of the diagonal term for the Cramer minor of u_1")
for slot, lab in ((0, "first"), (1, "middle"), (2, "last")):
    ser = [(K, by[K][slot][1]) for K in sorted(by) if len(by[K]) > slot]
    print(
        f"{lab} state of the family: "
        + ", ".join(f"K {K}: t* {t:.4f} (K(t* - 1) = {K * (t - 1):.3f})" for K, t in ser)
    )
    if len(ser) >= 2:
        (K0, t0), (K1, t1) = ser[0], ser[-1]
        print(
            f"   exponent p of t* - 1 ~ K^-p between K {K0} and K {K1}: {math.log((t0 - 1) / (t1 - 1)) / math.log(K1 / K0):.3f}; mean step of t* per family {(t1 - t0) / (K1 - K0):+.5f}"
        )
print("all N1(1) positive:", all(r[3] == "True" for r in rows), "; rows", len(rows))
print("END q991 READING (no verdict)")
