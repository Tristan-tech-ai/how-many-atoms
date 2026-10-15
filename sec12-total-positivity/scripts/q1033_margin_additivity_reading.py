"""q1033 (7 Oct 2026): READING (no verdict): is the margin of the Cramer minor N1 additive over the atoms?
Input: the per-atom critical factors f_p* of q989 (scaling d_p of atom p alone) for the stored states K 40 (A 29.85), K 65 (A 43.5803), K 70 (A 46.5787), K 80 (A 51.513057), and the crossing t* of q988 for the same states.
If N1 were linear in the diagonal term, then 1 / (t* - 1) = sum over p of 1 / (f_p* - 1). Printed: that sum, 1 / (t* - 1), their ratio, the contribution of the outermost 10 atoms to the sum and the number of atoms m.
Usage: py q1033_margin_additivity_reading.py"""

import re

A = "scratch/"
cases = [
    ("K 40 A 29.85", "q989_results_q207_gauss_K40_A29.8500_out.txt", 1.2524),
    ("K 65 A 43.5803", "q989_K65_A43.5803_out.txt", 1.1364),
    ("K 70 A 46.5787", "q989_results_q207_gauss_K70_A46.5787_out.txt", 1.1489),
    ("K 80 A 51.513057", "q989_results_q206_gauss_K80_A51.513057_out.txt", 1.1214),
]
print("READING q1033: additivity of the margin of N1 over the atoms")
for name, f, tstar in cases:
    fs = []
    for line in open(A + f, encoding="utf-8").read().splitlines():
        m = re.match(
            r"\s*atom (\d+): d_p [0-9.e+-]+; critical factor for this atom alone (infinity.*|[0-9.]+)", line
        )
        if m:
            fs.append(float("inf") if m.group(2).startswith("inf") else float(m.group(2)))
    if not fs:
        print("VOID: no atoms read from", f)
        raise SystemExit(2)
    s = [0.0 if x == float("inf") else 1 / (x - 1) for x in fs]
    S = sum(s)
    outer = sum(s[-10:])
    print(
        f"{name}: m {len(fs)}, sum 1/(f* - 1) {S:.4f}, 1/(t* - 1) {1 / (tstar - 1):.4f}, ratio sum/(1/(t*-1)) {S * (tstar - 1):.4f}, outermost 10 atoms contribute {outer:.4f}, outermost atom 1/(f*-1) {s[-1]:.4f}"
    )
print("END q1033 READING (no verdict)")
