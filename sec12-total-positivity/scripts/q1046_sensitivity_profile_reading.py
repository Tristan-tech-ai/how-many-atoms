# READING q1046: per-atom sensitivity s_p = 1/(f_p* - 1) of the Cramer minor N1 against the depth from the wall A - u_p, at the stored states K 40 (A 29.85), K 65 (A 43.5803), K 70 (A 46.5787), K 80 (A 51.513057) (q989 outputs).
# Printed: s_p at depths 5, 10, 15, 20, 25, 30 (linear interpolation in depth), the sum, and the sum of the atoms deeper than 20 and than 25. No extrapolation, no verdict.
import re, sys

A = "scratch/"
cases = [
    ("K 40", 29.85, "q989_results_q207_gauss_K40_A29.8500_out.txt"),
    ("K 65", 43.580315, "q989_K65_A43.5803_out.txt"),
    ("K 70", 46.5787, "q989_results_q207_gauss_K70_A46.5787_out.txt"),
    ("K 80", 51.513057, "q989_results_q206_gauss_K80_A51.513057_out.txt"),
]
print("READING q1046: s_p against the depth A - u_p")
prof = {}
for name, Aval, f in cases:
    pts = []
    for line in open(A + f, encoding="utf-8").read().splitlines():
        m = re.match(r"\s*atom (\d+): u\s+([0-9.]+)?", line)
        m2 = re.match(
            r"\s*atom (\d+): d_p [0-9.e+-]+; critical factor for this atom alone (infinity.*|[0-9.]+)", line
        )
        if m2:
            pts.append(
                (int(m2.group(1)), float("inf") if m2.group(2).startswith("inf") else float(m2.group(2)))
            )
    prof[name] = (Aval, pts)
    print(name, "atoms read", len(pts))
# the q989 files print the atom position only in the K 65 file; read the positions from the state files
import json

paths = {
    "K 40": "results_q207_gauss_K40_A29.8500.json",
    "K 65": "tower/K65/results_q220_gauss_K65_A43.5803.json",
    "K 70": "results_q207_gauss_K70_A46.5787.json",
    "K 80": "results_q206_gauss_K80_A51.513057.json",
}
for name, (Aval, pts) in prof.items():
    u = [float(x) for x in json.load(open(paths[name]))["u"]]
    if len(u) != len(pts):
        print("VOID: atom count differs for", name, len(u), len(pts))
        sys.exit(2)
    depth = [Aval - x for x in u]
    s = [0.0 if f == float("inf") else 1 / (f - 1) for _, f in pts]
    order = sorted(range(len(u)), key=lambda i: depth[i])
    dd = [depth[i] for i in order]
    ss = [s[i] for i in order]

    def at(x):
        for (x0, y0), (x1, y1) in zip(zip(dd, ss), zip(dd[1:], ss[1:])):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return float("nan")

    tail20 = sum(y for x, y in zip(dd, ss) if x > 20)
    tail25 = sum(y for x, y in zip(dd, ss) if x > 25)
    print(
        f"{name}: m {len(u)}, s at depth 5, 10, 15, 20, 25, 30: "
        + ", ".join(f"{at(x):.4f}" for x in (5, 10, 15, 20, 25, 30))
        + f"; sum {sum(ss):.4f}; sum over depth > 20 {tail20:.4f}, > 25 {tail25:.4f}"
    )
print("END q1046 READING (no verdict)")
