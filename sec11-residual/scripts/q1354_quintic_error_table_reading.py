"""q1354 (8 Oct 2026, EXPLORATORY table, amendment 2121; no item, no verdict): the error of the quintic prediction a_med against the converged value (spline degree 19) and its size relative to the residual R, for the capacity, from q1351_out.txt (a_med(degree) - a_med(13) in 1e-12) and q1352_events.json (capacity R with degree 5 in 1e-9). Printed per K_b of the table: a_med(5) - a_med(19) in 1e-12, R(degree 5) in 1e-9 (nearest stored event), and the ratio.
Control (first): the table has the ten columns and the rows for the degrees 5 and 19; the degree-17 and degree-19 rows differ by less than 1.0e-12 in every column except K_b 20.
"""

import json, re, sys

t = open("q1351_out.txt", encoding="utf-8").read()
blk = t.split("hard wall")[0]
cols = [int(x) for x in re.search(r"columns K_b \[([^\]]*)\]", blk).group(1).split(",")]
rows = {
    int(m.group(1)): [float(x) for x in m.group(2).split()]
    for m in re.finditer(r"degree\s+(\d+):\s+([^\n]+)", blk)
}
ctl = (
    len(cols) == 10
    and 5 in rows
    and 19 in rows
    and all(abs(a - b) < 1.0 for a, b, K in zip(rows[17], rows[19], cols) if K != 20)
)
print(f"control (ten columns, rows 5 and 19, degrees 17 and 19 within 1.0e-12 except K_b 20): {ctl}")
if not ctl:
    print("READING q1354: VOID (control)")
    sys.exit(2)
ev = json.load(open("q1352_events.json"))["capacity_deg5"]
R = {int(K): r for K, r, v in ev}
print("K_b   a_med(5) - a_med(19) [1e-12]   R(deg 5) [1e-9]   ratio")
for i, K in enumerate(cols):
    d = rows[5][i] - rows[19][i]
    r = R.get(K)
    print(
        f"{K:3d}   {d:12.2f}   {r:10.4f}   {d * 1e-3 / r:+.4f}" if r else f"{K:3d}   {d:12.2f}   (no event)"
    )
print("READING q1354 (exploratory, no verdict)")
