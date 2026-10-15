"""q884 (6 Oct 2026): T12, the branch-0 signature, written before any output was read (registered in amendment 1650).
Branch k = 0 means that g/C - 1 has no critical point other than the atoms (maxima) and one minimum between consecutive atoms (and the symmetric minimum at 0 when no
centre atom exists). A higher branch would add extra maxima/minima pairs between atoms (the chirped combs of 656). Rows: a fixed list of eight rows of results_q584.json
(indices 0, 100, 200, 300, 400, 500, 600 and the last), 200-bit, atoms as stored, g' evaluated in arb on the grid y = 0.005, 0.010, ..., a + 3 (the same two-term kernel sums as
q579_immune_mp.IMP.gs). The number of strict sign changes of g'(y) over (0, a + 3] is counted from the midpoints (a grid point whose ball contains 0 and is not an atom
position counts as undecided).
Expected count for branch 0: 2J - 1 when there is no centre atom (J atoms plus J - 1 minima between them; the minimum at y = 0 is not in the open range), 2J when there is a
centre atom (J atoms plus J minima, one between the centre atom and the first positive atom).
Criterion: for every row the count equals the expected number and no undecided grid point other than those within 1e-9 of an atom exists. PASS = all eight rows.
FAIL records the row and the position of the extra sign changes. Resolution limit, stated in advance: two extra critical points closer than 0.005 would be missed.
Usage: py q884_T12_critical_points.py"""

import json
import numpy as np
import q579_immune_mp as M
from flint import arb

rows = json.load(open("results_q584.json"))["rows"]
idx = [0, 100, 200, 300, 400, 500, 600, len(rows) - 1]
allok = True
for i in idx:
    r = rows[i]
    a = float(r["a"])
    centre = bool(r["centre"])
    J = len(r["p"])
    I = M.IMP(arb(a), 9)
    pp, vv, res = I.polish([float(t) for t in r["p"]], [float(t) for t in r["v"]], centre)
    c = (arb(1) - sum(vv)) if centre else None
    grid = [arb(float(y)) for y in np.arange(0.005, a + 3, 0.005)]
    gp, C = I.gs(pp, vv, c, grid)
    d1 = [t[1] for t in gp]
    mids = [float(t.mid()) for t in d1]
    atoms = [float(t.mid()) for t in pp]
    changes = []
    und = 0
    prev = None
    for k, (t, y) in enumerate(zip(d1, np.arange(0.005, a + 3, 0.005))):
        if t.contains(0) and min(abs(y - q) for q in atoms + [0.0]) > 0.0051:
            und += 1
        s = np.sign(mids[k])
        if prev is not None and s != 0 and s != prev:
            changes.append(float(y))
        if s != 0:
            prev = s
    exp = 2 * J if centre else 2 * J - 1
    ok = (len(changes) == exp) and und == 0
    allok = allok and ok
    print(
        f"row {i}: a {a:.6f}, K {2 * J + int(centre)}, centre {centre}: sign changes of g' {len(changes)} (expected {exp}), undecided {und} -> {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    if not ok:
        print("   changes at", [round(x, 3) for x in changes][:40])
print("T12 VERDICT:", "PASS" if allok else "FAIL")
