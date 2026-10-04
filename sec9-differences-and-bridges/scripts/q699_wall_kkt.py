"""q699 (exploratory, Amendment 1035): wall support of a q696/q698 solution read from the KKT function D on the wall. Support = wall
points with D > max D - K x (bracket width); pieces = maximal runs of such points around the closed curve; profile = max(D - max D)
per 2.5-degree bin of the folded angle (0 = edge midpoint, 45 = corner direction), in units of the bracket width.
Usage: py q699_wall_kkt.py K FILE [FILE ...]"""

import sys
import numpy as np

K = float(sys.argv[1])
for f in sys.argv[2:]:
    z = np.load(f)
    Db = z["Db"]
    bx = z["bx"]
    by = z["by"]
    gap = float(z["hi"] - z["lo"])
    P = float(z["P"])
    s = Db - Db.max()
    on = s > -K * gap
    pieces = "whole" if on.all() else int(np.sum(on & ~np.roll(on, 1)))
    ang = np.degrees(np.mod(np.arctan2(by, bx), np.pi / 2))
    ang = np.minimum(ang, 90 - ang)
    e = np.linspace(0, 45, 19)
    prof = [s[(ang >= e[k]) & (ang < e[k + 1] + (k == 17) * 1e-9)].max() / gap for k in range(18)]
    print(
        f"p {P:g}: bracket {gap:.1e}; pieces (D within {K:g} brackets of its max): {pieces}; profile in brackets:",
        np.round(prof, 1).tolist(),
    )
