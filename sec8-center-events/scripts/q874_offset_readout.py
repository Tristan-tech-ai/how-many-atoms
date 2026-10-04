"""q874 (1590): implied LFP offsets delta_impl = 2 j_{d/2-1,1}/sqrt(d - B) - m at the events of a q869 chain log, and the 1590 readout.
Usage: py q874_offset_readout.py LOG d
"""

import re, sys
import numpy as np
from scipy.optimize import brentq
from scipy.special import jv
from q868_lfp_events_dimd import solve_inner, risk

D = 3.34776
f, d = sys.argv[1], int(sys.argv[2])
j = brentq(lambda x: jv(d / 2 - 1, x), 1.0, 5.0) if d > 1 else np.pi / 2
arr = lambda s: [float(x) for x in s.replace("[", " ").replace("]", " ").split()]
pat = re.compile(
    r"CHAIN d (\d+) e(\d+) (birth|split): m ([0-9.]+); support \[([^\]]*)\]; weights \[([^\]]*)\]"
)
rows = []
for line in open(f, encoding="utf-8", errors="replace"):
    g = pat.search(line)
    if not g:
        continue
    n, kind, m, sup, wts = int(g.group(2)), g.group(3), float(g.group(4)), arr(g.group(5)), arr(g.group(6))
    cen = sup[0] == 0.0
    a = sup[1:-1] if cen else sup[:-1]
    w = wts[1:-1] if cen else wts[:-1]
    w0 = wts[0] if cen else None
    sol, at, wt, res = solve_inner(d, m, a, w, w0)
    dfc = d - risk(d, m, at, wt)
    off = 2 * j / np.sqrt(dfc) - m if d > 1 else np.pi / np.sqrt(dfc) - m
    rows.append((n, kind, m, dfc, off, res))
    print(f"e{n:2d} {kind:6s} m {m:.6f} deficit {dfc:.9f} offset {off:.5f} residual {res:.1e}", flush=True)
if d in (3, 4, 5):
    if d == 3:  # 1590
        L = lambda m: D + 0.5006 / m - 1.8141 / m**2
        S = lambda m: D + 0.0465 - 0.0053 / m - 0.4767 / m**2
        mlate = 7.0
    elif d == 5:  # 1597 (chain starts at 5D e8; the maximum includes the known 5D offsets of 1590 (1))
        L = lambda m: D + 1.1208 / m - 3.5232 / m**2
        S = lambda m: D + 0.0661 + 0.3627 / m - 1.4321 / m**2
        mlate = 7.7
        for mm, oo in [
            (7.715125, 3.4368),
            (6.658386, 3.4361),
            (6.114747, 3.4346),
            (4.956883, 3.4288),
            (4.402800, 3.4223),
        ]:
            rows.insert(0, (0, "known", mm, None, oo, 0.0))
    else:  # 1593 (offsets of the chain start at 4D e8; the chain maximum includes the known e7 offset 3.4108)
        L = lambda m: D + 0.7883 / m - 2.5327 / m**2
        S = lambda m: D + 0.0539 + 0.2086 / m - 1.0404 / m**2
        mlate = 7.3
        rows.insert(0, (7, "birth", 7.33592033, None, 3.4108, 0.0))
    win = [r for r in rows if 9.5 <= r[2] <= 13.5]
    offs = [r[4] for r in rows]
    okL = bool(win) and all(abs(r[4] - L(r[2])) <= 0.004 for r in win) and (max(offs) - offs[-1] >= 0.001)
    late = [r[4] for r in rows if r[2] >= mlate]
    okS = (
        bool(win)
        and all(abs(r[4] - S(r[2])) <= 0.004 for r in win)
        and all(late[i + 1] - late[i] >= -0.0005 for i in range(len(late) - 1))
    )
    for r in win:
        print(
            f"   window m {r[2]:.4f}: offset {r[4]:.5f}; L {L(r[2]):.5f} ({r[4] - L(r[2]):+.5f}); S {S(r[2]):.5f} ({r[4] - S(r[2]):+.5f})"
        )
    print(
        f"max offset {max(offs):.5f}; last {offs[-1]:.5f}; L {'PASS' if okL else 'FAIL'}; S {'PASS' if okS else 'FAIL'}"
    )
