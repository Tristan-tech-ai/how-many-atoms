"""q873 (1584): LFP deficit d - B at the tracked event states against 4 j_{d/2-1,1}^2/(m + delta)^2.
Each event state is re-solved at its m with q868.solve_inner (support and weights parsed from the logs as the guess), B = r(m).
Usage: py q873_lfp_deficit_events.py
"""

import re, glob
import numpy as np
from scipy.optimize import brentq
from scipy.special import jv
from q868_lfp_events_dimd import solve_inner, risk, two, interior_excess

DELTA = 3.34776


def pred(d, m):
    j = brentq(lambda x: jv(d / 2 - 1, x), 2.0, 5.0) if d > 1 else np.pi / 2
    return 4 * j * j / (m + DELTA) ** 2


def arr(s):
    return [float(x) for x in s.replace("[", " ").replace("]", " ").split()]


rows = []
pat = re.compile(r"d (\d+) (birth|split) event: m ([0-9.]+); support \[([^\]]*)\]; weights \[([^\]]*)\]")
patc = re.compile(
    r"CHAIN d (\d+) e\d+ (birth|split): m ([0-9.]+); support \[([^\]]*)\]; weights \[([^\]]*)\]"
)
for f in sorted(glob.glob("q868_runs/*.log")):
    if "check" in f:
        continue
    for line in open(f, encoding="utf-8", errors="replace"):
        g = pat.search(line) or patc.search(line)
        if not g:
            continue
        d, kind, m, sup, wts = (
            int(g.group(1)),
            g.group(2),
            float(g.group(3)),
            arr(g.group(4)),
            arr(g.group(5)),
        )
        cen = sup[0] == 0.0
        a = sup[1:-1] if cen else sup[:-1]
        w = wts[1:-1] if cen else wts[:-1]
        w0 = wts[0] if cen else None
        sol, at, wt, res = solve_inner(d, m, a, w, w0)
        rows.append((d, m, kind, d - risk(d, m, at, wt), res, interior_excess(d, at, wt, m), f))
for d, m in [(2, 1.53499461), (3, 1.90797975), (4, 2.22326275), (5, 2.50080379)]:
    rows.append((d, m, "e1", d - risk(d, m, [m], [1.0]), 0.0, 0.0, "e1 formula"))
for d, m in [(2, 2.31714899), (3, 2.59289334), (4, 2.84176358), (5, 3.07029462)]:
    at, w = two(d, m)
    rows.append((d, m, "e2", d - risk(d, m, at, w), 0.0, 0.0, "e2 formula"))
seen = set()
out = {}
for d, m, kind, dfc, res, ex, f in sorted(rows):
    key = (d, round(m, 6))
    if key in seen:
        continue
    seen.add(key)
    r = dfc / pred(d, m)
    out.setdefault(d, []).append((m, r))
    print(
        f"d {d} m {m:.6f} {kind:6s} deficit {dfc:.8f} predicted {pred(d, m):.8f} ratio {r:.5f} residual {res:.1e} excess {ex:+.1e}"
    )
for d in sorted(out):
    if d == 1:
        continue
    sc = [(m, r) for m, r in out[d] if m >= 4]
    lo, hi = (0.98, 1.02) if d <= 3 else (0.95, 1.05)
    ok = all(lo <= r <= hi for m, r in sc)
    print(
        f"d {d}: scored (m >= 4) {len(sc)}; ratios {[round(r, 4) for m, r in sc]}; band [{lo}, {hi}]: {'PASS' if sc and ok else 'FAIL'}"
    )
