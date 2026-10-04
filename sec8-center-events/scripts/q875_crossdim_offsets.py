"""q875 (1591): 3D LFP offsets predicted from the 1D and 2D long chains, offset_3D = offset_1D + 2 kappa_2D/(m + delta).
Offsets as in q874 (each chain event re-solved with q868.solve_inner). Usage: py q875_crossdim_offsets.py
"""

import re
import numpy as np
from scipy.optimize import brentq
from scipy.special import jv
from q868_lfp_events_dimd import solve_inner, risk

D = 3.34776
arr = lambda s: [float(x) for x in s.replace("[", " ").replace("]", " ").split()]
pat = re.compile(
    r"CHAIN d (\d+) e(\d+) (birth|split): m ([0-9.]+); support \[([^\]]*)\]; weights \[([^\]]*)\]"
)


def offsets(f, d):
    j = brentq(lambda x: jv(d / 2 - 1, x), 1.0, 5.0) if d > 1 else np.pi / 2
    out = []
    for line in open(f, encoding="utf-8", errors="replace"):
        g = pat.search(line)
        if not g:
            continue
        m, sup, wts = float(g.group(4)), arr(g.group(5)), arr(g.group(6))
        cen = sup[0] == 0.0
        a = sup[1:-1] if cen else sup[:-1]
        w = wts[1:-1] if cen else wts[:-1]
        w0 = wts[0] if cen else None
        sol, at, wt, res = solve_inner(d, m, a, w, w0)
        dfc = d - risk(d, m, at, wt)
        out.append((m, (2 * j if d > 1 else np.pi) / np.sqrt(dfc) - m))
    return out


def guarded(f):
    """event lines of a chain log that pass the 1592 guard (radius >= 0, weights in [0, 1], residual <= 1e-8), as (m, line)"""
    out = []
    try:
        text = re.sub(
            r"\n[ \t]+", " ", open(f, encoding="utf-8", errors="replace").read()
        )  # join wrapped CHAIN entries (1601)
    except FileNotFoundError:
        return out
    for line in text.split("\n"):
        g = re.search(
            r"CHAIN d \d+ e\d+ (birth|split): m ([0-9.]+); support \[([^\]]*)\]; weights \[([^\]]*)\]; residual ([0-9.e+-]+)",
            line,
        )
        if not g:
            continue
        s, w, r = arr(g.group(3)), arr(g.group(4)), float(g.group(5))
        if min(s) >= 0 and min(w) >= -1e-12 and max(w) <= 1 and r <= 1e-8:
            out.append((float(g.group(2)), line.rstrip("\n") + "\n"))
        else:
            print(f"guard rejects {f} event at m {g.group(2)}")
            break
    return out


# 1594 rule for the 1D input: first chain's guard-valid events, then the second chain's events beyond the first chain's last valid one
A1 = guarded("q868_runs/chain_d1_long.log")
B1 = guarded("q868_runs/chain_d1_long_b.log")
last = A1[-1][0] if A1 else 0.0
for mB, lB in B1:
    for mA, lA in A1:
        if abs(mA - mB) < 0.05 and abs(mA - mB) > 1e-6:
            print(f"1D chains disagree near m {mA:.6f} / {mB:.6f}: 1D input ambiguous above")
merged = [l for m_, l in A1] + [l for m_, l in B1 if m_ > last + 1e-6]
C1 = guarded("q868_runs/chain_d1_long_c.log")  # 1600: third 1D chain from the backup's e11 birth state
lastB = max([last] + [m_ for m_, l in B1])
merged += [l for m_, l in C1 if m_ > lastB + 1e-6]
open("q868_runs/chain_d1_merged_1594.log", "w", encoding="utf-8").write("".join(merged))
O1 = offsets("q868_runs/chain_d1_merged_1594.log", 1)
O2 = offsets("q868_runs/chain_d2_merged_1596.log", 2)
O3 = offsets("q868_runs/chain_d3_merged_1595.log", 3)
m1 = np.array([m for m, o in O1])
x1 = 1 / (m1 + D)
c = np.polyfit(x1, [o for m, o in O1], 3)
off1 = lambda m: np.polyval(c, 1 / (m + D))
m2 = np.array([m for m, o in O2])
k2 = np.array([(o - off1(m)) * (m + D) for m, o in O2])
print("1D offsets:", [(round(m, 3), round(o, 5)) for m, o in O1])
print("2D kappa:", [(round(m, 3), round(k, 4)) for m, k in zip(m2, k2)])
rows = []
for m, o in O3:
    if not (7.5 <= m <= 13.5) or m < m2.min() or m > m2.max() or m > m1.max():
        continue
    k = np.interp(m, m2, k2)
    p = off1(m) + 2 * k / (m + D)
    rows.append((m, o, p, o - p, (o - off1(m)) * (m + D) / 2))
for m, o, p, df, k3 in rows:
    print(f"3D m {m:.4f}: offset {o:.5f} predicted {p:.5f} ({df:+.5f}); kappa_3D {k3:.4f}")
print(
    f"scored {len(rows)}; within 0.003: {sum(abs(r[3]) <= 0.003 for r in rows)}; verdict "
    f"{'PASS' if rows and all(abs(r[3]) <= 0.003 for r in rows) else 'FAIL'}"
)
