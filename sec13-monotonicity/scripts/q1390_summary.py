"""q1390 (9 Oct 2026, summary by script, no verdict; py q1390_summary.py OUTFILE... from temporal_within/small_gap_q206/): parse the printed output of q1389 and tabulate, per state, K, A, flow, the D1-basis cancellation ratio and positive/negative ratio, and the sign structure; print totals, minima, maxima and exceptions to the sign structure (D' contributions all >= 0 and difference-row contributions all <= 0 in the D1 basis)."""

import re, sys

rows = []
cur = None
for fn in sys.argv[1:]:
    for line in open(fn, encoding="utf-8"):
        m = re.match(r"K (\d+) (\S+) A (\S+) \((odd|even), \S+\), stored-system flow (\S+), J", line)
        if m:
            cur = dict(
                K=int(m.group(1)),
                label=m.group(2),
                A=float(m.group(3)),
                parity=m.group(4),
                flow=float(m.group(5)),
                file=fn,
            )
            rows.append(cur)
            continue
        m = re.match(
            r"\s+D1\s+: sum \S+ \(control (\w+), relative difference (\S+)\); sum\|c\|/\|sum c\| = (\S+); positive part (\S+), negative part (\S+), ratio (\S+);",
            line,
        )
        if m and cur is not None:
            cur.update(
                ctl=m.group(1),
                cancel=float(m.group(3)),
                P=float(m.group(4)),
                N=float(m.group(5)),
                ratio=float(m.group(6)),
                reldiff=float(m.group(2)),
            )
            continue
        m = re.match(
            r"\s+signs of c in D1, D block: (\S+) \| D' block: (\S+)(?: \| D\(0\) row: (\S+))?", line
        )
        if m and cur is not None:
            cur.update(sD=m.group(1) + (m.group(3) or ""), sDp=m.group(2))
n = len(rows)
print("states", n, "with D1 data", sum(1 for r in rows if "ratio" in r))
assert n > 0 and all("ratio" in r and "sD" in r for r in rows), "incomplete parse"
print(
    "control relative differences: min %.3g, max %.3g"
    % (min(r["reldiff"] for r in rows), max(r["reldiff"] for r in rows))
)
print("controls not OK:", [(r["K"], r["A"]) for r in rows if r["ctl"] != "OK"])
print(
    "flow > 0 in",
    sum(1 for r in rows if r["flow"] > 0),
    "of",
    n,
    "; smallest flow",
    min(r["flow"] for r in rows),
)
print(
    "ratio P/N: min %.5f, max %.5f; cancellation sum|c|/|sum c|: min %.2f, max %.2f"
    % (
        min(r["ratio"] for r in rows),
        max(r["ratio"] for r in rows),
        min(r["cancel"] for r in rows),
        max(r["cancel"] for r in rows),
    )
)
exc = [(r["K"], r["A"], r["sD"], r["sDp"]) for r in rows if ("+" in r["sD"]) or ("-" in r["sDp"])]
print(
    "sign structure exceptions (a positive difference-row contribution or a negative D' contribution):",
    len(exc),
)
[print("   ", e) for e in exc]
byK = {}
for r in rows:
    byK.setdefault(r["K"], []).append(r["ratio"])
for K in sorted(byK):
    print(f"K {K}: {len(byK[K])} states, P/N from {min(byK[K]):.5f} to {max(byK[K]):.5f}")
