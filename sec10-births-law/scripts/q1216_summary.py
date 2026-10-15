"""Summary of q1216_out.txt (reading, no verdict): maxima of |dev(16)| by K_b range and the K_b sets where L1 or L2 is False."""

import re

rows = {"capacity": {}, "npmle": {}}
for line in open("q1216_out.txt"):
    m = re.match(
        r"(capacity|npmle) K_b (\d+): A_stored ([\d.]+); dev\(12\) ([-+\d.]+)e-9, dev\(16\) ([-+\d.]+)e-9 \(bar ([\d.]+)e-9\) L1 (\w+); n_opt (\d+) dev\(opt\) ([-+\d.]+)e-9 \(bar ([\d.]+)e-9\) L1' (\w+); L2 (\w+)",
        line,
    )
    if m:
        g = m.groups()
        rows[g[0]][int(g[1])] = dict(
            d12=float(g[3]),
            d16=float(g[4]),
            bar=float(g[5]),
            L1=g[6] == "True",
            L1o=g[10] == "True",
            L2=g[11] == "True",
        )
for name, r in rows.items():
    print(name, "events", len(r), "K_b range", min(r), max(r))
    for lo, hi in ((10, 23), (24, 39), (40, 59), (6, 28), (29, 35), (36, 52)):
        s = [abs(v["d16"]) for k, v in r.items() if lo <= k <= hi]
        if s:
            print(
                f"  K_b {lo}-{hi}: events {len(s)}, max |dev16| {max(s):.4f}e-9, median {sorted(s)[len(s)//2]:.4f}e-9, positive {sum(v['d16'] > 0 for k, v in r.items() if lo <= k <= hi)}"
            )
    print("  L1 False at K_b:", [k for k, v in sorted(r.items()) if not v["L1"]])
    print("  L2 False at K_b:", [k for k, v in sorted(r.items()) if not v["L2"]])
    print("  L1' False at K_b:", [k for k, v in sorted(r.items()) if not v["L1o"]])
    s = [(k, v["d16"] / v["bar"]) for k, v in sorted(r.items()) if 10 <= k <= 28]
    print("  dev16/bar, K_b 10-28:", ", ".join(f"{k}: {x:.2f}" for k, x in s))
