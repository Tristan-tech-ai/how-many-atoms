"""q1606 (10 Oct 2026, HELPER, no verdict word): where the ended capacity chains of the block B131 stopped along their family, relative to the registered prediction of their birth. As q1495_birth_eta.py, it reads from the chain logs only the family position A of the lines '   A = <A>: residual ... [<t>s]' and the registered prediction table q1451 (stages), and prints positions and distances, never an event amplitude. Usage: py q1606_chain_end_check.py (from small_gap_q206/)."""

import json, re, os

tab = json.load(open("../centre_scripts/q1451_cap_b20_table.json"))["stages"]["134"]
PAT = re.compile(
    r"\s+A = ([0-9.]+): residual [0-9.e+-]+ in \d+ it, signal [^ ]+ ?,? ?h [0-9.e+-]+\s+\[(\d+)s\]"
)


def fam(fn):
    return [
        (float(m.group(1)), int(m.group(2)))
        for m in map(PAT.match, open(fn, encoding="utf8", errors="replace"))
        if m
    ]


print("table keys near the block:", sorted(k for k in tab if 128 <= int(k) <= 140))
for name, K in (("capc_alt_132", 132), ("capc_alt_134", 134), ("capc_alt_138b", 138), ("capc_alt_136b", 136)):
    fn = f"cap_logs/{name}.log"
    if not os.path.exists(fn):
        print(name, "no log")
        continue
    pts = fam(fn)
    if not pts:
        print(f"{name}: no family lines")
        continue
    last = pts[-1][0]
    first = pts[0][0]
    pred = float(tab[str(K)]) if str(K) in tab else float("nan")
    print(
        f"{name}: K_b {K}: {len(pts)} family lines, first A {first:.4f}, last A {last:.4f}, registered prediction {pred:.4f}, last A minus prediction {last - pred:+.4f}"
    )
print("HELPER q1606 done (no verdict word)")
