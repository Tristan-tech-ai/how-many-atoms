"""Fig. 4 of the paper: the ratio Sigma+/Sigma- of Observation 2 (the positive over the negative part of the velocity of
the innermost unknown, in the formulation of Section VII) against K. Data: the 47 states with K from 19 to 81 of q1389_out_all.txt,
q1389_out_gap.txt and q1389_out_big.txt, and the 48 rows with K from 83 to 130 of q1416_rows.json whose control passed (field PN).
Checked numerically. Writes fig_margin_C.pdf, fig_margin_C.png and fig_margin_C_numbers.json. Usage: python make_fig_margin_C.py"""
import json
import os
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
from layout import find, state, FIGSTYLE_DIRS  # noqa: E402
sys.path[:0] = FIGSTYLE_DIRS
from figstyle import apply, OK, WIDTH

INK = "#1F2A36"
rows = []
for fn in ("q1389_out_all.txt", "q1389_out_gap.txt", "q1389_out_big.txt"):
    cur = None
    for line in open(find(fn), encoding="utf-8"):
        m = re.match(r"K (\d+) (\S+) A (\S+) \((odd|even), \S+\), stored-system flow (\S+), J", line)
        if m:
            cur = {"K": int(m.group(1)), "A": float(m.group(3))}; rows.append(cur); continue
        m = re.match(r"\s+D1\s+: sum \S+ \(control (\w+), relative difference (\S+)\); sum\|c\|/\|sum c\| = (\S+); positive part (\S+), negative part (\S+), ratio (\S+);", line)
        if m and cur is not None:
            cur["ratio"] = float(m.group(6))
small = [r for r in rows if "ratio" in r]
big = [r for r in json.load(open(find("q1416_rows.json"))) if 83 <= r["K"] <= 130 and r.get("control_ok") is True]
q = open(find("q1496_out.txt"), encoding="utf-8").read()
lim = float(re.search(r"smallest limit: [\d.]+ ([\d.]+)", q).group(1))
print(f"states K 19-81: {len(small)} (ratio {min(r['ratio'] for r in small):.5f} to {max(r['ratio'] for r in small):.5f}); "
      f"rows K 83-130: {len(big)} (smallest {min(r['PN'] for r in big):.6f}); limit {lim}")
if len(small) != 47 or len(big) != 48:
    print("VOID: counts differ from the paper"); sys.exit(2)
apply()
fig, ax = plt.subplots(figsize=(WIDTH, 2.1))
ax.axhline(1.0, color="#9AA3AE", lw=0.8)
ax.plot([r["K"] for r in small], [r["ratio"] for r in small], "o", ms=2.6, color=OK["blue"], mec="none")
ax.plot([r["K"] for r in big], [r["PN"] for r in big], "s", ms=2.2, color=INK, mec="none")
ax.text(131, 1.0 + 0.002, "ratio 1: no room", fontsize=7, color="#6B7480", ha="right", va="bottom")
ax.text(24, 1.097, "stored states", fontsize=7, color=OK["blue"], ha="left", va="bottom")
ax.text(84, 1.062, "one state per family", fontsize=7, color=INK, ha="left", va="bottom")
ax.set_xlim(15, 132); ax.set_ylim(0.99, 1.11)
ax.set_xlabel("number of mass points $K$"); ax.set_ylabel(r"$\Sigma_+/\Sigma_-$")
fig.savefig(os.path.join(HERE, "fig_margin_C.pdf")); fig.savefig(os.path.join(HERE, "fig_margin_C.png"), dpi=220)
json.dump({"n_small": len(small), "n_big": len(big), "small_min": min(r["ratio"] for r in small),
           "small_max": max(r["ratio"] for r in small), "big_min": min(r["PN"] for r in big), "limit": lim},
          open(os.path.join(HERE, "fig_margin_C_numbers.json"), "w"), indent=1)
print("fig_margin_C written")
