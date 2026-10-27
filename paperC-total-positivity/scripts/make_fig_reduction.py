"""Fig. 2 of the paper: the two quantities of Proposition 1 along the families of Fig. 1. Top: the mass w0 of the center
point along every family with odd K. Bottom: the innermost positive position u1 along every family with even K >= 4. A count could
decrease only if such a curve fell to zero at the right end of a family; one such ending is drawn dashed for illustration (the curve
of one family mirrored in A). Data: q1382_states_to15p8.json; checked numerically. Prints, per family, the first and last values and
whether the curve increases on the grid. Writes fig_reduction.pdf, fig_reduction.png and fig_reduction_numbers.json.
Usage: python make_fig_reduction.py"""
import json
import os
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
d = json.load(open(find("q1382_states_to15p8.json"), encoding="utf-8"))
S = sorted(d["states"], key=lambda s: s["A"])
fams = []
for s in S:
    if not fams or fams[-1][0]["K"] != s["K"]:
        fams.append([])
    fams[-1].append(s)
apply()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(WIDTH, 2.7), sharex=True, gridspec_kw={"hspace": 0.12})
rows = []
for f in fams:
    K = f[0]["K"]; A = [s["A"] for s in f]
    if f[0]["centre"]:
        v = [s["wc"] for s in f]; ax = a1; col = OK["vermillion"]
    elif K >= 4:
        v = [s["u"][0] for s in f]; ax = a2; col = OK["blue"]
    else:
        continue
    inc = all(b > a for a, b in zip(v, v[1:]))
    rows.append({"K": K, "A_start": A[0], "A_end": A[-1], "first": v[0], "last": v[-1], "increasing_on_grid": inc, "states": len(f)})
    ax.plot(A, v, color=col, lw=1.1)
mid = [f for f in fams if f[0]["centre"] and f[0]["K"] == 11][0]
A = [s["A"] for s in mid]; v = [s["wc"] for s in mid]
a1.plot([A[0] + A[-1] - a for a in A], v, color="#9AA3AE", lw=0.9, ls=(0, (3, 2)))
a1.text(A[-1] + 0.05, max(v) * 1.02, "excluded ending", fontsize=7, color="#6B7480", ha="left", va="bottom")
a1.set_ylabel("center mass $w_0$"); a2.set_ylabel("innermost $u_1$")
a1.text(0.98, 0.95, "odd $K$: (S1) $w_0$ grows", transform=a1.transAxes, fontsize=7.5, color=OK["vermillion"], va="top", ha="right")
a2.text(0.98, 0.95, "even $K$: (S2) $u_1$ grows", transform=a2.transAxes, fontsize=7.5, color=OK["blue"], va="top", ha="right")
a1.set_ylim(0, None); a2.set_ylim(0, None)
a2.set_xlim(0, S[-1]["A"]); a2.set_xlabel("amplitude $A$")
fig.savefig(os.path.join(HERE, "fig_reduction.pdf")); fig.savefig(os.path.join(HERE, "fig_reduction.png"), dpi=220)
out = {"families": rows, "all_increasing": all(r["increasing_on_grid"] for r in rows), "n_families": len(rows),
       "largest_first_value": max(r["first"] for r in rows)}
json.dump(out, open(os.path.join(HERE, "fig_reduction_numbers.json"), "w"), indent=1)
print({k: v for k, v in out.items() if k != "families"})
for r in rows:
    print(r)
