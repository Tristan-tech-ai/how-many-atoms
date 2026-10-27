"""Table I of the paper, read from the center of the certified box (ring32_R2fix_polished.json; the same values as
sec5-certificate/results/rerun_ring32_polished.json). Writes table_ring.tex (the parameters t_j in radians and the masses, rounded to
20 decimals with exact decimal arithmetic), numbers_ring.json (smallest mass, its orbit and angle, largest radius, the capacity to
30 digits, the number of mass points) and a plain ring figure fig_ring.pdf. Usage: python make_ring_table_fig.py"""
import json
import math
import os
import sys
from decimal import Decimal, ROUND_HALF_EVEN, getcontext
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
from layout import find, state, FIGSTYLE_DIRS  # noqa: E402
sys.path[:0] = FIGSTYLE_DIRS
from figstyle import apply, OK, WIDTH

getcontext().prec = 120
d = json.load(open(find("ring32_polished.json"), encoding="utf-8"))
G = [Decimal(str(x)) for x in d["G"]]; m = [Decimal(str(x)) for x in d["m"]]
rp, rm = float(d["rp"]), float(d["rm"])
q = Decimal(10) ** -20
rows = []
for i, (g, w) in enumerate(zip(G, m), 1):
    rows.append("G%d & %s & %s \\\\" % (i, g.quantize(q, ROUND_HALF_EVEN), w.quantize(q, ROUND_HALF_EVEN)))
tab = "\n".join([
    "\\begin{table}[t]",
    "\\caption{The certified ring at $r_p=1$, $r_m=0.9597$: one representative of each orbit of four mass points, with its parameter $t_j$ and its mass. Values are the midpoints of the certified box, rounded to $20$ decimals.}\\label{tab:ring}",
    "\\centering\\footnotesize",
    "\\begin{tabular}{@{}ccc@{}}",
    "\\hline",
    "orbit & $t_j$ (rad) & mass of each mass point \\\\",
    "\\hline",
    *rows,
    "\\hline",
    "\\multicolumn{3}{l}{Mass points at $(\\cos t,0.9597\\sin t)$ with $t=\\pm t_j$ and $t=\\pi\\pm t_j$.}",
    "\\end{tabular}",
    "\\end{table}"])
open(os.path.join(HERE, "table_ring.tex"), "w", encoding="utf-8").write(tab + "\n")
r = [math.hypot(rp * math.cos(float(g)), rm * math.sin(float(g))) for g in G]
imin = min(range(len(m)), key=lambda i: m[i])
nums = {"n_points": 4 * len(G), "n_orbits": len(G), "total_mass": float(4 * sum(m)), "min_mass": float(m[imin]), "min_orbit": imin + 1,
        "min_angle_deg": math.degrees(float(G[imin])), "max_radius": max(r), "C": str(d["C"])[:32], "rm": float(d["rm"])}
json.dump(nums, open(os.path.join(HERE, "numbers_ring.json"), "w", encoding="utf-8"), indent=1)
apply()
fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
tt = [2 * math.pi * i / 720 for i in range(721)]
ax.plot([rp * math.cos(t) for t in tt], [rm * math.sin(t) for t in tt], color="0.6", lw=0.6)
for g, w in zip(G, m):
    for t in (float(g), -float(g), math.pi - float(g), math.pi + float(g)):
        ax.plot(rp * math.cos(t), rm * math.sin(t), "o", ms=2.0 + 60 * float(w), color="k")
ax.plot([rp, -rp], [0, 0], "x", ms=6, color=OK["vermillion"], mew=1.0)
ax.annotate("no mass point at $(\\pm1, 0)$", xy=(0.97 * rp, 0), xytext=(0.0, 0.12), fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", lw=0.5))
ax.set_aspect("equal"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_ring.pdf")); fig.savefig(os.path.join(HERE, "fig_ring.png"), dpi=200)
print(json.dumps(nums))
