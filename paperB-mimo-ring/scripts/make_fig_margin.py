"""Fig. 2 of the paper: C - D along the boundary parameter t in degrees on a log axis, from margin_along_boundary.json.
The eight mass points of the quarter are the valleys, where C - D = 0 (drawn down to the floor of the axis); the end of the major axis
t = 0 is marked. Writes fig_margin.pdf, fig_margin.png and fig_margin_numbers.json next to itself. Usage: python make_fig_margin.py"""
import json
import math
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

FLOOR = 1e-58
INK = "#1F2A36"
d = json.load(open(os.path.join(HERE, "margin_along_boundary.json")))
rows = d["rows"]
tdeg = [math.degrees(r[0]) for r in rows]
gap = [max(-r[1], FLOOR) for r in rows]
apply()
fig, ax = plt.subplots(figsize=(WIDTH, 2.2))
ax.fill_between(tdeg, FLOOR, gap, color=OK["sky"], alpha=0.25, lw=0)
ax.semilogy(tdeg, gap, color=OK["blue"], lw=0.9)
for g in d["mass_point_t"]:
    ax.plot([math.degrees(g)], [FLOOR * 1.6], marker="^", ms=3.5, color=OK["blue"], mec="none", clip_on=False)
v0 = -d["vertex_t0"]
ax.plot([0], [v0], "o", ms=4, color=OK["vermillion"], mec="none", clip_on=False, zorder=5)
ax.annotate("end of the major axis:\n$C-D=1.13\\times10^{-50}$", xy=(0.3, v0), xytext=(12, 3e-47), fontsize=7.5,
            color=OK["vermillion"], arrowprops=dict(arrowstyle="-", lw=0.5, color=OK["vermillion"]))
ax.text(46, 2.5e-57, "mass points (32 on the whole ellipse; 8 per quarter)", fontsize=7, color=OK["blue"], ha="center", bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
ax.set_xlim(0, 90); ax.set_ylim(FLOOR, 1e-45)
ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
ax.set_xlabel("boundary parameter $t$ (degrees)"); ax.set_ylabel("$C-D_\\pi(\\cos\\,t,\\;0.9597\\,\\sin\\,t)$")
fig.savefig(os.path.join(HERE, "fig_margin.pdf")); fig.savefig(os.path.join(HERE, "fig_margin.png"), dpi=220)
nums = {"vertex_gap": v0, "largest_gap": -d["min_on_grid"], "smallest_gap_away_from_zones": -d["max_on_grid_away"],
        "n_points": d["n_points"], "bits": d["bits"], "h": d["h"], "Y": d["Y"], "max_abs_at_mass_points": d["max_abs_at_mass_points"]}
json.dump(nums, open(os.path.join(HERE, "fig_margin_numbers.json"), "w"), indent=1)
print(json.dumps(nums, indent=1))
