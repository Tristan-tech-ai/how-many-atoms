"""Fig. 3 of the paper: where the published argument for Theorem 4 of [1] breaks, drawn on the certified ring near the
right end of the major axis (true positions, horizontal direction stretched). The support points of largest abscissa are the pair at
x-bar = cos(2.5864 deg) = 0.998981, x2 = +-0.043307; on their horizontal lines every point with x1 > x-bar lies outside the ellipse.
On the line x2 = 0 the ellipse contains points with x-bar < x1 <= 1, but (x-bar, 0) is not a support point.
Writes fig_gap.pdf and fig_gap.png next to itself. Usage: python make_fig_gap.py"""
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

INK = "#1F2A36"
d = json.load(open(find("ring32_polished.json"), encoding="utf-8"))
g = float(d["G"][0]); w = float(d["m"][0])
rp, rm = float(d["rp"]), float(d["rm"])
xb, yb = rp * math.cos(g), rm * math.sin(g)
X0, X1, Y0, Y1 = 0.9972, 1.0008, -0.07, 0.07
apply()
fig, ax = plt.subplots(figsize=(WIDTH, 2.2))
ys = [Y0 + (Y1 - Y0) * i / 400 for i in range(401)]
bx = [rp * math.sqrt(max(0.0, 1 - (y / rm) ** 2)) for y in ys]
ax.fill_betweenx(ys, X0, bx, color="#F3F5F8", lw=0)
ax.fill_betweenx(ys, bx, X1, color="#E3E6EA", lw=0)
ax.plot(bx, ys, color="#9AA3AE", lw=0.6)
ax.axvline(xb, color="#9AA3AE", lw=0.6, ls=(0, (3, 2)))
for s in (1, -1):
    ax.annotate("", xy=(X1 - 0.0002, s * yb), xytext=(xb, s * yb),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=INK, shrinkA=4, shrinkB=0))
ax.plot([xb, rp], [0, 0], color=OK["vermillion"], lw=2.0, solid_capstyle="butt")
ax.scatter([xb, xb], [yb, -yb], s=900 * w, color=OK["blue"], lw=0, zorder=4)
ax.scatter([xb], [0], s=14, facecolor="white", edgecolor=INK, lw=0.8, zorder=4)
ax.scatter([rp], [0], s=22, facecolor="white", edgecolor=OK["vermillion"], lw=1.0, zorder=5)
ax.text(X1 - 0.00005, yb + 0.006, "no feasible point to the right", fontsize=7, color=INK, ha="right", va="bottom")
ax.text(xb - 0.00008, yb, "support point", fontsize=7, color=OK["blue"], ha="right", va="center")
ax.text(xb - 0.00008, 0.0, "$(\\bar x,0)$: not a support point", fontsize=7, color=INK, ha="right", va="center")
ax.text((xb + rp) / 2, -0.012, "feasible", fontsize=7, color=OK["vermillion"], ha="center", va="top")
ax.text(xb, Y0 + 0.003, "$x_1=\\bar x$", fontsize=7, color="#6B7480", ha="right", va="bottom")
ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
ax.set_xticks([0.998, xb, 1.0]); ax.set_xticklabels(["0.998", "", "1"])
ax.set_yticks([-yb, 0, yb]); ax.set_yticklabels(["$-0.0433$", "0", "0.0433"])
ax.set_xlabel("$x_1$ (stretched)"); ax.set_ylabel("$x_2$")
fig.savefig(os.path.join(HERE, "fig_gap.pdf")); fig.savefig(os.path.join(HERE, "fig_gap.png"), dpi=220)
print(f"x-bar {xb:.6f}, x2 {yb:.6f}, mass {w:.6f}")
