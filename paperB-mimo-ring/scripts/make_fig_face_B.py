"""Fig. 1 of the paper: the certified ring of 32 mass points on the ellipse with semi-axes 1 and 0.9597, marker area
proportional to the mass, with a zoom on the right end of the major axis (horizontal direction stretched): the nearest pair at
t = +-2.5864 degrees and the end point (1, 0), which carries no mass. Data: the center of the certified box. Writes fig_face_B.pdf
and fig_face_B.png next to itself and prints the numbers used. Usage: python make_fig_face_B.py"""
import json
import math
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
from layout import find, state, FIGSTYLE_DIRS  # noqa: E402
sys.path[:0] = FIGSTYLE_DIRS
from figstyle import apply, OK, WIDTH

INK = "#1F2A36"
d = json.load(open(find("ring32_polished.json"), encoding="utf-8"))
G = [float(g) for g in d["G"]]; m = [float(x) for x in d["m"]]
rp, rm = float(d["rp"]), float(d["rm"])
pts = [(rp * math.cos(s), rm * math.sin(s), w) for g, w in zip(G, m) for s in (g, -g, math.pi - g, math.pi + g)]
SCALE = 900.0                      # marker area in points^2 per unit mass
apply()
fig, ax = plt.subplots(figsize=(WIDTH, 3.25))
tt = [2 * math.pi * i / 1440 for i in range(1441)]
ax.fill([rp * math.cos(t) for t in tt], [rm * math.sin(t) for t in tt], color="#F3F5F8", lw=0)
ax.plot([rp * math.cos(t) for t in tt], [rm * math.sin(t) for t in tt], color="#9AA3AE", lw=0.6)
ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=[SCALE * p[2] for p in pts], color=OK["blue"], lw=0, zorder=3)
for x in (rp, -rp):
    ax.scatter([x], [0], s=22, facecolor="white", edgecolor=OK["vermillion"], lw=1.0, zorder=4)
ax.set_aspect("equal"); ax.set_xlim(-1.12, 1.12); ax.set_ylim(-1.05, 1.05)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_xticks([-1, 0, 1]); ax.set_yticks([-0.9597, 0, 0.9597]); ax.set_yticklabels(["$-0.9597$", "0", "$0.9597$"])
# zoom on the right end of the major axis, drawn inside the ellipse
X0, X1, Y0, Y1 = 0.9962, 1.0010, -0.075, 0.075
ins = ax.inset_axes([-0.62, -0.50, 1.04, 1.00], transform=ax.transData)
ys = [Y0 + (Y1 - Y0) * i / 400 for i in range(401)]
ins.fill_betweenx(ys, [rp * math.sqrt(max(0.0, 1 - (y / rm) ** 2)) for y in ys], X1, color="#E3E6EA", lw=0)
ins.plot([rp * math.sqrt(max(0.0, 1 - (y / rm) ** 2)) for y in ys], ys, color="#9AA3AE", lw=0.6)
near = [p for p in pts if p[0] > X0 and Y0 < p[1] < Y1]
ins.scatter([p[0] for p in near], [p[1] for p in near], s=[SCALE * p[2] for p in near], color=OK["blue"], lw=0, zorder=3)
ins.scatter([rp], [0], s=26, facecolor="white", edgecolor=OK["vermillion"], lw=1.0, zorder=4)
ins.text(rp - 0.00025, 0.0, "(1, 0): no mass", fontsize=7, color=OK["vermillion"], ha="right", va="center")
p1 = max(near, key=lambda p: p[1])
ins.text(p1[0] - 0.00025, p1[1], "$t=2.5864^\\circ$", fontsize=7, color=OK["blue"], ha="right", va="center")
ins.text(X1 - 0.0001, Y0 + 0.006, "outside", fontsize=6.5, color="#6B7480", ha="right", va="bottom")
ins.set_xlim(X0, X1); ins.set_ylim(Y0, Y1); ins.set_facecolor("#F3F5F8")
ins.set_xticks([0.997, 1.0]); ins.set_xticklabels(["0.997", "1"], fontsize=6.5)
ins.set_yticks([-0.05, 0, 0.05]); ins.set_yticklabels(["$-0.05$", "0", "0.05"], fontsize=6.5)
for s in ("top", "right"):
    ins.spines[s].set_visible(True)
for s in ins.spines.values():
    s.set_linewidth(0.6); s.set_color("#6B7480")
ax.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fill=False, ec="#6B7480", lw=0.6, zorder=5))
for (xa, ya), (xb, yb) in (((X0, Y1), (0.0, 1.0)), ((X0, Y0), (0.0, 0.0))):
    ax.add_artist(ConnectionPatch(xyA=(xa, ya), coordsA=ax.transData, xyB=(xb, yb), coordsB=ins.transAxes, color="#9AA3AE", lw=0.5))
fig.savefig(os.path.join(HERE, "fig_face_B.pdf")); fig.savefig(os.path.join(HERE, "fig_face_B.png"), dpi=220)
print(f"points {len(pts)}, total mass {sum(p[2] for p in pts):.12f}, nearest pair at x1 = {p1[0]:.6f}, x2 = +-{p1[1]:.6f}, "
      f"t = {math.degrees(G[0]):.4f} deg, mass {m[0]:.6f}")
