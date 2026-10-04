"""Figures 1-2 of paper 1 (rewritten 4 Oct 2026 for the figure gate: every plotted number is read by code from a primary file).
  fig_ring32.pdf: the certified 32-ring, agents/F_work/ring32_R2fix_polished.json (the state certified in ring32_R2fix_cert.log)
  fig_count.pdf : the count against eps at r_p = 1:
     leading-order series to K = 320: centre_scripts/q811/R1.0_K320_dps150.log
     count rule, K/2 truncation: amendment 1305 table (K 16-32) and centre_scripts/q811/k2conv/rp10_K*_flank.log (K 34-42)
     solved after registration: midpoints of the final brackets, amendments 1350 (K 30, 32), 1413 (K 34), 1428 (K 36)
(fig_segment was removed from the paper on 4 Oct; its script part is removed here as well.)
Run from this folder: py make_figs.py"""
import json, math, os, sys, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "gate_truth"))
from figsrc import rows, amend, amend_rows, jsonv, TW
from figstyle import apply, OK, WIDTH
apply()

# Figure 1: the certified 32-ring
d = jsonv("agents/F_work/ring32_R2fix_polished.json")
rp, rm = float(d["rp"]), float(d["rm"]); G = [float(x) for x in d["G"]]; m = [float(x) for x in d["m"]]
fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
tt = [2*math.pi*i/720 for i in range(721)]
ax.plot([rp*math.cos(t) for t in tt], [rm*math.sin(t) for t in tt], color="0.6", lw=0.6)
for g, w in zip(G, m):
    for t in (g, -g, math.pi - g, math.pi + g):
        ax.plot(rp*math.cos(t), rm*math.sin(t), "o", ms=2.0 + 60*w, color="k")
ax.plot([rp, -rp], [0, 0], "x", ms=6, color=OK["vermillion"], mew=1.0)
ax.annotate("no atom at $(\\pm r_p, 0)$", xy=(0.97*rp, 0), xytext=(0.0, 0.12), fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", lw=0.5))
ax.set_aspect("equal"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_ring32.pdf")); fig.savefig(os.path.join(HERE, "fig_ring32.png"), dpi=300); plt.close(fig)

# Figure 2: the count against eps at r_p = 1
ser = rows("centre_scripts/q811/R1.0_K320_dps150.log", r"K +(\d+): first in-regime crossing eps ([0-9.]+)")
k2 = [(K, e) for K, e in amend_rows(1305, r"^\| (\d+) \| (?:vertex[^|]*|flank \(8G, no X, no Y\)) \| (0\.\d+) \| [-\d.]+ / [-\d.]+ \| yes \|")]
for f in sorted(glob.glob(os.path.join(TW, "centre_scripts", "q811", "k2conv", "rp10_K*_flank.log"))):
    k2 += rows(os.path.relpath(f, TW), r"^q795 cap rp 1\.0 K (\d+) flank: lost at eps ([0-9.]+)")
k2 = sorted(set(k2))
mid = lambda a, b: (a + b) / 2
solved = [(30, mid(amend(1350, r"\| vertex thirty-ring \| 0\.\d+ \| \[(0\.\d+), "), amend(1350, r"\| vertex thirty-ring \| 0\.\d+ \| \[0\.\d+, (0\.\d+)\)"))),
          (32, mid(amend(1350, r"\| flank 32-ring \| 0\.\d+ \| \[(0\.\d+), "), amend(1350, r"\| flank 32-ring \| 0\.\d+ \| \[0\.\d+, (0\.\d+)\)"))),
          (34, mid(amend(1413, r"34-ring finished.*?eps \[(0\.\d+), "), amend(1413, r"34-ring finished.*?eps \[0\.\d+, (0\.\d+)\)"))),
          (36, mid(amend(1428, r"final bracket rm \(0\.\d+, 0\.\d+\], eps in \[(0\.\d+), "),
                   amend(1428, r"final bracket rm \(0\.\d+, 0\.\d+\], eps in \[0\.\d+, (0\.\d+)\)")))]
fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
ax.loglog([e for K, e in ser], [K for K, e in ser], "-", color="0.6", lw=0.8, label="leading-order series ($R = 1$)")
ax.loglog([e for K, e in k2], [K for K, e in k2], "s", ms=3, mfc="none", color=OK["blue"], label="count rule, $K/2$ truncation")
ax.loglog([e for K, e in solved], [K for K, e in solved], "o", ms=3.5, color=OK["vermillion"], label="solved after registration")
ee = [0.05, 0.1]; ax.loglog(ee, [1.2/e for e in ee], "--", color="k", lw=0.6); ax.text(0.06, 24, "$K\\propto\\epsilon^{-1}$", fontsize=8)
ee = [0.001, 0.025]; ax.loglog(ee, [6.4/math.sqrt(e) for e in ee], ":", color="k", lw=0.8); ax.text(0.0015, 230, "$K\\propto\\epsilon^{-1/2}$", fontsize=8)
ax.set_xlabel("$\\epsilon = r_p - r_m$"); ax.set_ylabel("number of atoms $K$"); ax.legend(fontsize=8, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_count.pdf")); fig.savefig(os.path.join(HERE, "fig_count.png"), dpi=300); plt.close(fig)
print("written: fig_ring32.pdf fig_count.pdf;", "k2", [K for K, _ in k2], "solved", [(K, round(e, 7)) for K, e in solved])
