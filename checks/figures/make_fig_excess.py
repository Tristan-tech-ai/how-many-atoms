"""Fig. 6 of paper 1: excess of the LFP effective strip of the d-ball over the one-dimensional delta, at exact centre-event states.
Offsets delta_impl = 2 j_{d/2-1,1}/sqrt(d - B) - m, as printed by centre_scripts/q874_offset_readout.py; every value is read here
from its log by code (gate_truth/figsrc.py):
  d = 2: centre_scripts/q868_runs/q874_d2_check.log (amendment 1604 (2))
  d = 3: centre_scripts/q868_runs/q874_d3_1590.log (1601)
  d = 4: gate_truth/fig_data/q874_d4_early.log (q874 rerun on chain_d4.log, 4 Oct 19:2x) + q868_runs/q874_d4_1593.log (1603)
  d = 5: gate_truth/fig_data/q874_d5_early.log (q874 rerun on chain_d5.log) + q868_runs/q874_d5_1597.log (1604 (1))
delta: the constant D of q874_offset_readout.py (3.34776, the record's value, 729; 754 (4)). Writes fig_excess.pdf and .png."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "gate_truth"))
from figsrc import rows, num
from figstyle import apply, OK, WIDTH

apply()
EV = r"^e\s*\d+ (?:birth|split)\s+m ([\d.]+) deficit [\d.]+ offset ([\d.]+)"
Q = "centre_scripts/q868_runs/"
SRC = {2: [Q + "q874_d2_check.log"], 3: [Q + "q874_d3_1590.log"],
       4: ["gate_truth/fig_data/q874_d4_early.log", Q + "q874_d4_1593.log"],
       5: ["gate_truth/fig_data/q874_d5_early.log", Q + "q874_d5_1597.log"]}
DELTA = num("centre_scripts/q874_offset_readout.py", r"\nD = ([\d.]+)")
STYLE = {2: ("o", OK["blue"]), 3: ("s", OK["vermillion"]), 4: ("^", OK["green"]), 5: ("D", OK["purple"])}
fig, ax = plt.subplots(figsize=(WIDTH, 2.7))
for d in (2, 3, 4, 5):
    pts = sorted({p for f in SRC[d] for p in rows(f, EV)})
    m = [a for a, _ in pts]; e = [b - DELTA for _, b in pts]
    ax.plot(m, e, marker=STYLE[d][0], ms=3.0, lw=0.8, color=STYLE[d][1], label=f"$d = {d}$")
ax.axhline(0.0, color="0.6", lw=0.6, ls=":")
ax.text(8.6, 0.003, r"one-dimensional $\delta$", fontsize=8, color="0.35")
ax.set_xlim(3.4, 14.6); ax.set_ylim(-0.005, 0.125)
ax.set_xlabel(r"ball radius $m$"); ax.set_ylabel(r"strip excess over $\delta$")
ax.legend(title="dimension of the ball", ncol=4, loc="upper center", handlelength=1.6, columnspacing=1.0, title_fontsize=8)
fig.savefig(os.path.join(HERE, "fig_excess.pdf")); fig.savefig(os.path.join(HERE, "fig_excess.png"), dpi=300)
print("written fig_excess")
