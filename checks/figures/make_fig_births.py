"""Fig. 5 of paper 1: centre-birth curves r_m*(r_p) of the filled ellipse from the formal density, three problems. Every value is
read here by code (gate_truth/figsrc.py):
  capacity: centre_scripts/q816_curve_*.log, q816_rp28.log, q816_rp3.log ("rp X: centre birth predicted at rm = Y")
  LFP:      centre_scripts/births/lfp_rp*.log and q827_rp17_M6.log
  NPMLE (Gaussian source s = 2): centre_scripts/births/rd_rp*.log and q829_rp21_M8.log
  disc points (f = 1 exact): the disc thresholds of the registration log (amendment 1447: capacity, LFP, rd)
  open symbols: registered births observed on rings, gate_truth/births_check.json (zeros of the ring tables of 1406, 1443, 1447)
Writes fig_births.pdf and .png next to this script."""
import os, sys, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "gate_truth"))
from figsrc import rows, amend, jsonv, TW
from figstyle import apply, OK, WIDTH

apply()
CB = r"rp ([\d.]+): centre birth predicted at rm = ([\d.]+)"
RES = r" \(eps [\d.]+\); residual ([\d.e+-]+)"


def curve(files, rx, disc):
    """points (rp, rm*) whose solve converged: residual printed in the same log line at most 1e-8"""
    pts = {(a, b) for f in files for a, b, res in rows(os.path.relpath(f, TW), rx + RES) if res <= 1e-8}
    return sorted(pts | {(disc, disc)})


C = "centre_scripts/"
DISC = r"Disc thresholds: capacity (\d+\.\d+), LFP (\d+\.\d+), rd \(s 2\) (\d+\.\d+)"
cap = curve(glob.glob(os.path.join(TW, C, "q816_curve_*.log")) + [os.path.join(TW, C, "q816_rp28.log"), os.path.join(TW, C, "q816_rp3.log")],
            r"^" + CB, amend(1447, DISC, 1))
lfp = curve(glob.glob(os.path.join(TW, C, "births", "lfp_rp*.log")) + [os.path.join(TW, C, "q827_rp17_M6.log")],
            r"^LFP " + CB, amend(1447, DISC, 2))
npm = curve(glob.glob(os.path.join(TW, C, "births", "rd_rp*.log")) + [os.path.join(TW, C, "q829_rp21_M8.log")],
            r"^RD s 2\.0 " + CB, amend(1447, DISC, 3))
BC = "gate_truth/births_check.json"
obs = {lab: (jsonv(BC, key, "rp"), jsonv(BC, key, "observed")) for lab, key in
       (("capacity", "capacity ellipse 2.8"), ("LFP", "LFP ellipse 1.7"), ("NPMLE", "NPMLE ellipse 2.1"))}
fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
lo = 1.25; hi = 3.2
ax.plot([lo, hi], [lo, hi], color="0.7", lw=0.6, ls=":")
ax.text(1.93, 2.02, "disc", fontsize=8, color="0.35", rotation=40)
for data, lab, mk, col in ((cap, "capacity", "o", OK["blue"]), (lfp, "LFP", "s", OK["vermillion"]),
                           (npm, "NPMLE", "^", OK["green"])):
    x = [a for a, _ in data]; y = [b for _, b in data]
    ax.plot(x, y, marker=mk, ms=3.0, lw=1.0, color=col, label=lab)
    bx, by = obs[lab]
    ax.plot([bx], [by], marker=mk, ms=8, mfc="none", mec="k", mew=0.8, ls="none")
ax.plot([], [], marker="o", ms=8, mfc="none", mec="k", mew=0.8, ls="none", label="registered, observed")
ax.set_xlim(lo, hi); ax.set_ylim(lo, 2.6)
ax.set_xlabel(r"semi-major axis $r_p$"); ax.set_ylabel(r"semi-minor axis $r_m^\star$ at the birth")
ax.legend(loc="upper left")
fig.savefig(os.path.join(HERE, "fig_births.pdf")); fig.savefig(os.path.join(HERE, "fig_births.png"), dpi=300)
print("written fig_births", len(cap), len(lfp), len(npm))
