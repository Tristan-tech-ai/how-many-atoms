"""Fig. of the merged paper: the real positive roots t_k of the margin polynomial N1(t) at four stored states, and the eigenvalues 1/t_k.
Every value is parsed by code from the outputs of q1087 (paper_births/figures/figsrc.py, copied here as figsrc_births.py):
scratchpad_archive_729e03fd/q1087_results_q220_gauss_K65_A43.5803_out.txt and the three K 71 files (first, median and last state).
Writes fig_roots.pdf and .png next to this script."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(os.path.dirname(HERE)))   # temporal_within: the paths of figsrc_births are relative to it
sys.path.insert(0, HERE)
import figsrc_births as F
from figstyle import apply, OK, WIDTH

apply()
A = F.ARCH
files = [("$K=65$, $A=43.5803$", A + "/q1087_results_q220_gauss_K65_A43.5803_out.txt", "o", OK["blue"]),
         ("$K=71$, first state", A + "/q1087_results_q220_gauss_K71_A46.4359_out.txt", "s", OK["vermillion"]),
         ("$K=71$, median state", A + "/q1087_results_q220_gauss_K71_A46.6889_out.txt", "v", OK["green"]),
         ("$K=71$, last state", A + "/q1087_results_q220_gauss_K71_A46.9439_out.txt", "^", OK["purple"])]
fig, ax = plt.subplots(2, 1, figsize=(WIDTH, 4.3))
for lab, f, mk, col in files:
    ts, _ = F.roots(f)
    ax[0].semilogy(np.arange(1, len(ts) + 1), ts, mk + "-", ms=3, lw=0.7, color=col, label=lab)
    ax[1].plot(np.arange(1, 9), [1 / t for t in ts[:8]], mk + "-", ms=3, lw=0.7, color=col, label=lab)
ax[0].axhline(1.0, color="k", lw=0.5); ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].set_xlabel("index $k$ of the root"); ax[0].set_ylabel("root $t_k$"); ax[0].legend(loc="upper left")
ax[1].axhline(1.0, color="k", lw=0.5, ls="--")
ax[1].set_xlabel("index $k$"); ax[1].set_ylabel("reciprocal $1/t_k$"); ax[1].legend(loc="upper right")
# 15 Oct 2026 (REFPASS): panel letters, which the caption uses
ax[0].text(0.97, 0.06, "(a)", transform=ax[0].transAxes, ha="right", va="bottom")
ax[1].text(0.03, 0.06, "(b)", transform=ax[1].transAxes, ha="left", va="bottom")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_roots.pdf")); fig.savefig(os.path.join(HERE, "fig_roots.png"), dpi=200)
print("fig_roots written")
