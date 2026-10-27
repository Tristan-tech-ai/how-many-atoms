"""Fig. 2 of the paper: the plan of the proof of Theorem 1, drawn with the true formulas at A = 1000 (a schematic).
Top: the depth log(1/eta(x)), of order A^(2/3) (1 - x^2/A^2), of the flatness of the output density (Step 1). Middle: the windows of
length 2 A^(2/3) that tile the bulk (Section V); the number of dots in a window is proportional to the square root of the depth at its
center (Step 2), scaled so that the fullest window shows 12, while the true counts are of order A. Bottom: about A^(1/3) windows of
order A points each. Writes fig_map.pdf and fig_map.png next to itself. Usage: python make_fig_map.py"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
from layout import find, state, FIGSTYLE_DIRS  # noqa: E402
sys.path[:0] = FIGSTYLE_DIRS
from figstyle import apply, OK, WIDTH

A = 1000.0
C0 = 0.44          # below c0max = 0.441296728... (Appendix C); only the shape matters here
DELTA_W = 0.06     # windows kept where u_j >= delta_w, as in Section V
MAXDOTS = 12
INK = "#1F2A36"


def main():
    Lw = A ** (2 / 3)
    x = np.linspace(-A, A, 801)
    depth = C0 * A ** (2 / 3) * (1 - x ** 2 / A ** 2) / 3
    centres = np.arange(0, A, 2 * Lw + 1)
    centres = np.concatenate([-centres[:0:-1], centres])
    kept = [c for c in centres if 1 - ((abs(c) + Lw) / A) ** 2 >= DELTA_W]
    print(f"A {A}, L_w = A^(2/3) = {Lw:.1f}, windows kept {len(kept)} of {len(centres)}, A^(1/3) = {A ** (1 / 3):.1f}")
    apply()
    fig = plt.figure(figsize=(WIDTH, 2.75))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 0.8, 0.55], hspace=0.10)
    a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1], sharex=a1); a3 = fig.add_subplot(gs[2])
    a1.fill_between(x, 0, depth, color=OK["sky"], alpha=0.35, lw=0)
    a1.plot(x, depth, color=OK["blue"], lw=1.1)
    for c in kept:
        a1.plot([c, c], [0, C0 * A ** (2 / 3) * (1 - c ** 2 / A ** 2) / 3], color="#8C96A3", lw=0.5, ls=(0, (2, 2)))
    a1.set_ylim(0, depth.max() * 1.45); a1.set_yticks([])
    a1.spines["left"].set_visible(False)
    a1.tick_params(axis="x", labelbottom=False, length=0)
    a1.text(-A, depth.max() * 1.42, r"Step 1: $p$ is flat up to $\eta(x)$, with $\log\frac{1}{\eta}\asymp A^{2/3}\left(1-\frac{x^2}{A^2}\right)$",
            fontsize=7.5, va="top", ha="left", color=INK)
    rng = np.random.default_rng(7)
    dmax = max(np.sqrt(1 - c ** 2 / A ** 2) for c in kept)
    for c in centres:
        keep = c in kept
        g = 0.06 * Lw   # drawn gap between windows (schematic; the true gap has length 1)
        a2.add_patch(Rectangle((c - Lw + g, 0.12), 2 * Lw - 2 * g, 0.76, fc="white" if keep else "#F1F2F4",
                               ec=OK["blue"] if keep else "#C9CDD3", lw=0.7))
        if keep:
            n = max(1, int(round(MAXDOTS * np.sqrt(1 - c ** 2 / A ** 2) / dmax)))
            cols = 4
            for k in range(n):
                r, q = divmod(k, cols)
                a2.plot(c - Lw + g + (q + 0.5) * (2 * Lw - 2 * g) / cols, 0.72 - 0.17 * r, "o", ms=2.0,
                        color=OK["blue"], mec="none")
            a2.text(c, 0.17, str(n), fontsize=6.5, ha="center", va="bottom", color="#5B6673")
    a2.set_ylim(0, 1.25); a2.set_yticks([]); a2.spines["left"].set_visible(False)
    a2.set_xlim(-A * 1.02, A * 1.02)
    a2.set_xticks([-A, 0, A]); a2.set_xticklabels(["$-A$", "0", "$A$"])
    i = len(centres) // 2
    a2.annotate("", xy=(centres[i] - Lw, 1.05), xytext=(centres[i] + Lw, 1.05),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=INK, shrinkA=0, shrinkB=0))
    a2.text(centres[i], 1.08, r"$2A^{2/3}$", fontsize=7, ha="center", va="bottom", color=INK)
    a3.axis("off")
    a3.text(0.0, 0.30, r"Step 2: each window $W_j$ holds $\gtrsim \frac{l}{\pi}\sqrt{2\log(1/\eta_j)}\asymp A$ points;"
            "\n" r"about $A^{1/3}$ windows, so $K(A)\gtrsim A\cdot A^{1/3}=A^{4/3}$.",
            fontsize=7.5, ha="left", va="center", color=INK, transform=a3.transAxes)
    fig.savefig(os.path.join(HERE, "fig_map.pdf")); fig.savefig(os.path.join(HERE, "fig_map.png"), dpi=220)
    print("fig_map written")


if __name__ == "__main__":
    main()
