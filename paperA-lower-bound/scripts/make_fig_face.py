"""Fig. 1 of the paper: one computed optimal input (A = 29.85, K = 40 mass points) shown twice on the same axis.
Top: the mass points (position and mass). Bottom: the Gaussian bump w_j phi(y - x_j) of every mass point (grey), their sum, the output
density p (dark), and the flat level c* (dashed). The level c* is computed from the state itself: the Kuhn-Tucker conditions give
(phi * log p)(x_j) = log c* at every mass point x_j, so log c* is the integral of phi(y - x_j) log p(y) dy (trapezoid rule, step
0.002, double precision). The script prints this value at every mass point, its spread, and the largest |p/c* - 1| on the middle half
of the interval. Writes fig_face.pdf, fig_face.png and fig_face_numbers.json next to itself. Usage: python make_fig_face.py"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
from layout import find, state, FIGSTYLE_DIRS  # noqa: E402
sys.path[:0] = FIGSTYLE_DIRS
from figstyle import apply, OK, WIDTH

TARGET = 29.85
GREY = "#B8BEC6"
INK = "#1F2A36"


def resolve(f):
    for p in [q for q in (state(f),) if q]:
        if os.path.exists(p):
            return p
    return None


def load_state():
    D = json.load(open(find("count_law_data.json"), encoding="utf-8"))
    for A, K, f in sorted(D["states"], key=lambda s: abs(s[0] - TARGET)):
        p = resolve(f)
        if p is None:
            continue
        d = json.load(open(p, encoding="utf-8"))
        if d.get("verdict") == "OPTIMAL":
            return A, K, p, d
    print("VOID: no optimal state near the target"); sys.exit(2)


def phi(t):
    return np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi)


def main():
    A, K, path, d = load_state()
    u = [float(x) for x in d["u"]]; w = [float(x) for x in d["w"]]; we = float(d["we"])
    wc = None if d.get("wc") in (None, "None") else float(d["wc"])
    xs = [-A] + [-x for x in u[::-1]] + ([0.0] if wc is not None else []) + u + [A]
    ms = [we] + w[::-1] + ([wc] if wc is not None else []) + w + [we]
    xs, ms = np.array(xs), np.array(ms)
    if len(xs) != K or abs(ms.sum() - 1) > 1e-12:
        print(f"VOID: {path}: {len(xs)} points, total mass {ms.sum()}"); sys.exit(2)
    h = 0.002
    y = np.arange(-A - 12, A + 12 + h / 2, h)
    p = (ms[:, None] * phi(y[None, :] - xs[:, None])).sum(axis=0)
    logp = np.log(p)
    logc = np.array([h * np.sum(phi(y - x) * logp) for x in xs])
    cstar = float(np.exp(logc.mean()))
    spread = float(logc.max() - logc.min())
    bulk = np.abs(y) <= A / 2
    dev_bulk = float(np.max(np.abs(p[bulk] / cstar - 1)))
    bounds = (np.exp(-0.5) / (2 * A + np.sqrt(2 * np.pi)), 1 / (2 * A))
    print(f"state {os.path.basename(path)}: A {A} K {K} residual {d['residual']:.3e} total mass {ms.sum():.15f}")
    print(f"log c* at the {K} mass points: mean {logc.mean():.12f}, spread {spread:.3e}; c* = {cstar:.10f}; 1/(2A) = {1/(2*A):.10f}")
    print(f"Lemma 1 bounds hold: {bounds[0] <= cstar <= bounds[1]}; max |p/c* - 1| on |y| <= A/2: {dev_bulk:.3e}")
    print(f"smallest mass {ms.min():.6f} (centre), end mass {we:.6f}")

    apply()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(WIDTH, 2.75), sharex=True, gridspec_kw={"height_ratios": [1, 1.35], "hspace": 0.12})
    a1.vlines(xs, 0, ms, color=OK["blue"], lw=0.9)
    a1.plot(xs, ms, "o", ms=2.2, color=OK["blue"], mec="none")
    a1.set_ylim(0, 1.18 * ms.max()); a1.set_ylabel("mass")
    a1.text(0.5, 0.97, f"optimal input, $A={A:g}$, $K={K}$ mass points", transform=a1.transAxes, ha="center", va="top", fontsize=8)
    sel = (y > -A - 5) & (y < A + 5)
    for x, m in zip(xs, ms):
        a2.plot(y[sel], m * phi(y[sel] - x), color=GREY, lw=0.5)
    a2.plot(y[sel], p[sel], color=INK, lw=1.3)
    a2.axhline(cstar, color=OK["vermillion"], lw=0.8, ls=(0, (4, 2)))
    a2.axvspan(-A / 2, A / 2, color=OK["sky"], alpha=0.10, lw=0)
    a2.text(0, cstar * 0.93, "flat level $c^*$", color=OK["vermillion"], fontsize=8, ha="center", va="top")
    a2.text(-A / 2 + 0.5, cstar * 1.22, r"output density $p$ = sum of the bumps", color=INK, fontsize=8, ha="left", va="bottom")
    a2.set_ylim(0, 1.45 * cstar); a2.set_ylabel("density")
    a2.set_xlim(-A - 5, A + 5); a2.set_xlabel("position $y$")
    a2.set_xticks([-A, -A / 2, 0, A / 2, A]); a2.set_xticklabels(["$-A$", "$-A/2$", "0", "$A/2$", "$A$"])
    fig.savefig(os.path.join(HERE, "fig_face.pdf")); fig.savefig(os.path.join(HERE, "fig_face.png"), dpi=220)
    json.dump({"A": A, "K": K, "file": os.path.basename(path), "residual": d["residual"], "cstar": cstar, "logc_spread": spread,
               "max_dev_central_half": dev_bulk, "one_over_2A": 1 / (2 * A), "smallest_mass": float(ms.min()), "end_mass": we},
              open(os.path.join(HERE, "fig_face_numbers.json"), "w", encoding="utf-8"), indent=1)
    print("fig_face written")


if __name__ == "__main__":
    main()
