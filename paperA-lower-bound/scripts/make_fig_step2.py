"""Fig. 4 of the paper: the idea of Theorem 2 on a window of a computed optimum. The window [-l, l] with l = 2 at the
center of the optimum at A = 29.85 holds k = 4 mass points. Top: the mass points and the output density p. Bottom: the nonnegative
trigonometric polynomial Q(theta) = prod_j 4 sin^2(omega0 (theta - x_j)/2), omega0 = pi/l, which vanishes at every mass point of
the window, scaled to mean one. Prints the points, Q at the points, the ratio of int Q p to c* int Q, the relative L2 error eta of p on
the window and the leading count (l/pi) sqrt(2 log(1/eta)). Writes fig_step2.pdf, fig_step2.png and fig_step2_numbers.json.
Usage: python make_fig_step2.py"""
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
from make_fig_face import load_state, phi

L = 2.0
INK = "#1F2A36"


def main():
    A, K, path, d = load_state()
    u = [float(x) for x in d["u"]]; w = [float(x) for x in d["w"]]; we = float(d["we"])
    wc = None if d.get("wc") in (None, "None") else float(d["wc"])
    xs = np.array([-A] + [-x for x in u[::-1]] + ([0.0] if wc is not None else []) + u + [A])
    ms = np.array([we] + w[::-1] + ([wc] if wc is not None else []) + w + [we])
    h = 0.002
    y = np.arange(-A - 12, A + 12 + h / 2, h)
    p = (ms[:, None] * phi(y[None, :] - xs[:, None])).sum(axis=0)
    cstar = float(np.exp(np.mean([h * np.sum(phi(y - x) * np.log(p)) for x in xs])))
    inW = np.abs(xs) <= L
    xw, mw = xs[inW], ms[inW]
    om = np.pi / L
    th = np.linspace(-L, L, 4001)
    Q = np.prod([4 * np.sin(om * (th - x) / 2) ** 2 for x in xw], axis=0)
    Q /= Q.mean()
    pw = (ms[:, None] * phi(th[None, :] - xs[:, None])).sum(axis=0)
    ratio = float(np.trapezoid(Q * pw, th) / (cstar * np.trapezoid(Q, th)))
    qat = [float(np.prod([4 * np.sin(om * (a - x) / 2) ** 2 for x in xw])) for a in xw]
    print(f"state {os.path.basename(path)}: A {A} K {K}; window [-{L}, {L}] holds k = {len(xw)} points at {np.round(xw, 6).tolist()}")
    print(f"masses {np.round(mw, 6).tolist()}; Q at the points {qat}; int Q p / (c* int Q) = {ratio:.6f}")
    eta = float(np.sqrt(np.trapezoid((pw - cstar) ** 2, th) / (cstar ** 2 * 2 * L)))
    lead = L / np.pi * np.sqrt(2 * np.log(1 / eta))
    print(f"relative L2 error on W: eta = {eta:.3e}; leading count (l/pi) sqrt(2 log(1/eta)) = {lead:.4f} <= k = {len(xw)}")
    apply()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(WIDTH, 2.35), sharex=True, gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})
    sel = np.abs(y) <= L + 0.6
    a1.plot(y[sel], p[sel] / cstar, color=INK, lw=1.2)
    a1.vlines(xw, 0, mw / mw.max() * 0.8, color=OK["blue"], lw=1.0)
    a1.plot(xw, mw / mw.max() * 0.8, "o", ms=2.6, color=OK["blue"], mec="none")
    a1.set_ylim(0, 1.3); a1.set_yticks([0, 1]); a1.set_yticklabels(["0", "$c^*$"])
    a1.text(-L - 0.55, 1.08, "output density $p$", fontsize=7.5, color=INK, va="bottom")
    a1.text(L + 0.55, 0.62, "mass\npoints", fontsize=7.5, color=OK["blue"], ha="right", va="center")
    a2.fill_between(th, 0, Q, color=OK["vermillion"], alpha=0.15, lw=0)
    a2.plot(th, Q, color=OK["vermillion"], lw=1.1)
    for x in xw:
        a2.axvline(x, color=OK["blue"], lw=0.5, ls=(0, (2, 2)))
        a1.axvline(x, color=OK["blue"], lw=0.5, ls=(0, (2, 2)))
    a2.set_ylim(0, Q.max() * 1.42); a2.set_yticks([0])
    a2.text(-L - 0.55, Q.max() * 1.38, "test function $Q\\geq 0$, zero at every mass point", fontsize=7.5, color=OK["vermillion"],
            va="top", zorder=5, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    for a in (a1, a2):
        a.axvspan(-L - 0.6, -L, color="#EEF0F3", lw=0); a.axvspan(L, L + 0.6, color="#EEF0F3", lw=0)
    a2.set_xlim(-L - 0.6, L + 0.6)
    a2.set_xticks([-L, 0, L]); a2.set_xticklabels(["$-l$", "0", "$l$"]); a2.set_xlabel("position in the window $W$")
    fig.savefig(os.path.join(HERE, "fig_step2.pdf")); fig.savefig(os.path.join(HERE, "fig_step2.png"), dpi=220)
    json.dump({"A": A, "K": K, "l": L, "k": int(len(xw)), "points": xw.tolist(), "ratio": ratio, "eta": eta, "leading_count": float(lead)},
              open(os.path.join(HERE, "fig_step2_numbers.json"), "w", encoding="utf-8"), indent=1)
    print("fig_step2 written")


if __name__ == "__main__":
    main()
