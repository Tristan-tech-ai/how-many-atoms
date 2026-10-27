"""Fig. 3 of the paper: Step 1 in computed optima. For the stored optima nearest to A = 17, 29.85, 40.95 and 52, the
relative deviation |p(y)/c* - 1| of the output density from its flat level, on a log scale, against y/A. The level c* is computed
from each state as in make_fig_face.py. The deviation oscillates with the spacing of the mass points, so the curve drawn is its
maximum over a sliding window of length one. Double precision; values below 1e-12 are not drawn.
Writes fig_flat.pdf, fig_flat.png and fig_flat_numbers.json next to itself. Usage: python make_fig_flat.py"""
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

TARGETS = [17.0, 29.85, 40.95, 52.0]
FLOOR = 1e-12


def resolve(f):
    for p in [q for q in (state(f),) if q]:
        if os.path.exists(p):
            return p
    return None


def phi(t):
    return np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi)


def atoms(d, A):
    u = [float(x) for x in d["u"]]; w = [float(x) for x in d["w"]]; we = float(d["we"])
    wc = None if d.get("wc") in (None, "None") else float(d["wc"])
    xs = [-A] + [-x for x in u[::-1]] + ([0.0] if wc is not None else []) + u + [A]
    ms = [we] + w[::-1] + ([wc] if wc is not None else []) + w + [we]
    return np.array(xs), np.array(ms)


def main():
    D = json.load(open(find("count_law_data.json"), encoding="utf-8"))
    rows = []
    used = set()
    for t in TARGETS:
        for A, K, f in sorted(D["states"], key=lambda s: abs(s[0] - t)):
            p = resolve(f)
            if p is None or p in used:
                continue
            d = json.load(open(p, encoding="utf-8"))
            if d.get("verdict") != "OPTIMAL":
                continue
            used.add(p)
            xs, ms = atoms(d, A)
            if len(xs) != K:
                print(f"VOID: {p}"); sys.exit(2)
            h = 0.002
            y = np.arange(-A - 12, A + 12 + h / 2, h)
            pd = (ms[:, None] * phi(y[None, :] - xs[:, None])).sum(axis=0)
            logc = np.array([h * np.sum(phi(y - x) * np.log(pd)) for x in xs])
            cstar = float(np.exp(logc.mean()))
            dev = np.abs(pd / cstar - 1)
            n = int(round(1.0 / h))
            env = np.array([dev[max(0, i - n // 2): i + n // 2 + 1].max() for i in range(0, len(y), 25)])
            ye = y[::25]
            keep = np.abs(ye) <= A
            rows.append((A, K, p, ye[keep] / A, env[keep], float(logc.max() - logc.min()), cstar,
                         float(env[keep][np.argmin(np.abs(ye[keep]))])))
            print(f"A {A} K {K} {os.path.basename(p)} residual {d['residual']:.2e} logc spread {logc.max()-logc.min():.2e} "
                  f"deviation envelope at the centre {rows[-1][7]:.3e}, at |y| = A/2 "
                  f"{float(env[keep][np.argmin(np.abs(np.abs(ye[keep]) - A/2))]):.3e}")
            break
    apply()
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    shades = ["#9BB7D4", "#5E8DBF", "#2F649F", OK["blue"]]
    for (A, K, p, x, e, *_), c in zip(rows, shades):
        m = e > FLOOR
        ax.semilogy(x[m], e[m], color=c, lw=1.0)
        i = np.argmin(np.abs(x + 0.5))
        lab = f"{A:.2f}".rstrip("0").rstrip(".")
        ax.text(-0.5, e[i], f"$A={lab}$", color=c, fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))
    ax.set_xlim(-1, 1); ax.set_ylim(1e-9, 2)
    ax.set_xlabel("scaled position $y/A$"); ax.set_ylabel(r"$|p(y)/c^*-1|$")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    fig.savefig(os.path.join(HERE, "fig_flat.pdf")); fig.savefig(os.path.join(HERE, "fig_flat.png"), dpi=220)
    json.dump([{"A": r[0], "K": r[1], "file": os.path.basename(r[2]), "logc_spread": r[5], "cstar": r[6], "centre_envelope": r[7]}
               for r in rows], open(os.path.join(HERE, "fig_flat_numbers.json"), "w", encoding="utf-8"), indent=1)
    print("fig_flat written")


if __name__ == "__main__":
    main()
