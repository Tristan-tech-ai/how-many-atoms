"""Run one figure script and record, by code, what it reads and what it draws.

Usage: py gate_figs_runner.py SCRIPT.py OUT.json
Records every file opened for reading (builtins.open / io.open) and, at every Figure.savefig, the content of each axes: axis
labels, limits and scales, every Line2D (data, colour, line style, marker, label), every text in data coordinates, and the legend
entries. Nothing is judged here; gate_figs.py reads OUT.json and applies the checks.
"""
import sys, os, io, json, builtins, runpy

script, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
reads, figs = [], {}
_open = builtins.open


def rec_open(file, mode="r", *a, **k):
    if isinstance(file, (str, bytes, os.PathLike)) and not any(c in str(mode) for c in "wax+"):
        reads.append(os.path.abspath(os.fsdecode(file)))
    return _open(file, mode, *a, **k)


builtins.open = rec_open
io.open = rec_open

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
from matplotlib.colors import to_hex
import numpy as np


def hexc(c):
    try:
        return to_hex(c, keep_alpha=False)
    except Exception:
        return str(c)


def floats(v):
    a = np.asarray(v, dtype=float).ravel()
    return [float(x) for x in a]


def describe(fig):
    axes = []
    for ax in fig.axes:
        lines = []
        for ln in ax.get_lines():
            try:
                x, y = floats(ln.get_xdata()), floats(ln.get_ydata())
            except Exception:
                continue
            lines.append({"x": x, "y": y, "color": hexc(ln.get_color()), "ls": str(ln.get_linestyle()), "marker": str(ln.get_marker()),
                          "mfc": hexc(ln.get_markerfacecolor()), "mec": hexc(ln.get_markeredgecolor()), "label": str(ln.get_label()),
                          "data_coords": bool(ln.get_transform() == ax.transData or ln.get_transform().contains_branch(ax.transData))})
        texts = []
        for t in ax.texts:
            d = {"text": t.get_text(), "data_coords": bool(t.get_transform() == ax.transData)}
            d["x"], d["y"] = [float(v) for v in t.get_position()]
            if hasattr(t, "xy"):                                    # annotation: the point it marks
                d["xy"] = [float(v) for v in t.xy]
                d["data_coords"] = True
            texts.append(d)
        leg = ax.get_legend()
        legend = []
        if leg is not None:
            hs = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", [])
            for txt, h in zip(leg.get_texts(), hs):
                e = {"text": txt.get_text()}
                for k, f in (("ls", "get_linestyle"), ("marker", "get_marker"), ("color", "get_color"), ("mfc", "get_markerfacecolor"),
                             ("mec", "get_markeredgecolor")):
                    if hasattr(h, f):
                        v = getattr(h, f)()
                        e[k] = hexc(v) if k in ("color", "mfc", "mec") else str(v)
                legend.append(e)
        axes.append({"xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(), "xlim": [float(v) for v in ax.get_xlim()],
                     "ylim": [float(v) for v in ax.get_ylim()], "xscale": ax.get_xscale(), "yscale": ax.get_yscale(),
                     "aspect_equal": ax.get_aspect() == 1.0, "lines": lines, "texts": texts, "legend": legend,
                     "n_collections": len(ax.collections)})
    return axes


_save = matplotlib.figure.Figure.savefig


def rec_save(self, fname, *a, **k):
    name = os.path.basename(os.fsdecode(fname))
    figs[name] = describe(self)
    return _save(self, fname, *a, **k)


matplotlib.figure.Figure.savefig = rec_save
os.chdir(os.path.dirname(script))
sys.path.insert(0, os.path.dirname(script))
err = None
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit:
    pass
except Exception as e:
    err = f"{type(e).__name__}: {e}"
builtins.open = _open
txt = json.dumps({"script": script, "error": err, "reads": sorted(set(reads)), "figures": figs}, indent=1, default=str)
_open(out, "w", encoding="utf-8").write(txt)
print("captured", os.path.basename(script), "figures:", sorted(figs), "error:", err)
