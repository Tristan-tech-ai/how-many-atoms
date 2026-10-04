"""Figure gate: pure code, no judgement. For every figure a paper includes (\\includegraphics{figs/NAME.pdf}):
  F1 regenerated  the figure is written by a script in the paper's figs/ folder during this run (gate_figs_runner.py), so the PDF
                  the paper uses is a function of the current data
  F2 no typed data the script holds no literal collection with three or more decimal numbers (a line marked
                  "# fig-gate: selection" may name which series of a loaded file are drawn)
  F3 provenance   every file the script reads is a primary source of truth_index.pkl (machine output, registration log, literature
                  text, or code read for a parameter) or a gate builder output in gate_truth/
  F4 labelled     every data series is identified: a legend entry, a text placed at it, a colour shared with an identified series
                  plus a legend entry for its line style, a legend entry for its marker style, or it is the only series of its axes
  F5 visible      no point of a data series lies outside the axes limits
  F6 axes         every axes that shows data has an x and a y label
Guides (lines with at most two points, or grey lines) are not data series; they are listed. Writes report_figs.md and
report_figs.json; exit code 1 if any check fails.
Run: py gate_figs.py ../paper_onelaw/paper1.tex ../paper_neuron/paper2.tex
"""
import os, re, sys, ast, json, glob, math, pickle, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(HERE)
ROOT = os.path.dirname(TW)
IDX = pickle.load(open(os.path.join(HERE, "truth_index.pkl"), "rb"))
OK_KINDS = {"PRIMARY-OUT", "PRIMARY-LOG", "LIT", "CODE"}
PRIMARY = {os.path.normcase(os.path.abspath(s[2])) for s in IDX["sources"] if s[0] in OK_KINDS}
SECONDARY = {os.path.normcase(os.path.abspath(s[2])) for s in IDX["sources"] if s[0] == "SECONDARY"}
NEAR = 0.12                          # a text within 12 % of the axes span of a series end identifies it


def literal_floats(node):
    n = 0
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, float):
            n += 1
    return n


def typed_data(path):
    src = open(path, encoding="utf-8").read()
    lines = src.split("\n")
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            k = literal_floats(node)
            if k >= 3 and "fig-gate: selection" not in lines[node.lineno - 1]:
                out.append((node.lineno, k, lines[node.lineno - 1].strip()[:90]))
    # keep the outermost literal of each line only
    seen, res = set(), []
    for ln, k, txt in sorted(out, key=lambda t: (t[0], -t[1])):
        if ln not in seen:
            seen.add(ln)
            res.append((ln, k, txt))
    return res


def allowed_read(p):
    q = os.path.normcase(os.path.abspath(p))
    if not q.startswith(os.path.normcase(ROOT)):
        return True, "outside the research folder (library or font)"
    if q.endswith(".py"):
        return True, "script"
    if q in PRIMARY:
        return True, "primary"
    g = os.path.normcase(HERE)
    if q.startswith(g) and (q.endswith(".json") or os.sep + "fig_data" + os.sep in q):
        return True, "gate builder output"
    if q in SECONDARY:
        return False, "SECONDARY source"
    return False, "not a primary source"


def grey(c):
    c = c.lstrip("#")
    if len(c) < 6:
        return False
    r, g, b = c[0:2], c[2:4], c[4:6]
    return r == g == b and c[:6] != "000000"


def norm_xy(ax, x, y):
    def f(v, scale, lim):
        lo, hi = lim
        if scale == "log":
            if v <= 0 or lo <= 0:
                return float("nan")
            v, lo, hi = math.log10(v), math.log10(lo), math.log10(hi)
        return (v - lo) / (hi - lo) if hi != lo else 0.0
    return f(x, ax["xscale"], ax["xlim"]), f(y, ax["yscale"], ax["ylim"])


def series_of(ax):
    data, guides = [], []
    groups = {}
    for ln in ax["lines"]:
        n = len(ln["x"])
        if n == 0:
            continue                                         # legend proxy
        if n == 2 or grey(ln["color"]) and n > 1:
            guides.append(ln)
            continue
        if n == 1:
            key = (ln["marker"], ln["mfc"], ln["mec"], ln["color"], ln["label"] if not ln["label"].startswith("_") else "")
            g = groups.setdefault(key, dict(ln, x=[], y=[], single=True))
            g["x"] += ln["x"]; g["y"] += ln["y"]
            continue
        data.append(dict(ln, single=False))
    return data + list(groups.values()), guides


def identified(ax, s, ident, data):
    legend_txt = {e["text"] for e in ax["legend"]}
    if s["label"] and not s["label"].startswith("_") and s["label"] in legend_txt:
        return "legend"
    pts = list(zip(s["x"], s["y"]))
    ends = pts if s["single"] else [pts[0], pts[-1]]
    for t in ax["texts"]:
        if not t.get("data_coords"):
            continue
        for tx, ty in [t.get("xy", (t["x"], t["y"])), (t["x"], t["y"])]:
            a = norm_xy(ax, tx, ty)
            for ex, ey in ends:
                b = norm_xy(ax, ex, ey)
                if all(map(math.isfinite, a + b)) and math.hypot(a[0] - b[0], a[1] - b[1]) < NEAR:
                    return f"text '{t['text'][:30]}'"
    if not s["single"]:
        if any(o is not s and o["color"] == s["color"] and o in ident for o in data) and \
                any(e.get("ls") == s["ls"] and e.get("marker") in ("None", "none", "") for e in ax["legend"]):
            return "colour of an identified series + legend line style"
    else:
        if any(e.get("mfc") == s["mfc"] and e.get("mec") == s["mec"] for e in ax["legend"]):
            return "legend marker style"
    if len(data) == 1:
        return "only series of its axes (identified by the caption)"
    return None


def check_fig(name, axes):
    issues, notes = [], []
    for i, ax in enumerate(axes):
        data, guides = series_of(ax)
        if not data:
            continue
        if not ax["xlabel"].strip() or not ax["ylabel"].strip():
            issues.append(f"F6 axes {i}: missing axis label (x '{ax['xlabel']}', y '{ax['ylabel']}')")
        ident = []
        for _ in range(3):                                   # propagate the colour rule
            for s in data:
                if s not in ident and identified(ax, s, ident, data):
                    ident.append(s)
        for s in data:
            how = identified(ax, s, ident, data)
            desc = f"axes {i} series ({len(s['x'])} pts, colour {s['color']}, ls {s['ls']}, marker {s['marker']}, label '{s['label']}')"
            if how:
                notes.append(f"{desc}: {how}")
            else:
                issues.append(f"F4 {desc}: not identified by legend, text or caption rule")
            out = 0
            for x, y in zip(s["x"], s["y"]):
                a = norm_xy(ax, x, y)
                if not all(math.isfinite(v) for v in a) or min(a) < -1e-9 or max(a) > 1 + 1e-9:
                    out += 1
            if out:
                issues.append(f"F5 {desc}: {out} point(s) outside the axes limits")
        for g in guides:
            notes.append(f"axes {i} guide ({len(g['x'])} pts, colour {g['color']}, ls {g['ls']})")
    return issues, notes


def main(papers):
    rows, nfail = [], 0
    for tex in papers:
        figdir = os.path.join(os.path.dirname(os.path.abspath(tex)), "figs")
        used = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figs/([^}]+?)(?:\.pdf)?\}", open(tex, encoding="utf-8").read())
        caps, typed = {}, {}
        for sc in sorted(glob.glob(os.path.join(figdir, "*.py"))):
            if os.path.basename(sc) == "figstyle.py":
                continue
            outj = os.path.join(HERE, "fig_capture_" + os.path.basename(sc).replace(".py", ".json"))
            r = subprocess.run([sys.executable, os.path.join(HERE, "gate_figs_runner.py"), sc, outj], capture_output=True, text=True)
            cap = json.load(open(outj, encoding="utf-8"))
            caps[sc] = cap
            typed[sc] = typed_data(sc)
        for name in used:
            fn = name + ".pdf"
            src = [sc for sc, c in caps.items() if fn in c["figures"]]
            row = {"paper": os.path.basename(tex), "figure": fn, "script": os.path.basename(src[0]) if src else None, "issues": [], "notes": []}
            if not src:
                row["issues"].append("F1 no figure script in figs/ writes this file in this run")
            else:
                sc = src[0]; cap = caps[sc]
                if cap["error"]:
                    row["issues"].append(f"F1 script error: {cap['error']}")
                for ln, k, txt in typed[sc]:
                    row["issues"].append(f"F2 typed data, line {ln} ({k} decimals): {txt}")
                for p in cap["reads"]:
                    ok, why = allowed_read(p)
                    if not ok:
                        row["issues"].append(f"F3 reads {os.path.relpath(p, ROOT)}: {why}")
                    elif why in ("primary", "gate builder output"):
                        row["notes"].append(f"reads {os.path.relpath(p, ROOT)} ({why})")
                iss, nts = check_fig(fn, cap["figures"][fn])
                row["issues"] += iss; row["notes"] += nts
            nfail += bool(row["issues"])
            rows.append(row)
    md = ["# Figure gate report", "", "Checks F1-F6 of gate_figs.py; no judgement is involved.", ""]
    for r in rows:
        md.append(f"## {r['paper']}: {r['figure']} ({r['script']}) - {'FAIL' if r['issues'] else 'PASS'}")
        md += [f"- FAIL {i}" for i in r["issues"]] + [f"- {n}" for n in r["notes"]] + [""]
    open(os.path.join(HERE, "report_figs.md"), "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    json.dump(rows, open(os.path.join(HERE, "report_figs.json"), "w", encoding="utf-8"), indent=1)
    for r in rows:
        print(f"{r['paper']:11s} {r['figure']:16s} {'FAIL' if r['issues'] else 'PASS'}" + "".join(f"\n      {i}" for i in r["issues"]))
    print("FAIL" if nfail else "PASS", nfail, "figures with issues")
    return nfail


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1:]) else 0)
