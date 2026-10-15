"""make_bindings_unified.py (10 Oct 2026): writes bindings_unified.json for the merged paper (paper name unified_flat).
Part 1: every binding of paper 1 (bindings_paper1*.json) is copied with the paper name changed. The binding keeps its printed number,
its context substring and its formula over primary sources, so it only fires where the text of paper 1 is unchanged.
Part 2: bindings for the new sections are read from unified_new_src.json (written by hand or by script, one record per number).
Usage: py make_bindings_unified.py"""
import os, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
out = []
for p in sorted(glob.glob(os.path.join(HERE, "bindings_paper1*.json"))):
    for b in json.load(open(p, encoding="utf-8")):
        b = dict(b); b["paper"] = "unified_flat"; b["from"] = os.path.basename(p); out.append(b)
new = os.path.join(HERE, "unified_new_src.json")
if os.path.exists(new):
    for b in json.load(open(new, encoding="utf-8")):
        b = dict(b); b["paper"] = "unified_flat"; out.append(b)
json.dump(out, open(os.path.join(HERE, "bindings_unified.json"), "w", encoding="utf-8"), indent=1)
print("bindings_unified.json:", len(out), "records")
