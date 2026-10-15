"""q1462 registration builder (9 Oct 2026, run ONCE before any row K >= 115 exists): writes q1462_series_table.json and puts its sha256 into the gate; refuses when a row K >= 115 exists."""

import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q1462_margin_series_verdict as G

rows = json.load(open(G.LIVE))
if any(r["K"] >= 115 for r in rows):
    print("REFUSED: a row K >= 115 exists")
    sys.exit(2)
tj = G.load_hashed(G.TRAIN, G.REG_TRAIN)
tab = G.build(tj["rows"])
tab["train_sha256"] = tj["sha256"]
tab["q1452_sha256"] = G.REG_Q1452
tab["sha256"] = G.sha_of(tab)
json.dump(tab, open(G.TABLE, "w"), indent=1)
src = open(os.path.join(HERE, "q1462_margin_series_verdict.py"), encoding="utf-8", newline="").read()
src = re.sub(r'REG_TABLE = "[^"]*"', f'REG_TABLE = "{tab["sha256"]}"', src)
open(os.path.join(HERE, "q1462_margin_series_verdict.py"), "w", encoding="utf-8", newline="").write(src)
print("table sha256", tab["sha256"])
for K in G.TARGET:
    print(K, "C", {q: round(tab["C"][str(K)][q], 9) for q in G.QUANT})
print("bars", {p: {q: float("%.3g" % tab["bar"][p][q]) for q in G.QUANT} for p in G.PARITY})
print("limits", {p: {q: round(tab["limit"][p][q], 5) for q in G.QUANT} for p in G.PARITY})
