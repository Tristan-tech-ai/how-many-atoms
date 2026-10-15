"""q1452 registration builder (9 Oct 2026, run ONCE before any row K >= 115 exists): writes the training snapshot and the table of q1452_margin_law_ext_verdict.py and puts their sha256 into the gate. Refuses to run when a row K >= 115 exists in the live file."""

import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q1452_margin_law_ext_verdict as G

rows = json.load(open(G.LIVE))
if any(r["K"] >= 115 for r in rows):
    print("REFUSED: a row K >= 115 exists")
    sys.exit(2)
train = sorted(
    ({"K": r["K"], "parity": r["parity"], "PN": r["PN"], "Re": r["Re"]} for r in rows if 83 <= r["K"] <= 114),
    key=lambda r: r["K"],
)
assert train and train[0]["K"] == 83 and train[-1]["K"] == 114 and len(train) == 32, (
    len(train),
    train[0]["K"],
    train[-1]["K"],
)
assert all(r["control_ok"] for r in rows if 83 <= r["K"] <= 114)
tj = {
    "note": "rows K = 83 to 114 of small_gap_q206/q1416_rows.json at registration, 9 Oct 2026",
    "rows": train,
}
tj["sha256"] = G.sha_of(tj)
json.dump(tj, open(G.TRAIN, "w"), indent=1)
tab = G.build(train)
tab["train_sha256"] = tj["sha256"]
tab["sha256"] = G.sha_of(tab)
json.dump(tab, open(G.TABLE, "w"), indent=1)
src = open(os.path.join(HERE, "q1452_margin_law_ext_verdict.py"), encoding="utf-8", newline="").read()
src = re.sub(r'REG_TRAIN = "[^"]*"', f'REG_TRAIN = "{tj["sha256"]}"', src)
src = re.sub(r'REG_TABLE = "[^"]*"', f'REG_TABLE = "{tab["sha256"]}"', src)
open(os.path.join(HERE, "q1452_margin_law_ext_verdict.py"), "w", encoding="utf-8", newline="").write(src)
print("training sha256", tj["sha256"])
print("table sha256", tab["sha256"])
for K in G.TARGET:
    print(
        K,
        "A",
        {q: round(tab["A"][str(K)][q], 7) for q in G.QUANT},
        "B",
        {q: round(tab["B"][str(K)][q], 7) for q in G.QUANT},
        "null",
        {q: round(tab["null"][str(K)][q], 7) for q in G.QUANT},
    )
print("bars A", {p: {q: float("%.3g" % tab["bar_A"][p][q]) for q in G.QUANT} for p in G.PARITY})
print("bars B", {p: {q: float("%.3g" % tab["bar_B"][p][q]) for q in G.QUANT} for p in G.PARITY})
print("free exponents p_B", {p: {q: round(tab["p_B"][p][q], 3) for q in G.QUANT} for p in G.PARITY})
