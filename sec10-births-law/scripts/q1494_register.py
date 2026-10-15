"""q1494 registration builder (9 Oct 2026, run ONCE before any row K >= 123 exists): snapshots the rows K = 83 to 122 of small_gap_q206/q1416_rows.json into q1494_training_rows.json, writes q1494_series_table.json, and puts both sha256 into the gate; refuses when a row K >= 123 exists, when a control of a row failed or when the rows are not exactly K = 83 to 122."""

import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q1494_margin_series_ext_verdict as G

rows = json.load(open(G.LIVE))
if any(r["K"] >= 123 for r in rows):
    print("REFUSED: a row K >= 123 exists")
    sys.exit(2)
if sorted(r["K"] for r in rows) != list(range(83, 123)):
    print("REFUSED: the rows are not exactly K = 83 to 122")
    sys.exit(2)
if any(r.get("control_ok") is not True for r in rows):
    print("REFUSED: a row has a failed control")
    sys.exit(2)
tr = {
    "note": "rows K = 83 to 122 of small_gap_q206/q1416_rows.json (PN, Re, parity), 40 rows, snapshot for the gate q1494 (9 Oct 2026); the rows 115 to 122 were the blind rows of q1462",
    "rows": [
        {"K": r["K"], "parity": r["parity"], "PN": float(r["PN"]), "Re": float(r["Re"])}
        for r in sorted(rows, key=lambda r: r["K"])
    ],
}
tr["sha256"] = G.sha_of(tr)
json.dump(tr, open(G.TRAIN, "w"), indent=1)
tab = G.build(tr["rows"])
tab["train_sha256"] = tr["sha256"]
tab["sha256"] = G.sha_of(tab)
json.dump(tab, open(G.TABLE, "w"), indent=1)
p = os.path.join(HERE, "q1494_margin_series_ext_verdict.py")
src = open(p, encoding="utf-8", newline="").read()
src, nsub = re.subn(
    r"REG_TRAIN = [^;\n]*; REG_TABLE = [^\n]*",
    f'REG_TRAIN = "{tr["sha256"]}"; REG_TABLE = "{tab["sha256"]}"',
    src,
)
if nsub != 1:
    print("REFUSED: the line with REG_TRAIN and REG_TABLE was not found exactly once")
    sys.exit(2)
open(p, "w", encoding="utf-8", newline="").write(src)
print("training sha256", tr["sha256"])
print("table sha256", tab["sha256"])
for K in G.TARGET:
    print(K, "C", {q: round(tab["C"][str(K)][q], 9) for q in G.QUANT})
print("bars", {p_: {q: float("%.3g" % tab["bar"][p_][q]) for q in G.QUANT} for p_ in G.PARITY})
print("limits", {p_: {q: round(tab["limit"][p_][q], 5) for q in G.QUANT} for p_ in G.PARITY})
