"""q1446 registration builder (9 Oct 2026, run ONCE before any row K >= 109 exists): writes the training snapshot and the table of q1446_margin_law_verdict.py and puts their sha256 into the gate. Refuses to run when a row K >= 109 exists in the live file."""

import hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q1446_margin_law_verdict as G

rows = json.load(open(G.LIVE))
if any(r["K"] >= 109 for r in rows):
    print("REFUSED: a row K >= 109 exists")
    sys.exit(2)
train = sorted(
    ({"K": r["K"], "parity": r["parity"], "PN": r["PN"], "Re": r["Re"]} for r in rows if r["K"] >= 83),
    key=lambda r: r["K"],
)
assert train and train[0]["K"] == 83 and train[-1]["K"] == 108 and len(train) == 26, (
    len(train),
    train[0]["K"],
    train[-1]["K"],
)
tj = {
    "note": "rows K = 83 to 108 of small_gap_q206/q1416_rows.json at registration, 9 Oct 2026",
    "rows": train,
}
tj["sha256"] = G.sha_of(tj)
json.dump(tj, open(G.TRAIN, "w"), indent=1)
tab = G.build(train)
tab["exponent"] = "2/3"
tab["null_exponent"] = 1.0
tab["train_sha256"] = tj["sha256"]
tab["sha256"] = G.sha_of(tab)
json.dump(tab, open(G.TABLE, "w"), indent=1)
src = open(os.path.join(HERE, "q1446_margin_law_verdict.py"), encoding="utf-8", newline="").read()
src = re.sub(r'REG_TRAIN = "[^"]*"', f'REG_TRAIN = "{tj["sha256"]}"', src)
src = re.sub(r'REG_TABLE = "[^"]*"', f'REG_TABLE = "{tab["sha256"]}"', src)
open(os.path.join(HERE, "q1446_margin_law_verdict.py"), "w", encoding="utf-8", newline="").write(src)
print("training sha256", tj["sha256"])
print("table sha256", tab["sha256"])
for K in G.TARGET:
    print(
        K,
        {q: round(tab["pred"][str(K)][q], 7) for q in G.QUANT},
        "null",
        {q: round(tab["null"][str(K)][q], 7) for q in G.QUANT},
    )
print("bars", {par: {q: float("%.3g" % tab["bar"][par][q]) for q in G.QUANT} for par in G.PARITY})
