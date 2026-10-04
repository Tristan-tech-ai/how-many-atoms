"""Q493: the 631(6) forward test on the MULTIPRECISION transitions of q492 (results_q492_lfp_mpcont.json), against the
frozen predictions (lfp_forward_predictions.json, read only, restored per amendment 634). Also lists, for each
transition q486 located in double precision, the multiprecision value and the difference (q492's pre-registered (C):
within 1e-5). Rule as in q488: PASS for the law if its max |error| over the located transitions in (18.7, 30] is <= 0.04
and the m ln m fit's exceeds twice the law's; FAIL if the law's exceeds 0.08."""

import json

fp = json.load(open("lfp_forward_predictions.json"))
pred = {p["K_before"]: p for p in fp["predictions"]}
assert fp["written"] == "2026-09-23 04:47:51"
mp = json.load(open("results_q492_lfp_mpcont.json"))
trm = mp["transitions"]
dbl = {t[1]: t[0] for t in json.load(open("results_q486_lfp_sym.json"))["transitions"]}
print(
    f"frozen predictions written {fp['written']}; multiprecision rows to m = {mp['rows'][-1]['m']}, {len(trm)} transitions"
)
print("(C) multiprecision vs double:")
worstC = 0.0
for mt, K, K1, kind in trm:
    if K in dbl:
        dC = mt - dbl[K]
        worstC = max(worstC, abs(dC))
        print(
            f"   {K} -> {K1} {kind:6s}: mp {mt:.8f}  double {dbl[K]:.8f}  diff {dC:+.1e}  {'ok' if abs(dC) <= 1e-5 else 'DISAGREE'}"
        )
print(f"   worst |diff| {worstC:.1e}: {'PASS' if worstC <= 1e-5 else 'FAIL'} so far")
errs = {"LAW": [], "MLNM": [], "FREEP": []}
for mt, K, K1, kind in trm:
    if mt <= 18.7 or K not in pred:
        continue
    p = pred[K]
    e = {k: p[k] - mt for k in errs}
    for k in errs:
        errs[k].append(e[k])
    print(
        f"   {K} -> {K1} {kind:6s} at {mt:11.7f}:  LAW {e['LAW']:+.4f}   MLNM {e['MLNM']:+.4f}   FREEP {e['FREEP']:+.4f}"
    )
if errs["LAW"]:
    mx = {k: max(abs(x) for x in v) for k, v in errs.items()}
    print("max |error|: " + ", ".join(f"{k} {v:.4f}" for k, v in mx.items()))
    verdict = (
        "PASS"
        if (mx["LAW"] <= 0.04 and mx["MLNM"] > 2 * mx["LAW"])
        else ("FAIL" if mx["LAW"] > 0.08 else "UNDECIDED so far")
    )
    print(
        f"631(6) rule on the multiprecision transitions so far: {verdict} ({len(errs['LAW'])} of the 20 predicted transitions located)"
    )
# ---- (30, 40]: the second frozen test, amendment 636(5) ----
import os

if os.path.exists("lfp_forward_predictions_30_40.json"):
    f2 = json.load(open("lfp_forward_predictions_30_40.json"))
    pred2 = {p["K_before"]: p for p in f2["predictions"]}
    e2 = {"LAW": [], "MLNM": [], "FREEP": []}
    for mt, K, K1, kind in trm:
        if mt <= 30 or K not in pred2:
            continue
        p = pred2[K]
        e = {k: p[k] - mt for k in e2}
        for k in e2:
            e2[k].append(e[k])
        print(
            f"   (30,40] {K} -> {K1} {kind:6s} at {mt:11.7f}:  LAW {e['LAW']:+.4f}   MLNM {e['MLNM']:+.4f}   FREEP {e['FREEP']:+.4f}"
        )
    if e2["LAW"]:
        mx = {k: max(abs(x) for x in v) for k, v in e2.items()}
        pref = "LAW PREFERRED" if mx["LAW"] < mx["MLNM"] / 2 else "LAW NOT PREFERRED"
        hold = (
            "HOLDS to 0.08"
            if mx["LAW"] <= 0.08
            else ("FAILS (> 0.12)" if mx["LAW"] > 0.12 else "between 0.08 and 0.12")
        )
        print(
            f"636(5) on (30, 40] ({len(e2['LAW'])} of {len(pred2)} located; written {f2['written']}): max |error| "
            + ", ".join(f"{k} {v:.4f}" for k, v in mx.items())
            + f"; {pref}; LAW {hold}"
        )
