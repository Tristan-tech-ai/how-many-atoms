"""Q495: scoring of the THIRD frozen LFP forward test on (40, 50] (pred_lfp_40_50.txt) against the multiprecision transitions
of q492 (results_q492_lfp_mpcont.json), and of the frozen offset extrapolation (pred_lfp_offset_extrap.txt). Reads only;
imports nothing that writes (amendment 634). Rule (636(5) form): LAW PREFERRED if its max |error| < half the m ln m fit's;
LAW HOLDS to 0.08, FAILS above 0.12. Trend prediction: LAW error 0.0464 + 0.00072 n +- 0.004 (n = transitions after 67 -> 68).
"""

import json, mpmath as mp

mp.mp.dps = 30
F = json.load(open("lfp_forward_predictions_40_50.json"))
assert F["written"] == "2026-09-25 10:07:36"
pred = {p["K_before"]: p for p in F["predictions"]}
R = json.load(open("results_q492_lfp_mpcont.json"))
trm = R["transitions"]
e = {"LAW": [], "MLNM": [], "FREEP": []}
trend_out = 0
for mt, K, K1, kind in trm:
    if mt <= 40 or K not in pred:
        continue
    p = pred[K]
    err = {k: p[k] - mt for k in e}
    n = K - 67
    for k in e:
        e[k].append(err[k])
    tp = 0.0464 + 0.00072 * n
    ok = abs(err["LAW"] - tp) <= 0.004
    trend_out += not ok
    print(
        f"   (40,50] {K} -> {K1} {kind:6s} at {mt:11.7f}:  LAW {err['LAW']:+.4f} (trend {tp:.4f}, {'in' if ok else 'OUT'})   MLNM {err['MLNM']:+.4f}   FREEP {err['FREEP']:+.4f}"
    )
if e["LAW"]:
    mx = {k: max(abs(x) for x in v) for k, v in e.items()}
    pref = "LAW PREFERRED" if mx["LAW"] < mx["MLNM"] / 2 else "LAW NOT PREFERRED"
    hold = (
        "HOLDS to 0.08"
        if mx["LAW"] <= 0.08
        else ("FAILS (> 0.12)" if mx["LAW"] > 0.12 else "between 0.08 and 0.12")
    )
    print(
        f"(40, 50] so far ({len(e['LAW'])} of {len(pred)} located): max |error| "
        + ", ".join(f"{k} {v:.4f}" for k, v in mx.items())
        + f"; {pref}; LAW {hold}; trend: {trend_out} of {len(e['LAW'])} outside the band (FAIL if more than 2)"
    )
else:
    print("(40, 50]: no transition located yet")
# offset extrapolation
tg = {
    41.0: (mp.mpf("3.3467076"), mp.mpf("5e-7")),
    43.0: (mp.mpf("3.3467966"), mp.mpf("1e-6")),
    45.0: (mp.mpf("3.3468748"), mp.mpf("2e-6")),
}
rows = {round(float(r["m"]), 4): r for r in R["rows"]}
for mm, (bp, tol) in tg.items():
    r = rows.get(round(mm, 4))
    if r is None:
        print(f"offset at m = {mm}: row not stored yet")
        continue
    rs = mp.mpf(r["r"])
    bl = mp.pi / mp.sqrt(1 - rs) - mm
    print(
        f"offset at m = {mm}: beta_L = {mp.nstr(bl, 10)}, predicted {bp} +- {tol}: diff {mp.nstr(bl - bp, 3)} -> {'PASS' if abs(bl - bp) <= tol else 'FAIL'}"
    )
