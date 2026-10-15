"""q1292 (8 Oct 2026, data acquisition, no item, no claim): a_med of the quintic median-sum law for the functionals k = 1.5, 2.5, 4, 5 at K_b 14 to 28 (all four) and K_b 37 to 44 (k = 1.5, 2.5), and the four launch scripts run1292a.sh (k = 1.5), run1292b.sh (k = 2.5), run1292c.sh (k = 4), run1292d.sh (k = 5) with the 160-bit full-chain locators c35; start a0 = a_med - 0.0045 (amendment 2066: a start 0.002 above the transition can fail, 0.0045 below works), step 0.003.
Purpose: events over the whole range v^2 = 69 to 115 for several k, to fit sigma_inf(k) and g(k) of the closure R_k = sigma_k Da (1 + g / v^2) (amendment 2070).
Control (first): a_med of k = 5 at K_b 36 equals the value in q1276_predictions.json within 1e-12, and of k = 4 at K_b 36 the value in q1273_predictions.json.
Usage: py q1292_prepare.py  (from temporal_within/centre_scripts/)"""

import json, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
import q1250_quintic_residual as Q
import q1273_k4_curvature_blind as W

SPEC = {
    1.5: ("k15", list(range(14, 29)) + list(range(37, 45))),
    2.5: ("k25", list(range(14, 29)) + list(range(37, 45))),
    4.0: ("k4", list(range(14, 29))),
    5.0: ("k5", list(range(14, 29))),
}
law = {
    1.5: json.load(open("q1276_predictions.json"))["1.5"]["law"],
    2.5: json.load(open("q1276_predictions.json"))["2.5"]["law"],
    4.0: json.load(open("q1273_predictions.json"))["law"],
    5.0: json.load(open("q1276_predictions.json"))["5.0"]["law"],
}
stage = {1.5: "c33_k15c.json", 2.5: "c33_k25c.json", 4.0: "c33_k4c.json", 5.0: "c33_k5c.json"}
N5 = {k: W.nhalf(stage[k], 92.0, 5, tuple(law[k])) for k in SPEC}
ok = (
    abs(Q.root(N5[5.0], 36) - json.load(open("q1276_predictions.json"))["5.0"]["36"]["a_med"]) < 1e-12
    and abs(Q.root(N5[4.0], 36) - json.load(open("q1273_predictions.json"))["36"]["a_med"]) < 1e-12
)
print(f"control: a_med k = 5 and k = 4 at K_b 36 reproduce the registered tables: {ok}")
if not ok:
    print("READING q1292: VOID (control)")
    sys.exit(2)
amed = {}
for i, (k, (tag, ks)) in enumerate(SPEC.items()):
    nm = "run1292" + "abcd"[i]
    L = [
        "#!/bin/bash",
        f"# {nm}.sh (8 Oct 2026): data acquisition k = {k}, start a0 = a_med - 0.0045 (q1292_prepare.py)",
        'cd "./centre_scripts"',
        f"export KEXP={k} HL_JSON={stage[k]} HL_STAGE=92 D0={law[k][0]:.5f} NU0={law[k][1]:.5f}",
    ]
    amed[str(k)] = {}
    for K in ks:
        am = Q.root(N5[k], K)
        amed[str(k)][str(K)] = am
        L.append(
            f"py c35_chain_k.py {K} {K // 2} 160 {am - 0.0045:.6f} 0.003 u1292_{tag}_K{K} > q607_u1292_{tag}_K{K}.log 2>&1"
        )
    L.append(f"echo done $(date +%H:%M:%S) > {nm}.done")
    open(nm + ".sh", "w", newline="\n").write("\n".join(L) + "\n")
json.dump(amed, open("q1292_amed.json", "w"), indent=1)
print("scripts run1292a.sh to run1292d.sh written; events:", {k: len(v[1]) for k, v in SPEC.items()})
print("READING q1292 (no verdict)")
