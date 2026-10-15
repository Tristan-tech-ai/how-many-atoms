"""q1256 (8 Oct 2026, a reading, no item): deviations A_meas - a_pred(order n) of the births law with the derived series for capacity K_b 61 to 80, orders 12 to 20 (coefficients d22_out_20.json), corrected split amplitudes (tower_K<K_b+1>_As2.json), cubic spline of N_half as q1209 and q1235. Draft reading 4 finding 10.
Control (first): at order 16 the deviations of the 20 events equal those of q1235_out.txt (printed order-16 list) to 0.001e-9.
Usage: py q1256_order20_deviations.py  (from temporal_within/centre_scripts/)"""

import json, os, re, sys

os.environ["E_FILE"] = "d22_out_20.json"
sys.argv = ["x"]
import q1209_births_law_derived_series as G

A = "../scratch"


def meas(K):
    if K % 2 == 0:
        return json.load(open(f"{A}/tower_K{K + 1}_birth_out.json"))["A_cross"]
    return json.load(open(f"{A}/tower_K{K + 1}_As2.json"))["A_s"]


d = {n: {K: (meas(K) - G.pred_series(G.Ncap, K, n)) * 1e9 for K in range(61, 81)} for n in range(12, 21)}
txt = open(f"{A}/q1235_out.txt", encoding="utf-8").read()
m = re.search(r"order 16 corrected deviations: ([^\n]*)", txt)
ref = {int(k): float(v) for k, v in re.findall(r"(\d+): ([-+]\d+\.\d+)", m.group(1))}
ok = len(ref) == 20 and all(abs(d[16][K] - ref[K]) < 0.001 + 5e-4 for K in ref)
print(f"control: order-16 deviations equal q1235_out.txt at {len(ref)} events to 0.001e-9: {ok}")
if not ok:
    print("READING q1256: VOID (control)")
    sys.exit(2)
sp, br = range(61, 80, 2), range(62, 81, 2)
for n in range(12, 21):
    print(
        f"order {n}: max |dev| splits {max(abs(d[n][K]) for K in sp):.4f}, births {max(abs(d[n][K]) for K in br):.4f}, all 20 {max(abs(v) for v in d[n].values()):.4f} (1e-9)"
    )
print("split K_b 61 at orders 16 to 20: " + ", ".join(f"{d[n][61]:+.4f}" for n in range(16, 21)))
for K in (63, 65, 73):
    print(f"split K_b {K} at orders 16 to 20: " + ", ".join(f"{d[n][K]:+.4f}" for n in range(16, 21)))
print("births K_b 66, 68 at order 20: " + ", ".join(f"{K}: {d[20][K]:+.4f}" for K in (66, 68)))
print("order-20 splits: " + ", ".join(f"{K}: {d[20][K]:+.4f}" for K in sp))
print("READING q1256 (no verdict)")
