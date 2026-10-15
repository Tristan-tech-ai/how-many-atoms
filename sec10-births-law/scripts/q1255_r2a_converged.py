"""q1255 (8 Oct 2026, a reading, no item): max |dev_med| over the 17 stored capacity events K_b 40 to 59 for the spline degrees k = 3, 5, 7, 9, from q1251_dev.json (item R2a of amendment 1977 has the bar 0.4e-9).
Control: the k = 3 maximum is 0.4510e-9 (q1220_out.txt, K_b 40) within 0.0005e-9.
Usage: py q1255_r2a_converged.py  (from temporal_within/centre_scripts/)"""

import json, sys

d = json.load(open("q1251_dev.json"))
mx = {}
for k in ("3", "5", "7", "9"):
    a = d["capacity_" + k]
    r = [(abs(a[str(K)]), K) for K in range(40, 60) if str(K) in a]
    mx[k] = max(r)
    n = len(r)
ok = abs(mx["3"][0] - 0.4510) < 5e-4 and n == 17
print(f"control (cubic maximum {mx['3'][0]:.4f}e-9 at K_b {mx['3'][1]}, {n} events): {ok}")
if not ok:
    print("READING q1255: VOID (control)")
    sys.exit(2)
for k, (v, K) in mx.items():
    print(f"k = {k}: max |dev_med| = {v * 1e-9:.3e} ({v:.4f}e-9) at K_b {K}")
print("READING q1255 (no verdict)")
