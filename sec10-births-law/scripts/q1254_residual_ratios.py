"""q1254 (8 Oct 2026, a reading, no item): ratios of the quintic median-sum residuals of the immune and NPMLE problems to the capacity residual at K_b 35 to 47, and the immune minus capacity difference; from q1250_dev.json (k = 5).
Control: the file carries the capacity values +0.2013 at K_b 40 and the immune value +1.1444 at K_b 40 (q1250_out.txt), within 5e-4.
Usage: py q1254_residual_ratios.py  (from temporal_within/centre_scripts/)"""

import json, sys

d = json.load(open("q1250_dev.json"))
c, i, n = d["capacity_5"], d["immune_5"], d["npmle_5"]
ok = abs(c["40"] - 0.2013) < 5e-4 and abs(i["40"] - 1.1444) < 5e-4
print("control (K_b 40 values of q1250_out.txt):", ok)
if not ok:
    print("READING q1254: VOID (control)")
    sys.exit(2)
ri = {K: i[str(K)] / c[str(K)] for K in range(35, 48) if str(K) in i and str(K) in c}
rn = {K: n[str(K)] / c[str(K)] for K in range(35, 48) if str(K) in n and str(K) in c}
print(
    "immune / capacity: "
    + ", ".join(f"{K}: {r:.2f}" for K, r in ri.items())
    + f"; range {min(ri.values()):.2f} to {max(ri.values()):.2f}"
)
print(
    "NPMLE / capacity: "
    + ", ".join(f"{K}: {r:.2f}" for K, r in rn.items())
    + f"; range {min(rn.values()):.2f} to {max(rn.values()):.2f}"
)
print(
    f"immune - capacity: K_b 35 {(i['35'] - c['35']) * 1e-9:.4e} (= {i['35'] - c['35']:.4f}e-9), K_b 47 {(i['47'] - c['47']) * 1e-9:.4e} (= {i['47'] - c['47']:.4f}e-9)"
)
print("READING q1254 (no verdict)")
