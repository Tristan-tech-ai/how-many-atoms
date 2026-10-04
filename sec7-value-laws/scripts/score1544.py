import json, math, os
import numpy as np

os.chdir(r"centre_scripts")
e = 2.0
s = json.load(open(f"q857_runs/state_R18.0_r8.0_e{e}_stage1.json"))
x = np.array(s["x"])
y = np.array(s["y"])
phi = np.degrees(np.arctan2(y, x - e))
rho = np.hypot(x - e, y)
counts = []
for a0 in range(0, 180, 5):
    k = (phi >= a0) & (phi < a0 + 5)
    rr = np.sort(rho[k])
    n = 0 if len(rr) == 0 else len(np.split(rr, np.where(np.diff(rr) > 0.5)[0] + 1))
    counts.append(n)
    print(
        f"phi {a0:3d}-{a0+5:3d}: rings {n}  radii {[round(float(g.mean()), 2) for g in np.split(rr, np.where(np.diff(rr) > 0.5)[0] + 1)] if len(rr) else []}"
    )
print("Gt", s["Gt"], "support", len(x))


def change(lo, hi):
    for i in range(1, len(counts) - 2):
        if counts[i - 1] == lo and all(c == hi for c in counts[i : i + 3]):
            return 5 * i
    return None


c34 = change(3, 4)
c45 = change(4, 5)
print("change 3->4 at boundary", c34, "deg (band 21.57-34.99, widened by 5: 16.57-39.99)")
print("change 4->5 at boundary", c45, "deg (band 98.45-104.17, widened by 5: 93.45-109.17)")
order_ok = counts[0] == 3 and counts[-1] == 5
p34 = c34 is not None and 16.57 <= c34 <= 39.99
p45 = c45 is not None and 93.45 <= c45 <= 109.17
print(
    "counts start/end", counts[0], counts[-1], " VERDICT:", "PASS" if (order_ok and p34 and p45) else "FAIL"
)
