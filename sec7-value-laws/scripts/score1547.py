import json, math, os
import numpy as np

os.chdir(r"centre_scripts")
e = 3.0
s = json.load(open(f"q857_runs/state_R18.0_r8.0_e{e}_stage1.json"))
x = np.array(s["x"])
y = np.array(s["y"])
d = s["Gt"] - (-49.799040)
print(
    f"(V) Gt {s['Gt']:.6f}  Delta_obs {d:+.6f}  Delta_mid -0.191141  diff {d + 0.191141:+.4f} (band 0.0202) -> {'PASS' if abs(d + 0.191141) <= 0.0202 else 'FAIL'};"
    f" H_null {'in' if abs(d) <= 0.0202 else 'OUT'}; outer {d + 0.249731:+.4f}, inner {d + 0.132550:+.4f}"
)
phi = np.degrees(np.arctan2(y, x - e))
rho = np.hypot(x - e, y)
counts = []
for a0 in range(0, 180, 5):
    k = (phi >= a0) & (phi < a0 + 5)
    rr = np.sort(rho[k])
    g = np.split(rr, np.where(np.diff(rr) > 0.5)[0] + 1) if len(rr) else []
    counts.append(len(g))
    print(f"phi {a0:3d}-{a0+5:3d}: rings {len(g)}  radii {[round(float(v.mean()), 2) for v in g]}")


def change(lo, hi):
    for i in range(1, len(counts) - 2):
        if counts[i - 1] == lo and all(c == hi for c in counts[i : i + 3]):
            return 5 * i
    return None


ok = counts[0] == 3 and counts[-1] == 6
for lo, hi, b0, b1 in [(3, 4, 55.37, 60.32), (4, 5, 98.27, 102.04), (5, 6, 139.75, 145.29)]:
    c = change(lo, hi)
    inb = c is not None and b0 - 5 <= c <= b1 + 5
    ok = ok and inb
    print(
        f"change {lo}->{hi} at {c} deg (band {b0}-{b1}, widened {b0-5:.2f}-{b1+5:.2f}): {'in' if inb else 'OUT'}"
    )
print("counts start/end", counts[0], counts[-1], "(S):", "PASS" if ok else "FAIL")
