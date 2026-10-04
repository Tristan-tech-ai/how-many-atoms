import json, os
import numpy as np

os.chdir(r"centre_scripts")
e = 2.5
s = json.load(open(f"q857_runs/state_R18.0_r8.0_e{e}_stage1.json"))
x = np.array(s["x"])
y = np.array(s["y"])
d = s["Gt"] - (-49.799040)
dm, band = -0.095441, 0.0106
print(
    f"(V) Gt {s['Gt']:.6f}  Delta_obs {d:+.6f}  Delta_mid {dm:+.6f}  diff {d - dm:+.4f} (band {band}) -> {'PASS' if abs(d - dm) <= band else 'FAIL'};"
    f" H_null {'in' if abs(d) <= band else 'OUT'}; outer {d + 0.130001:+.4f}, inner {d + 0.060880:+.4f}"
)
phi = np.degrees(np.arctan2(y, x - e))
rho = np.hypot(x - e, y)
c = []
for a0 in range(0, 180, 5):
    k = (phi >= a0) & (phi < a0 + 5)
    rr = np.sort(rho[k])
    g = np.split(rr, np.where(np.diff(rr) > 0.5)[0] + 1) if len(rr) else []
    c.append(len(g))
    print(f"phi {a0:3d}-{a0+5:3d}: rings {len(g)}  radii {[round(float(v.mean()), 2) for v in g]}")
b = (
    [c[0]]
    + [(min(c[i - 1], c[i + 1]) if c[i] < min(c[i - 1], c[i + 1]) else c[i]) for i in range(1, len(c) - 1)]
    + [c[-1]]
)
print("bridged:", b)


def change(lo, hi):
    for i in range(1, len(b) - 2):
        if b[i - 1] == lo and all(v == hi for v in b[i : i + 3]):
            return 5 * i
    return None


ok = b[0] == 3 and b[-1] == 6
for lo, hi, b0, b1 in [(3, 4, 44.27, 51.11), (4, 5, 98.18, 102.73), (5, 6, 152.70, 163.64)]:
    ch = change(lo, hi)
    inb = ch is not None and b0 - 5 <= ch <= b1 + 5
    ok = ok and inb
    print(
        f"change {lo}->{hi} at {ch} deg (band {b0}-{b1}, widened {b0-5:.2f}-{b1+5:.2f}): {'in' if inb else 'OUT'}"
    )
print("(S):", "PASS" if ok else "FAIL")
