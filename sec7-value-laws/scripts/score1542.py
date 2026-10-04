import os, re

os.chdir(r"centre_scripts")


def gt(f):
    t = open(f).read()
    m = re.search(r"support (\d+) orbits; KKT max - 1 ([-+0-9.e]+); L = [-0-9.]+; Gt = ([-+0-9.]+)", t)
    return (float(m.group(3)), int(m.group(1)), float(m.group(2))) if m else None


g0 = gt("q857_runs/validate_e0_H0.2_stage2.log")
PRED = {1: (-0.148442, 0.01589, -0.203388, -0.093497), 2: (-0.094945, 0.01054, -0.130474, -0.059417)}
print("e 0 stage 2:", g0)
for e, (dm, band, dout, din) in PRED.items():
    g = gt(f"q857_runs/stage2_e{e}.log")
    if not g:
        print(e, "no result")
        continue
    d = g[0] - g0[0]
    print(
        f"e {e}: Gt {g[0]:+.6f} (support {g[1]}, KKT {g[2]:+.1e})  Delta_obs {d:+.6f}  Delta_mid {dm:+.6f}  diff {d - dm:+.4f} (band {band})  "
        f"H_local {'in' if abs(d - dm) <= band else 'OUT'}  H_null {'in' if abs(d) <= band else 'OUT'}  [outer {d - dout:+.4f}, inner {d - din:+.4f}]"
    )
for e in (1, 2):
    s = gt(f"q857_runs/SEALED_e{e}_H0.2.log")
    s0 = gt("q857_runs/validate_e0_H0.2.log")
    if s and s0:
        print(f"stage 1 only, e {e}: Delta {s[0] - s0[0]:+.6f}")
