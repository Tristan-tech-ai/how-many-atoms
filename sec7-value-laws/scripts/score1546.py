import os, re, math

os.chdir(r"centre_scripts")
h2 = math.log(2 * math.pi * math.e)
Cref = 3.7194191566182786


def br(f):
    t = open(f).read()
    m = re.search(r"final C in \[([0-9.]+), ([0-9.]+)\]", t)
    return (float(m.group(1)), float(m.group(2))) if m else None


b0 = br("q859_runs/validate_e0.log")
if not b0:
    raise SystemExit("control not finished")
lo, hi = b0
gap = hi - lo
print(
    f"control E 0: bracket [{lo:.10f}, {hi:.10f}] gap {gap:.2e}; exact {Cref:.10f}; inside widened bracket: {lo - gap <= Cref <= hi + gap}"
)
Z0 = math.exp((lo + hi) / 2 + h2)
PRED = {0.5: -0.154151, 1.0: -0.550621}
VAR = {0.5: (-0.166849, -0.141453), 1.0: (-0.615752, -0.485490)}
for E, dm in PRED.items():
    b = br(f"q859_runs/e{E:g}.log")
    if not b:
        print(E, "not finished")
        continue
    Z = math.exp((b[0] + b[1]) / 2 + h2)
    d = Z - Z0
    band = math.exp(sum(b) / 2 + h2) * ((b[1] - b[0]) + gap) / 2 + 0.15 * abs(dm)
    print(
        f"E {E}: bracket [{b[0]:.10f}, {b[1]:.10f}] gap {b[1]-b[0]:.2e}; Delta_obs {d:+.6f}  Delta_mid {dm:+.6f}  diff {d - dm:+.4f}  band {band:.4f}  "
        f"H_local {'in' if abs(d - dm) <= band else 'OUT'}  H_null {'in' if abs(d) <= band else 'OUT'}  [outer {d - VAR[E][0]:+.4f}, inner {d - VAR[E][1]:+.4f}]"
    )
