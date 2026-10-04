import os, re

os.chdir(r"centre_scripts")
PRED = {"e5_3.5": (0.410547, 1.400974, 0.407864), "e6_4": (0.344331, 1.039255, 0.340144)}
for k, (pv, un, eqd) in PRED.items():
    t = open(f"q863_runs/{k}.log").read()
    m = re.search(r"Bayes risk ([0-9.]+); 2 - B ([0-9.]+); KKT gap ([0-9.e+-]+)", t)
    if not m:
        print(k, "no result")
        continue
    B, gap = float(m.group(1)), float(m.group(3))
    dmid = 2 - (B + gap / 2)
    acc = gap < 1e-3
    ratio = dmid / pv
    print(
        f"{k}: B {B:.8f} gap {gap:.2e} ({'accepted' if acc else 'NOT accepted'}); 2 - B* in [{2 - B - gap:.6f}, {2 - B:.6f}], midpoint {dmid:.6f}; "
        f"prediction {pv:.6f}; ratio {ratio:.4f} -> {'PASS' if acc and 0.97 <= ratio <= 1.04 else 'FAIL'}; unwidened ratio {dmid/un:.4f}; equal-area disc ratio {dmid/eqd:.4f}"
    )
