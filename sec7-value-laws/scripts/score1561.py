import os, re

os.chdir(r"centre_scripts")
PRED = {"a6_10": (0.325735, 2.449698, 0.719866), "a8_11": (0.407508, 4.375890, 0.971533)}
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
        f"prediction {pv:.6f}; ratio {ratio:.4f} -> {'PASS' if acc and 0.97 <= ratio <= 1.04 else 'FAIL'}; unwidened ratio {dmid/un:.4f}; outward-only ratio {dmid/eqd:.4f}"
    )
