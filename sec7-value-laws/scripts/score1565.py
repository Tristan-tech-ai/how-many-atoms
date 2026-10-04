import os, re

os.chdir(r"centre_scripts")
PRED = {
    "a7_10": (0.426799, 0.403856, 0.403856),
    "a9_13": (0.345854, 0.336029, 0.336029),
    "a10_15": (0.288764, 0.281571, 0.281571),
}
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
        f"prediction {pv:.6f}; ratio {ratio:.4f} -> {'PASS' if acc and 0.99 <= ratio <= 1.03 else 'FAIL'}; uncoupled ratio {dmid/un:.4f}"
    )
