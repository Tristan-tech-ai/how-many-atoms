import os, re, math

os.chdir(r"centre_scripts\q861b_runs")
j = 2.404825557695773
dl = 3.34776
PRED_RINGS = {2.0: 1, 3.5: 2, 4.0: 3, 6.5: 4}
nacc = 0
ok_r = True
ok_v = True
for m, pr in PRED_RINGS.items():
    t = open(f"m{m}.log").read()
    mm = re.search(
        r"^m [0-9.]+: rings (\d+) radii \[([^\]]*)\] masses \[([^\]]*)\]; B ([0-9.]+); .*?residual ([0-9.e+-]+); KKT max r - B ([-+0-9.e]+);.*?(NOT ACCEPTED|ACCEPTED)",
        t,
        re.M,
    )
    if not mm:
        print(m, "no result line")
        continue
    radii = [float(v) for v in mm.group(2).split(",")]
    acc = mm.group(7) == "ACCEPTED"
    B = float(mm.group(4))
    rings = sum(1 for r in radii if r > 1e-9)
    centre = any(r <= 1e-9 for r in radii)
    pv = 4 * j * j / (m + dl) ** 2
    ratio = (2 - B) / pv
    print(
        f"m {m}: {mm.group(7)} (res {mm.group(5)}, KKT {mm.group(6)}); radii {radii}; rings without centre {rings} (pred {pr}), centre atom {centre}; "
        f"B {B:.10f}; 2 - B {2 - B:.6f}; pred {pv:.6f}; ratio {ratio:.4f}; no-shift ratio {(2 - B)/(4*j*j/m**2):.4f}; half-shift ratio {(2 - B)/(4*j*j/(m + dl/2)**2):.4f}"
    )
    if acc:
        nacc += 1
        ok_r = ok_r and rings == pr
        ok_v = ok_v and 0.85 <= ratio <= 1.15
print(
    f"accepted {nacc}; 1562 rings: {'PASS' if ok_r and nacc >= 3 else ('NOT DECIDABLE' if nacc < 3 else 'FAIL')}; 1562 deficit: {'PASS' if ok_v and nacc >= 1 else 'FAIL'}"
)
