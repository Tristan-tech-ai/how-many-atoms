import os, re, math

os.chdir(r"centre_scripts\q865_runs")
d = 3.34776
nacc = 0
ok = True
for m in (2.5, 3.5, 4.5):
    t = open(f"m{m}.log").read() if os.path.exists(f"m{m}.log") else ""
    mm = re.search(
        r"^m [0-9.]+: rings (\d+) radii \[([^\]]*)\].*?; B ([0-9.]+); .*?residual ([0-9.e+-]+); KKT max r - B ([-+0-9.e]+);.*?(NOT ACCEPTED|ACCEPTED)",
        t,
        re.M,
    )
    if not mm:
        print(m, "no result")
        continue
    B = float(mm.group(3))
    acc = mm.group(6) == "ACCEPTED"
    pv = 4 * math.pi**2 / (m + d) ** 2
    ratio = (3 - B) / pv
    print(
        f"m {m}: {mm.group(6)} (res {mm.group(4)}, KKT {mm.group(5)}); shells {mm.group(2)}; B {B:.10f}; 3 - B {3 - B:.6f}; pred {pv:.6f}; ratio {ratio:.4f}; "
        f"no-shift ratio {(3 - B)/(4*math.pi**2/m**2):.4f}; half-shift ratio {(3 - B)/(4*math.pi**2/(m + d/2)**2):.4f}"
    )
    if acc:
        nacc += 1
        ok = ok and 0.85 <= ratio <= 1.15
print(f"accepted {nacc}; 1566: {'PASS' if ok and nacc >= 1 else 'FAIL / not decidable'}")
