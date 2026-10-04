import os, re

os.chdir(r"centre_scripts")
REG = {
    (12, 30): (-80.486516, 6.8e-4),
    (15, 30): (-86.232811, 3.5e-4),
    (20, 30): (-95.811154, 1.6e-4),
    (15, 25): (-76.653713, 3.5e-4),
}
for (r1, r2), (c, band) in REG.items():
    txt = open(f"q847_annulus_{r1}_{r2}_1530.log").read()
    m = re.search(r"rings (\d+).*?; (NOT ACCEPTED|ACCEPTED);.*?Gt ([-+0-9.]+)", txt)
    if not m:
        print((r1, r2), "no result line yet")
        continue
    gt = float(m.group(3))
    print(
        f"({r1}, {r2}) rings {m.group(1)} {m.group(2)}: Gt {gt:.8f} centre {c:.6f} diff {gt - c:+.2e} band {band:.1e} {'IN' if abs(gt - c) <= band else 'OUT'}"
    )
