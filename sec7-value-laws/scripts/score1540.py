import json, os, re

SP = r"scratch"
os.chdir(r"centre_scripts")
for o in json.load(open(os.path.join(SP, "pred1540_frozen.json"))):
    f = f"q856_runs/shell_{o['R1']}_{o['R2']}.log"
    t = open(f).read() if os.path.exists(f) else ""
    m = re.search(r"rings (\d+).*?; (NOT ACCEPTED|ACCEPTED);.*?Gt ([-+0-9.]+);", t)
    if not m:
        print(f, "no result yet")
        continue
    gt = float(m.group(3))
    acc = m.group(2)
    if o["coupling"] == 0:
        print(
            f"({o['R1']}, {o['R2']}) {m.group(1)} shells {acc}: Gt {gt:.8f}  obs-P0 {gt - o['P0']:+.4e} (+-{o['base']:.3f})  "
            f"obs-centre {gt - o['centre_alt']:+.4f}  obs-oneE0 {gt - o['one_E0_alt']:+.4f}  obs-noflip {gt - o['noflip_alt']:+.4f}"
        )
    else:
        print(
            f"({o['R1']}, {o['R2']}) {m.group(1)} shells {acc}: Gt {gt:.8f}  obs-P1 {gt - o['P1']:+.4e} (+-{o['band1']:.3f})  "
            f"obs-P0 {gt - o['P0']:+.4e}  coupling {o['coupling']:+.4e}  (obs-P1)/coupling {(gt - o['P1'])/o['coupling']:+.3f}"
        )
