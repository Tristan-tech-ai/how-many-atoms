"""Scoring of 1533: reads the 2D C values (run only after 1533 is appended)."""

import json, math, os

SP = r"scratch"
os.chdir(r"centre_scripts")
pred = json.load(open(os.path.join(SP, "pred1533.json")))
files = {a: json.load(open(f"q686_annulus_a{a}_d2_SEALED.json"))["states"] for a in (10, 20)}
h2 = math.log(2 * math.pi * math.e)
nd = n1 = n0 = 0
for o in pred:
    if "P0" not in o:
        continue
    C = files[o["a"]][o["k"]]["C"]
    obs = math.exp(C + h2)
    r0 = obs - o["P0"]
    r1 = obs - o["P1"]
    in0 = abs(r0) <= o["base"]
    in1 = abs(r1) <= o["band1"]
    if o["decisive"]:
        nd += 1
        n0 += in0
        n1 += in1
    print(
        f"a {o['a']} W {o['W']:4.1f}: obs {obs:.6f}  obs-P0 {r0:+.3e} (+-{o['base']:.1e}) {'in' if in0 else 'OUT'}   "
        f"obs-P1 {r1:+.3e} (+-{o['band1']:.1e}) {'in' if in1 else 'OUT'}  coupling {o['coupling']:+.3e} {'D' if o['decisive'] else ''}"
    )
print(f"decisive states {nd}: within P1 band {n1}, within P0 band {n0}")
