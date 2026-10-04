"""Scoring of 1535: reads the 3D shell C values (run only after 1535 is appended)."""

import json, math, os

SP = r"scratch"
os.chdir(r"centre_scripts")
pred = json.load(open(os.path.join(SP, "pred1535_frozen.json")))
st = json.load(open("q686_annulus_a20_d3_SEALED.json"))["states"]
h3 = 1.5 * math.log(2 * math.pi * math.e)
nd = n1 = n0 = 0
seen = {}
for o in pred:
    if "P0" not in o:
        continue
    obs = math.exp(st[o["k"]]["C"] + h3)
    r0 = obs - o["P0"]
    r1 = obs - o["P1"]
    in0 = abs(r0) <= o["base"]
    in1 = abs(r1) <= o["band1"]
    if o["decisive"]:
        nd += 1
        n0 += in0
        n1 += in1
    seen.setdefault(o["W"], (r0, r1, o))
    print(
        f"W {o['W']:4.1f}: obs {obs:.5f}  obs-P0 {r0:+.3e} (+-{o['base']:.2e}) {'in' if in0 else 'OUT'}   obs-P1 {r1:+.3e} "
        f"(+-{o['band1']:.2e}) {'in' if in1 else 'OUT'}  coupling {o['coupling']:+.3e} {'D' if o['decisive'] else ''}"
    )
print(f"decisive entries {nd}: within P1 band {n1}, within P0 band {n0}")
D = [v for v in seen.values() if v[2]["decisive"]]
rms = lambda xs: math.sqrt(sum(x * x for x in xs) / len(xs))
print(
    f"distinct decisive {len(D)}: rms obs-P0 {rms([v[0] for v in D]):.3e}, rms obs-P1 {rms([v[1] for v in D]):.3e}"
)
for W, (r0, r1, o) in sorted(seen.items()):
    if o["decisive"]:
        print(f"   W {W}: (obs-P1)/coupling {r1 / o['coupling']:+.3f}")
