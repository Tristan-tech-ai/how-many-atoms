import json, math, os, re

SP = r"scratch"
os.chdir(r"centre_scripts")
beta = 3.98160794709
k = -0.63226
k2 = 0.049
h1 = 0.5 * math.log(2 * math.pi * math.e)


def Zinf(A, t):
    if t == 0:
        return 2 * A + beta
    bR = beta / 2 + k * t + k2 * t * t
    bL = beta / 2 - k * t + k2 * t * t
    return (math.exp(t * (A + bR)) - math.exp(-t * (A + bL))) / t


def onerun(tag, A):
    txt = open(f"q850_tilt1538/{tag}_A{A}.log").read()
    rows = {}
    for m in re.finditer(
        r"A ([0-9.]+) t ([0-9.]+): atoms (\d+).*?residual ([0-9.e+-]+); KKT max ([-+0-9.e]+); I \+ tE = ([0-9.]+)",
        txt,
    ):
        t = float(m.group(2))
        acc = float(m.group(4)) <= 1e-10 and float(m.group(5)) <= 1e-9
        rows[t] = (acc, math.exp(float(m.group(6)) + h1) - Zinf(A, t))
    return rows


sets = {
    "2D10": (
        "pred1533_frozen.json",
        10,
        "q686_annulus_a10_d2_SEALED.json",
        math.log(2 * math.pi * math.e),
        1,
    ),
    "2D20": (
        "pred1533_frozen.json",
        20,
        "q686_annulus_a20_d2_SEALED.json",
        math.log(2 * math.pi * math.e),
        1,
    ),
    "3D20": (
        "pred1535_frozen.json",
        20,
        "q686_annulus_a20_d3_SEALED.json",
        1.5 * math.log(2 * math.pi * math.e),
        2,
    ),
}
for tag, (pf, a, sf, h, dm) in sets.items():
    st = json.load(open(sf))["states"]
    seen = {}
    for o in json.load(open(os.path.join(SP, pf))):
        if not o.get("decisive") or o.get("a", 20) != a:
            continue
        W = round(o["W"], 6)
        if W in seen:
            continue
        A = W / 2
        Rm = a + W / 2
        t = round(dm / Rm, 6)
        obs = math.exp(st[o["k"]]["C"] + h)
        try:
            rows = onerun(tag, A)
        except FileNotFoundError:
            print(tag, W, "no log")
            continue
        r0 = rows.get(0.0)
        rt = rows.get(t)
        if not r0 or not rt or not (r0[0] and rt[0]):
            print(f"{tag} W {W}: 1D rows missing or not accepted {r0} {rt}")
            continue
        S = 2 * math.pi * Rm if dm == 1 else 4 * math.pi * Rm**2
        P2 = o["P1"] + S * (rt[1] - r0[1])
        seen[W] = (obs - o["P1"], obs - P2, S * (rt[1] - r0[1]), o["coupling"])
        print(
            f"{tag} W {W:4.1f} t {t:.4f}: eps_0 {r0[1]:+.3e} eps_t {rt[1]:+.3e}  tilt term {S * (rt[1] - r0[1]):+.3e}  obs-P1 {obs - o['P1']:+.3e}  obs-P2 {obs - P2:+.3e}"
        )
    if seen:
        rms = lambda i: math.sqrt(sum(v[i] ** 2 for v in seen.values()) / len(seen))
        print(
            f"== {tag}: {len(seen)} widths  RMS obs-P1 {rms(0):.4e}  RMS obs-P2 {rms(1):.4e}  ratio {rms(1) / rms(0):.3f}"
        )
