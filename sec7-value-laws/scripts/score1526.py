import os, re, math

os.chdir(r"centre_scripts")
D = -0.304944111846
Dp, D2, D3 = -0.041318, -3.552e-3, 1.106e-2
for A in (20, 28):
    txt = open(f"q854_a{A}_1526.log").read().splitlines()
    rows = {}
    for i, line in enumerate(txt):
        m = re.match(r"a ([0-9.]+) t ([0-9.]+): atoms (\d+);.*?; (NOT ACCEPTED|ACCEPTED);", line)
        if not m:
            continue
        d = re.search(r"Def/\(f\(a\) \+ f\(-a\)\) = ([-+0-9.]+); Dslope = ([-+0-9.na]+)", txt[i - 1])
        rows[float(m.group(2))] = (
            m.group(4) == "ACCEPTED",
            int(m.group(3)),
            float(d.group(1)),
            float(d.group(2)) if float(m.group(2)) > 0 else None,
        )
    acc0, n0, r0, _ = rows[0.0]
    d0 = r0 - D
    print(f"a {A}: t = 0 {'ACC' if acc0 else 'NOT'} atoms {n0} delta0 {d0:+.2e} (limit 1e-7)")
    for t in (0.01, 0.02, 0.03):
        acc, n, r, ds = rows[t]
        c = ds - d0 / math.tanh(t * A) / t
        pred = Dp + D2 / 2 * t / math.tanh(t * A) + D3 / 6 * t * t
        print(
            f"   t {t}: {'ACC' if acc else 'NOT'} printed {ds:+.6f} corrected {c:+.6f} predicted {pred:+.6f} diff {c - pred:+.1e}"
        )
