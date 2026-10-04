import glob, os, re, math
import numpy as np

os.chdir(r"centre_scripts")
D = -0.304944111846
rows = []
files = sorted(glob.glob("q854_a*_1524.log")) + [
    "q854_t0.01.log",
    "q854_t0.02.log",
    "q854_t0.05.log",
    "q854_t0.log",
]
for f in files:
    txt = open(f).read().splitlines()
    for i, line in enumerate(txt):
        m = re.match(r"a ([0-9.]+) t ([0-9.]+): atoms (\d+);.*?; (NOT ACCEPTED|ACCEPTED);", line)
        if not m:
            continue
        a, t, acc = float(m.group(1)), float(m.group(2)), m.group(4) == "ACCEPTED"
        dl = [l for l in txt[max(0, i - 1) : i] if "defect:" in l]
        d = (
            re.search(r"Def/\(f\(a\) \+ f\(-a\)\) = ([-+0-9.]+); Dslope = ([-+0-9.na]+)", dl[-1])
            if dl
            else None
        )
        ratio = float(d.group(1)) if d else float("nan")
        ds = float(d.group(2)) if d and t > 0 else float("nan")
        print(
            f"{f:22s} a {a:4.0f} t {t:.2f} atoms {m.group(3):>3s} {'ACC' if acc else 'NOT'}  Def/(fa+fma) - D {ratio - D:+.1e}  Dslope {ds:+.6f}"
        )
        if acc:
            rows.append((a, t, ds, ratio))
R = [r for r in rows if r[1] > 0]
a = np.array([r[0] for r in R])
t = np.array([r[1] for r in R])
y = np.array([r[2] for r in R])
X = np.c_[np.ones_like(t), t / np.tanh(t * a) / 2, t**2 / 6]
c = np.linalg.lstsq(X, y, rcond=None)[0]
print(
    f"fit over {len(R)} rows: D'(0) = {c[0]:.6f}  D''(0) = {c[1]:.3e}  D'''(0) = {c[2]:.3e}  maxres {np.abs(X @ c - y).max():.1e}"
)
X2 = np.c_[np.ones_like(t), t**2 / 6]
c2 = np.linalg.lstsq(X2, y, rcond=None)[0]
print(f"H_0 model (no coth term): D'(0) = {c2[0]:.6f}  maxres {np.abs(X2 @ c2 - y).max():.1e}")
print(
    "H_bias: D'(0) [-0.041328, -0.041308], D''(0) [-4.8e-3, -2.8e-3];  H_0: D'(0) [-0.041448, -0.041428], |D''(0)| < 3e-4"
)
