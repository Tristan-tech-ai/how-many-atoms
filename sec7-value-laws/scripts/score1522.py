import glob, os, re, math
import numpy as np

os.chdir(r"centre_scripts")
D = -0.304944111846
G = {}
for f in sorted(glob.glob("q848_R*.log")):
    for line in open(f):
        m = re.match(r"R ([0-9.]+): rings (\d+).*?; (NOT ACCEPTED|ACCEPTED);.*Gt ([-+0-9.]+);", line)
        if m:
            R = float(m.group(1))
            acc = m.group(3) == "ACCEPTED"
            print(f"{f:24s} R {R:5.1f} rings {m.group(2):>3s} {m.group(3)}")
            if acc:
                G.setdefault(R, float(m.group(4)))
Rs = np.array(sorted(G))
Gt = np.array([G[r] for r in Rs])


def fit(lo, Dfree=False, nE=3):
    s = Rs >= lo
    R = Rs[s]
    y = Gt[s] - (0 if Dfree else 4 * math.pi * R**2 * D)
    cols = ([R**2] if Dfree else []) + [R] + [R**-j for j in range(nE)]
    X = np.c_[cols].T
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    return c, np.abs(X @ c - y).max(), s.sum()


for lo in [15, 20]:
    c, r, n = fit(lo)
    print(
        f"R >= {lo} ({n} states) 4-par: a3 = {c[0]/(8*math.pi):.7f}  E0_3 {c[1]:+.5f} E1 {c[2]:+.4f} E2 {c[3]:+.3f} maxres {r:.1e}"
    )
for lo in [15, 20]:
    c, r, n = fit(lo, nE=2)
    print(f"R >= {lo} 3-par: a3 = {c[0]/(8*math.pi):.7f} maxres {r:.1e}")
c, r, n = fit(15, Dfree=True)
print(
    f"control R>=15, R^2 free: {c[0]:.8f} vs 4 pi D {4*math.pi*D:.8f} rel {c[0]/(4*math.pi*D)-1:+.1e}; a3 {c[1]/(8*math.pi):.7f}"
)
print("H_c band [-0.041468, -0.041408]; H_e band [-0.041373, -0.041284]")
