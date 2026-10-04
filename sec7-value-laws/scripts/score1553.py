import numpy as np, os, re, math

os.chdir(r"centre_scripts")


def pieces(dens):
    on = dens > 0.01 * dens.max()
    if on.all():
        return 1, []
    # count maximal runs on a circle
    n = len(on)
    starts = [i for i in range(n) if on[i] and not on[i - 1]]
    gaps = [i for i in range(n) if not on[i]]
    return len(starts), gaps


def wall1d(A):
    t = open(f"q850_scan1546/A{A}.log").read()
    m = re.search(r"walls w ([0-9.]+), ([0-9.]+)", t)
    return float(m.group(1))


for E in (0.5, 1.0):
    f = f"q859_runs/e{E:g}.npz"
    if not os.path.exists(f):
        print(f, "missing")
        continue
    z = np.load(f)
    wb = z["wb"]
    MO = int(z["MO"])
    bx, by = z["bx"], z["by"]
    for name, sl, cx, rad in (("outer", slice(0, MO), 0.0, 15.0), ("inner", slice(MO, None), E, 10.0)):
        ang = np.degrees(np.arctan2(by[sl], bx[sl] - cx)) % 360
        w = wb[sl]
        dens = np.array([w[(ang >= a) & (ang < a + 2.5)].sum() for a in np.arange(0, 360, 2.5)]) / (
            rad * math.radians(2.5)
        )
        npc, gaps = pieces(dens)
        print(
            f"E {E} {name}: wall mass {w.sum():.4f}; pieces {npc}{' (gaps at bins ' + str([g*2.5 for g in gaps][:12]) + ')' if gaps else ''}; "
            f"density narrow side (0 deg) {dens[0]:.4e}, wide side (180 deg) {dens[72]:.4e}, ratio wide/narrow {dens[72]/dens[0]:.3f}; min/max {dens.min()/dens.max():.3f}"
        )
    A0, A1 = round((15 - E - 10) / 2, 2), round((15 + E - 10) / 2, 2)
    try:
        print(
            f"   1D wall mass at A {A0}: {wall1d(A0):.6f}, at A {A1}: {wall1d(A1):.6f}, ratio {wall1d(A1)/wall1d(A0):.3f} (reported only)"
        )
    except Exception as ex:
        print("   1D wall masses unavailable:", ex)
