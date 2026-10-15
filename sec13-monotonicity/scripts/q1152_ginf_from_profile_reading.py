"""q1152 (7 Oct 2026, registered in amendment 1893): READING (no verdict) T126c. g_inf = G(0) = f'(0) / (4 pi), G(xi) = f(xi) / (4 pi xi), from the extrapolation of the universal profile in xi.
Estimators (G fitted on xi): E1 linear, first atoms (j = 1) of all families; E2 quadratic, j = 1; E3 linear, j <= 2 and xi <= 0.25; E4 quadratic, j <= 2 and xi <= 0.25; E5 local quadratic regression of f / xi on xi, all atoms with xi <= 0.35 (value at 0).
Spread: leave-one-family-out jackknife. Controls S1, S2 on synthetic profiles first. Usage: py q1152_ginf_from_profile_reading.py [--registration-run]
"""

import sys, math
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "paper_births/figures")
import figsrc

P4 = 4 * math.pi
ONE6 = 1 / 6
KMEAN = 0.16623


def pts(fams, jmax, xmax):
    out = []
    for name, K, xi, f in fams:
        for j in range(min(jmax, len(xi))):
            if xi[j] <= xmax:
                out.append(((name, K), xi[j], f[j] / (P4 * xi[j])))
    return out


def est(points, order):
    x = np.array([p[1] for p in points])
    y = np.array([p[2] for p in points])
    c = np.polyfit(x, y, order)
    return float(c[-1]), float(np.sqrt(np.mean((np.polyval(c, x) - y) ** 2)))


def estimator(fams, which):
    spec = {"E1": (1, 9, 1), "E2": (1, 9, 2), "E3": (2, 0.25, 1), "E4": (2, 0.25, 2), "E5": (10, 0.35, 2)}[
        which
    ]
    jmax, xmax, order = spec
    return est(pts(fams, jmax, xmax if jmax > 1 else 9), order)


def with_jack(fams, which):
    v, r = estimator(fams, which)
    keys = sorted({(n, K) for n, K, _, _ in fams})
    loo = []
    for k in keys:
        sub = [t for t in fams if (t[0], t[1]) != k]
        loo.append(estimator(sub, which)[0])
    loo = np.array(loo)
    n = len(keys)
    se = float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))
    return v, se, r


def synth(fams, c1):
    out = []
    for name, K, xi, f in fams:
        xi = np.array(xi)
        f = (
            1
            - (1 + c1 * xi + 8.0 * xi**2) * np.exp(-2 * math.pi * xi)
            + np.random.default_rng(K).normal(0, 0.0005, xi.size)
        )
        out.append((name, K, list(xi), list(f)))
    return out


def main():
    prof, _ = figsrc.event_profiles(10)
    if not prof:
        print("READING q1152: VOID (no family read)")
        sys.exit(2)
    S1 = synth(prof, 4.0734)
    S2 = synth(prof, 4 * math.pi / 3)
    c1 = {w: abs(estimator(S1, w)[0] - 0.17585) < 0.003 for w in ("E1", "E2", "E3", "E4", "E5")}
    c2 = {w: abs(estimator(S2, w)[0] - ONE6) < 0.003 for w in ("E1", "E2", "E3", "E4", "E5")}
    print("controls: S1 (cubic profile, G(0) 0.17585)", c1, "| S2 (f'(0) = 2 pi / 3, G(0) 1/6)", c2)
    if not (all(c1.values()) and all(c2.values())):
        print("READING q1152: VOID (a control does not behave)")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("READING q1152: controls only")
        sys.exit(0)
    print("families:", len(prof), "| first-atom points:", len(pts(prof, 1, 9)))
    for w in ("E1", "E2", "E3", "E4", "E5"):
        v, se, r = with_jack(prof, w)
        print(
            f"{w}: G(0) {v:.5f}, jackknife se {se:.5f}, residual rms {r:.5f}; from 1/6: {(v - ONE6) / se:+.2f} se; from {KMEAN}: {(v - KMEAN) / se:+.2f} se"
        )
    print("END READING q1152 (no verdict)")


if __name__ == "__main__":
    main()
