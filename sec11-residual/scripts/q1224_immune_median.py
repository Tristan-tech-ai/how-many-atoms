"""q1224 (8 Oct 2026, registered in amendment 1982): the third functional (immune) through the median-sum law of q1220.
N_half: Ni of q1201 (c33_immc_D146 stage D = 92, atoms with depth <= 72, edge numbers (0.22573, 0.06806)). Stored events: q624_immune_table.json (45 transitions K_b 8 to 99; entry = [a_t, K_b, rho, error]).
dev_med = a_t - a_med with a_med the root of 2 Ni(a) - K_b - 1/6 - S_med(1/v^2) = 0 (table of q1220, theta 0.3), v = 4 pi dNi/da, bracket +-2.5 around the zeroth-order root.
Item I1: dev_med > 0 at every stored event with K_b >= 14 (the sign of all 40 NPMLE events). Readings without item: dev_med of the events, the ratio dev_med(immune)/dev_med(NPMLE) at the NPMLE events K_b 40, 44, 48 interpolated linearly in K_b of the immune table, and kappa of q1222 for the immune events K_b 14 to 48.
Controls (first): the hash of q1220_table.json equals c5102c2a...; the root function with S_med = 0 returns the zeroth-order root; theta 0.5 changes a_med by less than 1e-10 at K_b 40.
Usage: py q1224_immune_median.py  (from temporal_within/centre_scripts/)"""

import hashlib, json, sys
import numpy as np
from scipy.optimize import brentq

sys.argv = ["x"]
import q1220_borel_pade_births_law as Q20
import q1222_stokes_ratio as Q22
import mpmath as mp

G = Q20.G
T = G.T
Ni = T.Ni
H = G.H
REG = "c5102c2a416bbf077013b7f7c2a7c30a44646d02e18361bbc3919eba4f6a0564"


def root(tab, K):
    z = lambda a: 2 * Ni(a) - K - 1 / 6
    a0 = brentq(z, 1.0, 140.0, xtol=1e-12)
    if tab is None:
        return a0
    f = lambda a: z(a) - float(tab(1.0 / (4 * np.pi * (Ni(a + H) - Ni(a - H)) / (2 * H)) ** 2))
    return brentq(f, a0 - 2.5, a0 + 2.5, xtol=1e-14)


def main():
    js = json.load(open("q1220_table.json"))
    sh = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    T3 = np.polynomial.chebyshev.Chebyshev(js["0.3"], domain=js["domain"])
    T5 = np.polynomial.chebyshev.Chebyshev(js["0.5"], domain=js["domain"])
    a0 = root(None, 40)
    c1 = h == sh == REG
    c2 = abs(2 * Ni(a0) - 40 - 1 / 6) < 1e-10
    c3 = abs(root(T3, 40) - root(T5, 40)) < 1e-10
    print(
        f"control: table hash equals the registered one: {c1}; zeroth-order root satisfies 2 N = K + 1/6: {c2}; theta 0.3 against 0.5 at K_b 40: {c3}"
    )
    if not (c1 and c2 and c3):
        print("READING q1224: VOID (control)")
        sys.exit(2)
    tb = json.load(open("q624_immune_table.json"))
    ev = {int(e[1]): float(e[0]) for e in tb}
    dev = {K: a - root(T3, K) for K, a in sorted(ev.items())}
    print("immune dev_med (1e-9):", ", ".join(f"{K}: {d * 1e9:+.4g}" for K, d in dev.items()))
    i1 = all(d > 0 for K, d in dev.items() if K >= 14)
    print(
        f"item I1 (dev_med > 0 at every stored immune event with K_b >= 14; {sum(1 for K in dev if K >= 14)} events): {i1}; negative at K_b: {[K for K, d in dev.items() if K >= 14 and d <= 0]}"
    )
    npm = {}
    import re

    for line in open("../scratch/q1220_out.txt"):
        m = re.match(r"npmle K_b (\d+): dev_med ([-+\d.]+)e-9", line)
        if m:
            npm[int(m.group(1))] = float(m.group(2)) * 1e-9
    ks = sorted(K for K in dev if K >= 14)
    for K in (40, 44, 48):
        if K in npm and K >= ks[0]:
            lo = max(k for k in ks if k <= K)
            hi = min(k for k in ks if k >= K)
            w = 0 if hi == lo else (K - lo) / (hi - lo)
            di = (1 - w) * dev[lo] + w * dev[hi]
            print(
                f"ratio immune/NPMLE at K_b {K}: immune (interpolated between {lo} and {hi}) {di * 1e9:+.4f}e-9, NPMLE {npm[K] * 1e9:+.4f}e-9, ratio {di / npm[K]:.3f}"
            )
    Bf, p, q = Q20.machine(Q20.coef_series(), 8, 8)
    rt = mp.polyroots(list(reversed(q)), maxsteps=200, extraprec=200)
    poles = [mp.re(z) for z in rt if abs(mp.im(z)) < 1e-20 and mp.re(z) > 0]
    kap = {}
    for K in ks:
        if K > 48:
            continue
        a = ev[K]
        v = 4 * np.pi * (Ni(a + H) - Ni(a - H)) / (2 * H)
        kap[K] = 2 * (v / (4 * np.pi)) * dev[K] / (Q22.jump(v, poles, p, q) / 2)
    print(
        "kappa immune (q1222 definition), K_b 14 to 48:", ", ".join(f"{K}: {x:.3f}" for K, x in kap.items())
    )
    print("READING q1224: I1 " + str(i1) + " (no verdict)")


if __name__ == "__main__":
    main()
