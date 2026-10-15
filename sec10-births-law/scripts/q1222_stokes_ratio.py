"""q1222 (8 Oct 2026, registered in amendment 1980): is the residual dev_med of q1220 a constant multiple of the jump of the lateral Borel sums across the real poles of the Pade approximant?
J(x) = 2 pi |sum over real positive poles y_k of B of t_k^(b-1) exp(-t_k) r_k / x|, t_k = y_k / x, r_k the residue of B at y_k, b = 1.5, B the [8/8] Pade of q1220 (coefficients e_2 .. e_18).
kappa(K_b) = 2 rho dev_med / (J/2), rho = v/(4 pi), dev_med in A units, 2 rho dev_med the same residual in units of delta (the law is 2 N - K - 1/6 - delta = 0).
Item K1: for NPMLE K_b 14 to 40 (stored events, dev_med > 0) max(kappa)/min(kappa) < 1.15. Capacity K_b 24 to 50 (dev_med > 0): reported, no item.
Controls (first): synthetic dev_med = 0.3 (J/2)/(2 rho) returns a spread below 1.0001; synthetic dev_med proportional to exp(-v^2/9) returns a spread above 1.15.
Usage: py q1222_stokes_ratio.py  (from temporal_within/centre_scripts/)"""

import json, re
import numpy as np
import mpmath as mp

mp.mp.dps = 40
import q1220_borel_pade_births_law as Q20


def jump(v, poles, p, q):
    x = 1 / mp.mpf(v) ** 2
    s = mp.mpc(0)
    for y0 in poles:
        r = y0**2 * Q20.horner(p, y0) / Q20.horner([k * q[k] for k in range(1, len(q))], y0)
        t0 = y0 / x
        s += t0 ** (Q20.B - 1) * mp.exp(-t0) * r / x
    return float(2 * mp.pi * abs(s))


def main():
    Bf, p, q = Q20.machine(Q20.coef_series(), 8, 8)
    rt = mp.polyroots(list(reversed(q)), maxsteps=200, extraprec=200)
    poles = [mp.re(z) for z in rt if abs(mp.im(z)) < 1e-20 and mp.re(z) > 0]
    print(
        "real positive poles y_k (1/y_k):", ", ".join(f"{mp.nstr(y, 8)} ({mp.nstr(1 / y, 8)})" for y in poles)
    )
    pred = json.load(open("q1216_predictions.json"))
    dev = {"capacity": {}, "npmle": {}}
    for line in open("../scratch/q1220_out.txt"):
        m = re.match(r"(capacity|npmle) K_b (\d+): dev_med ([-+\d.]+)e-9", line)
        if m:
            dev[m.group(1)][int(m.group(2))] = float(m.group(3)) * 1e-9
    vv = lambda n, k: pred[n][str(k)]["v"]
    kap = lambda d, n, k: 2 * (vv(n, k) / (4 * np.pi)) * d / (jump(vv(n, k), poles, p, q) / 2)
    ks = [k for k in (14, 16, 18, 20, 22, 24, 26, 28, 31, 34, 36, 38, 40)]
    c1 = [
        kap(0.3 * jump(vv("npmle", k), poles, p, q) / 2 / (2 * vv("npmle", k) / (4 * np.pi)), "npmle", k)
        for k in ks
    ]
    c2 = [kap(1e-9 * np.exp(-vv("npmle", k) ** 2 / 9), "npmle", k) for k in ks]
    ok = max(c1) / min(c1) < 1.0001 and max(c2) / min(c2) > 1.15
    print(
        f"control: synthetic constant ratio spread {max(c1) / min(c1):.6f}; synthetic exp(-v^2/9) spread {max(c2) / min(c2):.3f}: {ok}"
    )
    if not ok:
        print("READING q1222: VOID (control)")
        raise SystemExit(2)
    res = {}
    for name, lo, hi in (("npmle", 14, 40), ("capacity", 24, 50)):
        k_ = [k for k in sorted(dev[name]) if lo <= k <= hi and dev[name][k] > 0]
        kp = [kap(dev[name][k], name, k) for k in k_]
        res[name] = max(kp) / min(kp)
        print(
            f"{name} K_b {lo} to {hi} ({len(k_)} events): kappa min {min(kp):.4g} max {max(kp):.4g} spread {res[name]:.3f}; kappa at K_b {k_[0]}, {k_[len(k_) // 2]}, {k_[-1]}: {kp[0]:.4g}, {kp[len(k_) // 2]:.4g}, {kp[-1]:.4g}"
        )
    print(f"item K1 (NPMLE spread < 1.15): {res['npmle'] < 1.15}")
    print("READING q1222: K1 " + str(res["npmle"] < 1.15) + " (no verdict)")


if __name__ == "__main__":
    main()
