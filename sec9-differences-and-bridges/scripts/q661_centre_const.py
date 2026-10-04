"""q661: empirical centre constant e = 2 N(A) - K_eff at each event, N from the d-dimensional bulk law fitted to the state's own bulk
rings (c = d - 1 fixed, two free constants, rings with depth >= 6 and radius >= 6) and integrated to depth A (the centre).
Control: the same procedure on 1D capacity states at their transitions (c = 0; c60_small_K*.json, last converged step before the event):
the 1D births law says e = 1/6 + delta(rho) there.
Usage: py q661_centre_const.py control | 2d FILE | 3d FILE | 4d FILE | ..."""

import sys, json, glob
import numpy as np
from scipy.integrate import solve_ivp

src = open("q653_bulk2d_fit.py", encoding="utf-8").read()
src = src[: src.index('if sys.argv[1] == "control":')]
_argv = sys.argv
sys.argv = ["q653", "none"]
ns = {}
exec(src, ns)
sys.argv = _argv  # q653 definitions, copied as text
PI = np.pi
s0 = ns["s0"]
f1 = ns["f1"]


def centre_e(A, depths, c, K):
    d = np.sort(depths)
    N = np.arange(len(d)) + 0.5
    import os

    m = (d >= 6.0) & (A - d >= float(os.environ.get("RMIN", 6.0)))
    if m.sum() < 3:
        return np.nan, np.nan, np.nan, int(m.sum())
    rms, p = ns["fit"](A, d[m], N[m], c)
    import os

    if os.environ.get("FORM") == "corr":  # 1007 T2: arctan at r + S/rho, S from the law (one step)

        def rhs(x, y):
            r = max(A - x, 1e-12)
            S = f1(y[1]) - c * np.arctan(4 * PI * y[1] / r) / (2 * PI)
            return [
                y[1],
                s0 / y[1] ** 2 * (f1(y[1]) - c * np.arctan(4 * PI * y[1] / (r + S / y[1])) / (2 * PI)),
            ]

    else:
        rhs = lambda x, y: [
            y[1],
            s0 / y[1] ** 2 * (f1(y[1]) - c * np.arctan(4 * PI * y[1] / max(A - x, 1e-12)) / (2 * PI)),
        ]
    s = solve_ivp(rhs, (d[m][0], A), [p[0], p[1]], rtol=1e-11, atol=1e-13, method="DOP853")
    return 2 * s.y[0][-1] - K, s.y[1][-1], rms, int(m.sum())


if sys.argv[1] == "control":
    for f in sorted(glob.glob("c60_small_K*.json"), key=lambda x: int(x.split("K")[-1][:-5])):
        D = json.load(open(f))
        K = D["K"]
        h = [x for x in D["hist"] if x["conv"]]
        st = [x for x in h if x["A"] <= D["A_t"]][-1]
        A = st["A"]
        p = np.array(st["p"], float)
        depths = np.concatenate([[0.0], A - np.sort(p)[::-1]])  # wall atom and interior atoms from one edge
        e, rho, rms, n = centre_e(A, depths, 0.0, K)
        v = 4 * PI * rho
        delta = 3.0879296713 / v**4 + 50.783470776 / v**6 + 535.47708947 / v**8 - 4254.4559505 / v**10
        print(
            f"1D K {K:2d} A {A:9.5f}: e = {e:.5f}  (1/6 + delta(rho) = {1/6 + delta:.5f}, rho_c {rho:.4f}; fit rms {rms:.1e} on {n})"
        )
else:
    dim = {"2d": 2, "3d": 3}.get(sys.argv[1]) or int(sys.argv[1].rstrip("d"))  # 2d, 3d, 4d, 5d ...
    ev = json.load(open(sys.argv[2]))["events"]
    K = 3
    for e_ in ev:
        if "r" not in e_:
            continue
        A = e_["A_lo"]
        r = np.array(e_["r"])
        depths = A - r[r > 1e-12] if "o" in e_["types"] else A - r
        K = sum(
            1 if t == "o" else 2 for t in e_["types"]
        )  # K_eff from the event state itself (runs start at different counts)
        if A >= 12:
            e, rho, rms, n = centre_e(A, depths, float(dim - 1), K)
            print(
                f"{dim}D {e_['kind']:8s} K_eff {K:2d}->{K + 1:2d} A {A:9.5f}: e = 2N(A) - K_eff = {e:+.5f}  rho_c {rho:.4f}  (fit rms {rms:.1e} on {n})"
            )
        K += 2 if e_["kind"] == "interior" else 1
