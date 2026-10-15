"""q1218 (8 Oct 2026, prepared under amendment 1974): slope of the logarithm of the deviation of the births law (derived series, order 16) against v^2, the exponent of a term of the size exp(-v^2/s) with s from the large-order growth of the series.
Data: dev(16) of scratch/q1216_out.txt and v of q1216_predictions.json (the reading of amendment 1973).
s: the coefficient alpha of the model r_n = e_(n+1)/e_n = alpha n + beta + gamma/n of q1217 (alpha = 4.4450 from n = 10 to 17); the expected exponent is -1/alpha = -0.2250.
Fits (ordinary least squares of ln(deviation in 1e-9) on v^2): NPMLE K_b 12 to 28 (all deviations positive), capacity K_b 24 to 39 (all deviations negative; ln of the absolute value).
Items: X1a the NPMLE slope lies in [-0.259, -0.191] (15 percent around -1/alpha); X1b the capacity slope lies in the same interval.
Controls (first): the fit routine returns a slope inside the interval for synthetic data y = 5 - 0.225 x with noise 0.05, and outside it for y = 5 - 0.6 x with the same noise.
Usage: py q1218_excess_slope.py   (from temporal_within/centre_scripts/)"""

import json, re
import numpy as np

ALPHA = 4.4450
LO, HI = -1 / ALPHA * 1.15, -1 / ALPHA * 0.85


def slope(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    c, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    se = float(np.sqrt(np.sum(r**2) / (len(x) - 2) / np.sum((x - x.mean()) ** 2)))
    return float(c[0]), se


def main():
    rng = np.random.default_rng(1)
    x = np.linspace(60, 100, 17)
    b1, _ = slope(x, 5 - 0.225 * x + 0.05 * rng.standard_normal(17))
    b2, _ = slope(x, 5 - 0.6 * x + 0.05 * rng.standard_normal(17))
    ok = LO < b1 < HI and not (LO < b2 < HI)
    print(f"control: synthetic slope {b1:+.4f} inside [{LO:.3f}, {HI:.3f}] and {b2:+.4f} outside: {ok}")
    if not ok:
        print("READING q1218: VOID (control)")
        raise SystemExit(2)
    pred = json.load(open("q1216_predictions.json"))
    dev = {"capacity": {}, "npmle": {}}
    for line in open("../scratch/q1216_out.txt"):
        m = re.match(
            r"(capacity|npmle) K_b (\d+): A_stored [\d.]+; dev\(12\) [-+\d.]+e-9, dev\(16\) ([-+\d.]+)e-9",
            line,
        )
        if m:
            dev[m.group(1)][int(m.group(2))] = float(m.group(3))
    res = {}
    for name, ks, sign in (("npmle", range(12, 29), 1), ("capacity", range(24, 40), -1)):
        ks = [k for k in ks if k in dev[name]]
        d = np.array([dev[name][k] for k in ks])
        v = np.array([pred[name][str(k)]["v"] for k in ks])
        if not np.all(sign * d > 0):
            print(f"{name}: not all deviations have the sign {sign}")
            raise SystemExit(2)
        b, se = slope(v**2, np.log(sign * d))
        res[name] = LO < b < HI
        print(
            f"{name} K_b {ks[0]} to {ks[-1]} ({len(ks)} events): slope of ln(deviation) against v^2 = {b:+.4f} (standard error {se:.4f}); expected {-1 / ALPHA:+.4f}; interval [{LO:.3f}, {HI:.3f}]; item {'X1a' if name == 'npmle' else 'X1b'}: {res[name]}"
        )
    print("READING q1218: X1a " + str(res["npmle"]) + ", X1b " + str(res["capacity"]) + " (no verdict)")


if __name__ == "__main__":
    main()
