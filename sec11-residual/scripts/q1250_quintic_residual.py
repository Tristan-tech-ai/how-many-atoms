"""q1250 (8 Oct 2026, registered in amendment 2033): the residual of the median-sum law (table q1220) and its birth/split alternation with the counting function interpolated by the cubic spline (as q1220, q1225, q1229) and by the quintic B-spline (k = 5), for the three problems.
Draft reading 4 (finding 1) reports that the alternation and the quantities S and D are the interpolation error of the cubic spline at the births. Test:
 capacity: stored transitions K_b 10 to 59 (files of q1216), half-line c50c D = 110; NPMLE: K_b 6 to 52, c33_npmc_D110; immune: K_b 24 to 48 (q624 table with the precise values K_b 29 to 37 of c19_u1231), stage D = 92 atoms <= 72 of q1201.
 alt(K) = ln dev(K) - (ln dev(K-1) + ln dev(K+1))/2 over triples with all three dev above 0.05e-9 (capacity, NPMLE) or 0.1e-9 (immune), as q1225.
Items (registered expectation of the draft reading, which we have not yet reproduced): Z1 capacity with the quintic spline: the parity rule (alt positive at even K_b and negative at odd K_b, at least five triples) is False and max |alt| over the triples of K_b 24 to 46 other than K_b 38 and 39 is below 0.1; Z2 NPMLE (triples K_b 34 to 48): max |alt| below 0.1; Z3 immune K_b 39 to 47: max |alt| below 0.1.
Controls (first): the cubic variant reproduces dev_med of q1220_out.txt for capacity K_b 40 (+0.4510e-9) and K_b 24 (+17.4265e-9, same table q1220_table.json) within 0.001e-9, and the parity rule with the cubic spline holds for capacity (25 triples) as in q1225_cap2_out.txt.
Usage: py q1250_quintic_residual.py  (from temporal_within/centre_scripts/)"""

import json, os, re, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.optimize import brentq
import q1239_interpolation_sensitivity as C
import q1240_npmle_interpolation as N
import q1224_immune_median as Q24

G = C.G
H = G.H
pi = np.pi
s0 = 1 / (16 * pi**2)
js = json.load(open("q1220_table.json"))
js.pop("sha256")
T3 = np.polynomial.chebyshev.Chebyshev(js["0.3"], domain=js["domain"])


def rho4(d, law):
    return brentq(
        lambda r: r**3 - 3 * r / (8 * pi**2) - 1 / (32 * pi**2 * r) - 3 * s0 * (d + law[0]), 0.02, 400
    )


def N4(d, law):
    r = rho4(d, law)
    return 4 * pi**2 * r**4 - r**2 + np.log(r) / 6 + law[1]


def make(Nlaw, X, k):
    R = np.arange(1, len(X) + 1) - 0.5 - np.array([Nlaw(x) for x in X])
    s = CubicSpline(X, R) if k == 3 else make_interp_spline(X, R, k=k)
    return lambda a: Nlaw(a) + float(s(a))


def root(Nf, K):
    z = lambda a: 2 * Nf(a) - K - 1 / 6
    a0 = brentq(z, 1.0, 140.0, xtol=1e-12)
    f = lambda a: z(a) - float(T3(1.0 / (4 * pi * (Nf(a + H) - Nf(a - H)) / (2 * H)) ** 2))
    return brentq(f, a0 - 2.5, a0 + 2.5, xtol=1e-14)


def alts(dev, lo, hi, floor):
    out = {}
    for K in range(lo, hi + 1):
        if all(k in dev and dev[k] > floor for k in (K - 1, K, K + 1)):
            out[K] = np.log(dev[K]) - (np.log(dev[K - 1]) + np.log(dev[K + 1])) / 2
    return out


def parity(a):
    return len(a) >= 5 and all((x > 0) == (K % 2 == 0) and x != 0 for K, x in a.items())


def main():
    capN = lambda a: C.Q.N_of(a, *C.CAPL)
    impts = {}
    atoms_i = None
    sn = [s for s in json.load(open("c33_immc_D146.json"))["stages"] if s["conv"] and s["D"] == 92.0][0]
    pi_ = np.array(sn["p"][: sn["nfree"]])
    pi_ = pi_[pi_ <= 72]
    LI = (0.22573, 0.06806)
    NL = {
        "capacity": (capN, C.X),
        "npmle": (lambda a: N4(a, N.LAW), N.pn),
        "immune": (lambda a: N4(a, LI), pi_),
    }
    stored = {
        "capacity": {
            K: json.load(open(f + ".json")).get("A_t") or json.load(open(f + ".json")).get("a_t")
            for K, f in __import__("q1216_derived_series_lower_kb").CAP_FILES.items()
            if 10 <= K <= 59
        },
        "npmle": {
            K: (lambda j: j.get("a_t") or j.get("A_t"))(json.load(open(f + ".json")))
            for K, f in __import__("q1216_derived_series_lower_kb").NPM_FILES.items()
            if 6 <= K <= 52
        },
    }
    ev = {int(e[1]): float(e[0]) for e in json.load(open("q624_immune_table.json"))}
    for K in range(29, 38):
        ev[K] = json.load(open(f"c19_u1231_K{K}.json"))["a_t"]
    stored["immune"] = {K: ev[K] for K in range(24, 49) if K in ev}
    dev = {}
    for name in NL:
        for k in (3, 5):
            Nf = make(NL[name][0], NL[name][1], k)
            dev[(name, k)] = {K: (a - root(Nf, K)) * 1e9 for K, a in stored[name].items()}
    c1 = abs(dev[("capacity", 3)][40] - 0.4510) < 0.001 and abs(dev[("capacity", 3)][24] - 17.4265) < 0.001
    a3 = alts(dev[("capacity", 3)], 24, 50, 0.05)
    c2 = parity(a3) and len(a3) >= 20
    print(
        f"control: cubic capacity dev(40) {dev[('capacity', 3)][40]:+.4f}, dev(24) {dev[('capacity', 3)][24]:+.4f} (q1220: +0.4510, +17.4265): {c1}; cubic parity rule holds with {len(a3)} triples: {c2}"
    )
    if not (c1 and c2):
        print("READING q1250: VOID (control)")
        sys.exit(2)
    res = {}
    for name, (lo, hi, fl) in (
        ("capacity", (24, 46, 0.05)),
        ("npmle", (34, 48, 0.05)),
        ("immune", (39, 47, 0.1)),
    ):
        for k in (3, 5):
            a = alts(dev[(name, k)], lo, hi, fl)
            res[(name, k)] = a
            print(
                f"{name} k = {k}: {len(a)} triples; alt "
                + ", ".join(f"{K}: {x:+.3f}" for K, x in sorted(a.items()))
                + f"; parity rule {parity(a)}; max |alt| {max(abs(x) for x in a.values()):.3f}"
            )
    ex = lambda a: [abs(x) for K, x in a.items() if K not in (38, 39)]
    z1 = (not parity(res[("capacity", 5)])) and max(ex(res[("capacity", 5)])) < 0.1
    z2 = max(abs(x) for x in res[("npmle", 5)].values()) < 0.1
    z3 = max(abs(x) for x in res[("immune", 5)].values()) < 0.1
    print(
        "quintic residual (1e-9): capacity K_b 24, 30, 40, 50: "
        + ", ".join(f"{dev[('capacity', 5)][K]:+.4f}" for K in (24, 30, 40, 50))
        + "; NPMLE K_b 20, 28, 40: "
        + ", ".join(f"{dev[('npmle', 5)][K]:+.4f}" for K in (20, 28, 40))
        + "; immune K_b 28, 40, 47: "
        + ", ".join(f"{dev[('immune', 5)][K]:+.4f}" for K in (28, 40, 47))
    )
    print(f"item Z1 (capacity): {z1}; item Z2 (NPMLE): {z2}; item Z3 (immune): {z3}")
    json.dump(
        {f"{n}_{k}": {str(K): v for K, v in d.items()} for (n, k), d in dev.items()},
        open("q1250_dev.json", "w"),
    )
    print("READING q1250: Z1 " + str(z1) + ", Z2 " + str(z2) + ", Z3 " + str(z3) + " (no verdict)")


if __name__ == "__main__":
    main()
