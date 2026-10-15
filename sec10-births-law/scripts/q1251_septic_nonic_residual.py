"""q1251 (8 Oct 2026, registered in amendment 2035): convergence of the residual dev_med with the degree of the interpolating spline (k = 5, 7, 9) and the parity rule of the alternation for k = 7.
Same atoms and stored amplitudes as q1250 (functions imported from it).
Controls (first): C1 |dev(k=3) - dev(k=5)| at capacity K_b 24 above 1e-9 (positive: the instrument sees a change of interpolant); C2 dev(k=5) recomputed twice is identical (negative: no hidden randomness).
Items: W1 capacity K_b 30 to 50: max |dev5 - dev7| < 0.01e-9; W2 capacity K_b 30 to 50: max |dev7 - dev9| < 0.01e-9; W3 parity rule False for k = 7 in capacity, NPMLE, immune; W4 max |alt(k=7)| below 0.1 in NPMLE and immune.
Usage: py q1251_septic_nonic_residual.py  (from temporal_within/centre_scripts/)"""

import json, sys
import numpy as np
import q1250_quintic_residual as Q


def devs(name, k, NL, stored):
    Nf = Q.make(NL[name][0], NL[name][1], k)
    return {K: (a - Q.root(Nf, K)) * 1e9 for K, a in stored[name].items()}


def main():
    C, N = Q.C, Q.N
    capN = lambda a: C.Q.N_of(a, *C.CAPL)
    sn = [s for s in json.load(open("c33_immc_D146.json"))["stages"] if s["conv"] and s["D"] == 92.0][0]
    pi_ = np.array(sn["p"][: sn["nfree"]])
    pi_ = pi_[pi_ <= 72]
    NL = {
        "capacity": (capN, C.X),
        "npmle": (lambda a: Q.N4(a, N.LAW), N.pn),
        "immune": (lambda a: Q.N4(a, (0.22573, 0.06806)), pi_),
    }
    m = __import__("q1216_derived_series_lower_kb")
    gv = lambda f: (lambda j: j.get("a_t") or j.get("A_t"))(json.load(open(f + ".json")))
    stored = {
        "capacity": {K: gv(f) for K, f in m.CAP_FILES.items() if 10 <= K <= 59},
        "npmle": {K: gv(f) for K, f in m.NPM_FILES.items() if 6 <= K <= 52},
    }
    ev = {int(e[1]): float(e[0]) for e in json.load(open("q624_immune_table.json"))}
    for K in range(29, 38):
        ev[K] = json.load(open(f"c19_u1231_K{K}.json"))["a_t"]
    stored["immune"] = {K: ev[K] for K in range(24, 49) if K in ev}
    d = {(n, k): devs(n, k, NL, stored) for n in NL for k in (3, 5, 7, 9)}
    c1 = abs(d[("capacity", 3)][24] - d[("capacity", 5)][24]) > 1.0
    d5b = devs("capacity", 5, NL, stored)
    c2 = all(d5b[K] == d[("capacity", 5)][K] for K in d5b)
    print(
        f"control C1 (|dev3 - dev5| at K_b 24 = {abs(d[('capacity', 3)][24] - d[('capacity', 5)][24]):.4f}e-9 > 1): {c1}; control C2 (repeat identical): {c2}"
    )
    if not (c1 and c2):
        print("READING q1251: VOID (control)")
        sys.exit(2)
    r = range(30, 51)
    w1 = max(abs(d[("capacity", 5)][K] - d[("capacity", 7)][K]) for K in r)
    w2 = max(abs(d[("capacity", 7)][K] - d[("capacity", 9)][K]) for K in r)
    print(
        f"capacity K_b 30-50: max |dev5 - dev7| {w1:.5f}e-9 (W1 < 0.01: {w1 < 0.01}); max |dev7 - dev9| {w2:.5f}e-9 (W2 < 0.01: {w2 < 0.01})"
    )
    print(
        "reading: max |dev5 - dev7| capacity K_b 24-29: "
        + ", ".join(f"{K}: {abs(d[('capacity', 5)][K] - d[('capacity', 7)][K]):.4f}" for K in range(24, 30))
        + "; NPMLE K_b 35-47: "
        + ", ".join(
            f"{K}: {abs(d[('npmle', 5)][K] - d[('npmle', 7)][K]):.5f}"
            for K in range(35, 48)
            if K in d[("npmle", 5)]
        )
    )
    ok3, ok4 = True, True
    for n, (lo, hi, fl) in (
        ("capacity", (24, 46, 0.05)),
        ("npmle", (34, 48, 0.05)),
        ("immune", (39, 47, 0.1)),
    ):
        a = Q.alts(d[(n, 7)], lo, hi, fl)
        p = Q.parity(a)
        mx = max(abs(x) for x in a.values())
        print(
            f"{n} k = 7: {len(a)} triples; alt "
            + ", ".join(f"{K}: {x:+.3f}" for K, x in sorted(a.items()))
            + f"; parity {p}; max |alt| {mx:.3f}"
        )
        ok3 = ok3 and not p
        if n != "capacity":
            ok4 = ok4 and mx < 0.1
    print(f"item W1: {w1 < 0.01}; W2: {w2 < 0.01}; W3: {ok3}; W4: {ok4}")
    json.dump(
        {f"{n}_{k}": {str(K): v for K, v in x.items()} for (n, k), x in d.items()},
        open("q1251_dev.json", "w"),
    )
    print(f"READING q1251: W1 {w1 < 0.01}, W2 {w2 < 0.01}, W3 {ok3}, W4 {ok4} (no verdict)")


if __name__ == "__main__":
    main()
