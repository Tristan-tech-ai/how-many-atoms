"""q1114 (7 Oct 2026): READING (no verdict): the logarithmic flow g = (dp_1/dA) / (p_1 h), h = u_2 - u_1, of the innermost position along each capacity family K 63..73 and its value at the event that ends the family.
p_1(A) and h(A) come from the state jsons listed in scratch/tower_family_list_K<K>.txt (path of the json after the A value); dp_1/dA is the Lagrange derivative at the last node of four consecutive states (error of order step^3, the steps are the gaps of the list). The event that ends the family: the A_s of tower_K<K+1>_As.json for odd K (split; K 69 from tower_K70_As_T83.json), the A_cross of tower_K<K+1>_birth_out.json for even K (birth).
Prints per family: the number of states, the distance d of the last state to the event, g at the last six states with their distances, and the linear extrapolation of g to d = 0 from the last two windows (the last two states). Counts what it reads; VOID on a missing file or zero states.
Usage: py q1114_log_flow_along_family_reading.py"""

import json, os, sys

ARCH = "scratch"


def deriv_last(a, p):
    n = len(a)
    x = a[-1]
    tot = 0.0
    for i in range(n):
        s = 0.0
        for k in range(n):
            if k == i:
                continue
            prod = 1.0
            for m in range(n):
                if m != i and m != k:
                    prod *= x - a[m]
            s += prod
        den = 1.0
        for m in range(n):
            if m != i:
                den *= a[i] - a[m]
        tot += p[i] * s / den
    return tot


def event(K):
    if K % 2 == 1:
        f = f"{ARCH}/tower_K{K + 1}_As.json"
        key = "A_s"
        if K == 69:
            f = f"{ARCH}/tower_K70_As_T83.json"
    else:
        f = f"{ARCH}/tower_K{K + 1}_birth_out.json"
        key = "A_cross"
    if not os.path.exists(f):
        return None, f
    return float(json.load(open(f))[key]), f


def main():
    allg = {1: [], 0: []}
    KMAX = (
        int(sys.argv[1]) if len(sys.argv) > 1 else 74
    )  # 7 Oct 16:24: optional upper bound (exclusive); the default reproduces the first output
    for K in range(63, KMAX):
        lst = f"{ARCH}/tower_family_list_K{K}.txt"
        if not os.path.exists(lst):
            if K >= 74:
                continue
            print(f"VOID q1114: {lst} missing")
            sys.exit(2)
        st = []
        for line in open(lst):
            t = line.split()
            if len(t) < 2:
                continue
            d = json.load(open(t[1]))
            u = [float(x) for x in d["u"]]
            st.append((float(d["A"]), u[0], u[1] - u[0]))
        if len(st) < 8:
            print(f"VOID q1114: K {K} has {len(st)} states")
            sys.exit(2)
        Ae, fe = event(K)
        if Ae is None:
            print(f"K {K}: event file {fe} missing, skipped")
            continue
        gs = []
        for i in range(3, len(st)):
            a = [st[j][0] for j in range(i - 3, i + 1)]
            p = [st[j][1] for j in range(i - 3, i + 1)]
            gs.append((st[i][0], deriv_last(a, p) / (st[i][1] * st[i][2])))
        last = gs[-6:]
        (a1, g1), (a2, g2) = gs[-2], gs[-1]
        slope = (g2 - g1) / (a2 - a1)
        g0 = g2 + slope * (Ae - a2)
        print(
            f"K {K}: {len(st)} states, event {Ae:.6f}, last state {a2:.4f}, distance {Ae - a2:.4f}; g at the last six states: "
            + ", ".join(f"{g:.5f} (d {Ae - a:.4f})" for a, g in last)
            + f"; linear extrapolation to d = 0: {g0:.5f}"
        )
        allg[K % 2].append((g2, g0))
    for par, name in ((1, "odd"), (0, "even")):
        v = allg[par]
        if v:
            print(
                f"{name} K: last-state g mean {sum(x[0] for x in v) / len(v):.5f} (min {min(x[0] for x in v):.5f}, max {max(x[0] for x in v):.5f}); extrapolated to the event mean {sum(x[1] for x in v) / len(v):.5f} (min {min(x[1] for x in v):.5f}, max {max(x[1] for x in v):.5f}); n {len(v)}"
            )
    print("number to compare with: 1/6 =", f"{1 / 6:.7f}")
    print("END q1114 READING (no verdict)")


if __name__ == "__main__":
    main()
