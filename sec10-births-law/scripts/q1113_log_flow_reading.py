"""q1113 (7 Oct 2026): READING (no verdict): the logarithmic flow of the innermost position at the last state of a family, g = (dp_1/da) / (p_1 h) with h = p_2 - p_1, in three problems.
Capacity: u_end, u_1 and h = u_2 - u_1 are parsed from the output q1111_spacing_law_out.txt (u_end has four digits); for even K the position p_1 is u_1 and the output prints 2 u_1, which is read as is.
Immune (results_q584_ext.json) and NPMLE (results_q601_ext.json): the derivative of p_1 against a at the last row of each completed family from the Lagrange formula through the last three rows (error of order step^2) and through the last four rows (order step^3); only families whose last three steps in a are at most 0.0125 are used (the finer steps), the others are counted and named.
Prints g per family and the mean, minimum and maximum per problem and parity; 1/6 = 0.1666667 is printed only as a number to compare with, no claim about its origin.
Usage: py q1113_log_flow_reading.py"""

import json, re, sys


def deriv_last(a, p):
    """derivative at the last node by the Lagrange formula through all given nodes (a list of nodes, p list of values)"""
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


def families(rows):
    fam = []
    cur = [rows[0]]
    for r in rows[1:]:
        if r["K"] == cur[-1]["K"] and r["centre"] == cur[-1]["centre"]:
            cur.append(r)
        else:
            fam.append(cur)
            cur = [r]
    fam.append(cur)
    return fam


def summary(tag, v):
    for par in (True, False):
        x = v[par]
        if x:
            print(
                f"  {tag} centre {par}: g mean {sum(x) / len(x):.5f}, min {min(x):.5f}, max {max(x):.5f}, n {len(x)}"
            )


def main():
    pat = re.compile(r"K (\d+): u_end ([0-9.]+), u_1 ([0-9.]+), u_2 - u_1 ([0-9.]+)")
    cap = {True: [], False: []}
    txt = open("scratch/q1111_spacing_law_out.txt", encoding="utf-8").read()
    rows = pat.findall(txt)
    if not rows:
        print("VOID q1113: zero capacity rows parsed")
        sys.exit(2)
    print(f"capacity: {len(rows)} families parsed from q1111_spacing_law_out.txt")
    for K, ue, u1, h in rows:
        K = int(K)
        ue = float(ue)
        u1 = float(u1)
        h = float(h)
        odd = K % 2 == 1
        p1 = u1 if odd else u1  # u_1 is the first positive atom in both parities
        g = ue / (p1 * h)
        cap[odd].append(g)
        print(f"  K {K}: u_end {ue}, p_1 {p1}, h {h}, g {g:.5f}")
    summary("capacity", cap)
    for name, f in (("immune", "results_q584_ext.json"), ("NPMLE", "results_q601_ext.json")):
        d = json.load(open(f))
        fam = families(d["rows"])[:-1]
        res3 = {True: [], False: []}
        res4 = {True: [], False: []}
        skipped = []
        for c in fam:
            if len(c) < 5:
                skipped.append((c[0]["K"], "rows<5"))
                continue
            a = [r["a"] for r in c[-4:]]
            st = [a[i + 1] - a[i] for i in range(3)]
            if max(st) > 0.0125:
                skipped.append((c[0]["K"], f"step {max(st):.4f}"))
                continue
            p = [r["p"][0] for r in c[-4:]]
            h = c[-1]["p"][1] - c[-1]["p"][0]
            cen = bool(c[-1]["centre"])
            g3 = deriv_last(a[-3:], p[-3:]) / (p[-1] * h)
            g4 = deriv_last(a, p) / (p[-1] * h)
            res3[cen].append(g3)
            res4[cen].append(g4)
            print(
                f"  {name} K {c[-1]['K']} centre {cen}: g (3 nodes) {g3:.5f}, g (4 nodes) {g4:.5f}, last a {a[-1]:.5f}"
            )
        print(f"  {name}: used {sum(len(v) for v in res3.values())} families, skipped {skipped}")
        summary(name + " 3 nodes", res3)
        summary(name + " 4 nodes", res4)
    print("number to compare with: 1/6 =", f"{1 / 6:.7f}")
    print("END q1113 READING (no verdict)")


if __name__ == "__main__":
    main()
