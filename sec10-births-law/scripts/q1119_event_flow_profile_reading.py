"""q1119 (7 Oct 2026): READING (no verdict): the flows f_j = dp_j/da of the innermost six positions at the event that ends the family, in the capacity problem (families K 63 to 73, the states of tower_family_list_K<K>.txt) and in the two chains (completed families with steps of at most 0.0125 and a last row within 0.03 of the event, as in q1115).
f_j at each of the last two states from the Lagrange derivative through four consecutive states; linear extrapolation in a to the event (events as in q1114 and q1115). Prints for each family: K, p_1 / h, and f_1 ... f_6 at the event, the ratios f_j / f_1, and the products f_j / (p_j h) (the generalisation of g to the atom j), then per problem and parity the mean of each column over the families.
Counts what it reads; VOID on zero families. Usage: py q1119_event_flow_profile_reading.py"""

import json, os, sys
from q1113_log_flow_reading import deriv_last, families
from q1114_log_flow_along_family_reading import event

ARCH = "scratch"
NJ = 6


def flows_at_event(st, Ae):
    """st: list of (A, [p_1..p_NJ+1]); returns (f_1..f_NJ at the event, p at the last state)"""
    out = []
    for i in (len(st) - 2, len(st) - 1):
        a = [st[j][0] for j in range(i - 3, i + 1)]
        out.append((a[-1], [deriv_last(a, [st[j][1][k] for j in range(i - 3, i + 1)]) for k in range(NJ)]))
    (a1, f1), (a2, f2) = out
    return [f2[k] + (f2[k] - f1[k]) / (a2 - a1) * (Ae - a2) for k in range(NJ)], st[-1][1]


def report(tag, K, f, p):
    h = p[1] - p[0]
    print(
        f"  {tag} K {K}: p_1/h {p[0] / h:.4f}; f_1..f_6 "
        + " ".join(f"{x:.5f}" for x in f)
        + "; f_j/f_1 "
        + " ".join(f"{x / f[0]:.4f}" for x in f)
        + "; f_j/(p_j h) "
        + " ".join(f"{f[k] / (p[k] * h):.5f}" for k in range(NJ))
    )
    return [f[0]] + [x / f[0] for x in f[1:]] + [f[k] / (p[k] * h) for k in range(NJ)]


def mean_cols(tag, rows):
    if rows:
        m = [sum(r[i] for r in rows) / len(rows) for i in range(len(rows[0]))]
        print(
            f"  {tag} mean over {len(rows)} families: f_1 {m[0]:.5f}; f_j/f_1 (j = 2..6) "
            + " ".join(f"{x:.4f}" for x in m[1:NJ])
            + "; f_j/(p_j h) (j = 1..6) "
            + " ".join(f"{x:.5f}" for x in m[NJ:])
        )


def main():
    used = 0
    acc = {}
    print("capacity")
    for K in range(63, 74):
        lp = f"{ARCH}/tower_family_list_K{K}.txt"
        Ae, fe = event(K)
        if not os.path.exists(lp) or Ae is None:
            print(f"VOID q1119: missing {lp} or {fe}")
            sys.exit(2)
        st = []
        for line in open(lp):
            t = line.split()
            if len(t) < 2:
                continue
            d = json.load(open(t[1]))
            u = [float(x) for x in d["u"]]
            st.append((float(d["A"]), u[: NJ + 1]))
        f, p = flows_at_event(st, Ae)
        acc.setdefault(("capacity", K % 2), []).append(report("capacity", K, f, p))
        used += 1
    for name, fn in (("immune", "results_q584_ext.json"), ("NPMLE", "results_q601_ext.json")):
        print(name)
        d = json.load(open(fn))
        fam = families(d["rows"])[:-1]
        ev = {}
        for e in d["events"]:
            if e["kind"] in ("centre appear", "centre split"):
                ev.setdefault(e["K_before"], e["a_lo"])
        for c in fam:
            K = c[-1]["K"]
            if K not in ev or len(c) < 8:
                continue
            a = [r["a"] for r in c[-7:]]
            st = [a[i + 1] - a[i] for i in range(6)]
            if max(st[1:]) > 0.0125 or ev[K] - a[-1] > 0.03:
                continue
            S = [(r["a"], [float(x) for x in r["p"][: NJ + 1]]) for r in c[-8:]]
            f, p = flows_at_event(S, ev[K])
            acc.setdefault((name, K % 2), []).append(report(name, K, f, p))
            used += 1
    if used == 0:
        print("VOID q1119: zero families")
        sys.exit(2)
    print("means")
    for (tag, par), rows in sorted(acc.items()):
        mean_cols(f"{tag} {'odd' if par else 'even'}", rows)
    print(f"families used {used}")
    print("END q1119 READING (no verdict)")


if __name__ == "__main__":
    main()
