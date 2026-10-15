"""q1117 (7 Oct 2026): verdict script of T125 (registered in the amendment that follows 1858): the logarithmic flow g = (dp_1/dA) / (p_1 h) of the innermost position of a capacity family, extrapolated to the event that ends the family, follows the law fitted on the immune and NPMLE chains only (reading q1116): g_ev(K) = 0.16619 - 0.0815 / K for even K and g_ev(K) = 0.16641 - 0.1863 / K for odd K.
Procedure of g_ev (the procedure of the reading q1114, copied): states from the family list (family_list_K61.txt, family_list_K62.txt, tower_family_list_K<K>.txt for K >= 63), p_1 = u_1 and h = u_2 - u_1 of each state; dp_1/dA at each of the last two states from the Lagrange formula through four consecutive states; linear extrapolation of g from these two states to the event A_ev (A_s of the file tower_K<K+1>_As.json for odd K, K 69 from tower_K70_As_T83.json; A_cross of tower_K<K+1>_birth_out.json for even K).
Items: E1 for each of the families K = 61, 62, 74, 75, 76, 77, 78, 79, |g_ev(K) - model(K)| <= 0.00020. Families 61 and 62 are existing data that no reading of this chain has used; the families 74 to 79 are counted present when their family list file exists (it is written after the family is complete).
Verdict (last line): PASS if all eight families are present and E1 holds for all; FAIL if a measured item fails; INCOMPLETE if no measured item fails and a family is missing; VOID if a control misbehaves, a list has fewer than 8 states or a file lacks its key.
Controls (no output file of the families under test is read in the registration run): PC synthetic g = model for all eight passes; NC1 K 61 at +0.0004 fails; NC2 K 78 at -0.0004 fails; NC3 K 79 missing is INCOMPLETE; PC2 the procedure reproduces on K 63 and K 64 the values of q1114 (0.16336 and 0.16492, tolerance 0.00001).
Usage: py q1117_event_log_flow_law_verdict.py [--registration-run]"""

import json, os, sys

ARCH = "scratch"
KS = (61, 62, 74, 75, 76, 77, 78, 79)
TOL = 0.00020
MODEL = {0: (0.16619, 0.0815), 1: (0.16641, 0.1863)}


def model(K):
    g, c = MODEL[K % 2]
    return g - c / K


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


def list_path(K):
    return f"{ARCH}/family_list_K{K}.txt" if K in (61, 62) else f"{ARCH}/tower_family_list_K{K}.txt"


def event_path(K):
    if K % 2 == 1:
        return (f"{ARCH}/tower_K70_As_T83.json" if K == 69 else f"{ARCH}/tower_K{K + 1}_As.json"), "A_s"
    return f"{ARCH}/tower_K{K + 1}_birth_out.json", "A_cross"


def g_ev(K):
    """returns g_ev or None when the list or the event file is missing; exits with VOID on a defect"""
    lp = list_path(K)
    ep, key = event_path(K)
    if not os.path.exists(lp) or not os.path.exists(ep):
        return None
    st = []
    for line in open(lp):
        t = line.split()
        if len(t) < 2:
            continue
        d = json.load(open(t[1]))
        u = [float(x) for x in d["u"]]
        st.append((float(d["A"]), u[0], u[1] - u[0]))
    if len(st) < 8:
        print(f"VERDICT q1117 T125: VOID (K {K}: list has {len(st)} states)")
        sys.exit(2)
    ev = json.load(open(ep))
    if key not in ev:
        print(f"VERDICT q1117 T125: VOID (no key {key} in {ep})")
        sys.exit(2)
    Ae = float(ev[key])
    gs = []
    for i in (len(st) - 2, len(st) - 1):
        a = [st[j][0] for j in range(i - 3, i + 1)]
        p = [st[j][1] for j in range(i - 3, i + 1)]
        gs.append((st[i][0], deriv_last(a, p) / (st[i][1] * st[i][2])))
    (a1, g1), (a2, g2) = gs
    return g2 + (g2 - g1) / (a2 - a1) * (Ae - a2)


def verdict(g):
    res = {}
    for K in KS:
        if g.get(K) is None:
            res[K] = "MISSING"
            continue
        res[K] = "PASS" if abs(g[K] - model(K)) <= TOL else "FAIL"
    s = list(res.values())
    return ("FAIL" if "FAIL" in s else ("INCOMPLETE" if "MISSING" in s else "PASS")), res


def main():
    reg = "--registration-run" in sys.argv
    good = {K: model(K) for K in KS}
    c = {
        "PC": verdict(good)[0] == "PASS",
        "NC1": verdict({**good, 61: good[61] + 0.0004})[0] == "FAIL",
        "NC2": verdict({**good, 78: good[78] - 0.0004})[0] == "FAIL",
        "NC3": verdict({**good, 79: None})[0] == "INCOMPLETE",
    }
    g63 = g_ev(63)
    g64 = g_ev(64)
    c["PC2"] = (
        g63 is not None
        and g64 is not None
        and abs(g63 - 0.16336) <= 0.00001
        and abs(g64 - 0.16492) <= 0.00001
    )
    print("controls:", c)
    if not all(c.values()):
        print("VERDICT q1117 T125: VOID (a control does not behave)")
        sys.exit(2)
    if reg:
        print("VERDICT q1117 T125: INCOMPLETE (controls run only)")
        sys.exit(3)
    g = {K: g_ev(K) for K in KS}
    v, res = verdict(g)
    for K in KS:
        print(
            f"   K {K}: "
            + (
                "missing"
                if g[K] is None
                else f"g_ev {g[K]:.5f}, model {model(K):.5f}, difference {g[K] - model(K):+.5f}"
            )
            + f" -> {res[K]}"
        )
    print(f"VERDICT q1117 T125: {v}")
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
