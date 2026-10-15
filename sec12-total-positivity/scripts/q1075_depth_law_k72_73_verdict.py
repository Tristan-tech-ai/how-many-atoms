"""q1075 (7 Oct 2026): verdict script of T109 (registered in the amendment that follows 1823; a copy of q1010 of T89): blind test that the depth of the centre signal obeys ln|s_min| = a - b/Delta^2 with b near 4 pi^2 = 39.48 on the families K 72 and K 73 of the tower (T104, 1822), with K 70 and K 71 as the starting families (s_min of K 70 and K 71 was read in 1820 and in q953; K 72 and K 73 do not exist at the registration).
For a family, s_min is the smallest centre signal of its OPTIMAL states and Delta = (u_3 - u_1)/2 of that state (u_1, u_2, u_3 the three innermost positive positions). N1: the slope d ln|s_min| / d(1/Delta^2) between the even families K 70 and K 72 lies in [-45, -33]. N2: the same slope between the odd families K 71 and K 73 lies in [-45, -33]. The bars are those of T89 (the slopes measured there were -38.34 and -39.22).
A family counts as complete when a state with the verdict "centre positive" lies above all its OPTIMAL states.
Verdict (last line): PASS if both slopes are measured and inside their bars; FAIL if a measured slope is outside; INCOMPLETE if no measured slope fails and a family is missing or not complete; VOID if a file is unreadable or a control does not behave.
Controls (synthetic): PC the law value -39.48 for both must PASS; NC1 slope -20 must FAIL; NC2 slope -60 must FAIL; NC3 K 73 missing must be INCOMPLETE. Usage: py q1075_depth_law_k72_73_verdict.py [--registration-run] (controls only with the flag)
"""

import glob, json, math, sys

FAM = {70: "tower/K70", 71: "tower/K71", 72: "tower/K72", 73: "tower/K73"}
LO, HI = -45.0, -33.0
cl = lambda v: float(str(v).replace("np.float64(", "").rstrip(")"))


def family(K, d):
    st = []
    non = []
    for f in glob.glob(d + f"/results_q220_gauss_K{K}_A*.json"):
        j = json.load(open(f))
        a = cl(j["A"])
        if str(j["verdict"]) == "OPTIMAL":
            st.append((a, float(j["centre_signal"]), [cl(t) for t in j["u"]]))
        else:
            non.append((a, str(j["verdict"])))
    if not st:
        return None
    st.sort()
    done = any(a > st[-1][0] and v == "centre positive" for a, v in non)
    a, c, u = min(st, key=lambda s: s[1])
    return {"complete": done, "s": c, "delta": (u[2] - u[0]) / 2, "A": a}


def slope(f0, f1):
    return (math.log(abs(f1["s"])) - math.log(abs(f0["s"]))) / (1 / f1["delta"] ** 2 - 1 / f0["delta"] ** 2)


def verdict(s1, s2):
    items = {"N1": None if s1 is None else (LO <= s1 <= HI), "N2": None if s2 is None else (LO <= s2 <= HI)}
    if any(v is False for v in items.values()):
        return "FAIL", items
    if any(v is None for v in items.values()):
        return "INCOMPLETE", items
    return "PASS", items


def main():
    try:
        f66 = family(70, FAM[70])
        f67 = family(71, FAM[71])
    except Exception as e:
        print(f"VERDICT q1075 T109: VOID (unreadable files: {e})")
        sys.exit(2)
    if f66 is None or f67 is None:
        print("VERDICT q1075 T109: VOID (K 70 or K 71 missing)")
        sys.exit(2)
    c = {
        "PC": verdict(-39.48, -39.48)[0] == "PASS",
        "NC1": verdict(-20.0, -39.48)[0] == "FAIL",
        "NC2": verdict(-39.48, -60.0)[0] == "FAIL",
        "NC3": verdict(-39.48, None)[0] == "INCOMPLETE",
    }
    print("controls:", c)
    if not all(c.values()):
        print("VERDICT q1075 T109: VOID (a control does not behave)")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("VERDICT q1075 T109: INCOMPLETE (controls run only)")
        sys.exit(3)
    out = {}
    for K in (72, 73):
        try:
            out[K] = family(K, FAM[K])
        except Exception as e:
            print(f"VERDICT q1075 T109: VOID (unreadable {FAM[K]}: {e})")
            sys.exit(2)
    s1 = s2 = None
    if out[72] and out[72]["complete"]:
        s1 = slope(f66, out[72])
        print(
            f"   even K 70 -> 72: s_min {f66['s']:.4e} -> {out[72]['s']:.4e}, Delta {f66['delta']:.5f} -> {out[72]['delta']:.5f}, slope {s1:.2f}"
        )
    else:
        print("   K 72 not complete")
    if out[73] and out[73]["complete"]:
        s2 = slope(f67, out[73])
        print(
            f"   odd K 71 -> 73: s_min {f67['s']:.4e} -> {out[73]['s']:.4e}, Delta {f67['delta']:.5f} -> {out[73]['delta']:.5f}, slope {s2:.2f}"
        )
    else:
        print("   K 73 not complete")
    v, it = verdict(s1, s2)
    print(f"VERDICT q1075 T109: {v}; items {it}")
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
