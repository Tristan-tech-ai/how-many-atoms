"""q1010 (7 Oct 2026): verdict script of T89 (amendment 1773): blind test that the depth of the centre signal obeys ln|s_min| = a - b/Delta^2 with b near 4 pi^2 = 39.48 (reading q1009 on K 61 to K 67: slopes -37.6, -37.6, -33.7 for the odd and -35.2, -39.2 for the even neighbours, -36.2 and -37.1 from the first to the last family of a parity).
For a family, s_min is the smallest centre signal of its OPTIMAL states and Delta = (u_3 - u_1)/2 of that state (u_1, u_2, u_3 the three innermost positive positions). N1: the slope d ln|s_min| / d(1/Delta^2) between the even families K 66 and K 68 lies in [-45, -33]; N2: the same between the odd families K 67 and K 69.
Instrument size (computed before registration, q1009 and the check of 1773): the state of the minimum is known to one state, which moves Delta by 0.0011 and 1/Delta^2 by 0.0018; over two events 1/Delta^2 changes by about 0.03, so the slope is known to about 9 percent; the bar is 15 percent around the law value.
A family counts as complete when a state with the verdict "centre positive" lies above all its OPTIMAL states.
Verdict (last line): PASS if both slopes are measured and inside their bars; FAIL if a measured slope is outside; INCOMPLETE if no measured slope fails and a family is missing or not complete; VOID if a file is unreadable or a control does not behave.
Controls (synthetic, built from the K 66 and K 67 values): PC the law value -39.48 for both must PASS; NC1 slope -20 must FAIL; NC2 slope -60 must FAIL; NC3 K 69 missing must be INCOMPLETE. Usage: py q1010_depth_law_blind_verdict.py
"""

import glob, json, math, sys

FAM = {66: "tower/K66", 67: "tower/K67", 68: "tower/K68", 69: "tower/K69"}
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
        f66 = family(66, FAM[66])
        f67 = family(67, FAM[67])
    except Exception as e:
        print(f"VERDICT q1010 T89: VOID (unreadable files: {e})")
        sys.exit(2)
    if f66 is None or f67 is None:
        print("VERDICT q1010 T89: VOID (K 66 or K 67 missing)")
        sys.exit(2)
    c = {
        "PC": verdict(-39.48, -39.48)[0] == "PASS",
        "NC1": verdict(-20.0, -39.48)[0] == "FAIL",
        "NC2": verdict(-39.48, -60.0)[0] == "FAIL",
        "NC3": verdict(-39.48, None)[0] == "INCOMPLETE",
    }
    print("controls:", c)
    if not all(c.values()):
        print("VERDICT q1010 T89: VOID (a control does not behave)")
        sys.exit(2)
    out = {}
    for K in (68, 69):
        try:
            out[K] = family(K, FAM[K])
        except Exception as e:
            print(f"VERDICT q1010 T89: VOID (unreadable {FAM[K]}: {e})")
            sys.exit(2)
    s1 = s2 = None
    if out[68] and out[68]["complete"]:
        s1 = slope(f66, out[68])
        print(
            f"   even K 66 -> 68: s_min {f66['s']:.4e} -> {out[68]['s']:.4e}, Delta {f66['delta']:.5f} -> {out[68]['delta']:.5f}, slope {s1:.2f}"
        )
    else:
        print("   K 68 not complete")
    if out[69] and out[69]["complete"]:
        s2 = slope(f67, out[69])
        print(
            f"   odd K 67 -> 69: s_min {f67['s']:.4e} -> {out[69]['s']:.4e}, Delta {f67['delta']:.5f} -> {out[69]['delta']:.5f}, slope {s2:.2f}"
        )
    else:
        print("   K 69 not complete")
    v, it = verdict(s1, s2)
    print(f"VERDICT q1010 T89: {v}; items {it}")
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
