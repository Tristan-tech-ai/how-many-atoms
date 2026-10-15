"""q1099 (7 Oct 2026): verdict script of T121 (registered in the amendment that follows 1846): the smallest flow du_j / dA over all positions and all states of a capacity family changes slowly with K and stays well above zero in the families K 73 to K 79 of the tower.
Series read before the registration (the line "du_j/dA: smallest over all positions and states" of the readings files q953 for odd K and q954 for even K): odd K 63, 65, 67, 69, 71: 0.1948, 0.1910, 0.1873, 0.1852, 0.1829 (steps -0.0038, -0.0037, -0.0021, -0.0023); even K 64, 66, 68, 70, 72: 0.0972, 0.0954, 0.0937, 0.0929, 0.0914 (steps -0.0018, -0.0017, -0.0008, -0.0015). A linear extrapolation of the odd series reaches zero near K = 217.
Items: F1 the smallest flow of the family K 73 lies in [0.178, 0.183]; F2 the smallest flow of the family K 74 lies in [0.0885, 0.0914]; F3 the smallest flows of the families K 75, K 77 and K 79 are at least 0.165 and those of K 76 and K 78 are at least 0.080.
Source: scratch/tower_K<K>_readings.txt (the output of q953_readings_odd.py or q954_readings_even.py, produced after the family is complete), first match of "smallest over all positions and states ([0-9.]+)".
Verdict (last line): PASS if all seven families have a readings file and F1 to F3 hold; FAIL if a measured item fails; INCOMPLETE if no measured item fails and a file is missing; VOID if a readings file has no match or a control misbehaves.
Controls (synthetic text with the same parser and rule): PC values 0.1805, 0.0900, 0.1750, 0.0850, 0.1700, 0.0820, 0.1680 must PASS; NC1 K 73 at 0.170 must FAIL; NC2 K 76 at 0.060 must FAIL; NC3 K 79 missing must be INCOMPLETE.
Usage: py q1099_min_flow_extension_verdict.py [--registration-run] (controls only with the flag; no file is read)
"""

import os, re, sys

ARCH = "scratch"
KS = (73, 74, 75, 76, 77, 78, 79)
PAT = re.compile(r"smallest over all positions and states ([0-9.]+)")


def parse(text):
    m = PAT.search(text)
    return None if m is None else float(m.group(1))


def verdict(vals):
    """vals: dict K -> float or None (missing)."""
    res = {}

    def chk(K, ok):
        res[K] = "MISSING" if vals.get(K) is None else ("PASS" if ok(vals[K]) else "FAIL")

    chk(73, lambda x: 0.178 <= x <= 0.183)
    chk(74, lambda x: 0.0885 <= x <= 0.0914)
    for K in (75, 77, 79):
        chk(K, lambda x: x >= 0.165)
    for K in (76, 78):
        chk(K, lambda x: x >= 0.080)
    s = list(res.values())
    return ("FAIL" if "FAIL" in s else ("INCOMPLETE" if "MISSING" in s else "PASS")), res


def main():
    reg = "--registration-run" in sys.argv
    good = {73: 0.1805, 74: 0.0900, 75: 0.1750, 76: 0.0850, 77: 0.1700, 78: 0.0820, 79: 0.1680}
    parsed = {
        K: parse(f"K4 du_j/dA: smallest over all positions and states {v}, largest 9.9")
        for K, v in good.items()
    }
    c = {
        "PC": verdict(parsed)[0] == "PASS",
        "NC1": verdict({**parsed, 73: 0.170})[0] == "FAIL",
        "NC2": verdict({**parsed, 76: 0.060})[0] == "FAIL",
        "NC3": verdict({**parsed, 79: None})[0] == "INCOMPLETE",
        "PARSE_NONE": parse("no such line") is None,
    }
    print("controls:", c)
    if not all(c.values()):
        print("VERDICT q1099 T121: VOID (a control does not behave)")
        sys.exit(2)
    if reg:
        print("VERDICT q1099 T121: INCOMPLETE (controls run only)")
        sys.exit(3)
    vals = {}
    for K in KS:
        f = f"{ARCH}/tower_K{K}_readings.txt"
        if not os.path.exists(f):
            vals[K] = None
            continue
        v = parse(open(f, encoding="utf-8").read())
        if v is None:
            print(f"VERDICT q1099 T121: VOID (no match in {f})")
            sys.exit(2)
        vals[K] = v
    v, res = verdict(vals)
    for K in KS:
        print(f"   K {K}: smallest flow {vals[K]} -> {res[K][0] if isinstance(res[K], tuple) else res[K]}")
    print(f"VERDICT q1099 T121: {v}")
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
