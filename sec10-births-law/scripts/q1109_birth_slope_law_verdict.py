"""q1109 (7 Oct 2026): verdict script of T124 (registered in the amendment that follows 1855): the slope dwc/dA of the centre mass at a birth times the density rho of the births law at the event is nearly constant, slope * rho = 0.3702, 0.3704, 0.3705, 0.3707, 0.3709 and 0.3710 at K_b = 62, 64, 66, 68, 70 and 72 (rising by 0.00016 per birth), so that the birth slope falls like 1 / rho.
Items: B1 for the births K 74 -> 75, K 76 -> 77 and K 78 -> 79 (K_b = 74, 76, 78) the product slope * rho lies in [0.3705, 0.3722].
The slope is the key slope_wc of scratch/tower_K<K_b + 1>_birth_out.json (output of q1063 of the stage of q1083); rho = rho(K_b) is read from q635_capacity_K74_79_predictions.json (sha256 a0b13d05...).
Verdict (last line): PASS if the three files exist and B1 holds for all; FAIL if a measured item fails; INCOMPLETE if no measured item fails and a file is missing; VOID if the prediction file is missing or has another sha256, a file has no key slope_wc or a control misbehaves.
Controls (synthetic slopes with the same rule): PC slope = 0.3712 / rho for the three births passes; NC1 slope = 0.3650 / rho at K_b 74 fails; NC2 slope = 0.3780 / rho at K_b 78 fails; NC3 K_b 78 missing is INCOMPLETE.
Usage: py q1109_birth_slope_law_verdict.py [--registration-run] (controls only with the flag; no output file is read)
"""

import hashlib, json, os, sys

ARCH = "scratch"
P3 = f"{ARCH}/q635_capacity_K74_79_predictions.json"
SHA3 = "a0b13d053585eadc282e0e38a89ee1324f0be4a4015f8fb72c8e9b177c65310f"
KB = (74, 76, 78)
LO, HI = 0.3705, 0.3722


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def verdict(slopes, rho):
    res = {}
    for K in KB:
        if slopes.get(K) is None:
            res[K] = "MISSING"
            continue
        res[K] = "PASS" if LO <= slopes[K] * rho[K] <= HI else "FAIL"
    s = list(res.values())
    return ("FAIL" if "FAIL" in s else ("INCOMPLETE" if "MISSING" in s else "PASS")), res


def main():
    reg = "--registration-run" in sys.argv
    if not os.path.exists(P3):
        print("VERDICT q1109 T124: VOID (prediction file missing)")
        sys.exit(2)
    if sha(P3) != SHA3:
        print("VERDICT q1109 T124: VOID (sha256 of the prediction file differs from the registered one)")
        sys.exit(2)
    rho = {int(k): v["rho"] for k, v in json.load(open(P3))["capacity"]["pred"].items() if int(k) in KB}
    good = {K: 0.3712 / rho[K] for K in KB}
    c = {
        "PC": verdict(good, rho)[0] == "PASS",
        "NC1": verdict({**good, 74: 0.3650 / rho[74]}, rho)[0] == "FAIL",
        "NC2": verdict({**good, 78: 0.3780 / rho[78]}, rho)[0] == "FAIL",
        "NC3": verdict({**good, 78: None}, rho)[0] == "INCOMPLETE",
    }
    print("controls:", c)
    if not all(c.values()):
        print("VERDICT q1109 T124: VOID (a control does not behave)")
        sys.exit(2)
    if reg:
        print("VERDICT q1109 T124: INCOMPLETE (controls run only)")
        sys.exit(3)
    sl = {}
    for K in KB:
        f = f"{ARCH}/tower_K{K + 1}_birth_out.json"
        if not os.path.exists(f):
            sl[K] = None
            continue
        d = json.load(open(f))
        if "slope_wc" not in d:
            print(f"VERDICT q1109 T124: VOID (no key slope_wc in {f})")
            sys.exit(2)
        sl[K] = float(d["slope_wc"])
    v, res = verdict(sl, rho)
    for K in KB:
        print(
            f"   K_b {K}: slope {sl[K]}, rho {rho[K]:.5f}"
            + ("" if sl[K] is None else f", slope * rho = {sl[K] * rho[K]:.5f}")
            + f" -> {res[K]}"
        )
    print(f"VERDICT q1109 T124: {v}")
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
