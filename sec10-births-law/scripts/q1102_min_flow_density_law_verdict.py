"""q1102 (7 Oct 2026): verdict script of T122 (registered in the amendment that follows 1848): the smallest flow of a family, which is the flow of the innermost atom at the last optimal state of the family, follows u_end = C / rho^2, where rho is the density of the births law at the event that ends the family (the column rho of the prediction files).
Series read before the registration: u_end * rho^2 is 0.17519, 0.17448, 0.17371, 0.17431, 0.17461 for the odd families K 63, 65, 67, 69, 71 (mean 0.174461) and 0.08811, 0.08782, 0.08755, 0.08807, 0.08787 for the even families K 64, 66, 68, 70, 72 (mean 0.087882); the log-log slopes of u_end against rho are -2.106 and -2.037. The flow of the innermost atom at the last state was read directly at K 71 (0.182918) and K 72 (0.091392) (q1101).
Items: G1 for the odd families K 73, 75, 77 and 79, |u_end rho^2 / 0.174461 - 1| <= 0.015; G2 for the even families K 74, 76 and 78, |u_end rho^2 / 0.087882 - 1| <= 0.015. The values u_end are the first match of "smallest over all positions and states" in scratch/tower_K<K>_readings.txt (q953 or q954 readings, run after the family is complete); rho(K_b = K) is read from q635_capacity_K72_74_predictions.json (K 73, sha256 b0a3c934...) and q635_capacity_K74_79_predictions.json (K 74 to 79, sha256 a0b13d05...).
Verdict (last line): PASS if all seven families have a readings file and G1 and G2 hold; FAIL if a measured item fails; INCOMPLETE if no measured item fails and a file is missing; VOID if a prediction file is missing or has another sha256, a readings file has no match or a control misbehaves.
Controls (synthetic u_end values with the same rule): PC u_end = C / rho^2 for all seven families passes; NC1 K 73 at 3 percent above fails; NC2 K 78 at 3 percent below fails; NC3 K 79 missing is INCOMPLETE.
Usage: py q1102_min_flow_density_law_verdict.py [--registration-run] (controls only with the flag; no readings file is read)
"""

import hashlib, json, os, re, sys

ARCH = "scratch"
KS = (73, 74, 75, 76, 77, 78, 79)
P2 = f"{ARCH}/q635_capacity_K72_74_predictions.json"
P3 = f"{ARCH}/q635_capacity_K74_79_predictions.json"
SHA2 = "b0a3c9346ea2c463af9f8e4458cf4e50b505c140129438fc9fa853718ef617da"
SHA3 = "a0b13d053585eadc282e0e38a89ee1324f0be4a4015f8fb72c8e9b177c65310f"
C = {1: 0.174461, 0: 0.087882}
TOL = 0.015
PAT = re.compile(r"smallest over all positions and states ([0-9.]+)")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def parse(text):
    m = PAT.search(text)
    return None if m is None else float(m.group(1))


def verdict(u, rho):
    res = {}
    for K in KS:
        if u.get(K) is None:
            res[K] = "MISSING"
            continue
        res[K] = "PASS" if abs(u[K] * rho[K] ** 2 / C[K % 2] - 1) <= TOL else "FAIL"
    s = list(res.values())
    return ("FAIL" if "FAIL" in s else ("INCOMPLETE" if "MISSING" in s else "PASS")), res


def load_rho():
    for p, s in ((P2, SHA2), (P3, SHA3)):
        if not os.path.exists(p):
            print(f"VERDICT q1102 T122: VOID (prediction file {p} missing)")
            sys.exit(2)
        if sha(p) != s:
            print(f"VERDICT q1102 T122: VOID (sha256 of {p} differs from the registered one)")
            sys.exit(2)
    d2 = json.load(open(P2))["capacity"]["pred"]
    d3 = json.load(open(P3))["capacity"]["pred"]
    rho = {73: d2["73"]["rho"], **{int(k): v["rho"] for k, v in d3.items() if int(k) in KS}}
    if set(rho) != set(KS):
        print("VERDICT q1102 T122: VOID (rho missing for some family)")
        sys.exit(2)
    return rho


def main():
    reg = "--registration-run" in sys.argv
    rho = load_rho()
    good = {K: C[K % 2] / rho[K] ** 2 for K in KS}
    c = {
        "PC": verdict(good, rho)[0] == "PASS",
        "NC1": verdict({**good, 73: good[73] * 1.03}, rho)[0] == "FAIL",
        "NC2": verdict({**good, 78: good[78] * 0.97}, rho)[0] == "FAIL",
        "NC3": verdict({**good, 79: None}, rho)[0] == "INCOMPLETE",
    }
    print("controls:", c)
    if not all(c.values()):
        print("VERDICT q1102 T122: VOID (a control does not behave)")
        sys.exit(2)
    if reg:
        print("VERDICT q1102 T122: INCOMPLETE (controls run only)")
        sys.exit(3)
    u = {}
    for K in KS:
        f = f"{ARCH}/tower_K{K}_readings.txt"
        if not os.path.exists(f):
            u[K] = None
            continue
        v = parse(open(f, encoding="utf-8").read())
        if v is None:
            print(f"VERDICT q1102 T122: VOID (no match in {f})")
            sys.exit(2)
        u[K] = v
    v, res = verdict(u, rho)
    for K in KS:
        extra = "" if u[K] is None else f", u_end rho^2 / C = {u[K] * rho[K] ** 2 / C[K % 2]:.5f}"
        print(f"   K {K}: u_end {u[K]}, rho {rho[K]:.5f}{extra} -> {res[K]}")
    print(f"VERDICT q1102 T122: {v}")
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
