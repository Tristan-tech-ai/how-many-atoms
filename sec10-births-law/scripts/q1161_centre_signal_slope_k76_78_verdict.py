"""q1161 (7 Oct 2026, registered in amendment 1902): verdict script of T136. The slope of the centre signal of the last two states of the even families K = 76 and 78 against 32 pi^2 rho_pred(K) exp(-v^2 / 4), v = 4 pi rho.
Item: the ratio lies in [0.95, 1.05] for both. Last line: VERDICT q1161 T136: PASS / FAIL / INCOMPLETE / VOID. Usage: py q1161_centre_signal_slope_k76_78_verdict.py [--registration-run]
"""

import sys, json, math, os
import numpy as np

sys.path.insert(0, ".")
import q1155_apriori_lattice_k76_79_verdict as T
import q1117_event_log_flow_law_verdict as Q

KS = (76, 78)
LO, HI = 0.95, 1.05


def pred(rho):
    v = 4 * math.pi * rho
    return 32 * math.pi**2 * rho * math.exp(-v * v / 4)


def ratio_from_series(cs, A, rho):
    return (cs[-1] - cs[-2]) / (A[-1] - A[-2]) / pred(rho)


def judge(ratios):
    s = ["MISSING" if r is None else ("PASS" if LO <= r <= HI else "FAIL") for r in ratios.values()]
    return "FAIL" if "FAIL" in s else ("INCOMPLETE" if "MISSING" in s else "PASS")


def synth(rho, k, f=1.0):
    p = pred(rho) * f
    A = [48.0 + 0.005 * i for i in range(3)]
    z = A[-1] + 0.002
    cs = [-(p * (z - a)) for a in A]
    return cs, A


def controls(rho):
    c = {}
    good = {K: ratio_from_series(*synth(rho[K], K), rho[K]) for K in KS}
    c["PC"] = judge(good) == "PASS"
    bad = {K: ratio_from_series(*synth(rho[K], K, 1.10), rho[K]) for K in KS}
    c["NC1"] = judge(bad) == "FAIL"
    c["NC2"] = judge({76: good[76], 78: None}) == "INCOMPLETE"
    cal = []
    for K in range(62, 76, 2):
        st = [json.load(open(l.split()[1])) for l in open(Q.list_path(K)) if len(l.split()) >= 2][-2:]
        cal.append(
            ratio_from_series([float(d["centre_signal"]) for d in st], [float(d["A"]) for d in st], rho[K])
        )
    c["NC3"] = all(0.95 <= x <= 1.05 for x in cal) and not all(0.99 <= x <= 1.01 for x in cal)
    return c


def measured(K, rho):
    lp = Q.list_path(K)
    if not os.path.exists(lp):
        return None
    st = [json.load(open(l.split()[1])) for l in open(lp) if len(l.split()) >= 2]
    if len(st) < 8:
        print(f"VERDICT q1161 T136: VOID (K {K}: list has {len(st)} states)")
        sys.exit(2)
    st = st[-2:]
    return ratio_from_series([float(d["centre_signal"]) for d in st], [float(d["A"]) for d in st], rho[K])


def main():
    rho = T.rho_pred()
    c = controls(rho)
    print("controls:", {k: bool(v) for k, v in c.items()})
    if not all(c.values()):
        print("VERDICT q1161 T136: VOID (a control does not behave)")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("VERDICT q1161 T136: INCOMPLETE (controls run only)")
        sys.exit(3)
    ratios = {K: measured(K, rho) for K in KS}
    for K, r in ratios.items():
        print(
            f"   K {K}: "
            + (
                "missing"
                if r is None
                else f"ratio {r:.4f} ({'in' if LO <= r <= HI else 'outside'} [{LO}, {HI}])"
            )
        )
    v = judge(ratios)
    print(
        f"VERDICT q1161 T136: {v} ("
        + "; ".join(
            f"K {K} {'MISSING' if r is None else ('PASS' if LO <= r <= HI else 'FAIL')}"
            for K, r in ratios.items()
        )
        + ")"
    )
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
