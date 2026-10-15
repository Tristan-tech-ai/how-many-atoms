"""q1210 (8 Oct 2026, scores items D2 of amendment 1968 and its successor; text copy of q1208): reads the locator outputs q607_u1207_K<K_b>.json (K_b 67 to 100) and the control q607_u1207c_K50.json, compares a_t with q1207_predictions.json.
Located: every state of the hist converged (res < 1e-20), a_t differs from the last secant a by less than 1e-9 and the last |s| is below 1e-18. N1: |a_t - a_pred| < 0.5e-9; N2: < 2e-9. Class: at least 24 of 34 located and D2 true for all located (bar 0.3e-9 up to K_b 99, 1e-9 for K_b 100).
Control: u1205c_K50 reproduces 36.4666031158 within 1e-9 (else VOID). Item-function controls: a deviation of 0.4e-9 gives N1 True, 0.6e-9 gives N1 False and N2 True, 2.5e-9 gives both False.
Last line: 'VERDICT q1210 D2 class: PASS|FAIL|INCOMPLETE|VOID ...'. Usage: py q1206_npmle_score.py  (from temporal_within/centre_scripts/)
"""

import json, os, sys


def items(d, K=0):
    return abs(d) < (0.3e-9 if K <= 99 else 1e-9), abs(d) < 2e-9


def located(j):
    h = j["hist"]
    res_ok = all(x.get("res", 1) < 1e-20 for x in h if "res" in x)
    a = [x["a"] for x in h if "a" in x]
    sl = [x["s"] for x in h if "s" in x]
    br = (
        abs(j["a_t"] - a[-1]) < 1e-9 and abs(sl[-1]) < 1e-18 if len(a) >= 2 else False
    )  # repaired 8 Oct (see amendment 1966): the locator prints no bracket; the final secant step is below 1e-9 in a and the last |s| below 1e-18
    return res_ok and br


def main():
    ok = True
    for d, exp in ((0.2e-9, (True, True)), (0.4e-9, (False, True)), (2.5e-9, (False, False))):
        g = items(d) == exp
        ok = ok and g
        print(f"item-function control {d:.1e}: {items(d)} expected {exp} -> {g}")
    c = json.load(open("q607_u1207c_K50.json"))
    dc = c["a_t"] - 36.4666031158
    g = abs(dc) < 1e-9 and located(c)
    ok = ok and g
    print(
        f"control u1205c_K50: a_t {c['a_t']:.10f}, deviation from 36.4666031158 {dc * 1e9:+.3f}e-9, located {located(c)} -> {g}"
    )
    if not ok:
        print("VERDICT q1210 D2 class: VOID (control)")
        sys.exit(2)
    PF = os.environ.get("PRED_FILE", "q1209_predictions.json")
    pj = json.load(open(PF))["npmle"]
    pred = {k: {"a_pred": v, "rho_c": 0.0} for k, v in pj.items()}
    rows = []
    nloc = 0
    n1 = True
    n2 = True
    for K in range(67, 101):
        f = f"q607_u1207_K{K}.json"
        if not os.path.exists(f):
            print(f"K_b {K}: output missing")
            continue
        j = json.load(open(f))
        if not isinstance(j, dict) or j.get("a_t") is None:
            print(f"K_b {K}: NOT LOCATED (output incomplete or states unconverged)")
            continue
        loc = located(j)
        d = j["a_t"] - pred[str(K)]["a_pred"]
        a, b = items(d, K)
        print(
            f"K_b {K}: a_t {j['a_t']:.10f}, a_pred {pred[str(K)]['a_pred']:.10f}, deviation {d * 1e9:+.3f}e-9, located {loc}, N1 {a}, N2 {b}"
        )
        if loc:
            nloc += 1
            n1 = n1 and a
            n2 = n2 and b
    cls = nloc >= 24 and n1
    print(f"located {nloc} of 34; N1 for all located {n1}; N2 for all located {n2}")
    print(
        f"VERDICT q1210 D2 class: {'PASS' if cls else ('INCOMPLETE' if nloc < 24 else 'FAIL')}; located {nloc} of 34; N1 {n1}; N2 {n2}"
    )


if __name__ == "__main__":
    main()
