"""q1133 (7 Oct 2026): verdict script of T129 (registered in the amendment that follows 1868): the flows f_j = dp_j/dA of the ten innermost positions at the event that ends a capacity family, plotted against xi_j = p_j h / (4 pi) (h = p_2 - p_1 of the last state), lie on the universal curve of the existing families (reading q1120: 30 families of three problems, largest difference 0.004).
Reference curve: the polynomial f = a_1 xi + ... + a_8 xi^8 (no constant term) fitted by least squares to the points j = 1..10 of the 30 existing families (capacity K 63 to 73 with tower_family_list_K<K>.txt and the events of q1114; the chain families of results_q584_ext.json and results_q601_ext.json with steps of at most 0.0125 in the last five rows and a last row within 0.03 of the event, as q1115), restricted to xi <= 1.2. The flows at the event are computed by the procedure of q1119 (four-node windows at the last two states, linear extrapolation in A to the event).
Items: P1 for each capacity family K = 74, 75, 76, 77, 78, 79 and each j = 1..10 with xi_j <= 1.2, |f_j - reference(xi_j)| <= 0.0060. The families count as present when tower_family_list_K<K>.txt exists (written after the family is complete).
Verdict (last line): PASS if all six families are present and P1 holds for all items; FAIL if a measured item fails; INCOMPLETE if no measured item fails and a family is missing; VOID if a control misbehaves, the reference fit has fewer than 250 points or its largest residual on the fitted points exceeds 0.0060.
Controls (the files of the families under test are not read in the registration run): PC synthetic flows equal to the reference at the xi of K 73 pass; NC1 the same flows with +0.010 at j = 5 fail; NC2 the flows evaluated at 1.10 xi fail; NC3 K 79 missing is INCOMPLETE; PC2 the reference reproduces the flows of the existing families K 72 and K 73 within 0.0060 for j = 1..10 (these are fitted data and not items).
Usage: py q1133_flow_profile_collapse_verdict.py [--registration-run]"""

import json, os, sys, math
import numpy as np
import q1119_event_flow_profile_reading as Q19
from q1113_log_flow_reading import families
from q1114_log_flow_along_family_reading import event

Q19.NJ = 10
NJ = 10
ARCH = "scratch"
KS = (74, 75, 76, 77, 78, 79)
TOL = 0.0060
XMAX = 1.2
DEG = 8


def capacity_points(K):
    lp = f"{ARCH}/tower_family_list_K{K}.txt"
    Ae, fe = event(K)
    if not os.path.exists(lp) or Ae is None:
        return None
    lines = [l.split() for l in open(lp) if l.strip()]
    if len(lines) < 8:
        print(f"VERDICT q1133 T129: VOID (K {K}: list has {len(lines)} states)")
        sys.exit(2)
    st = []
    for t in lines:
        d = json.load(open(t[1]))
        u = [float(x) for x in d["u"]]
        st.append((float(d["A"]), u[: NJ + 1]))
    f, p = Q19.flows_at_event(st, Ae)
    h = p[1] - p[0]
    return [(p[k] * h / (4 * math.pi), f[k]) for k in range(NJ)]


def chain_points():
    out = []
    for fn in ("results_q584_ext.json", "results_q601_ext.json"):
        d = json.load(open(fn))
        ev = {}
        for e in d["events"]:
            if e["kind"] in ("centre appear", "centre split"):
                ev.setdefault(e["K_before"], e["a_lo"])
        for c in families(d["rows"])[:-1]:
            K = c[-1]["K"]
            if K not in ev or len(c) < 8:
                continue
            a = [r["a"] for r in c[-7:]]
            s = [a[i + 1] - a[i] for i in range(6)]
            if max(s[1:]) > 0.0125 or ev[K] - a[-1] > 0.03:
                continue
            S = [(r["a"], [float(x) for x in r["p"][: NJ + 1]]) for r in c[-8:]]
            f, p = Q19.flows_at_event(S, ev[K])
            h = p[1] - p[0]
            out.append([(p[k] * h / (4 * math.pi), f[k]) for k in range(NJ)])
    return out


def fit_reference():
    fams = [capacity_points(K) for K in range(63, 74)]
    if any(f is None for f in fams):
        print("VERDICT q1133 T129: VOID (an existing capacity family is missing)")
        sys.exit(2)
    fams += chain_points()
    pts = [q for f in fams for q in f if q[0] <= XMAX]
    if len(pts) < 250:
        print(f"VERDICT q1133 T129: VOID (reference fit has {len(pts)} points)")
        sys.exit(2)
    x = np.array([q[0] for q in pts])
    y = np.array([q[1] for q in pts])
    M = np.vstack([x**n for n in range(1, DEG + 1)]).T
    c, *_ = np.linalg.lstsq(M, y, rcond=None)
    res = float(np.max(np.abs(M @ c - y)))
    if res > TOL:
        print(f"VERDICT q1133 T129: VOID (reference residual {res:.4f} above the tolerance)")
        sys.exit(2)
    return c, len(pts), res, fams


def ref(c, x):
    return sum(c[n - 1] * np.asarray(x, float) ** n for n in range(1, DEG + 1))


def verdict(meas, c):
    res = {}
    for K in KS:
        if meas.get(K) is None:
            res[K] = "MISSING"
            continue
        res[K] = "PASS" if all(abs(f - float(ref(c, x))) <= TOL for x, f in meas[K] if x <= XMAX) else "FAIL"
    s = list(res.values())
    return ("FAIL" if "FAIL" in s else ("INCOMPLETE" if "MISSING" in s else "PASS")), res


def main():
    reg = "--registration-run" in sys.argv
    c, n, rmax, fams = fit_reference()
    base = fams[10]  # K 73
    good = {K: [(x, float(ref(c, x))) for x, _ in base] for K in KS}
    bad1 = {**good, 74: [(x, f + (0.010 if j == 4 else 0.0)) for j, (x, f) in enumerate(good[74])]}
    bad2 = {**good, 75: [(x, float(ref(c, 1.10 * x))) for x, _ in base]}
    ctl = {
        "PC": verdict(good, c)[0] == "PASS",
        "NC1": verdict(bad1, c)[0] == "FAIL",
        "NC2": verdict(bad2, c)[0] == "FAIL",
        "NC3": verdict({**good, 79: None}, c)[0] == "INCOMPLETE",
    }
    ctl["PC2"] = all(
        abs(f - float(ref(c, x))) <= TOL for fam in (fams[9], fams[10]) for x, f in fam if x <= XMAX
    )
    print(
        f"reference fit: {n} points from {len(fams)} families, degree {DEG}, largest residual {rmax:.4f}; controls:",
        ctl,
    )
    if not all(ctl.values()):
        print("VERDICT q1133 T129: VOID (a control does not behave)")
        sys.exit(2)
    if reg:
        print("VERDICT q1133 T129: INCOMPLETE (controls run only)")
        sys.exit(3)
    meas = {K: capacity_points(K) for K in KS}
    v, res = verdict(meas, c)
    for K in KS:
        if meas[K] is None:
            print(f"   K {K}: missing -> MISSING")
            continue
        d = [f - float(ref(c, x)) for x, f in meas[K] if x <= XMAX]
        print(
            f"   K {K}: differences f_j - reference "
            + " ".join(f"{t:+.4f}" for t in d)
            + f" (tolerance {TOL}) -> {res[K]}"
        )
    print(f"VERDICT q1133 T129: {v}")
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
