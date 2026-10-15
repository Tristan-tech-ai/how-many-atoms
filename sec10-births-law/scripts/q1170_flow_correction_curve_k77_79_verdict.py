"""q1170 (7 Oct 2026, registered in amendment 1910): verdict script of T140. The flows of the ten innermost levels at the event of the families K = 77, 78, 79 are F(xi_j) + P_par(xi_j) / (4 pi^2 rho^2), with F the closed form of T133, xi_j = p_j h / (4 pi) (positions at the last state),
rho = rho_pred(K), and P_par a universal curve per parity fitted on the calibration families K 63 to 76 (reading q1169b: the product (f_j - F(xi_j)) 4 pi^2 rho^2 varies with xi_j from -0.078 to +0.090 and its spread over K at fixed level is 0.002 to 0.009).
P_par is a polynomial of degree DEG in xi, least squares over the 10 levels of all calibration families of the parity (flows at the event of q1119.flows_at_event, as in figsrc.event_profiles).
Items for K = 77, 78, 79: max over j = 1..10 of |f_j - F(xi_j) - P_par(xi_j) / (4 pi^2 rho^2)| <= TOLF. Verdict: PASS if all three families are present and every item holds; FAIL if a present family fails; INCOMPLETE if none fails and one is missing; VOID if a control misbehaves or a list has fewer than 8 states.
Controls (registration run, no K 77 to 79 value): PC the prediction itself passes; NC1 prediction + 2.5 TOLF at level 3 fails; NC2 prediction - 2.5 TOLF at level 8 fails; NC3 a missing family is INCOMPLETE; NC4 leave-one-family-out (all levels of the left-out family, positions and flows as measured, distance of the last state 0.0006 to 0.0043)
stays within TOLF; NC5 the closed form F alone (P = 0) fails for the calibration families (the curve is needed). Last line: VERDICT q1170 T140: PASS / FAIL / INCOMPLETE / VOID.
Usage: py q1170_flow_correction_curve_k77_79_verdict.py [--registration-run] [--degrees]"""

import sys, json, math, os
import numpy as np

sys.path.insert(0, ".")
import q1119_event_flow_profile_reading as Q19
import q1114_log_flow_along_family_reading as Q14
import q1155_apriori_lattice_k76_79_verdict as T
import q1153_profile_closed_form_verdict as Z

KS = (77, 78, 79)
NJ = 10
DEG = 6
TOLF = 0.0009
F4 = 4 * math.pi**2
Q19.NJ = NJ


def family(K):
    lp = f"scratch/tower_family_list_K{K}.txt"
    if not os.path.exists(lp):
        return None
    Ae, fe = Q14.event(K)
    if Ae is None or not os.path.exists(fe):
        return None
    st = []
    for line in open(lp):
        t = line.split()
        if len(t) < 2:
            continue
        d = json.load(open(t[1]))
        u = [float(x) for x in d["u"]]
        st.append((float(d["A"]), u[: NJ + 1]))
    if len(st) < 8:
        print(f"VERDICT q1170 T140: VOID (K {K}: list has {len(st)} states)")
        sys.exit(2)
    f, p = Q19.flows_at_event(st, Ae)
    h = p[1] - p[0]
    return [p[j] * h / (4 * math.pi) for j in range(NJ)], f[:NJ]


def calib(rho, skip=None):
    pts = {0: ([], []), 1: ([], [])}
    for K in range(63, 77):
        if K == skip:
            continue
        fm = family(K)
        if fm is None:
            continue
        for j in range(NJ):
            pts[K % 2][0].append(fm[0][j])
            pts[K % 2][1].append((fm[1][j] - float(Z.f_pred(fm[0][j]))) * F4 * rho[K] ** 2)
    return pts


def fit(pts, deg=None):
    return {par: np.polyfit(np.array(x), np.array(y), deg or DEG) for par, (x, y) in pts.items()}


def predict(m, K, rho, xi):
    return [float(Z.f_pred(x)) + float(np.polyval(m[K % 2], x)) / (F4 * rho**2) for x in xi]


def item(fm, pr):
    return max(abs(a - b) for a, b in zip(fm[1], pr))


def judge(res):
    s = ["MISSING" if x is None else ("PASS" if x <= TOLF else "FAIL") for x in res.values()]
    return "FAIL" if "FAIL" in s else ("INCOMPLETE" if "MISSING" in s else "PASS")


def loo(rho, deg):
    worst = 0.0
    for K in range(63, 77):
        fm = family(K)
        if fm is None:
            continue
        m = fit(calib(rho, skip=K), deg)
        worst = max(worst, item(fm, predict(m, K, rho[K], fm[0])))
    return worst


def controls(rho):
    c = {}
    m = fit(calib(rho))
    fm0 = family(76)
    pr = predict(m, 76, rho[76], fm0[0])
    c["PC"] = judge({77: item((fm0[0], pr), pr), 78: 0.0, 79: 0.0}) == "PASS"
    c["NC1"] = (
        judge({77: item((fm0[0], [x + (2.5 * TOLF if j == 2 else 0) for j, x in enumerate(pr)]), pr)})
        == "FAIL"
    )
    c["NC2"] = (
        judge({77: item((fm0[0], [x - (2.5 * TOLF if j == 7 else 0) for j, x in enumerate(pr)]), pr)})
        == "FAIL"
    )
    c["NC3"] = judge({77: 0.0, 78: None, 79: 0.0}) == "INCOMPLETE"
    w = loo(rho, DEG)
    c["NC4_LOO"] = w <= TOLF
    print(f"leave-one-family-out (degree {DEG}), largest deviation of any level {w:.5f} (TOLF {TOLF})")
    zero = {0: np.array([0.0]), 1: np.array([0.0])}
    bad = [
        K
        for K in range(63, 77)
        if family(K) is not None and item(family(K), predict(zero, K, rho[K], family(K)[0])) <= TOLF
    ]
    c["NC5_curve_needed"] = len(bad) == 0
    print(f"closed form alone passes the tolerance for families: {bad}")
    return c, m


def main():
    rho = T.rho_pred()
    if "--degrees" in sys.argv:
        for d in (3, 4, 5, 6, 7, 8):
            print(f"degree {d}: leave-one-family-out largest deviation {loo(rho, d):.5f}")
        return
    c, m = controls(rho)
    print("controls:", c)
    if not all(c.values()):
        print("VERDICT q1170 T140: VOID (a control does not behave)")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("VERDICT q1170 T140: INCOMPLETE (controls only)")
        sys.exit(3)
    res = {}
    for K in KS:
        fm = family(K)
        if fm is None:
            res[K] = None
            continue
        pr = predict(m, K, rho[K], fm[0])
        res[K] = item(fm, pr)
        print(
            f"   K {K}: largest deviation over 10 levels {res[K]:.5f} (TOLF {TOLF}); differences "
            + " ".join(f"{a - b:+.5f}" for a, b in zip(fm[1], pr))
        )
    v = judge(res)
    print(
        "VERDICT q1170 T140: "
        + v
        + " ("
        + "; ".join(
            f"K {K} {'MISSING' if x is None else ('PASS' if x <= TOLF else 'FAIL')}" for K, x in res.items()
        )
        + ")"
    )
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
