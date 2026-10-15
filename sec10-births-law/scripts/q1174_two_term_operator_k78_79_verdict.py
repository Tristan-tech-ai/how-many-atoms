"""q1174 (7 Oct 2026, registered in amendment 1913): verdict script of T141. Reading q1173: the first-order correction of the flows at the event is a differential operator of two terms acting on the closed form,
f_j = F(xi_j) + [a_par xi_j F'(xi_j) + b xi_j^2 F''(xi_j)] / (4 pi^2 rho^2), with (a, b) fitted by least squares on the capacity families K = 63 to 77 (reading q1173 on K 63 to 76: a = -0.2530 even, -0.2907 odd; b = -0.1900 even, -0.1912 odd; rms 0.0052, 0.0055).
Three numbers are fitted: a_even, a_odd and one b shared by both parities (the two b of the reading agree within one standard error 0.002). Test families K = 78 and 79 (K 77 is in the calibration): flows at the event of the ten innermost levels (q1119.flows_at_event as in figsrc.event_profiles), xi_j = p_j h / (4 pi) of the last state, rho = rho_pred(K).
Item per family: the largest over the ten levels of |f_j - prediction| <= TOLF2. Verdict: PASS if both families are present and in the tolerance; FAIL if a present family is outside; INCOMPLETE if none fails and one is missing; VOID if a control misbehaves or a list has fewer than 8 states.
Controls (registration run, no K 78 or 79 value): PC the prediction itself passes; NC1 prediction + 2.5 TOLF2 at level 3 fails; NC2 prediction - 2.5 TOLF2 at level 8 fails; NC3 a missing family is INCOMPLETE; NC4 leave-one-family-out over K 63 to 77 within TOLF2; NC5 the closed form alone (no operator terms) fails TOLF2 for every calibration family.
Last line: VERDICT q1174 T141: PASS / FAIL / INCOMPLETE / VOID. Usage: py q1174_two_term_operator_k78_79_verdict.py [--registration-run]
"""

import sys, math
import numpy as np
import mpmath as mp

sys.path.insert(0, ".")
import q1170_flow_correction_curve_k77_79_verdict as V
import q1155_apriori_lattice_k76_79_verdict as T
import q1153_profile_closed_form_verdict as Z

KS = (78, 79)
TOLF2 = 0.0009
F4 = 4 * math.pi**2
NJ = 10
Fm = lambda x: mp.coth(mp.pi * x) - mp.pi * x / mp.sinh(mp.pi * x) ** 2
_c = {}


def ops(x):
    if x not in _c:
        _c[x] = (x * float(mp.diff(Fm, x)), x * x * float(mp.diff(Fm, x, 2)))
    return _c[x]


def calib(rho, skip=None):
    rows = []
    for K in range(63, 78):
        if K == skip:
            continue
        fm = V.family(K)
        if fm is None:
            continue
        for j in range(NJ):
            o = ops(fm[0][j])
            rows.append((K % 2, o[0], o[1], (fm[1][j] - float(Z.f_pred(fm[0][j]))) * F4 * rho[K] ** 2))
    return rows


def fit(rows):
    X = np.array([[1.0 if r[0] == 0 else 0.0, 1.0 if r[0] == 1 else 0.0, 0.0] for r in rows])
    X[:, 0] *= np.array([r[1] for r in rows])
    X[:, 1] *= np.array([r[1] for r in rows])
    X[:, 2] = [r[2] for r in rows]
    return np.linalg.lstsq(X, np.array([r[3] for r in rows]), rcond=None)[0]


def predict(c, K, rho, xi):
    return [float(Z.f_pred(x)) + ((c[K % 2] * ops(x)[0]) + c[2] * ops(x)[1]) / (F4 * rho**2) for x in xi]


def loo(rho):
    worst = 0.0
    for K in range(63, 78):
        fm = V.family(K)
        if fm is None:
            continue
        c = fit(calib(rho, skip=K))
        worst = max(worst, V.item(fm, predict(c, K, rho[K], fm[0])))
    return worst


def judge(res):
    s = ["MISSING" if x is None else ("PASS" if x <= TOLF2 else "FAIL") for x in res.values()]
    return "FAIL" if "FAIL" in s else ("INCOMPLETE" if "MISSING" in s else "PASS")


def controls(rho):
    c = {}
    cf = fit(calib(rho))
    print(f"calibration (K 63 to 77): a_even {cf[0]:+.4f}, a_odd {cf[1]:+.4f}, b {cf[2]:+.4f}")
    fm0 = V.family(77)
    pr = predict(cf, 77, rho[77], fm0[0])
    c["PC"] = judge({78: V.item((fm0[0], pr), pr), 79: 0.0}) == "PASS"
    c["NC1"] = (
        judge({78: V.item((fm0[0], [x + (2.5 * TOLF2 if j == 2 else 0) for j, x in enumerate(pr)]), pr)})
        == "FAIL"
    )
    c["NC2"] = (
        judge({78: V.item((fm0[0], [x - (2.5 * TOLF2 if j == 7 else 0) for j, x in enumerate(pr)]), pr)})
        == "FAIL"
    )
    c["NC3"] = judge({78: 0.0, 79: None}) == "INCOMPLETE"
    w = loo(rho)
    c["NC4_LOO"] = w <= TOLF2
    print(f"leave-one-family-out, K 63 to 77: largest deviation of any level {w:.5f} (TOLF2 {TOLF2})")
    bad = [
        K
        for K in range(63, 78)
        if V.family(K) is not None
        and V.item(V.family(K), [float(Z.f_pred(x)) for x in V.family(K)[0]]) <= TOLF2
    ]
    c["NC5_operator_needed"] = len(bad) == 0
    print(f"closed form alone passes the tolerance for families: {bad}")
    return c, cf


def main():
    rho = T.rho_pred()
    c, cf = controls(rho)
    print("controls:", c)
    if not all(c.values()):
        print("VERDICT q1174 T141: VOID (a control does not behave)")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("VERDICT q1174 T141: INCOMPLETE (controls only)")
        sys.exit(3)
    res = {}
    for K in KS:
        fm = V.family(K)
        if fm is None:
            res[K] = None
            continue
        pr = predict(cf, K, rho[K], fm[0])
        res[K] = V.item(fm, pr)
        print(
            f"   K {K}: largest deviation over 10 levels {res[K]:.5f} (TOLF2 {TOLF2}); differences "
            + " ".join(f"{a - b:+.5f}" for a, b in zip(fm[1], pr))
        )
    v = judge(res)
    print(
        "VERDICT q1174 T141: "
        + v
        + " ("
        + "; ".join(
            f"K {K} {'MISSING' if x is None else ('PASS' if x <= TOLF2 else 'FAIL')}" for K, x in res.items()
        )
        + ")"
    )
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
