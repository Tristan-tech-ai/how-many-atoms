"""q1179 (7 Oct 2026, registered in amendment 1914): verdict script of T142. The first correction of the flows at the event from the next-order continuum problem (derivation of amendment 1914, code q1175): in the variable xi = p / v, v = 4 pi rho_pred(K), eps = 1 / v^2,
f_j = F(xi_j) + eps [chi1(xi_j) + 2 pi eta0'(xi_j) F(xi_j) - 4 pi xi_j] + kappa eps F''(xi_j),
chi1 = (2/3) xi F' + (1/3) xi^2 F'' + (8 pi / 3) xi^2 coth(pi xi) F', eta0' = 2 xi coth(pi xi), F = coth(pi xi) - pi xi / sinh^2(pi xi). chi1 solves the difference equation of the response at first order (residual 1e-13 in q1175); kappa is the one fitted number
(least squares over the ten levels of the capacity families K = 63 to 77, both parities together; reading q1178: the single-family values lie between -0.2517 and -0.2620, none depends on the parity).
Test families K = 78 and 79 (K 77 in the calibration): flows at the event of the ten innermost levels (q1119.flows_at_event as in figsrc.event_profiles), positions p_j of the last state. Item per family: the largest over the ten levels of |f_j - prediction| <= TOLF3.
Verdict: PASS if both families are present and inside; FAIL if a present family is outside; INCOMPLETE if none fails and one is missing; VOID if a control misbehaves or a list has fewer than 8 states.
Controls (registration run, no K 78 or 79 value): PC the prediction itself passes; NC1 prediction + 2.5 TOLF3 at level 3 fails; NC2 prediction - 2.5 TOLF3 at level 8 fails; NC3 a missing family is INCOMPLETE; NC4 leave-one-family-out over K 63 to 77 within TOLF3;
NC5 the closed form alone fails TOLF3 for every calibration family; NC6 the derived terms without the kappa term (kappa = 0) fail TOLF3 for at least one half of the calibration families (the fitted term is needed).
Last line: VERDICT q1179 T142: PASS / FAIL / INCOMPLETE / VOID. Usage: py q1179_first_order_response_k78_79_verdict.py [--registration-run]
"""

import sys, math
import numpy as np
import mpmath as mp

sys.path.insert(0, ".")
import q1170_flow_correction_curve_k77_79_verdict as V
import q1155_apriori_lattice_k76_79_verdict as T

KS = (78, 79)
TOLF3 = 0.0006
NJ = 10
mp.mp.dps = 25
pi = mp.pi
Fm = lambda z: mp.coth(pi * z) - pi * z / mp.sinh(pi * z) ** 2
_c = {}


def terms(x):
    """returns (F, derived part eps-coefficient, F'') at xi = x"""
    if x not in _c:
        xm = mp.mpf(x)
        F = Fm(xm)
        F1 = mp.diff(Fm, xm)
        F2 = mp.diff(Fm, xm, 2)
        cth = mp.coth(pi * xm)
        chi1 = mp.mpf(2) / 3 * xm * F1 + mp.mpf(1) / 3 * xm * xm * F2 + 8 * pi / 3 * xm * xm * cth * F1
        _c[x] = (float(F), float(chi1 + 4 * pi * xm * cth * F - 4 * pi * xm), float(F2))
    return _c[x]


def rows(K, rho):
    fm = V.family(K)
    if fm is None:
        return None
    eps = 1 / (4 * math.pi * rho[K]) ** 2
    v = 4 * math.pi * rho[K]
    # family() returns xi_h = p h / (4 pi); recover p / v from p h / (4 pi) with the spacing: use the positions directly
    return fm, eps, v


def positions(K):
    import json, os
    import q1119_event_flow_profile_reading as Q19
    import q1114_log_flow_along_family_reading as Q14

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
        print(f"VERDICT q1179 T142: VOID (K {K}: list has {len(st)} states)")
        sys.exit(2)
    Q19.NJ = NJ
    f, p = Q19.flows_at_event(st, Ae)
    return [float(x) for x in f[:NJ]], [float(x) for x in p[:NJ]]


def calib(rho, skip=None):
    out = []
    for K in range(63, 78):
        if K == skip:
            continue
        r = positions(K)
        if r is None:
            continue
        f, p = r
        v = 4 * math.pi * rho[K]
        eps = 1 / v**2
        for j in range(NJ):
            xi = p[j] / v
            F, der, F2 = terms(xi)
            out.append(((f[j] - F - eps * der) / eps, F2))
    return out


def fit_kappa(rows_):
    y = np.array([a for a, _ in rows_])
    b = np.array([c for _, c in rows_])
    return float(b @ y / (b @ b))


def predict(kappa, K, rho, p):
    v = 4 * math.pi * rho[K]
    eps = 1 / v**2
    out = []
    for pj in p:
        xi = pj / v
        F, der, F2 = terms(xi)
        out.append(F + eps * (der + kappa * F2))
    return out


def item(f, pr):
    return max(abs(a - b) for a, b in zip(f, pr))


def judge(res):
    s = ["MISSING" if x is None else ("PASS" if x <= TOLF3 else "FAIL") for x in res.values()]
    return "FAIL" if "FAIL" in s else ("INCOMPLETE" if "MISSING" in s else "PASS")


def controls(rho):
    c = {}
    kappa = fit_kappa(calib(rho))
    print(f"calibration (K 63 to 77): kappa {kappa:+.4f} (coefficient of eps F'' in f - F)")
    f77, p77 = positions(77)
    pr = predict(kappa, 77, rho, p77)
    c["PC"] = judge({78: item(pr, pr), 79: 0.0}) == "PASS"
    c["NC1"] = judge({78: item([x + (2.5 * TOLF3 if j == 2 else 0) for j, x in enumerate(pr)], pr)}) == "FAIL"
    c["NC2"] = judge({78: item([x - (2.5 * TOLF3 if j == 7 else 0) for j, x in enumerate(pr)], pr)}) == "FAIL"
    c["NC3"] = judge({78: 0.0, 79: None}) == "INCOMPLETE"
    worst = 0.0
    nbad_zero = 0
    nbad_F = 0
    n = 0
    for K in range(63, 78):
        r = positions(K)
        if r is None:
            continue
        f, p = r
        n += 1
        kk = fit_kappa(calib(rho, skip=K))
        worst = max(worst, item(f, predict(kk, K, rho, p)))
        nbad_zero += item(f, predict(0.0, K, rho, p)) > TOLF3
        v = 4 * math.pi * rho[K]
        nbad_F += item(f, [terms(pj / v)[0] for pj in p]) <= TOLF3
    c["NC4_LOO"] = worst <= TOLF3
    print(f"leave-one-family-out: largest deviation of any level {worst:.5f} (TOLF3 {TOLF3})")
    c["NC5_closed_form_needs_terms"] = nbad_F == 0
    print(f"closed form alone passes the tolerance for {nbad_F} of {n} families")
    c["NC6_kappa_needed"] = nbad_zero >= n / 2
    print(f"derived terms without kappa fail the tolerance for {nbad_zero} of {n} families")
    return c, kappa


def main():
    rho = T.rho_pred()
    c, kappa = controls(rho)
    print("controls:", {k: bool(v) for k, v in c.items()})
    if not all(c.values()):
        print("VERDICT q1179 T142: VOID (a control does not behave)")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("VERDICT q1179 T142: INCOMPLETE (controls only)")
        sys.exit(3)
    res = {}
    for K in KS:
        r = positions(K)
        if r is None:
            res[K] = None
            continue
        f, p = r
        pr = predict(kappa, K, rho, p)
        res[K] = item(f, pr)
        print(
            f"   K {K}: largest deviation over 10 levels {res[K]:.5f} (TOLF3 {TOLF3}); differences "
            + " ".join(f"{a - b:+.5f}" for a, b in zip(f, pr))
        )
    v = judge(res)
    print(
        "VERDICT q1179 T142: "
        + v
        + " ("
        + "; ".join(
            f"K {K} {'MISSING' if x is None else ('PASS' if x <= TOLF3 else 'FAIL')}" for K, x in res.items()
        )
        + ")"
    )
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
