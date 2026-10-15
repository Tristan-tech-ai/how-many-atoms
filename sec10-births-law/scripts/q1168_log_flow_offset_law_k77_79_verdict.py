"""q1168 (7 Oct 2026, registered in amendment 1909): verdict script of T139. The offset of the log flow at the event from the a priori value, g_ev - g_apriori, equals -(c + beta d) / (4 pi^2 rho^2) with a constant c per parity
(reading q1167: offset * 4 pi^2 rho^2 = -0.0449 for even K and -0.0483 for odd K, standard deviations 0.0005 and 0.0007, K 61 to 76) and a small linear term in the distance d of the last state to the event.
g_ev: the procedure of q1117 (copied): dp_1/dA at the last two states by the Lagrange derivative through four states, g = (dp_1/dA) / (p_1 h), linear extrapolation in A to the event amplitude; g_apriori from the sharp lattice with rho = rho_pred(K) (q1155).
Calibration: K 61 to 76 with the last 0, 1 and 2 states of each family dropped (distances d of 0.0006 to 0.014; K 77 to 79 have about 0.001 to 0.005): per parity y = offset * 4 pi^2 rho^2 = a + b d by least squares (24 points even, 24 odd).
Items for K = 77, 78, 79: |g_ev - g_apriori - (a + b d) / (4 pi^2 rho^2)| <= TOLG. Verdict: PASS if all three families are present and every item holds; FAIL if a present family fails; INCOMPLETE if none fails and one is missing; VOID if a control misbehaves or a list has fewer than 8 states.
Controls (registration run, no K 77 to 79 value): PC the prediction itself passes; NC1 prediction + 2.5 TOLG fails; NC2 prediction - 2.5 TOLG fails; NC3 a missing family is INCOMPLETE; NC4 leave-one-family-out with the last state dropped (the geometry of a family with a larger distance than the calibration median): the deviation is within TOLG for all 16 calibration families.
Last line: VERDICT q1168 T139: PASS / FAIL / INCOMPLETE / VOID. Usage: py q1168_log_flow_offset_law_k77_79_verdict.py [--registration-run]
"""

import sys, json, math, os
import numpy as np

sys.path.insert(0, ".")
import q1155_apriori_lattice_k76_79_verdict as T
import q1117_event_log_flow_law_verdict as Q

KS = (77, 78, 79)
TOLG = 0.0001
F4 = 4 * math.pi**2


def family(K, drop=0):
    lp = Q.list_path(K)
    ep, key = Q.event_path(K)
    if not (os.path.exists(lp) and os.path.exists(ep)):
        return None
    st = []
    for line in open(lp):
        t = line.split()
        if len(t) < 2:
            continue
        d = json.load(open(t[1]))
        u = [float(x) for x in d["u"]]
        st.append((float(d["A"]), u[0], u[1] - u[0]))
    if len(st) < 8:
        print(f"VERDICT q1168 T139: VOID (K {K}: list has {len(st)} states)")
        sys.exit(2)
    ev = json.load(open(ep))
    if key not in ev:
        print(f"VERDICT q1168 T139: VOID (no key {key} in {ep})")
        sys.exit(2)
    return st[: len(st) - drop], float(ev[key])


def gev(K, drop=0):
    f = family(K, drop)
    if f is None:
        return None
    st, Ae = f
    gs = []
    for i in (len(st) - 2, len(st) - 1):
        a = [st[j][0] for j in range(i - 3, i + 1)]
        p = [st[j][1] for j in range(i - 3, i + 1)]
        gs.append((st[i][0], Q.deriv_last(a, p) / (st[i][1] * st[i][2])))
    (a1, g1), (a2, g2) = gs
    return g2 + (g2 - g1) / (a2 - a1) * (Ae - a2), Ae - st[-1][0]


def calib_points(rho, skip=None, drops=(0, 1, 2)):
    pts = {0: [], 1: []}
    for K in range(61, 77):
        if K == skip:
            continue
        for drop in drops:
            r = gev(K, drop)
            if r is None:
                continue
            off = r[0] - T.sharp(rho[K], K)["g"]
            pts[K % 2].append((r[1], off * F4 * rho[K] ** 2))
    return pts


def fit(pts):
    m = {}
    for par, p in pts.items():
        X = np.array([[1.0, x[0]] for x in p])
        y = np.array([x[1] for x in p])
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        m[par] = (c, float(np.std(y - X @ c)), len(p))
    return m


def pred_off(m, K, rho, d):
    c = m[K % 2][0]
    return float((c[0] + c[1] * d) / (F4 * rho**2))


def judge(res):
    s = ["MISSING" if x is None else ("PASS" if abs(x[0] - x[1]) <= TOLG else "FAIL") for x in res.values()]
    return "FAIL" if "FAIL" in s else ("INCOMPLETE" if "MISSING" in s else "PASS")


def controls(rho):
    c = {}
    m = fit(calib_points(rho))
    print(
        "calibration per parity (a, b), residual std of offset*4pi^2rho^2, points:",
        {("odd" if p else "even"): (np.round(m[p][0], 5).tolist(), round(m[p][1], 5), m[p][2]) for p in m},
    )
    pr = {K: pred_off(m, K, rho[K], 0.003) for K in KS}
    c["PC"] = judge({K: (pr[K], pr[K]) for K in KS}) == "PASS"
    c["NC1"] = judge({K: (pr[K] + 2.5 * TOLG, pr[K]) for K in KS}) == "FAIL"
    c["NC2"] = judge({K: (pr[K] - 2.5 * TOLG, pr[K]) for K in KS}) == "FAIL"
    c["NC3"] = judge({77: (0, 0), 78: None, 79: (0, 0)}) == "INCOMPLETE"
    worst = 0.0
    ok = True
    for K in range(61, 77):
        mk = fit(calib_points(rho, skip=K))
        r = gev(K, 1)
        if r is None:
            continue
        dev = abs((r[0] - T.sharp(rho[K], K)["g"]) - pred_off(mk, K, rho[K], r[1]))
        worst = max(worst, dev)
        ok = ok and dev <= TOLG
    c["NC4_LOO"] = ok
    print(
        f"leave-one-family-out, last state dropped: largest deviation of the offset {worst:.2e} (TOLG {TOLG:.0e})"
    )
    return c, m


def main():
    rho = T.rho_pred()
    c, m = controls(rho)
    print("controls:", c)
    if not all(c.values()):
        print("VERDICT q1168 T139: VOID (a control does not behave)")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("VERDICT q1168 T139: INCOMPLETE (controls only)")
        sys.exit(3)
    res = {}
    for K in KS:
        r = gev(K)
        if r is None:
            res[K] = None
            continue
        off = r[0] - T.sharp(rho[K], K)["g"]
        pr = pred_off(m, K, rho[K], r[1])
        res[K] = (off, pr)
        print(
            f"   K {K}: distance {r[1]:.4f}, g_ev {r[0]:.5f}, offset {off:+.6f}, predicted {pr:+.6f}, difference {off - pr:+.2e} (TOLG {TOLG:.0e})"
        )
    v = judge(res)
    print(
        "VERDICT q1168 T139: "
        + v
        + " ("
        + "; ".join(
            f"K {K} {'MISSING' if x is None else ('PASS' if abs(x[0] - x[1]) <= TOLG else 'FAIL')}"
            for K, x in res.items()
        )
        + ")"
    )
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
