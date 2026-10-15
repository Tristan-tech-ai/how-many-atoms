"""q1166 (7 Oct 2026, registered in amendment 1907): verdict script of T138. The rate of the centre signal AT THE EVENT of the families K = 76, 77, 78, 79 against the closed-form response, with the ratio
R = (rate at the event) / sigma_pred(rho_pred(K)). sigma_pred: even K 32 pi^2 rho exp(-v^2/4) (v = 4 pi rho); odd K the exact quadrature of q1162 (functions of q1163).
Rate at the event: slopes of the last two pairs of states (centre_signal against A), extrapolated linearly in A from their midpoints to the event amplitude (A_cross of tower_K<K+1>_birth_out.json for even K, A_s of tower_K<K+1>_As.json for odd K), as in q1165.
Calibration (K 61 to 75, drop = 0, 1, 2 last states of each family; the dropped geometry reproduces distances d of the last state before the event of 0.0006 to 0.0143, K 76 has 0.0045): per parity the least-squares model R = a + b rho + c d.
Items per family: |R_meas - (a + b rho + c d)| <= TOL (TOL_EVEN, TOL_ODD). Verdict: PASS if all four families are present and every item holds; FAIL if a present family fails; INCOMPLETE if none fails and one is missing; VOID if a control misbehaves or a list has fewer than 8 states.
Controls (registration run, no K 76 to 79 value): PC synthetic series with the predicted R pass at d = 0.0009, 0.0045 and 0.0088; NC1 R + 2.5 TOL fails; NC2 R - 2.5 TOL fails; NC3 a missing family is INCOMPLETE; NC4 leave-one-family-out: each calibration family with its last state dropped (the geometry of K 76) is predicted by the model fitted on the other families within TOL.
Last line: VERDICT q1166 T138: PASS / FAIL / INCOMPLETE / VOID. Usage: py q1166_centre_rate_at_event_k76_79_verdict.py [--registration-run]
"""

import sys, json, math, os
import numpy as np

sys.path.insert(0, ".")
import q1155_apriori_lattice_k76_79_verdict as T
import q1117_event_log_flow_law_verdict as Q
import q1163_odd_centre_slope_k77_79_verdict as O

KS = (76, 77, 78, 79)
TOL = {0: 0.0025, 1: 0.002}


_CACHE = {}


def sig(K, rho):
    v = 4 * math.pi * rho
    if K % 2:
        if (K, rho) not in _CACHE:
            _CACHE[(K, rho)] = O.sigma_pred(rho)
        return _CACHE[(K, rho)]
    return 32 * math.pi**2 * rho * math.exp(-v * v / 4)


def ratio(A, cs, Aev, sp):
    s1 = (cs[2] - cs[1]) / (A[2] - A[1])
    s0 = (cs[1] - cs[0]) / (A[1] - A[0])
    m1 = 0.5 * (A[2] + A[1])
    m0 = 0.5 * (A[1] + A[0])
    return (s1 + (s1 - s0) / (m1 - m0) * (Aev - m1)) / sp, Aev - A[2]


def family(K, drop=0):
    lp = Q.list_path(K)
    ep, key = Q.event_path(K)
    if not (os.path.exists(lp) and os.path.exists(ep)):
        return None
    st = [json.load(open(l.split()[1])) for l in open(lp) if len(l.split()) >= 2]
    if len(st) < 8:
        print(f"VERDICT q1166 T138: VOID (K {K}: list has {len(st)} states)")
        sys.exit(2)
    st = st[: len(st) - drop][-3:]
    return (
        [float(d["A"]) for d in st],
        [float(d["centre_signal"]) for d in st],
        float(json.load(open(ep))[key]),
    )


def calib_points(rho, skip=None):
    pts = {0: [], 1: []}
    for K in range(61, 76):
        if K == skip:
            continue
        for drop in (0, 1, 2):
            f = family(K, drop)
            if f is None:
                continue
            r, d = ratio(f[0], f[1], f[2], sig(K, rho[K]))
            pts[K % 2].append((rho[K], d, r))
    return pts


def fit(pts):
    m = {}
    for par, p in pts.items():
        X = np.array([[1.0, x[0], x[1]] for x in p])
        y = np.array([x[2] for x in p])
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        m[par] = (c, float(np.std(y - X @ c)), len(p))
    return m


def predict(m, K, rho, d):
    c = m[K % 2][0]
    return float(c[0] + c[1] * rho + c[2] * d)


def judge_k(res):
    s = []
    for K, x in res.items():
        s.append("MISSING" if x is None else ("PASS" if abs(x[0] - x[1]) <= TOL[K % 2] else "FAIL"))
    return "FAIL" if "FAIL" in s else ("INCOMPLETE" if "MISSING" in s else "PASS")


def synth(K, rho, R, d, Aev=49.0):
    """synthetic centre signal with the rate ratio R at the event (sigma = sigma_pred R) and a curvature that lowers the previous-pair slope by about 5 percent; states 0.005 apart, the last d before the event"""
    sp = sig(K, rho[K])
    A = [Aev - d - 0.010, Aev - d - 0.005, Aev - d]
    q = 0.05 * sp * R / 0.0075
    return A, [-(sp * R * (Aev - a) - 0.5 * q * (Aev - a) ** 2) for a in A], Aev


def controls(rho):
    c = {}
    pts = calib_points(rho)
    m = fit(pts)
    print(
        "calibration fit per parity (a, b, c), residual std, points:",
        {("odd" if p else "even"): (np.round(m[p][0], 5).tolist(), round(m[p][1], 5), m[p][2]) for p in m},
    )

    def run(shift, ds=(0.0009, 0.0045, 0.0088)):
        out = {}
        for K in KS:
            for d in ds:
                pr = predict(m, K, rho[K], d)
                A, cs, Aev = synth(K, rho, pr + shift * TOL[K % 2], d)
                r, dd = ratio(A, cs, Aev, sig(K, rho[K]))
                out[(K, d)] = (r, pr)
        return out

    def judge_all(out):
        s = [abs(r - p) <= TOL[K % 2] for (K, d), (r, p) in out.items()]
        return "PASS" if all(s) else "FAIL"

    c["PC"] = judge_all(run(0.0)) == "PASS"
    c["NC1"] = all(judge_all({k: v}) == "FAIL" for k, v in run(2.5).items())
    c["NC2"] = all(judge_all({k: v}) == "FAIL" for k, v in run(-2.5).items())
    c["NC3"] = judge_k({76: (1.0, 1.0), 77: None, 78: (1.0, 1.0), 79: (1.0, 1.0)}) == "INCOMPLETE"
    worst = {0: 0.0, 1: 0.0}
    ok = True
    for K in range(61, 76):
        mk = fit(calib_points(rho, skip=K))
        f = family(K, 1)
        if f is None:
            continue
        r, d = ratio(f[0], f[1], f[2], sig(K, rho[K]))
        dev = abs(r - predict(mk, K, rho[K], d))
        worst[K % 2] = max(worst[K % 2], dev)
        ok = ok and dev <= TOL[K % 2]
    c["NC4_LOO"] = ok
    print(
        f"leave-one-family-out, last state dropped: largest deviation even {worst[0]:.4f} (TOL {TOL[0]}), odd {worst[1]:.4f} (TOL {TOL[1]})"
    )
    return c, m


def main():
    rho = T.rho_pred()
    c, m = controls(rho)
    print("controls:", c)
    if not all(c.values()):
        print("VERDICT q1166 T138: VOID (a control does not behave)")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("VERDICT q1166 T138: INCOMPLETE (controls only)")
        sys.exit(3)
    res = {}
    for K in KS:
        f = family(K)
        if f is None:
            res[K] = None
            continue
        r, d = ratio(f[0], f[1], f[2], sig(K, rho[K]))
        res[K] = (r, predict(m, K, rho[K], d))
        print(
            f"   K {K}: distance {d:.4f}, R measured {r:.4f}, predicted {res[K][1]:.4f}, difference {r - res[K][1]:+.4f} (TOL {TOL[K % 2]})"
        )
    v = judge_k(res)
    print(
        "VERDICT q1166 T138: "
        + v
        + " ("
        + "; ".join(
            f"K {K} {'MISSING' if x is None else ('PASS' if abs(x[0] - x[1]) <= TOL[K % 2] else 'FAIL')}"
            for K, x in res.items()
        )
        + ")"
    )
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
