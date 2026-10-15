"""q1163 (7 Oct 2026, registered in amendment 1903): verdict script of T137. The slope of the centre signal of the last two states of the odd families K = 77 and 79 against the response sigma_pred computed with the closed-form profile
(exact quadrature, functions copied as text from q1162). Item: the ratio lies in [0.92, 1.02]. Last line: VERDICT q1163 T137: PASS / FAIL / INCOMPLETE / VOID.
Usage: py q1163_odd_centre_slope_k77_79_verdict.py [--registration-run]"""

import sys, json, math, os
import mpmath as mp

sys.path.insert(0, ".")
import q1155_apriori_lattice_k76_79_verdict as T
import q1117_event_log_flow_law_verdict as Q

mp.mp.dps = 40
KS = (77, 79)
LO, HI = 0.92, 1.02


def eta_p(xi):
    z = mp.pi * xi
    return mp.mpf(2) / mp.pi if abs(xi) < mp.mpf("1e-20") else 2 * xi * mp.coth(z)


def F(xi):
    z = mp.pi * xi
    return 2 * z / 3 if abs(z) < mp.mpf("1e-8") else mp.coth(z) - z / mp.sinh(z) ** 2


def Fp(xi):
    z = mp.pi * xi
    if abs(z) < mp.mpf("1e-6"):
        return 2 * mp.pi / 3 - 4 * mp.pi * z**2 / 15
    return mp.pi * (-2 / mp.sinh(z) ** 2 + 2 * z * mp.cosh(z) / mp.sinh(z) ** 3)


def eta(xi):
    a = abs(xi)
    if a < mp.mpf("1e-30"):
        return mp.mpf(0)
    y = mp.e ** (-2 * mp.pi * a)
    val = a * a + mp.mpf(1) / 6 + (2 * a / mp.pi) * mp.log(1 - y) - mp.polylog(2, y) / mp.pi**2
    return val if xi > 0 else -val


def sigma_pred(rho):
    rho = mp.mpf(rho)
    v = 4 * mp.pi * rho
    psi = lambda u: mp.e ** (-u * u / 4) / mp.sqrt(4 * mp.pi)
    Mt = lambda u: rho * u - eta(u / v) / 2
    dM1 = lambda u: eta_p(0) / (2 * v) - eta_p(u / v) / (2 * v)
    pts = [mp.mpf(k) / 2 for k in range(0, 41)]
    dRe = 2 * mp.pi * mp.quad(lambda u: psi(u) * F(u / v) * mp.sin(2 * mp.pi * Mt(u)), pts)
    dIm = (
        2
        * 2
        * mp.pi
        * mp.quad(
            lambda u: psi(u)
            * (
                -(Fp(u / v) - Fp(0)) / (2 * v) * mp.cos(2 * mp.pi * Mt(u))
                + mp.pi * dM1(u) * F(u / v) * mp.sin(2 * mp.pi * Mt(u))
            ),
            pts,
        )
    )
    return float(2 * rho * (8 * mp.pi**2 * rho**2 * dRe + 8 * mp.pi * rho * dIm))


def judge(r):
    s = ["MISSING" if x is None else ("PASS" if LO <= x <= HI else "FAIL") for x in r.values()]
    return "FAIL" if "FAIL" in s else ("INCOMPLETE" if "MISSING" in s else "PASS")


def series_ratio(st, rho):
    return (
        (float(st[-1]["centre_signal"]) - float(st[-2]["centre_signal"]))
        / (float(st[-1]["A"]) - float(st[-2]["A"]))
        / sigma_pred(rho)
    )


def controls(rho):
    c = {}
    sp = {K: sigma_pred(rho[K]) for K in KS}

    def synth(K, f=1.0):
        A = [48.0 + 0.005 * i for i in range(3)]
        z = A[-1] + 0.002
        return [{"A": a, "centre_signal": -(sp[K] * f * (z - a))} for a in A]

    good = {K: series_ratio(synth(K), rho[K]) for K in KS}
    c["PC"] = judge(good) == "PASS"
    c["NC1"] = judge({K: series_ratio(synth(K, 0.85), rho[K]) for K in KS}) == "FAIL"
    c["NC2"] = judge({77: good[77], 79: None}) == "INCOMPLETE"
    cal = []
    for K in range(61, 76, 2):
        st = [json.load(open(l.split()[1])) for l in open(Q.list_path(K)) if len(l.split()) >= 2][-2:]
        cal.append(series_ratio(st, rho[K]))
    c["NC3"] = all(LO <= x <= HI for x in cal) and not all(0.97 <= x <= 1.01 for x in cal)
    return c


def measured(K, rho):
    lp = Q.list_path(K)
    if not os.path.exists(lp):
        return None
    st = [json.load(open(l.split()[1])) for l in open(lp) if len(l.split()) >= 2]
    if len(st) < 8:
        print(f"VERDICT q1163 T137: VOID (K {K}: list has {len(st)} states)")
        sys.exit(2)
    return series_ratio(st[-2:], rho[K])


def main():
    rho = T.rho_pred()
    c = controls(rho)
    print("controls:", c)
    if not all(c.values()):
        print("VERDICT q1163 T137: VOID (a control does not behave)")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("VERDICT q1163 T137: INCOMPLETE (controls run only)")
        sys.exit(3)
    r = {K: measured(K, rho) for K in KS}
    for K, x in r.items():
        print(
            f"   K {K}: "
            + (
                "missing"
                if x is None
                else f"ratio {x:.4f} ({'in' if LO <= x <= HI else 'outside'} [{LO}, {HI}])"
            )
        )
    v = judge(r)
    print(
        f"VERDICT q1163 T137: {v} ("
        + "; ".join(
            f"K {K} {'MISSING' if x is None else ('PASS' if LO <= x <= HI else 'FAIL')}" for K, x in r.items()
        )
        + ")"
    )
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
