"""q1154 (7 Oct 2026, registered in amendment 1897): verdict script of T134. The closed-form profile f_pred(xi) = coth(pi xi) - pi xi / sinh^2(pi xi) for the capacity families K 76 to 79.
Items per family: E1 rms of f_j - f_pred(xi_j) over the ten innermost atoms <= 0.0050 and largest |difference| <= 0.012; E2 |g_ev - g_pred| <= 0.0020 with g_pred = f_pred(xi_1) / (4 pi xi_1);
E3 |g_ev - (g_pred + d_lin(K))| <= 0.00020, d_lin = least-squares line through the offsets g_ev - g_pred of the same parity among K 61 to 75.
Last line: VERDICT q1154 T134: PASS if all four families are present and E1 to E3 hold, FAIL if a measured item fails, INCOMPLETE if no item fails and a family is missing, VOID if a control misbehaves.
Usage: py q1154_closed_form_k76_79_verdict.py [--registration-run]"""

import sys, math, json
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "paper_births/figures")
import q1117_event_log_flow_law_verdict as Q
import q1153_profile_closed_form_verdict as Z

KS = (76, 77, 78, 79)
FOUR_PI = 4 * math.pi


def gpred(xi1):
    return float(Z.f_pred(xi1)) / (FOUR_PI * xi1)


def calib_offsets():
    out = {}
    for K in range(61, 76):
        g = Q.g_ev(K)
        last = [l.split() for l in open(Q.list_path(K)) if len(l.split()) >= 2][-1]
        u = [float(x) for x in json.load(open(last[1]))["u"]]
        xi1 = u[0] * (u[1] - u[0]) / FOUR_PI
        out[K] = g - gpred(xi1)
    return out


def d_lin(offsets, K):
    ks = [k for k in offsets if k % 2 == K % 2]
    c = np.polyfit(ks, [offsets[k] for k in ks], 1)
    return float(np.polyval(c, K))


def judge(fam, offsets):
    """fam: {K: None or dict(g=g_ev, xi=[...], f=[...])}"""
    res = {}
    for K in KS:
        d = fam.get(K)
        if d is None:
            res[K] = "MISSING"
            continue
        xi = np.array(d["xi"])
        f = np.array(d["f"])
        e = f - Z.f_pred(xi)
        E1 = np.sqrt(np.mean(e**2)) <= 0.0050 and np.max(np.abs(e)) <= 0.012
        gp = gpred(xi[0])
        E2 = abs(d["g"] - gp) <= 0.0020
        E3 = abs(d["g"] - (gp + d_lin(offsets, K))) <= 0.00020
        res[K] = "PASS" if (E1 and E2 and E3) else "FAIL"
        res[(K, "items")] = (bool(E1), bool(E2), bool(E3))
    s = [v for k, v in res.items() if not isinstance(k, tuple)]
    return ("FAIL" if "FAIL" in s else ("INCOMPLETE" if "MISSING" in s else "PASS")), res


def synth(offsets, shift_profile=0.0, err3=0.0, missing=None):
    fam = {}
    xs = np.array([0.044, 0.131, 0.219, 0.307, 0.395, 0.484, 0.570, 0.657, 0.744, 0.832])
    for K in KS:
        if K == missing:
            fam[K] = None
            continue
        xi = xs * (2.0 if K % 2 else 1.0) * 0.99
        xi = np.array(xi)
        d = d_lin(offsets, K)
        g = gpred(xi[0]) + d + (err3 if K == 77 else 0.0)
        fam[K] = dict(g=g, xi=list(xi), f=list(Z.f_pred(xi) + shift_profile))
    return fam


def controls(offsets):
    c = {}
    c["PC"] = judge(synth(offsets), offsets)[0] == "PASS"
    v, r = judge(synth(offsets, err3=0.0003), offsets)
    c["NC1"] = v == "FAIL" and r[(77, "items")][1] and not r[(77, "items")][2]
    c["NC2"] = judge(synth(offsets, shift_profile=0.015), offsets)[0] == "FAIL"
    c["NC3"] = judge(synth(offsets, missing=79), offsets)[0] == "INCOMPLETE"
    return c


def load_real():
    import figsrc

    prof, _ = figsrc.event_profiles(10)
    fam = {K: None for K in KS}
    for pb, K, xi, f in prof:
        if pb == "capacity" and K in KS:
            g = Q.g_ev(K)
            if g is not None:
                fam[K] = dict(g=g, xi=list(xi), f=list(f))
    return fam


def main():
    offsets = calib_offsets()
    c = controls(offsets)
    print("calibration offsets (K: g_ev - g_pred):", {k: round(v, 5) for k, v in offsets.items()})
    print("controls:", c)
    if not all(c.values()):
        print("VERDICT q1154 T134: VOID (a control does not behave)")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("VERDICT q1154 T134: INCOMPLETE (controls run only)")
        sys.exit(3)
    fam = load_real()
    v, res = judge(fam, offsets)
    for K in KS:
        if fam[K] is None:
            print(f"   K {K}: missing")
            continue
        d = fam[K]
        gp = gpred(d["xi"][0])
        e = np.array(d["f"]) - Z.f_pred(np.array(d["xi"]))
        print(
            f"   K {K}: xi_1 {d['xi'][0]:.4f}, g_ev {d['g']:.5f}, g_pred {gp:.5f}, offset {d['g'] - gp:+.5f}, d_lin {d_lin(offsets, K):+.5f}; profile rms {np.sqrt(np.mean(e ** 2)):.5f}, largest {np.max(np.abs(e)):.5f}; items E1 E2 E3 {res[(K, 'items')]} -> {res[K]}"
        )
    print(f"VERDICT q1154 T134: {v} (" + "; ".join(f"K {K} {res[K]}" for K in KS) + ")")
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
