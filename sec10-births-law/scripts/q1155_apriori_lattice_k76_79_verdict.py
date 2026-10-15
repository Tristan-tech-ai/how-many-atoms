"""q1155 (7 Oct 2026, registered in amendment 1898): verdict script of T135. A priori lattice and log flow at the end of the families K 76 to 79 from the density rho_pred(K) alone.
Sharp lattice: positive atoms solve rho p_j = j - 1/2 + eta(p_j / v) / 2 (K even) or rho p_j = j + eta(p_j / v) / 2 (K odd), v = 4 pi rho, eta(xi) = xi^2 + 1/6 + (2 xi / pi) ln(1 - e^{-2 pi xi}) - Li2(e^{-2 pi xi}) / pi^2.
Items per family: H |h_meas / h_sharp - 1| <= 0.0015; S |p_1/h_meas - p_1/h_sharp| <= 0.0006; G |g_ev - (g_apriori + d_lin(K))| <= 0.00030, g_apriori = f_pred(xi_1) / (4 pi xi_1) with xi_1 = p_1 h / (4 pi) of the sharp lattice,
d_lin = least-squares line through the offsets g_ev - g_apriori of the same parity among K 61 to 75.
Last line: VERDICT q1155 T135: PASS / FAIL / INCOMPLETE / VOID as registered. Usage: py q1155_apriori_lattice_k76_79_verdict.py [--registration-run]
"""

import sys, math, json, os
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, ".")
import q1117_event_log_flow_law_verdict as Q
import q1153_profile_closed_form_verdict as Z

ARCH = "scratch"
KS = (76, 77, 78, 79)
FOUR_PI = 4 * math.pi


def eta(xi):
    xi = abs(xi)
    if xi < 1e-9:
        return 0.0
    y = math.exp(-2 * math.pi * xi)
    li2 = sum(y**n / n**2 for n in range(1, 80))
    return xi * xi + 1 / 6 + (2 * xi / math.pi) * math.log(1 - y) - li2 / math.pi**2


def sharp(rho, K):
    v = FOUR_PI * rho
    p = []
    for j in (1, 2):
        c = (j - 0.5) if K % 2 == 0 else j
        p.append(brentq(lambda x: rho * x - c - eta(x / v) / 2, 0.01, j + 2))
    h = p[1] - p[0]
    xi1 = p[0] * h / FOUR_PI
    return dict(h=h, r1=p[0] / h, xi1=xi1, g=float(Z.f_pred(xi1)) / (FOUR_PI * xi1))


def rho_pred():
    out = {}
    for fn in (
        "q635_capacity_K44_69_predictions.json",
        "q635_capacity_K67_72_predictions.json",
        "q635_capacity_K72_74_predictions.json",
        "q635_capacity_K74_79_predictions.json",
    ):
        p = f"{ARCH}/{fn}"
        if not os.path.exists(p):
            continue
        for K, v in json.load(open(p))["capacity"]["pred"].items():
            out.setdefault(int(K), v["rho"])
    return out


def last_state(K):
    last = [l.split() for l in open(Q.list_path(K)) if len(l.split()) >= 2][-1]
    u = [float(x) for x in json.load(open(last[1]))["u"]]
    h = u[1] - u[0]
    return h, u[0] / h


def measured(K, rho):
    lp = Q.list_path(K)
    ep, _ = Q.event_path(K)
    if not (os.path.exists(lp) and os.path.exists(ep)):
        return None
    g = Q.g_ev(K)
    if g is None:
        return None
    h, r1 = last_state(K)
    return dict(g=g, h=h, r1=r1)


def calib(rho):
    off = {}
    for K in range(61, 76):
        m = measured(K, rho[K])
        s = sharp(rho[K], K)
        off[K] = m["g"] - s["g"]
    return off


def d_lin(off, K):
    ks = [k for k in off if k % 2 == K % 2]
    return float(np.polyval(np.polyfit(ks, [off[k] for k in ks], 1), K))


def judge(fam, rho, off):
    res = {}
    for K in KS:
        m = fam.get(K)
        if m is None:
            res[K] = "MISSING"
            continue
        s = sharp(rho[K], K)
        H = abs(m["h"] / s["h"] - 1) <= 0.0015
        S = abs(m["r1"] - s["r1"]) <= 0.0006
        G = abs(m["g"] - (s["g"] + d_lin(off, K))) <= 0.00030
        res[K] = "PASS" if (H and S and G) else "FAIL"
        res[(K, "items")] = (bool(H), bool(S), bool(G))
    s_ = [v for k, v in res.items() if not isinstance(k, tuple)]
    return ("FAIL" if "FAIL" in s_ else ("INCOMPLETE" if "MISSING" in s_ else "PASS")), res


def synth(rho, off, dh=0.0, dr=0.0, dg=0.0, missing=None):
    fam = {}
    for K in KS:
        if K == missing:
            fam[K] = None
            continue
        s = sharp(rho[K], K)
        fam[K] = dict(h=s["h"] * (1 + dh), r1=s["r1"] + dr, g=s["g"] + d_lin(off, K) + dg)
    return fam


def controls(rho, off):
    c = {}
    c["PC"] = judge(synth(rho, off), rho, off)[0] == "PASS"
    c["NC1"] = judge(synth(rho, off, dh=0.003), rho, off)[0] == "FAIL"
    c["NC2"] = judge(synth(rho, off, dr=0.002), rho, off)[0] == "FAIL"
    c["NC3"] = judge(synth(rho, off, dg=0.0006), rho, off)[0] == "FAIL"
    c["NC4"] = judge(synth(rho, off, missing=79), rho, off)[0] == "INCOMPLETE"
    hd = [
        sharp(rho[K], K)["h"] / measured(K, rho[K])["h"] - 1 for K in (61, 62, 70, 75)
    ]  # sharp over measured, the sign of the table of amendment 1898
    c["PC2"] = (
        abs(hd[0] + 0.0006) < 0.0002
        and abs(hd[1] + 0.0005) < 0.0002
        and abs(hd[2]) < 0.0002
        and abs(hd[3] + 0.0005) < 0.0002
    )
    return c


def main():
    rho = rho_pred()
    missing = [K for K in list(range(61, 80)) if K not in rho]
    if missing:
        print("VERDICT q1155 T135: VOID (rho_pred missing for", missing, ")")
        sys.exit(2)
    off = calib(rho)
    c = controls(rho, off)
    print("calibration offsets g_ev - g_apriori:", {k: round(v, 5) for k, v in off.items()})
    print("controls:", c)
    if not all(c.values()):
        print("VERDICT q1155 T135: VOID (a control does not behave)")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("VERDICT q1155 T135: INCOMPLETE (controls run only)")
        sys.exit(3)
    fam = {K: measured(K, rho[K]) for K in KS}
    v, res = judge(fam, rho, off)
    for K in KS:
        if fam[K] is None:
            print(f"   K {K}: missing")
            continue
        s = sharp(rho[K], K)
        m = fam[K]
        print(
            f"   K {K}: rho {rho[K]:.4f}; h meas {m['h']:.4f} sharp {s['h']:.4f} (rel {m['h'] / s['h'] - 1:+.5f}); p1/h meas {m['r1']:.4f} sharp {s['r1']:.4f}; g_ev {m['g']:.5f}, g_apriori {s['g']:.5f}, d_lin {d_lin(off, K):+.5f}, residual {m['g'] - s['g'] - d_lin(off, K):+.6f}; items H S G {res[(K, 'items')]} -> {res[K]}"
        )
    print(f"VERDICT q1155 T135: {v} (" + "; ".join(f"K {K} {res[K]}" for K in KS) + ")")
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[v])


if __name__ == "__main__":
    main()
