"""q1201 (8 Oct 2026, to be registered in an amendment before the table is read): table of PREDICTED capacity event amplitudes K_b = 81 to K_max from the births law with the shared centre function,
2 N_half(a) = K_b + 1/6 + delta_der(rho_c(a)) r(rho_c(a)), the method of q1200 (amendments 1956 to 1958) with the immune ratio curve interpolated by a cubic spline of ln(1 - r) between its nodes (rho 0.8858 to 1.0641; the choice follows the post hoc reading of amendment 1958).
Beyond the last node (rho > 1.0641) ln(1 - r) is extrapolated as a straight line in ln v (v = 4 pi rho) with the slope of the last four nodes, anchored at the last node: a MODEL; the events with rho_c > 1.0641 are flagged 'extrapolated'.
The capacity half-line stage is the converged stage of the largest D of the file given on the command line (default centre_scripts/c50_capc_D110.json); the table runs to a_t <= D - 42 (convergence in D of the c50c residual: amendment 1954).
Output: q1201_table.json {D, kmax, a_pred, flag, sha256}. Controls (registration run; no table value beyond K_b 80 is printed): this interpolation reproduces the located amplitudes of amendment 962 (K_b 53, 55, 57, 59) within 5e-10 and the measured amplitudes of K_b 61 to 79 within 0.2e-9, and the extension is continuous at the last node.
Usage: py q1201_births_law_table_shared_centre.py [--registration-run] [FILE]  (from temporal_within/centre_scripts/)
"""

import hashlib, json, sys, os
import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, least_squares

pi = np.pi
s0 = 1 / (16 * pi**2)
k = 1 / (8 * pi**2)
k4 = 1 / (64 * pi**4) - 1 / (96 * pi**2)
c6 = 1 / (512 * pi**6) - 13 / (768 * pi**4)
c8 = 1 / (4096 * pi**8) - 17 / (1536 * pi**6) - 1 / (3072 * pi**4)
c10 = 1 / (32768 * pi**10) - 29 / (6144 * pi**8) - 109 / (368640 * pi**6)
E = [
    4 / 3 + 8 * pi**2 / 45,
    176 / 15 + 32 * pi**2 / 9 + 64 * pi**4 / 1575,
    23792 / 315 + 8416 * pi**2 / 135 - 4544 * pi**4 / 2025 + 2176 * pi**6 / 33075,
    248768 / 567
    + 2083072 * pi**2 / 2835
    - 112448 * pi**4 / 1215
    - 16524544 * pi**6 / 2679075
    + 345088 * pi**8 / 1091475,
]
R = np.linspace(0.45, 2.6, 500001)
f = 1 + k / R**2 + k4 / R**4 + c6 / R**6 + c8 / R**8 + c10 / R**10
D = cumulative_simpson(R**2 / (s0 * f), x=R, initial=0)
Nn = cumulative_simpson(R**3 / (s0 * f), x=R, initial=0)
rinv = CubicSpline(D, R)
Nr = CubicSpline(R, Nn)
dder = lambda r: sum(E[i] / (4 * pi * r) ** (4 + 2 * i) for i in range(4))


def offs(p, lo, hi, x0):
    p = np.asarray(p)
    j = np.arange(1, len(p) + 1) - 0.5
    m = (p >= lo) & (p <= hi)
    return least_squares(
        lambda q: Nr(rinv(p[m] + q[0])) + q[1] - j[m], x0, x_scale=[0.1, 0.1], xtol=1e-15, ftol=1e-15
    ).x


def mkN(p, law):
    def rho4(d):
        return brentq(
            lambda r: r**3 - 3 * r / (8 * pi**2) - 1 / (32 * pi**2 * r) - 3 * s0 * (d + law[0]), 0.02, 400
        )

    def N4(d):
        r = rho4(d)
        return 4 * pi**2 * r**4 - r**2 + np.log(r) / 6 + law[1]

    p = np.asarray(p)
    cs = CubicSpline(p, np.arange(1, len(p) + 1) - 0.5 - np.array([N4(d) for d in p]))
    return lambda a: N4(a) + float(cs(a))


si = [s for s in json.load(open("c33_immc_D146.json"))["stages"] if s["conv"] and s["D"] == 92.0][0]
pi_ = np.array(si["p"][: si["nfree"]])
pi_ = pi_[pi_ <= 72]
oi = offs(pi_, 15, 70, [-3.22, 1.31])
Ni = mkN(pi_, (0.22573, 0.06806))
ev = [(json.load(open(f"c19_fwd_K{kk}.json"))["a_t"], kk) for kk in range(38, 44)] + [
    (json.load(open(f"c19_hl_K{kk}.json"))["a_t"], kk) for kk in (44, 45, 46, 47, 48, 69, 79, 89, 99)
]
ev += [(json.load(open(f"c19_d38_K{kk}.json"))["a_t"], kk) for kk in (52, 55, 58, 62, 65)] + [
    (json.load(open("c19_far_K59.json"))["a_t"], 59)
]
pts = sorted((float(rinv(a + oi[0])), (2 * Ni(a) - K - 1 / 6) / dder(float(rinv(a + oi[0])))) for a, K in ev)
rr = np.array([p[0] for p in pts])
qq = np.array([p[1] for p in pts])
ly = np.log(1 - qq)
cs_ly = CubicSpline(rr, ly)
lv = np.log(4 * pi * rr[-4:])
slope, _ic = np.polyfit(lv, ly[-4:], 1)
icpt = ly[-1] - slope * np.log(
    4 * pi * rr[-1]
)  # slope from the last four nodes, line anchored at the last node (continuity)


def rcurve(r):
    if r <= rr[-1]:
        return 1 - float(np.exp(cs_ly(r)))
    return 1 - float(np.exp(icpt + slope * np.log(4 * pi * r)))


def stage_fn(fn):
    st = [s for s in json.load(open(fn))["stages"] if s["conv"]]
    sc = st[-1]
    pc = np.array(sc["d"][: 1 + sc["nfree"]])
    pc = pc[pc <= sc["D"] - 20]
    oc = offs(pc, 15, 70, [-3.42, 1.81])
    return sc["D"], oc, mkN(pc, (0.02997, 0.56293))


def pred(K, N, oc, lo=30.0, hi=140.0):
    return brentq(
        lambda a: 2 * N(a) - K - 1 / 6 - dder(float(rinv(a + oc[0]))) * rcurve(float(rinv(a + oc[0]))),
        lo,
        hi,
        xtol=1e-14,
    )


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fn = args[0] if args else "c50_capc_D110.json"
    Dm, oc, N = stage_fn(fn)
    located = {53: 37.4069796957, 55: 38.5030763179, 57: 39.5892471499, 59: 40.6659278448}
    ok = True
    for K in located:
        d = abs(pred(K, N, oc) - located[K])
        g = d < 5e-10
        ok = ok and g
        print(f"control K_b {K}: deviation from the located amplitude of amendment 962 {d:.1e} -> {g}")
    os.chdir("..")
    sys.path.insert(0, ".")
    import q1183_births_residual_halfline_depth_reading as Q

    dev = max(abs(Q.measured(K) - pred(K, N, oc)) for K in range(61, 80))
    g = dev < 2e-10
    ok = ok and g
    print(
        f"control: largest |A_meas - a_pred| over K_b 61 to 79 with this interpolation {dev:.3e} (registered below 2e-10) -> {g}"
    )
    eps = 1e-9
    g = abs(rcurve(rr[-1] - eps) - rcurve(rr[-1] + eps)) < 1e-6
    ok = ok and g
    print(f"control: the extension is continuous at the last node (rho {rr[-1]:.4f}) -> {g}")
    os.chdir("centre_scripts")
    if not ok:
        print("READING q1201: VOID (control)")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("READING q1201: controls only")
        sys.exit(0)
    amax = Dm - 42.0
    tab = {}
    flag = {}
    K = 81
    while True:
        try:
            a = pred(K, N, oc)
        except ValueError:
            break
        if a > amax:
            break
        tab[K] = a
        flag[K] = "extrapolated" if float(rinv(a + oc[0])) > rr[-1] else "interpolated"
        K += 1
    gaps = np.diff([tab[k] for k in sorted(tab)])
    print(
        f"stage D = {Dm}; table K_b 81 to {max(tab)}, a_t up to {tab[max(tab)]:.4f} (limit D - 42 = {amax}); first extrapolated K_b: {min([k for k in tab if flag[k] == 'extrapolated'], default=None)}; smallest gap between consecutive events {gaps.min():.6f}"
    )
    for kk in sorted(tab):
        if kk <= 90 or kk % 10 == 0:
            print(f"  K_b {kk}: a_pred {tab[kk]:.10f} {flag[kk]}")
    js = {
        "D": Dm,
        "kmax": max(tab),
        "a_pred": {str(kk): tab[kk] for kk in tab},
        "flag": {str(kk): flag[kk] for kk in tab},
    }
    sh = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    js["sha256"] = sh
    json.dump(js, open("q1201_table.json", "w"), indent=1)
    print("sha256", sh)
    print("END READING q1201 (predictions; no measurement beyond K_b 79 read)")


if __name__ == "__main__":
    main()
