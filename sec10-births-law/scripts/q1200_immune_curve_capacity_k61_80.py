"""q1200 (8 Oct 2026, registered in amendment 1956): the method of q645 (amendment 961, PASS 4 of 4 at K_b 53 to 59) applied to the capacity events K_b = 61 to 80: the births law 2 N_half(a) = K_b + 1/6 + delta_der(rho_c(a)) r(rho_c(a)) with
rho_c the density of the DERIVED bulk law (through rho^-10) at d = a + OFF1 (OFF1 and the counting offset fitted on the atoms with 15 <= d <= 70 of the c50c stage D = 110 of the capacity half-line), delta_der the series through e5 and r(rho) the IMMUNE ratio curve delta_imm / delta_der from the precisely located immune transitions only (linear in rho; same events as q645).
Nothing in this script reads a measured capacity amplitude except through the comparison lines (K_b 61 to 79 and the controls). The prediction for K_b 80 is printed without any comparison unless --compare is given.
Registered items (amendment 1956): I1 max over K_b 61 to 79 of |A_meas - a_pred| < 2e-9 (prior 0.60); I2 (blind) |A_meas(80) - a_pred(80)| < 2e-9 (prior 0.55).
Controls (registration run): the stage and curve reproduce the stored predictions of q645 for K_b 53, 55, 57, 59 (q645_predictions.json) within 1e-10 and the located amplitudes of amendment 962 (37.4069796957, 38.5030763179, 39.5892471499, 40.6659278448) within 5e-10.
Usage: py q1200_immune_curve_capacity_k61_80.py [--registration-run] [--compare]  (from temporal_within/centre_scripts/)
"""

import hashlib, json, sys
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
sc = [s for s in json.load(open("c50_capc_D110.json"))["stages"] if s["conv"]][-1]
pc = np.array(sc["d"][: 1 + sc["nfree"]])
pc = pc[pc <= sc["D"] - 20]
oi, oc = offs(pi_, 15, 70, [-3.22, 1.31]), offs(pc, 15, 70, [-3.42, 1.81])
Ni, Nc = mkN(pi_, (0.22573, 0.06806)), mkN(pc, (0.02997, 0.56293))
ev = [(json.load(open(f"c19_fwd_K{kk}.json"))["a_t"], kk) for kk in range(38, 44)] + [
    (json.load(open(f"c19_hl_K{kk}.json"))["a_t"], kk) for kk in (44, 45, 46, 47, 48, 69, 79, 89, 99)
]
ev += [(json.load(open(f"c19_d38_K{kk}.json"))["a_t"], kk) for kk in (52, 55, 58, 62, 65)] + [
    (json.load(open("c19_far_K59.json"))["a_t"], 59)
]
pts = sorted((float(rinv(a + oi[0])), (2 * Ni(a) - K - 1 / 6) / dder(float(rinv(a + oi[0])))) for a, K in ev)
rr = np.array([p[0] for p in pts])
qq = np.array([p[1] for p in pts])
rcurve = lambda r: float(np.interp(r, rr, qq))


def pred(K, N=Nc, o=oc):
    return brentq(
        lambda a: 2 * N(a) - K - 1 / 6 - dder(float(rinv(a + o[0]))) * rcurve(float(rinv(a + o[0]))),
        30.0,
        60.0,
        xtol=1e-14,
    )


def main():
    q645 = json.load(open("q645_predictions.json"))["capacity"]
    located = {53: 37.4069796957, 55: 38.5030763179, 57: 39.5892471499, 59: 40.6659278448}
    ok = True
    for K in (53, 55, 57, 59):
        a = pred(K)
        d1 = abs(a - q645[str(K)]["a_pred"])
        d2 = abs(a - located[K])
        good = d1 < 1e-10 and d2 < 5e-10
        ok = ok and good
        print(
            f"control K_b {K}: a_pred {a:.10f}; against the stored q645 value {d1:.1e}; against the located amplitude of amendment 962 {d2:.1e} -> {good}"
        )
    if not ok:
        print("READING q1200: VOID (control)")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("READING q1200: controls only")
        sys.exit(0)
    Ks = list(range(61, 81))
    P = {K: pred(K) for K in Ks}
    print(
        "immune curve range of rho:",
        f"{rr[0]:.3f} to {rr[-1]:.3f}; capacity rho_c at K_b 61, 79, 80:",
        [round(float(rinv(P[K] + oc[0])), 4) for K in (61, 79, 80)],
    )
    js = {str(K): P[K] for K in Ks}
    sh = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    print("predictions a_pred(K_b):", {K: round(P[K], 10) for K in Ks})
    print("sha256 of the 20 predictions", sh)
    if "--compare" not in sys.argv:
        print("READING q1200: predictions only (no measurement read)")
        sys.exit(0)
    import os

    os.chdir("..")
    sys.path.insert(0, ".")
    import q1183_births_residual_halfline_depth_reading as Q

    res = {}
    for K in Ks:
        try:
            res[K] = Q.measured(K) - P[K]
        except Exception as e:
            print(f"K_b {K}: no measured amplitude ({type(e).__name__})")
    print("A_meas - a_pred (1e-9):", {K: round(v * 1e9, 3) for K, v in res.items()})
    i1 = all(abs(res[K]) < 2e-9 for K in range(61, 80))
    print(
        f"I1 (max |A_meas - a_pred| < 2e-9 over K_b 61 to 79; largest {max(abs(res[K]) for K in range(61, 80)) * 1e9:.3f}e-9): {i1}"
    )
    if 80 in res:
        print(f"I2 (|A_meas(80) - a_pred(80)| < 2e-9; value {res[80] * 1e9:.3f}e-9): {abs(res[80]) < 2e-9}")
    print(f"READING q1200: I1 {i1}" + (f", I2 {abs(res[80]) < 2e-9}" if 80 in res else "") + " (no verdict)")


if __name__ == "__main__":
    main()
