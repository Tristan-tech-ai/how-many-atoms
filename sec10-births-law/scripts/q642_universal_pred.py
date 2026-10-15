"""q642: predictions for Amendment 950. Canonical rho (derived bulk law through rho^-10 at d = a + OFF1, OFF1 fitted on each functional's
own half-line). Immune ratio curve r(rho) = delta_imm/delta_der from precisely located immune transitions only (c19_fwd K38-43, c19_hl
K44-48/69/79/89/99, c19_d38 K52-65, c19_far_K59), linear in rho. Prediction for NPMLE K_b 50, 52 and capacity K_b 51, 52:
2 N_half(a) = K_b + 1/6 + delta_der(rho_c(a)) r(rho_c(a)). Null: r = 1 (derived series alone). Writes q642_predictions.json.
"""

import json, numpy as np
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
sn = [s for s in json.load(open("c33_npmc_D110.json"))["stages"] if s["conv"]][-1]
pn = np.array(sn["p"][: sn["nfree"]])
pn = pn[pn <= 90]
sc = [s for s in json.load(open("c50_cap_D100_norefit.json"))["stages"] if s["D"] == 92.0][0]
pc = np.array(sc["d"][: 1 + sc["nfree"]])
pc = pc[pc <= 72]
oi, on, oc = (
    offs(pi_, 15, 70, [-3.22, 1.31]),
    offs(pn, 15, 70, [-3.68, 1.40]),
    offs(pc, 15, 50, [-3.42, 1.81]),
)
Ni, Nnp, Nc = mkN(pi_, (0.22573, 0.06806)), mkN(pn, (-0.22676, 0.14770)), mkN(pc, (0.02997, 0.56293))
ev = [(json.load(open(f"c19_fwd_K{k}.json"))["a_t"], k) for k in range(38, 44)] + [
    (json.load(open(f"c19_hl_K{k}.json"))["a_t"], k) for k in (44, 45, 46, 47, 48, 69, 79, 89, 99)
]
ev += [(json.load(open(f"c19_d38_K{k}.json"))["a_t"], k) for k in (52, 55, 58, 62, 65)] + [
    (json.load(open("c19_far_K59.json"))["a_t"], 59)
]
pts = sorted((float(rinv(a + oi[0])), (2 * Ni(a) - K - 1 / 6) / dder(float(rinv(a + oi[0])))) for a, K in ev)
rr = np.array([p[0] for p in pts])
qq = np.array([p[1] for p in pts])
rcurve = lambda r: float(np.interp(r, rr, qq))
print("immune ratio curve:", " ".join(f"{r:.3f}:{q:.5f}" for r, q in pts))
out = {}
for name, N, o, Ks, (lo, hi) in (
    ("npmle", Nnp, on, (50, 52), (35, 39)),
    ("capacity", Nc, oc, (51, 52), (35, 38)),
):
    out[name] = {}
    for K in Ks:
        a_u = brentq(
            lambda a: 2 * N(a) - K - 1 / 6 - dder(float(rinv(a + o[0]))) * rcurve(float(rinv(a + o[0]))),
            lo,
            hi,
            xtol=1e-14,
        )
        a_n = brentq(lambda a: 2 * N(a) - K - 1 / 6 - dder(float(rinv(a + o[0]))), lo, hi, xtol=1e-14)
        rc = float(rinv(a_u + o[0]))
        out[name][K] = {"a_pred": a_u, "null_derived_only": a_n, "rho_c": rc}
        print(
            f"{name:8s} K_b {K}: a_pred = {a_u:.10f}  rho_c {rc:.4f}  ratio used {rcurve(rc):.5f}  null (derived series alone) {a_n - a_u:+.2e}"
        )
json.dump(out, open("q642_predictions.json", "w"), indent=1)
