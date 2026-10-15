"""q1223 (8 Oct 2026, registered in amendments 1981 and 2009; TEXT COPY of q1220 with e_2 .. e_20 from d22_out_20.json, inner series of 19 coefficients, Pade [9/9]; item R7 added to --read). Text of q1220 follows: (registered in amendments 1977 and 1978): Borel-Pade median summation of the centre series e_2 .. e_18 in the births law, read against the stored transitions.
Series: delta(x) = sum_{n=2}^{18} e_n x^n, x = 1/v^2, v = 4 pi dN_half/da. Borel weight Gamma(n + b), b = 1.5: B(y) = y^2 sum_{k=0}^{16} c_k y^k, c_k = e_(k+2) / Gamma(k + 2 + b), replaced by the diagonal Pade [8/8].
S_pm(x) = integral over t = s exp(+-i theta), s in (0, inf), of t^(b-1) exp(-t) B(x t) dt; S_med = Re (S_+ + S_-)/2; tabulated at 56 Chebyshev nodes on x in [0.0056, 0.0260] for theta = 0.3 and 0.5.
--registration-run : controls C1' (branch-point series, direct lateral average, 1e-9), C2' (Pade [2/2] must fail, 1e-8), C3 (table against direct, 1e-12), C4 (K_b 79: S_med against the sum to n = 18, 1e-11); writes q1223_table.json. No stored amplitude is read.
--read             : items R2a, R2b, R2c and the informational readings of amendment 1977 on the stored transitions of q1216.
Usage: py q1220_borel_pade_births_law.py --registration-run | --read   (from temporal_within/centre_scripts/)
"""

import hashlib, json, os, re, sys, time
import numpy as np
import mpmath as mp
from scipy.optimize import brentq

mode = sys.argv[1] if len(sys.argv) > 1 else "--registration-run"
sys.argv = ["x"]
import q1216_derived_series_lower_kb as Q6

G = Q6.G
mp.mp.dps = 40
B = mp.mpf("1.5")
ALPHA = mp.mpf("4.445")
XA, XB, NN = 0.0056, 0.0260, 56
E20 = {
    int(k[1:]): mp.mpf(v)
    for k, v in json.load(open("d22_out_20.json")).items()
    if re.fullmatch(r"e\d+", k) and k != "e1"
}


def horner(p, y):
    r = mp.mpc(0)
    for c in reversed(p):
        r = r * y + c
    return r


def machine(c, L, M):
    p, q = mp.pade(c, L, M)
    return (lambda y: y**2 * horner(p, y) / horner(q, y)), p, q


def ray(Bf, x, th):
    ph = mp.expj(th)
    return mp.quad(
        lambda s: (s * ph) ** (B - 1) * mp.exp(-s * ph) * Bf(x * s * ph) * ph, [0, 1, 4, 12, 40, 120, mp.inf]
    )


def smed(Bf, x, th):
    return mp.re((ray(Bf, x, th) + ray(Bf, x, -th)) / 2)


def coef_series():
    return [E20[k + 2] / mp.gamma(k + 2 + B) for k in range(19)]


def ctl_coefs():
    return [
        mp.gamma(m + mp.mpf("0.7")) * ALPHA**m / (mp.gamma(mp.mpf("0.7")) * mp.factorial(m))
        for m in range(2, 21)
    ]


def ctl_exact(y):
    return (1 - ALPHA * y) ** (-mp.mpf("0.7")) - 1 - mp.mpf("0.7") * ALPHA * y


def table(th):
    Bf, p, q = machine(coef_series(), 9, 9)
    nodes = [(XA + XB) / 2 + (XB - XA) / 2 * np.cos(np.pi * (j + 0.5) / NN) for j in range(NN)]
    ys = [float(smed(Bf, mp.mpf(x), th)) for x in nodes]
    return np.polynomial.chebyshev.Chebyshev.fit(nodes, ys, NN - 1, domain=[XA, XB]), Bf


def registration():
    t0 = time.time()
    ok = True
    cc = ctl_coefs()
    Bm, _, _ = machine(cc, 9, 9)
    B2, _, _ = machine(cc, 2, 2)
    th = 0.3
    for x in (0.0124, 0.0063):
        d = abs(smed(Bm, mp.mpf(x), th) - smed(ctl_exact, mp.mpf(x), th))
        g = d < 1e-9
        ok &= g
        print(f"control C1': x = {x}: machine against direct lateral average {float(d):.3e} (< 1e-9): {g}")
    d = abs(smed(B2, mp.mpf(0.0124), th) - smed(ctl_exact, mp.mpf(0.0124), th))
    g = d > 1e-8
    ok &= g
    print(f"control C2': Pade [2/2] differs by {float(d):.3e} (> 1e-8): {g}   [{time.time() - t0:.0f} s]")
    tabs = {}
    for th in (0.3, 0.5):
        tabs[th] = table(th)[0]
        print(f"table theta = {th} built [{time.time() - t0:.0f} s]")
    Bf = machine(coef_series(), 9, 9)[0]
    for x in (0.0071, 0.0093, 0.0121, 0.0150, 0.0190):
        d = abs(float(smed(Bf, mp.mpf(x), 0.3)) - tabs[0.3](x))
        g = d < 1e-12
        ok &= g
        if not g:
            print(f"control C3: x = {x} table minus direct {d:.3e} (< 1e-12): {g}")
    print("control C3: five points table against direct checked:", ok)
    v = mp.mpf("12.608284652941046")
    x = 1 / v**2
    d = abs(smed(Bf, x, 0.3) - sum(E20[n] * x**n for n in range(2, 21)))
    g = d < 1e-11
    ok &= g
    print(f"control C4: K_b 79: S_med against the sum to n = 18: {float(d):.3e} (< 1e-11): {g}")
    if not ok:
        print("REGISTRATION-RUN q1223: VOID (control)")
        sys.exit(2)
    js = {str(t): [float(c) for c in tabs[t].coef] for t in tabs}
    js["domain"] = [XA, XB]
    sh = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    js["sha256"] = sh
    json.dump(js, open("q1223_table.json", "w"))
    print("sha256", sh)
    print(f"REGISTRATION-RUN q1223: controls True [{time.time() - t0:.0f} s]")


def read():
    js = json.load(open("q1223_table.json"))
    sh = js.pop("sha256")
    h = hashlib.sha256(json.dumps(js, sort_keys=True).encode()).hexdigest()
    if h != sh:
        print("READING q1223: VOID (table hash)")
        sys.exit(2)
    T = {t: np.polynomial.chebyshev.Chebyshev(js[t], domain=js["domain"]) for t in ("0.3", "0.5")}
    N = {"capacity": G.Ncap, "npmle": G.Nnp}
    FILES = {"capacity": (Q6.CAP_FILES, 10, 59), "npmle": (Q6.NPM_FILES, 6, 52)}
    dev16 = {"capacity": {}, "npmle": {}}
    for line in open("../scratch/q1216_out.txt"):
        m = re.match(
            r"(capacity|npmle) K_b (\d+): A_stored [\d.]+; dev\(12\) [-+\d.]+e-9, dev\(16\) ([-+\d.]+)e-9",
            line,
        )
        if m:
            dev16[m.group(1)][int(m.group(2))] = float(m.group(3)) * 1e-9

    def root(Nf, K, tab):
        z = lambda a: 2 * Nf(a) - K - 1 / 6
        a0 = brentq(z, 1.0, 140.0, xtol=1e-12)
        f = lambda a: z(a) - float(tab(1.0 / (4 * np.pi * (Nf(a + G.H) - Nf(a - G.H)) / (2 * G.H)) ** 2))
        lo, hi = a0 - 2.5, a0 + 2.5
        return float("nan") if f(lo) * f(hi) > 0 else brentq(f, lo, hi, xtol=1e-14)

    R = {}
    for name in N:
        F, klo, khi = FILES[name]
        for K in sorted(F):
            if not klo <= K <= khi:
                continue
            j = json.load(open(F[K] + ".json"))
            m = j.get("a_t") or j.get("A_t")
            a3 = root(N[name], K, T["0.3"])
            a5 = root(N[name], K, T["0.5"])
            v = Q6.json.load(open("q1216_predictions.json"))[name][str(K)]["v"]
            R[(name, K)] = dict(dev=m - a3, shift=a5 - a3, d16=dev16[name].get(K), v=v)
    cap = {K: r for (n, K), r in R.items() if n == "capacity"}
    npm = {K: r for (n, K), r in R.items() if n == "npmle"}
    for name, d in (("capacity", cap), ("npmle", npm)):
        for K, r in sorted(d.items()):
            print(
                f"{name} K_b {K}: dev_med {r['dev'] * 1e9:+.4f}e-9, dev16 {r['d16'] * 1e9:+.4f}e-9, shift(theta 0.5 - 0.3) {r['shift'] * 1e9:+.5f}e-9"
            )
    a = max(abs(cap[K]["dev"]) for K in cap if 40 <= K <= 59) < 0.4e-9
    nb = sum(abs(cap[K]["dev"]) < abs(cap[K]["d16"]) for K in cap if 10 <= K <= 23)
    nev = sum(1 for K in cap if 10 <= K <= 23)
    b_ = nb >= 10
    c = max(abs(r["shift"]) for r in R.values()) < 1e-10
    print(
        f"item R2a (capacity K_b 40-59 max |dev_med| < 0.4e-9): max {max(abs(cap[K]['dev']) for K in cap if 40 <= K <= 59) * 1e9:.4f}e-9 -> {a}"
    )
    print(f"item R2b (capacity K_b 10-23: |dev_med| < |dev16| at >= 10 of {nev}): {nb} -> {b_}")
    print(
        f"item R2c (theta stability < 1e-10 at every stored event): max {max(abs(r['shift']) for r in R.values()) * 1e9:.5f}e-9 -> {c}"
    )
    ks = [K for K in range(12, 29) if K in npm and npm[K]["dev"] > 0]
    if len(ks) == 17:
        A = np.vstack([[npm[K]["v"] ** 2 for K in ks], np.ones(17)]).T
        cf = np.linalg.lstsq(A, np.log([npm[K]["dev"] * 1e9 for K in ks]), rcond=None)[0]
        print(f"information: NPMLE K_b 12-28 slope of ln(dev_med) against v^2 {cf[0]:+.4f}")
    else:
        print(
            "information: NPMLE K_b 12-28 deviations not all positive:",
            {K: round(npm[K]["dev"] * 1e9, 3) for K in range(12, 29) if K in npm},
        )
    print(
        "information: capacity dev_med sign counts (positive/negative): K_b 10-23",
        sum(cap[K]["dev"] > 0 for K in cap if 10 <= K <= 23),
        sum(cap[K]["dev"] < 0 for K in cap if 10 <= K <= 23),
        "; K_b 24-39",
        sum(cap[K]["dev"] > 0 for K in cap if 24 <= K <= 39),
        sum(cap[K]["dev"] < 0 for K in cap if 24 <= K <= 39),
    )
    Bf, p, q = machine(coef_series(), 9, 9)
    rt = mp.polyroots(list(reversed(q)), maxsteps=200, extraprec=200)
    near = sorted(rt, key=lambda z: abs(mp.im(z)))[:4]
    print(
        "information: Pade poles of B nearest to the real axis (y; 1/y):",
        "; ".join(f"{mp.nstr(z, 6)} ({mp.nstr(1 / z, 6)})" for z in near),
    )
    q20 = {}
    for line in open("../scratch/q1220_out.txt"):
        m = re.match(r"npmle K_b (\d+): dev_med ([-+\d.]+)e-9", line)
        if m:
            q20[int(m.group(1))] = float(m.group(2))
    rel = {K: abs(npm[K]["dev"] * 1e9 - q20[K]) / q20[K] for K in npm if 14 <= K <= 40 and K in q20}
    r7 = len(rel) >= 20 and max(rel.values()) < 0.05
    print(
        f"item R7 (NPMLE dev_med K_b 14-40, {len(rel)} events, change against q1220 below 5 percent): max {max(rel.values()) * 100:.3f} percent at K_b {max(rel, key=rel.get)} -> {r7}"
    )
    print(
        "information: relative change at K_b 14, 20, 28, 34, 40: "
        + ", ".join(f"{K}: {rel[K] * 100:.3f}" for K in (14, 20, 28, 34, 40) if K in rel)
    )
    capold = {}
    for line in open("../scratch/q1220_out.txt"):
        m = re.match(r"capacity K_b (\d+): dev_med ([-+\d.]+)e-9", line)
        if m:
            capold[int(m.group(1))] = float(m.group(2))
    print(
        "information: capacity dev_med new / old at K_b 24, 30, 36, 40: "
        + ", ".join(
            f"{K}: {cap[K]['dev'] * 1e9:+.4f} / {capold[K]:+.4f}" for K in (24, 30, 36, 40) if K in cap
        )
    )
    print(
        "READING q1223: R2a "
        + str(a)
        + ", R2b "
        + str(b_)
        + ", R2c "
        + str(c)
        + ", R7 "
        + str(r7)
        + " (no verdict)"
    )


if __name__ == "__main__":
    registration() if mode == "--registration-run" else read()
