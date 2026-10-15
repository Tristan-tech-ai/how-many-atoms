"""births_law (10 Oct 2026, INSTRUMENT / MODULE, no verdict): the births law in one self-contained file, without the exec-of-q1616 pattern.
For a capacity birth or split K_b the amplitude A solves 2 N(A) - K - 1/6 - Delta(v) = 0 with N(A) = N_law(rho(A)) + 1/2 - nu0, d_law(rho) + d0 = A, rho' = dN/dA by a central difference of step Hs, v = 4 pi rho', and Delta(v) = sum_n E_n / v^(4 + 2n), the centre series with E_n = e_(n+2) (derived, centre_scripts/q1634_out_8.json) plus optional extra terms. The bulk law is the law through c20 (q1583). The wall pair is the pair of q1610.
Usage: py births_law.py --selftest | --new K [K ...]  (from small_gap_q206/; --new prints only the difference predicted minus measured of the events that exist in fw_prod, never an amplitude).
"""

import json, os, sys
import numpy as np
from scipy.optimize import brentq

H = os.path.dirname(os.path.abspath(__file__))
PI = np.pi
k1 = 1 / (8 * PI**2)
k4 = 1 / (64 * PI**4) - 1 / (96 * PI**2)
c6 = 1 / (512 * PI**6) - 13 / (768 * PI**4)
c8 = 1 / (4096 * PI**8) - 17 / (1536 * PI**6) - 1 / (3072 * PI**4)
c10 = 1 / (32768 * PI**10) - 29 / (6144 * PI**8) - 109 / (368640 * PI**6)
f12 = -37 / (442368 * PI**6) - 617 / (393216 * PI**10) + 1 / (262144 * PI**12) + 229 / (327680 * PI**8)
f14 = (
    -33223 / (123863040 * PI**8)
    - 1393 / (3145728 * PI**12)
    + 1 / (2097152 * PI**14)
    + 3819 / (2621440 * PI**10)
)
f16 = (
    -16507 / (424673280 * PI**8)
    - 43261 / (123863040 * PI**10)
    - 175 / (1572864 * PI**14)
    + 1 / (16777216 * PI**16)
    + 42631 / (31457280 * PI**12)
)
f18 = (
    -54007 / (293601280 * PI**10)
    - 53549 / (165150720 * PI**12)
    - 215 / (8388608 * PI**16)
    + 1 / (134217728 * PI**18)
    + 71767 / (83886080 * PI**14)
)
f20 = (
    -64764727 / (190253629440 * PI**12)
    - 395711 / (13589544960 * PI**10)
    - 11659519 / (31708938240 * PI**14)
    - 2963 / (536870912 * PI**18)
    + 1 / (1073741824 * PI**20)
    + 848207 / (2013265920 * PI**16)
)
F = [1.0, k1, k4, c6, c8, c10, f12, f14, f16, f18, f20]
NG = 10
g = [1.0]
for n in range(1, NG + 1):
    g.append(-sum(F[j] * g[n - j] for j in range(1, n + 1)))


def dlaw(r):
    return (
        16 * PI**2 * (g[0] * r**3 / 3 + sum(g[n] * r ** (3 - 2 * n) / (3 - 2 * n) for n in range(1, NG + 1)))
    )


def Nlaw(r):
    return (
        16
        * PI**2
        * (
            g[0] * r**4 / 4
            + g[1] * r**2 / 2
            + g[2] * np.log(r)
            + sum(g[n] * r ** (4 - 2 * n) / (4 - 2 * n) for n in range(3, NG + 1))
        )
    )


_e = json.load(open(os.path.join(H, "..", "centre_scripts", "q1634_out_8.json")))
E_DER = {n: float(_e["e%d_num" % n]) for n in range(2, 9)}
WALL = (-0.093773648256, 0.001159913641)
LIMP = (-0.093773592, 0.001159849)


def a_pred(K, nu, d0, extra=(), Hs=1e-3, base=4):
    """predicted amplitude; base = number of derived terms e_2..e_(base+1) in Delta (default 4: E_0..E_3), extra = further coefficients E_4, E_5, ..."""
    Es = [E_DER[n + 2] for n in range(base)] + list(extra)
    rho = lambda a: brentq(lambda r: dlaw(r) + d0 - a, 0.05, 6.0, xtol=1e-15)
    Nn = lambda a: Nlaw(rho(a)) + 0.5 - nu

    def dd(r):
        v = 4 * PI * r
        return sum(E / v ** (4 + 2 * i) for i, E in enumerate(Es))

    return brentq(
        lambda a: 2 * Nn(a) - K - 1 / 6 - dd((Nn(a + Hs) - Nn(a - Hs)) / (2 * Hs)), 30, 85, xtol=1e-14
    )


def measured(K):
    f = os.path.join(
        os.environ.get("BL_FWPROD", os.path.join(H, "fw_prod")), "event_Kb%d.json" % K
    )  # env var added 10 Oct 09:15 for the selftest; the default is unchanged
    return float(json.load(open(f))["A_event"]) if os.path.exists(f) else None


def diffs(Ks, nu, d0, extra=(), base=4):
    out = {}
    for K in Ks:
        m = measured(K)
        if m is not None:
            out[K] = a_pred(K, nu, d0, extra, base=base) - m
    return out


def selftest():
    Ks = [K for K in range(83, 131) if measured(K) is not None]
    rms = lambda d: float(np.sqrt(np.mean(np.array(list(d.values())) ** 2)))
    d3 = diffs(Ks, *WALL, extra=[E_DER[6], E_DER[7], E_DER[8]])
    r3 = rms(d3)
    d2 = diffs([135], *WALL, extra=[-4.251e05, -6.538e07])
    bad = diffs(Ks, WALL[0] + 1e-6, WALL[1], extra=[E_DER[6], E_DER[7], E_DER[8]])
    rbad = rms(bad)
    print(
        "%d births; rms with E_4, E_5, E_6 = e_6, e_7, e_8: %.4e (q1630: 2.841e-10); K_b 135 frozen two-term difference %s (q1626: -6.76e-11); wrong pair (nu0 + 1e-6) rms %.3e"
        % (len(Ks), r3, ("%+.2e" % d2[135]) if 135 in d2 else "no event", rbad)
    )
    ok = (
        len(Ks) == 48
        and abs(r3 / 2.841e-10 - 1) < 2e-3
        and (135 not in d2 or abs(d2[135] / -6.76e-11 - 1) < 5e-3)
        and rbad > 10 * r3
    )
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a:
        sys.exit(0 if selftest() else 1)
    if "--new" in a:
        Ks = [int(x) for x in a[a.index("--new") + 1 :]]
        d = diffs(Ks, *WALL, extra=[E_DER[6], E_DER[7], E_DER[8]])
        for K in Ks:
            print(
                "K_b %d: %s"
                % (
                    K,
                    (
                        ("difference predicted minus measured %+.3e (derived E_4..E_6)" % d[K])
                        if K in d
                        else "no event"
                    ),
                )
            )
