"""q1616 (10 Oct 2026, READING, no verdict word; deep proof Q11 step 70): the births law with the pair of the exact wall block and the two centre terms E_4 and E_5.
q1614: with the c20 block pair fixed, E_4 alone leaves a residual rms of 3.02e-10 (E_4 = -8.006e+05). q1615: the offset without E_4 is one-sided and follows a pure power law c K^-p with p = 3.486 (c20 pair) and 3.539 (c24 pair), residual rms 2.4e-11 and 2.9e-11. The term E_4 / v^12 has the shape rho^-12 at the centre, and in the births equation it moves the amplitude A by about rho^-13, that is K^-3.25 (rho^4 is proportional to K). The next term E_5 / v^14 moves A by K^-3.75. The measured exponent 3.5 lies between them.
Test: fix the block pair and fit E_4 and E_5 together (linear least squares with explicit steps), then E_4, E_5 and E_6, and compare with the single-term fit.
Predictions written before the run:
 P1 with E_4 and E_5 free and the c20 block pair fixed, the residual rms over the 48 births is below 5e-11;
 P2 the residual rms of the two-term fit is below 0.3 of the rms of the one-term fit (3.02e-10);
 P3 the fitted E_4 and E_5 for the c24 block pair lie within 30 percent of those for the c20 block pair;
 P4 adding E_6 lowers the residual rms by less than a factor 2 compared with the two-term fit (the data do not need a third term).
Usage: py q1616_E4_E5.py (about 3 minutes; executes q1567 silently for its functions)."""

import io, contextlib, os
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(H, "q1567_missing_E4.py"), "__name__": "q1567ns"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(H, "q1567_missing_E4.py"), encoding="utf-8").read(), ns)
dfn, Nfn, E, brentq, PI, Ks, meas = (
    ns["dfn"],
    ns["Nfn"],
    ns["E"],
    ns["brentq"],
    ns["PI"],
    ns["Ks"],
    ns["meas"],
)


def a_pred(K, nu, d0, extra, Hs=1e-3):
    rho = lambda a: brentq(lambda r: dfn(r) + d0 - a, 0.05, 6.0, xtol=1e-15)
    Nn = lambda a: Nfn(rho(a)) + 0.5 - nu

    def dd(r):
        v = 4 * PI * r
        return sum(E[i] / v ** (4 + 2 * i) for i in range(4)) + sum(
            x / v ** (12 + 2 * j) for j, x in enumerate(extra)
        )

    return brentq(
        lambda a: 2 * Nn(a) - K - 1 / 6 - dd((Nn(a + Hs) - Nn(a - Hs)) / (2 * Hs)), 30, 85, xtol=1e-14
    )


offs = lambda pair, extra: np.array([a_pred(K, pair[0], pair[1], extra) - meas[K] for K in Ks])
rm = lambda x: float(np.sqrt((x**2).mean()))
scale = [1e5, 1e6, 1e7]


def fit(pair, n):
    base = offs(pair, [0.0] * n)
    cols = []
    for j in range(n):
        up = [0.0] * n
        dn = [0.0] * n
        up[j] = scale[j]
        dn[j] = -scale[j]
        cols.append((offs(pair, up) - offs(pair, dn)) / (2 * scale[j]))
    J = np.array(cols).T
    x, *_ = np.linalg.lstsq(J, -base, rcond=None)
    r = offs(pair, list(x))
    return x, r


out = {}
for name, pair in (
    ("c20 block pair", (-0.093773648256, 0.001159913641)),
    ("c24 block pair", (-0.093773650, 0.001159915)),
):
    for n in (1, 2, 3):
        x, r = fit(pair, n)
        out[(name, n)] = (x, rm(r))
        print(
            "%s, %d term(s): %s; residual rms %.2e, largest %.2e"
            % (name, n, ", ".join("E_%d = %.4g" % (4 + j, x[j]) for j in range(n)), rm(r), np.abs(r).max())
        )
x2, r2 = out[("c20 block pair", 2)]
x1, r1 = out[("c20 block pair", 1)]
x3, r3 = out[("c20 block pair", 3)]
y2 = out[("c24 block pair", 2)][0]
lab = lambda p: "holds" if p else "does not hold"
p1 = r2 < 5e-11
p2 = r2 < 0.3 * r1
p3 = all(abs(y2[j] / x2[j] - 1) < 0.3 for j in range(2))
p4 = r3 > r2 / 2
print(
    "P1 two-term rms below 5e-11: %s; P2 two-term rms below 0.3 of the one-term rms: %s; P3 c24 coefficients within 30 percent of c20: %s; P4 third term lowers the rms by less than a factor 2: %s"
    % (lab(p1), lab(p2), lab(p3), lab(p4))
)
print("READING q1616 done (labels by the script; no verdict word)")
