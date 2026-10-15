"""q1777 (10 Oct 2026, FACT TABLE, no predictions, no verdict word; task Q175 design step): the share s_j = b(j) u^ceil(j/2) / F_j of the bottom power in the exact bulk coefficient F_j (u = 1/pi^2), for j = 2..15, at 40 digits. Printed to design the registered rule for the numeric value of f32 = F_16. The values of f32 are not available.
Usage: py q1777_bottom_share_explore.py (from small_gap_q206/; seconds)."""

import os, re
import sympy as sp
import mpmath as mp

mp.mp.dps = 40
H = os.path.dirname(os.path.abspath(__file__))
x = sp.Symbol("x")


def coeffs(expr):
    return {m[0]: sp.Rational(c) for m, c in sp.Poly(sp.expand(expr.subs(sp.pi, 1 / x)), x).terms()}


src = open(os.path.join(H, "births_law.py"), encoding="utf-8").read().splitlines()
ns = {"PI": sp.pi}
for ln in src:
    if re.match(r"^(k1 =|c6 =|c10 =|f1[2-9] =|f20 =)", ln):
        exec(ln, ns)
F = {
    j: f
    for j, f in zip(
        range(2, 11),
        [ns["k4"], ns["c6"], ns["c8"], ns["c10"], ns["f12"], ns["f14"], ns["f16"], ns["f18"], ns["f20"]],
    )
}
txt = open(os.path.join(H, "q1557_stdout_ORD30.txt"), encoding="utf-8").read()
for j, n in zip(range(11, 16), (22, 24, 26, 28, 30)):
    m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, txt, re.M)
    F[j] = sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi})
u = 1 / mp.pi**2
print("j, F_j, bottom term b(j) u^ceil(j/2), share of the bottom, share of the two lowest powers")
for j in range(2, 16):
    cf = coeffs(F[j])
    pw = sorted(cf)
    Fj = mp.mpf(sp.N(F[j], 50))
    b0 = mp.mpf(sp.N(cf[pw[0]], 50)) * u ** (pw[0] // 2)
    b1 = mp.mpf(sp.N(cf[pw[1]], 50)) * u ** (pw[1] // 2) if len(pw) > 1 else 0
    print(
        "%2d: F = %s, bottom = %s, share %s, two lowest %s"
        % (j, mp.nstr(Fj, 6), mp.nstr(b0, 6), mp.nstr(b0 / Fj, 6), mp.nstr((b0 + b1) / Fj, 6))
    )
print("TABLE q1777 done (facts; no labels)")
