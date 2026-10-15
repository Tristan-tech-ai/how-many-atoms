"""q1771 (10 Oct 2026, FACT TABLE after q1770, no predictions, no verdict word): the log levels 1 and 2 in the variable y = w - 1 = x / (1 - x), x = u t / 8. The factor (w-1)^(2k) of the level k, seen in q1770 (multiplicity 2 for k = 1 and 4 for k = 2), is the support condition of the series (the power u^(n-k) of L_n exists only for n >= 2k, so the level starts at x^(2k)). The script prints L_1 and L_2 as polynomials in y with exact rational coefficients and checks the lowest power in y.
Control: the substitution reproduces the w-polynomials of q1768 and q1770 exactly.
Usage: py q1771_levels_in_y.py (from small_gap_q206/; seconds)."""

import sys
import sympy as sp

w = sp.Symbol("w")
y = sp.Symbol("y")
L1w = -8 * w**5 + sp.Rational(76, 3) * w**4 - 28 * w**3 + sp.Rational(34, 3) * w**2 - sp.Rational(2, 3)
L2w = (
    sp.Rational(3680, 3) * w**10
    - 8640 * w**9
    + sp.Rational(236456, 9) * w**8
    - sp.Rational(403472, 9) * w**7
    + 46520 * w**6
    - sp.Rational(1325456, 45) * w**5
    + sp.Rational(94202, 9) * w**4
    - sp.Rational(12472, 9) * w**3
    - sp.Rational(2380, 9) * w**2
    + sp.Rational(280, 3) * w
    - sp.Rational(214, 45)
)
ctl = (
    sp.expand(L1w - sp.Rational(2, 3) * (w - 1) ** 2 * (2 * w - 1) * (1 + 4 * w - 6 * w**2)) == 0
    and sp.expand(
        L2w
        - sp.Rational(2, 45)
        * (w - 1) ** 4
        * (27600 * w**6 - 84000 * w**5 + 89540 * w**4 - 36120 * w**3 + 1380 * w**2 + 1672 * w - 107)
    )
    == 0
)
print("CONTROL the w-polynomials equal the factored forms of q1768 and q1770: %s" % ctl)
if not ctl:
    print("VOID: control failed")
    sys.exit(2)
for name, Lw in (("L_1", L1w), ("L_2", L2w)):
    Ly = sp.expand(Lw.subs(w, 1 + y))
    P = sp.Poly(Ly, y)
    print(
        "%s(y) = %s ; lowest power of y: %d ; coefficients from the lowest power: %s"
        % (
            name,
            sp.collect(Ly, y),
            min(m[0] for m in P.monoms()),
            ", ".join(
                str(P.coeff_monomial(y**k)) for k in range(min(m[0] for m in P.monoms()), P.degree() + 1)
            ),
        )
    )
print("TABLE q1771 done (facts; no labels)")
