"""q1557 (10 Oct 2026, DERIVATION RUN, no verdict word; deep proof Q11 step 26, PROOF_TRACKS A step 32): the next coefficient f12 = c12 of the bulk law rho^2 rho'/s0 = 1 + f2/rho^2 + ... + f12/rho^12, derived symbolically (sympy, exact in pi) by the machine of centre_scripts/d20_c6_series.py (BULK_C6_DERIVATION.md section 2.7), which is copied here as text with two edits: the symbol list gets f12, and the output goes to q1557_out.txt. The frozen file d20_c6_series.py and its output are not touched. Run as: py q1557_f12_symbolic.py 12 varsonly (self-consistent profile only).
Predictions written before the run:
 P1 control: the run at ORD = 12 returns f2, f4, f6, f8 and f10 equal to the closed forms of section 0 of the note (exact sympy equality, f10 = -29/(6144 pi^8) - 109/(368640 pi^6) + 1/(32768 pi^10));
 P2 the numerical value of f12 agrees with the numerical estimate of d20_f12.py (Richardson (6,8): -3.0096808e-08) within 1 percent;
 P3 f12 contains exactly three powers of pi, namely pi^-12, pi^-10 and pi^-8, with rational coefficients, and the coefficient of pi^-12 is 1/(8^6) = 1/262144 (the highest power is k^6 with k = 1/(8 pi^2)).
Unrelated to the above, the original docstring follows.
"""

"""d20_c6_series: the bulk law rho^2 rho'/s0 = f(rho) = 1 + f2/rho^2 + f4/rho^4 + f6/rho^6 + f8/rho^8, carried from the record's
k4 derivation (Amendments 481, 485, 489-492) to rho^-6 and rho^-8, SYMBOLICALLY (sympy, exact in pi).

The record's k4 derivation, reconstructed:
  (i)   positional law  arg int psi(u) exp(2 pi i [N(z+u) - N(z)]) du = pi,  psi = N(0,2) density  (429, LFP 1.5, NPMLE lemma 1.3);
  (ii)  phase 2 pi [N(z+u) - N(z)] = alpha u + beta u^2 + sum_{n>=3} theta_n u^n, alpha = 2 pi rho, beta = pi rho';
  (iii) exact Gaussian factor  I2 = a^(1/2) exp(-alpha^2 a),  a = 1/(1 - 4 i beta), arg I2 = -PHI            (481);
  (iv)  G = E[exp(i R(u))] over the complex Gaussian u ~ N(2 i alpha a, 2a), i.e. sum_m c_m i^m a^(m/2) H_m(alpha sqrt a)  (485, 492);
  (v)   law:  PHI - arg G = pi                                                                                    (489);
  (vi)  the record evaluates theta_n on the CONSTANT-s family rho^3 = rho_c^3 + 3 s u                              (481, 485).
Here (vi) is replaced by the exact local Taylor series of the self-consistent profile rho' = s0 f(rho)/rho^2 (flag VARS=True),
so that the s-variation enters; VARS=False reproduces the constant-s family (f frozen at its local value) for the decomposition.

Scaling (x = 1/rho): theta_n ~ x^(3n-4), mean mu = 2 i alpha a ~ x^-1, so theta_n mu^(n-j) w^j ~ x^(2n-4+j). To reach x^ORD one needs
n <= ORD/2 + 2 and R^p with p <= ORD/2.

Usage: py d20_c6_series.py [ORD] [varsonly]   (ORD = 8 gives f2..f8; varsonly skips the constant-s decomposition). Output: d20_c6_series_out.txt (appended) and stdout.
"""
import sys, time
import sympy as sp

ORD = int(sys.argv[1]) if len(sys.argv) > 1 else 8
pi, I = sp.pi, sp.I
F = sp.symbols(
    "f2 f4 f6 f8 f10 f12 f14 f16 f18 f20 f22 f24 f26 f28 f30 f32", real=True
)  # unknown coefficients of f

s0 = 1 / (16 * pi**2)
NMAX = ORD // 2 + 2  # highest Taylor order theta_n needed
CAP = ORD + NMAX + 2  # working cap for series that later meet mu^-n


def S(d, cap):  # clean + truncate a dict series {power: coeff}
    out = {}
    for p, c in d.items():
        if p <= cap:
            c = sp.expand(c)
            if c != 0:
                out[p] = c
    return out


def add(*ds):
    out = {}
    for d in ds:
        for p, c in d.items():
            out[p] = out.get(p, 0) + c
    return out


def mul(d1, d2, cap):
    out = {}
    for p1, c1 in d1.items():
        for p2, c2 in d2.items():
            if p1 + p2 <= cap:
                out[p1 + p2] = out.get(p1 + p2, 0) + c1 * c2
    return S(out, cap)


def scal(d, k):
    return {p: k * c for p, c in d.items()}


def deriv(d):
    return {p - 1: p * c for p, c in d.items() if p != 0}


def powr(d, n, cap):
    out = {0: sp.Integer(1)}
    for _ in range(n):
        out = mul(out, d, cap)
    return out


def run(VARS):
    t0 = time.time()
    fser = {0: sp.Integer(1)}
    for i in range(1, ORD // 2 + 1):
        fser[2 * i] = F[i - 1]
    # rho' = s0 f x^2 ; D = d/dz = -s0 f x^4 d/dx (x = 1/rho). For VARS=False f is a constant C: rho' = s0 C x^2, D = -s0 C x^4 d/dx,
    # and C is set to f(x) at the evaluation point only after the derivatives are taken (the constant-s family).
    if VARS:
        fD = fser
    else:
        Csym = sp.Symbol("C", positive=True)
        fD = {0: Csym}
    rho_d = [None, S(mul(scal(fD, s0), {2: sp.Integer(1)}, CAP), CAP)]  # rho_d[k] = k-th z-derivative of rho
    Dop = S(mul(scal(fD, -s0), {4: sp.Integer(1)}, CAP), CAP)
    for k in range(2, NMAX):
        rho_d.append(mul(Dop, deriv(rho_d[k - 1]), CAP))
    theta = {n: scal(rho_d[n - 1], 2 * pi / sp.factorial(n)) for n in range(3, NMAX + 1)}
    beta = scal(rho_d[1], pi)
    if not VARS:  # substitute the constant C = f(x) (value at the evaluation point) afterwards
        sub = lambda d: S(mul_sub(d, Csym, fser), CAP)
        theta = {n: sub(theta[n]) for n in theta}
        beta = sub(beta)
    # a = 1/(1 - 4 i beta) = sum (4 i beta)^k
    fourib = scal(beta, 4 * I)
    a = {0: sp.Integer(1)}
    term = {0: sp.Integer(1)}
    for k in range(1, CAP // 2 + 1):
        term = mul(term, fourib, CAP)
        a = add(a, term)
    a = S(a, CAP)
    alpha_inv = {-1: 2 * pi}  # alpha = 2 pi / x
    mu = S(mul({-1: 4 * pi * I}, a, CAP - 1), CAP)  # mu = 2 i alpha a
    # R(mu + w) = sum_n theta_n (mu + w)^n  -> polynomial in w: Rw[j] = sum_n theta_n binom(n, j) mu^(n-j)
    mupow = {0: {0: sp.Integer(1)}}
    for m in range(1, NMAX + 1):
        mupow[m] = mul(mupow[m - 1], mu, CAP)
    Rw = {}
    for n in range(3, NMAX + 1):
        for j in range(0, n + 1):
            t = scal(mul(theta[n], mupow[n - j], ORD), sp.binomial(n, j))
            if t:
                Rw[j] = S(add(Rw.get(j, {}), t), ORD)
    Rw = {j: d for j, d in Rw.items() if d}
    # exp(i R) as polynomial in w with series coefficients: sum_p (iR)^p/p!, p <= ORD/2 (R starts at x^2)
    iR = {j: scal(d, I) for j, d in Rw.items()}

    def pmul(P1, P2):
        out = {}
        for j1, d1 in P1.items():
            for j2, d2 in P2.items():
                t = mul(d1, d2, ORD)
                if t:
                    out[j1 + j2] = S(add(out.get(j1 + j2, {}), t), ORD)
        return {j: d for j, d in out.items() if d}

    E = {0: {0: sp.Integer(1)}}
    Pp = {0: {0: sp.Integer(1)}}
    for p in range(1, ORD // 2 + 1):
        Pp = pmul(Pp, iR)
        for j, d in Pp.items():
            E[j] = S(add(E.get(j, {}), scal(d, sp.Rational(1, sp.factorial(p)))), ORD)
    # Gaussian moments E[w^(2m)] = (2a)^m (2m-1)!!, odd moments 0
    G = {}
    twoa = scal(a, 2)
    for j, d in E.items():
        if j % 2:
            continue
        m = j // 2
        mom = scal(powr(twoa, m, ORD), sp.factorial2(2 * m - 1) if m > 0 else 1)
        G = S(add(G, mul(d, mom, ORD)), ORD)
    # log G = sum (-1)^(k+1) (G-1)^k / k
    Gm1 = {p: c for p, c in G.items() if p != 0}
    assert sp.simplify(G.get(0, 0) - 1) == 0
    logG = {}
    term = {0: sp.Integer(1)}
    for k in range(1, ORD // 2 + 1):
        term = mul(term, Gm1, ORD)
        logG = S(add(logG, scal(term, sp.Rational((-1) ** (k + 1), k))), ORD)
    ImlogG = {p: sp.expand(sp.im(c)) for p, c in logG.items()}
    RelogG = {p: sp.expand(sp.re(c)) for p, c in logG.items()}
    # PHI = 4 alpha^2 beta/(1 + 16 beta^2) - (1/2) arctan(4 beta);  4 alpha^2 beta = (16 pi^2/x^2) beta
    b2 = mul(beta, beta, ORD)
    den = {0: sp.Integer(1)}
    term = {0: sp.Integer(1)}
    for k in range(1, ORD // 4 + 2):
        term = mul(term, scal(b2, -16), ORD)
        den = add(den, term)
    fourab = S(mul({-2: 16 * pi**2}, beta, ORD), ORD)
    PHI1 = mul(fourab, S(den, ORD), ORD)
    fb = scal(beta, 4)
    at = {}
    term = {0: sp.Integer(1)}
    for k in range(0, ORD // 2 + 1):  # arctan y = sum (-1)^k y^(2k+1)/(2k+1)
        tk = powr(fb, 2 * k + 1, ORD)
        at = add(at, scal(tk, sp.Rational((-1) ** k, 2 * k + 1)))
    PHI = S(add(PHI1, scal(S(at, ORD), -sp.Rational(1, 2))), ORD)
    eq = S(add(PHI, scal(ImlogG, -1), {0: -pi}), ORD)
    print(f"  [VARS={VARS}] series built in {time.time() - t0:.1f} s", flush=True)
    return eq, PHI, ImlogG, RelogG


def mul_sub(d, Csym, fser):
    """substitute C -> f(x) (a series) in a dict series whose coefficients are polynomials in C."""
    out = {}
    for p, c in d.items():
        poly = sp.Poly(sp.expand(c), Csym)
        for (e,), cc in poly.terms():
            t = mul({p: cc}, powr(fser, e, CAP), CAP)
            out = add(out, t)
    return out


def solve(eq):
    """the x^(2i) equation is linear in f_(2i) (it enters through PHI's pi f term): solve it by coefficient extraction."""
    sol = {}
    for i in range(1, ORD // 2 + 1):
        c = sp.expand(eq.get(2 * i, 0).subs(sol))
        A = sp.expand(c.coeff(F[i - 1]))
        B = sp.expand(c.subs(F[i - 1], 0))
        assert A != 0 and sp.expand(c - A * F[i - 1] - B) == 0, (i, A)
        sol[F[i - 1]] = sp.expand(sp.cancel(-B / A))
    return sol


if __name__ == "__main__":
    out = open("q1557_out.txt", "a")

    def say(*s):
        print(*s, flush=True)
        print(*s, file=out, flush=True)

    say(f"=== d20_c6_series, ORD = {ORD} ===")
    for VARS in ((True, False) if (len(sys.argv) < 3 or sys.argv[2] != "varsonly") else (True,)):
        eq, PHI, ImlogG, RelogG = run(VARS)
        for p in sorted(eq):
            if p % 2:
                assert sp.simplify(eq[p]) == 0, ("odd order nonzero", p, eq[p])
        sol = solve(eq)
        say(f"-- {'SELF-CONSISTENT profile (s varies)' if VARS else 'CONSTANT-s family (record 481/485)'}:")
        for k, v in sol.items():
            vv = sp.expand(sp.nsimplify(v))
            say(f"   {k} = {sp.expand(v)}   = {sp.N(v, 16)}")
        # decomposition at each order: PHI-only part vs arg G part
        for i in range(1, ORD // 2 + 1):
            p = 2 * i
            phi_part = sp.expand(PHI.get(p, 0).subs(sol))
            g_part = sp.expand(ImlogG.get(p, 0).subs(sol))
            say(
                f"   order x^{p}: [PHI]_{p} = {sp.N(phi_part, 12)}, [arg G]_{p} = {sp.N(g_part, 12)}, [Re log G]_{p} = {sp.N(RelogG.get(p, 0).subs(sol), 12)}"
            )
        if VARS:
            f2, f4, f6 = sol[F[0]], sol[F[1]], sol[F[2]]
            e6 = sp.expand(2 * f2 * f4 - f2**3 - f6)
            say(f"   predicted e6 = 2 f2 f4 - f2^3 - f6 = {e6} = {sp.N(e6, 12)}")
            if ORD >= 8:
                say(f"   (exact) e6 = {e6}")
                f8 = sol[F[3]]
                # rho^-6 coefficient of dd/drho = rho^2/(s0 f): e8 = -[x^8] (1/f)
                xx = sp.Symbol("xx")
                fx = 1 + f2 * xx**2 + f4 * xx**4 + f6 * xx**6 + f8 * xx**8
                fl = [1, 0, f2, 0, f4, 0, f6, 0, f8]
                b = [sp.Integer(1)] + [0] * 8
                for m in range(1, 9):
                    b[m] = sp.expand(-sum(fl[i] * b[m - i] for i in range(1, m + 1)))
                e8 = b[8]
                say(
                    f"   dd/drho = (rho^2 - f2 + (f2^2 - f4)/rho^2 + e6/rho^4 + e8/rho^6 + ...)/s0 with e8 = {e8} = {sp.N(e8, 12)}"
                )
    out.close()
