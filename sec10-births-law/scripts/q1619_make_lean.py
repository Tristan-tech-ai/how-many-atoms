"""q1619_make_lean.py (10 Oct 2026): builds q1619_e6_lean.py from d20_order_n.py by exact text replacement (every replaced block must match once; the old file is not touched).
Replacements: (1) laurent_at0 and laurent_ati use truncated exact series of coth (h(x) = x coth x, powers h^j as Fractions, only the needed exponents) instead of expanding the full substitution; (2) the two sp.series(...exp...) calls in inner_solve become truncated Taylor sums in delta; (3) say() prints the peak working set; (4) sympy cache cleared after every order; (5) the output file is q1619_out_<NMAX>.json (the old d20_out_5.json is kept).
"""

import os

H = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(H, "d20_order_n.py"), encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)


# header
i = s.index("import sympy as sp, json, time, sys")
j = s.index("\n", i)
head = """import sympy as sp, json, time, sys, gc, ctypes
from fractions import Fraction
from sympy.core.cache import clear_cache
def peak_mb():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    pm = PMC(); pm.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL('kernel32'); k32.GetCurrentProcess.restype = ctypes.c_void_p
    ps = ctypes.WinDLL('psapi'); ps.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]; ps.GetProcessMemoryInfo.restype = ctypes.c_int
    ok = ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pm), pm.cb)
    return pm.PeakWorkingSetSize / 1048576.0 if ok else -1.0"""
s = s[:i] + head + s[j:]
rep(
    "say = lambda *a: print(*a, flush=True)",
    'say = lambda *a: print(*a, "[peak %.0f MB]" % peak_mb(), flush=True)',
)
# laurent functions
a = s.index("def laurent_at0(eta, kmin):")
b = s.index("def derivs(F, n):")
new_laurent = '''_B = {}
def _bk(k):
    if k not in _B:
        v = sp.Rational(2 ** (2 * k)) * sp.bernoulli(2 * k) / sp.factorial(2 * k); _B[k] = Fraction(int(v.p), int(v.q))
    return _B[k]
_HP = {}
def hp(j, D):
    """coefficients r[0..D] of h(x)^j, h(x) = x coth x = sum b_k x^(2k); coth(pi z)^j = sum_d r[d] (pi z)^(d - j)"""
    if j in _HP and len(_HP[j]) > D: return _HP[j]
    D = max(D, 2 * NMAX + 8)
    h = [Fraction(0)] * (D + 1)
    for k in range(0, D // 2 + 1): h[2 * k] = _bk(k)
    r = [Fraction(1)] + [Fraction(0)] * D
    for _ in range(j):
        out = [Fraction(0)] * (D + 1)
        for p, rp in enumerate(r):
            if rp == 0: continue
            for q in range(0, D + 1 - p, 2): out[p + q] += rp * h[q]
        r = out
    _HP[j] = r
    return r
def _terms(eta):
    Pz = sp.Poly(sp.expand(eta), z, C)
    return [(m, j, c) for (m, j), c in Pz.terms()]
def _rat(f): return sp.Rational(f.numerator, f.denominator)
def laurent_at0(eta, kmin):
    acc = {}
    for m, j, c in _terms(eta):
        r = hp(j, j - 1 - m)
        for d in range(0, min(len(r) - 1, j - 1 - m) + 1):
            if r[d] == 0: continue
            e = m - j + d
            if e < kmin or e > -1: continue
            acc.setdefault(e, []).append(c * _rat(r[d]) * PI ** (d - j))
    return {k: sp.simplify(sp.expand(sp.Add(*acc.get(k, [0])))) for k in range(kmin, 0)}
def laurent_ati(eta, kmin, kmax):
    acc = {}
    for m, j, c in _terms(eta):
        for aa in range(m + 1):
            bn = sp.binomial(m, aa) * I ** (m - aa)
            r = hp(j, kmax + j - aa)
            for d in range(max(0, kmin + j - aa), min(len(r) - 1, kmax + j - aa) + 1):
                if r[d] == 0: continue
                acc.setdefault(aa + d - j, []).append(c * bn * _rat(r[d]) * PI ** (d - j))
    return {k: sp.simplify(sp.expand(sp.Add(*acc.get(k, [0])))) for k in range(kmin, kmax + 1)}
def trunc(expr, order):
    ex = sp.expand(expr)
    if ex == 0: return sp.Integer(0)
    Pd = sp.Poly(ex, dl)
    return sum((cf * dl ** k for (k,), cf in Pd.terms() if k <= order), sp.Integer(0))
def texp(x, order):
    E = sp.Integer(1); term = sp.Integer(1)
    for jj in range(1, order + 3):
        term = trunc(sp.expand(term * x), order) / jj
        if term == 0: break
        E += term
    return sp.expand(E)
'''
s = s[:a] + new_laurent + s[b:]
rep(
    "    body = sp.expand(sp.series(sp.expand(P*sp.series(sp.exp(rest.subs(Wv, Wsh)), dl, 0, order + 1).removeO()), dl, 0, order + 1).removeO())",
    "    body = trunc(sp.expand(P*texp(sp.expand(rest.subs(Wv, Wsh)), order)), order)",
)
rep(
    "    g0 = sp.expand(sp.series(avg*sp.series(sp.exp(sp.expand(lin**2)), dl, 0, order + 1).removeO(), dl, 0, order + 1).removeO())",
    "    g0 = trunc(sp.expand(avg*texp(sp.expand(lin**2), order)), order)",
)
rep(
    "    regc[n] = {k: Ln[k] for k in range(1, 2*NMAX - 2*n + 1)}",
    "    regc[n] = {k: Ln[k] for k in range(1, 2*NMAX - 2*n + 1)}\n    clear_cache(); gc.collect()",
)
rep(
    'json.dump(out, open(f"d20_out_{NMAX}.json", "w"), indent=1)',
    'json.dump(out, open(f"q1619_out_{NMAX}.json", "w"), indent=1)',
)
rep(
    'say("done", out, f"[{time.time() - T0:.0f} s]")',
    'say("done", out, f"[{time.time() - T0:.0f} s]")\nsay("PEAK_WORKING_SET_MB %.0f" % peak_mb())',
)
# docstring
doc = '"""q1619_e6_lean (generated by q1619_make_lean.py from d20_order_n.py; do not edit by hand). Memory-lean version of the centre-series script: exact truncated coth series in laurent_at0 and laurent_ati, truncated Taylor sums in inner_solve, sympy cache cleared per order, peak working set printed on every line.\nPredictions written before the first run (10 Oct 2026, READING, no verdict word; deep proof Q11 step 73):\n P1 CONTROL: for NMAX = 5 the lean script gives e_2..e_5 equal to the old values in d20_out_5.json (e_n_num to 12 significant digits relative, and the exact strings equal);\n P2 at NMAX = 4 the peak working set of the lean script is below 40 percent of the old script (old measured by q1619_run_old_peak.py);\n P3 NMAX = 6 finishes with a peak working set below 8 GB;\n P4 the derived e_6 lies between -1.5e+06 and -1e+05.\nUsage: py q1619_e6_lean.py NMAX (labels by q1619_check.py)."""\n'
k = s.index("import sympy as sp")
s = doc + s[k:]
open(os.path.join(H, "q1619_e6_lean.py"), "w", encoding="utf-8").write(s)
print("written", len(s))
