"""q920_test (7 Oct 2026): test of q920_cap_flint.Cap against q206's residual_and_jacobian (mpmath, 40 digits) and the stored dF/dA of q901d.
Criteria written before any output: T1 max |res_flint - res_mp| <= 1e-33; T2 max |J_flint - J_mp| / max(1, |J_mp|) <= 1e-30 on the even K 60 state of A = 41.1 and on the odd state (same state plus a centre atom wc = 0.003, other masses scaled);
T3 max |b_flint - b_stored| / max(1, |b|) <= 1e-30 against the b of q901d_K60_A41.1_state.json; T4 the flint residual and Jacobian take at most 60 s. Verdict: last line.
"""

import json, sys, time
from flint import arb
from mpmath import mp, mpf, log, pi
import q206_gauss_mp as G
import q920_cap_flint as F

mp.dps = 40
G.L2P = log(2 * pi) / 2
A_str, u, w, we, wc, d = F.load_state("results_q220_gauss_K60_A41.1000.json")
m = len(u)
okall = True
for odd in (False, True):
    cap = F.Cap(A_str, m, odd)
    if odd:
        c = arb("0.003")
        w2 = [t * (1 - c) for t in w]
        we2 = we * (1 - c)
        v = u + w2 + [we2, c]
    else:
        v = u + w + [we]
    t0 = time.time()
    res, J, b = cap.res_jac(v, True, True)
    tf = time.time() - t0
    mu = [mpf(t.str(45, radius=False)) for t in v[:m]]
    mw = [mpf(t.str(45, radius=False)) for t in v[m : 2 * m]]
    mwe = mpf(v[2 * m].str(45, radius=False))
    mwc = mpf(v[2 * m + 1].str(45, radius=False)) if odd else None
    t1 = time.time()
    resm, Jm = G.residual_and_jacobian(mpf(A_str), G.grid(mpf(A_str)), mu, mw, mwe, mwc)
    tm = time.time() - t1
    r_err = max(abs(mpf(res[i].str(45, radius=False)) - resm[i]) for i in range(len(resm)))
    n = len(resm)
    j_err = max(
        abs(mpf(J[i, k].str(45, radius=False)) - Jm[i, k]) / max(1, abs(Jm[i, k]))
        for i in range(n)
        for k in range(n)
    )
    T1 = r_err <= mpf("1e-33")
    T2 = j_err <= mpf("1e-30")
    T4 = tf <= 60
    msg = f"odd {odd}: residual diff {mp.nstr(r_err, 3)} ({T1}), Jacobian rel diff {mp.nstr(j_err, 3)} ({T2}), flint {tf:.1f}s vs mpmath {tm:.1f}s ({T4})"
    if not odd:
        bs = json.load(open("q901d_K60_A41.1_state.json"))["b"]
        b_err = max(
            abs(mpf(b[i].str(45, radius=False)) - mpf(bs[i])) / max(1, abs(mpf(bs[i])))
            for i in range(len(bs))
        )
        T3 = b_err <= mpf("1e-30")
        msg += f"; b rel diff {mp.nstr(b_err, 3)} ({T3})"
        okall = okall and T3
    print(msg, flush=True)
    okall = okall and T1 and T2 and T4
print("VERDICT q920_test:", "PASS" if okall else "FAIL")
sys.exit(0 if okall else 1)
