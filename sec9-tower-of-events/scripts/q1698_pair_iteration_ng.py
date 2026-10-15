"""q1698 (10 Oct 2026, READING, no verdict word, plan_c28_ladder.md): the self-consistent edge pair of the exact clean block (the iteration of q1610) with the bulk law through c20, c24 or c28.
Copy of q1610_pair_iteration.py. The law order comes from the environment Q1698_NG (10, 12 or 14, default 14), the block size from Q1698_N (90 or 120, default 90). The module is q1697_krawczyk_ng.py (copy of q1598 reading q1696_ball_rows_ng.py, control q1699: P1 to P5 hold).
Each round: fit the deviations of the atoms k = 30..60 of the stored clean block from the formal law of the current pair (arb formal depths, q.formal_tail), add the fit to the pair, solve the block again with the law atoms on the new pair (Newton from the previous solution, 300 bits). The start is the stored clean solution q1601_z_N{N}.json (law atoms on the c20 law). Five rounds, as in q1610.
Reference values (stored outputs, no new run): births pair (-0.0937736508372, +0.0011599160085) with combination d0 + 0.95 nu0 = -0.0879250522868; q1617 pairs of the stored blocks with the law c24 (window 30..60): N = 90 (-0.093773650189, +0.001159915388), rest rms 1.53e-11; N = 120 (-0.093773650464, +0.001159915665), rest rms 9.52e-12; window 45..75 of N = 120 (-0.093773650757, +0.001159915938).
Predictions written before any run with NG = 12 or 14 (no output of this script exists when this text is written, between 11:36 and 11:38 on 10 Oct):
 S1 (NG = 12, N = 90) the final pair differs from the q1617 pair of the same block and window by between 1e-10 and 6e-10 in each parameter, and its combination d0 + 0.95 nu0 by less than 5e-11 (basis: at the law c20 the q1610 pair (-0.093773648256, +0.001159913641) differs from the q1617 pair (-0.093773647967, +0.001159913353) by -2.9e-10 and +2.9e-10, combination 1.4e-11; an earlier version of this line, bar 1e-10 for 'does not move', was replaced at 11:38 before any output existed, after this c20 difference was noticed);
 S2 (NG = 12, N = 90) the rest rms of the last round lies between 0.8e-11 and 2.5e-11;
 C1 (NG = 14) the combination d0 + 0.95 nu0 of the final pair lies within 1e-10 of the births value -0.0879250522868;
 C2 (NG = 14, N = 120) the pair refitted on the window 45..75 of the final state (iterated pair plus the fit correction) differs from the births pair by less than 4e-10 in each parameter (the plan had 8e-11, the value of the stored blocks; bar widened at 11:38 before any output existed, because the re-solve moves the pair by about 3e-10 at the law c20);
 C3 (NG = 14, N = 120) the rest rms of the last round (window 30..60) lies between 1.0e-12 and 4.0e-12;
 C4 (every NG) every Newton solve of the run converges to a residual below 1e-60 within 8 updates.
Labels that do not apply to the configuration print 'not applicable'. Usage: Q1698_NG=12 py q1698_pair_iteration_ng.py (about 6 to 9 minutes at N = 90, one process); Q1698_NG=14 Q1698_N=120 for the second run.
"""

import os, sys, json, time, importlib.util

sys.dont_write_bytecode = True
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
NG = int(os.environ.get("Q1698_NG", "14"))
N = int(os.environ.get("Q1698_N", "90"))
os.environ["Q1696_NG"] = str(NG)
assert NG in (10, 12, 14) and N in (90, 120)
spec = importlib.util.spec_from_file_location("q1697m", os.path.join(here, "q1697_krawczyk_ng.py"))
q = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(q)
m = q.m
from flint import arb, arb_mat

n = 2 * N + 1
LO, HI = 30, 60
BNU, BD0 = -0.0937736508372, 0.0011599160085
BCOMB = -0.0879250522868
REF = {(12, 90): (-0.093773650189, 0.001159915388)}
comb = lambda nu, d0: d0 + 0.95 * nu


def set_block_clean(NN):
    m.NIN = NN
    m.NOUT = NN + 80
    m.KC = NN + 50


def fit(z, nu, d0, lo=LO, hi=HI):
    q.NU, q.D0 = nu, d0
    dt, wt = q.formal_tail(0, N + 1)
    d = np.array([float(z[k]) for k in range(N)])
    f = np.array([float(t) for t in dt])
    delta = d - f[:N]
    s = np.array([(f[k + 1] - f[k - 1]) / 2 if k > 0 else f[1] - f[0] for k in range(N)])
    idx = np.arange(lo - 1, hi)
    A = np.vstack([np.ones(len(idx)), s[idx]]).T
    coef, *_ = np.linalg.lstsq(A, delta[idx], rcond=None)
    res = delta[idx] - A @ coef
    return coef[0], coef[1], float(np.sqrt((res**2).mean())), float(np.abs(delta[idx]).max())


def run():
    t0 = time.time()
    z = [arb(s) for s in json.load(open(os.path.join(here, "q1601_z_N%d.json" % N)))]
    nu, d0 = q.NU, q.D0
    hist = []
    conv = True
    print("NG = %d, N = %d, formal start pair (%+.10f, %+.10f)" % (NG, N, nu, d0), flush=True)
    for r in range(5):
        dd0, dnu, rms, mx = fit(z, nu, d0)
        hist.append(dict(round=r, nu0=nu, d0=d0, fit_dd0=dd0, fit_dnu=dnu, rms=rms, maxdev=mx))
        nu_used, d0_used, zs = nu, d0, z
        print(
            "round %d: pair (%+.12f, %+.12f); fit corrections delta d0 %+.3e, delta nu0 %+.3e; rms of the rest %.2e; largest deviation %.2e [%.0fs]"
            % (r, nu, d0, dd0, dnu, rms, mx, time.time() - t0),
            flush=True,
        )
        nu, d0 = float(nu + dnu), float(d0 + dd0)
        if r == 4:
            break
        q.NU, q.D0 = nu, d0
        set_block_clean(N)
        WA = m.WallB(nu, d0, q.C0)
        z, h = q.newton_own(WA, z, n, 1e-70, 12)
        conv = conv and h[-1] < 1e-60 and len(h) - 1 <= 8
        hist[-1]["newton"] = [float(x) for x in h]
        print(
            "   Newton for the new pair: %d updates, residuals %s"
            % (len(h) - 1, ", ".join("%.1e" % x for x in h)),
            flush=True,
        )
    extra = {}
    for lo, hi in ((20, 40), (45, 75)):
        if hi <= N - 15:
            a, b, rm, _ = fit(zs, nu_used, d0_used, lo, hi)
            extra[(lo, hi)] = (nu_used + b, d0_used + a, rm)
            print(
                "window %d..%d on the state of the last round: pair (%+.12f, %+.12f), rest rms %.2e"
                % (lo, hi, nu_used + b, d0_used + a, rm),
                flush=True,
            )
    out = dict(
        NG=NG,
        N=N,
        hist=hist,
        final_pair=[nu, d0],
        newton_ok=conv,
        windows={"%d..%d" % k: v for k, v in extra.items()},
    )
    json.dump(out, open(os.path.join(here, "q1698_res_NG%d_N%d.json" % (NG, N)), "w"), indent=1)
    print(
        "final pair (nu0, d0) = (%.12f, %.12f); differences to the births pair: %+.3e, %+.3e; combination minus the births value %+.3e"
        % (nu, d0, nu - BNU, d0 - BD0, comb(nu, d0) - BCOMB)
    )
    lab = lambda x: "holds" if x else "does not hold"
    na = "not applicable"
    s1 = (
        lab(
            all(1e-10 < abs(x - r) < 6e-10 for x, r in zip((nu, d0), REF[(12, 90)]))
            and abs(comb(nu, d0) - comb(*REF[(12, 90)])) < 5e-11
        )
        if (NG, N) == (12, 90)
        else na
    )
    s2 = lab(0.8e-11 <= hist[-1]["rms"] <= 2.5e-11) if (NG, N) == (12, 90) else na
    c1 = lab(abs(comb(nu, d0) - BCOMB) < 1e-10) if NG == 14 else na
    c2 = (
        lab(all(abs(extra[(45, 75)][i] - (BNU, BD0)[i]) < 4e-10 for i in (0, 1)))
        if (NG == 14 and N == 120 and (45, 75) in extra)
        else na
    )
    c3 = lab(1.0e-12 <= hist[-1]["rms"] <= 4.0e-12) if (NG == 14 and N == 120) else na
    print("S1 %s; S2 %s; C1 %s; C2 %s; C3 %s; C4 %s" % (s1, s2, c1, c2, c3, lab(conv)))
    print("READING q1698 done (labels by the script; no verdict word)")


if __name__ == "__main__":
    run()
