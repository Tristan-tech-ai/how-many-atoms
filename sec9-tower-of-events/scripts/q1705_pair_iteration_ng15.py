"""q1705 (10 Oct 2026, READING, no verdict word): the self-consistent edge pair of the exact clean block (iteration of q1698) with the bulk law through c30 (NG = 15), N = 90 by default.
Copy of q1698_pair_iteration_ng.py (body changed only where named): the module is q1704_krawczyk_ng15.py (control q1706: P1 to P4 hold), the order comes from the environment Q1703_NG (set here from Q1705_NG, default 15), the block size from Q1705_N (90 or 120).
Reference values (stored outputs): births pair of the c30 refit (q1692, cell NG = 15, nmax 18) (-0.0937736508427, +0.0011599160135); the c28 pair of the same block from q1698_res_NG14_N90.json, rest rms from its last round.
Predictions written before any NG = 15 output exists (no q1705 output file exists when this text is written):
 P2 the final pair differs from the births pair of the c30 refit by less in absolute value than the c28 pair does, in both parameters (the c28 differences are computed by the script from q1698_res_NG14_N90.json);
 P3 the rest rms of the last round on the window 30..60 is below that of the c28 run (2.20e-12);
 C4 every Newton solve of the run converges to a residual below 1e-60 within 8 updates.
Usage: py q1705_pair_iteration_ng15.py (about 6 to 9 minutes at N = 90, one process)."""

import os, sys, json, time, importlib.util

sys.dont_write_bytecode = True
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
NG = int(os.environ.get("Q1705_NG", "15"))
N = int(os.environ.get("Q1705_N", "90"))
os.environ["Q1703_NG"] = str(NG)
assert NG in (14, 15) and N in (90, 120)
spec = importlib.util.spec_from_file_location("q1704m", os.path.join(here, "q1704_krawczyk_ng15.py"))
q = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(q)
m = q.m
from flint import arb, arb_mat

n = 2 * N + 1
LO, HI = 30, 60
BNU, BD0 = -0.0937736508427, 0.0011599160135
BCOMB = BD0 + 0.95 * BNU
REF = json.load(open(os.path.join(here, "q1698_res_NG14_N90.json")))
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
    json.dump(out, open(os.path.join(here, "q1705_res_NG%d_N%d.json" % (NG, N)), "w"), indent=1)
    print(
        "final pair (nu0, d0) = (%.12f, %.12f); differences to the births pair: %+.3e, %+.3e; combination minus the births value %+.3e"
        % (nu, d0, nu - BNU, d0 - BD0, comb(nu, d0) - BCOMB)
    )
    lab = lambda x: "holds" if x else "does not hold"
    c28 = REF["final_pair"]
    c28rms = REF["hist"][-1]["rms"]
    print(
        "c28 pair minus the c30 births pair: %+.3e, %+.3e; c28 rest rms %.3e"
        % (c28[0] - BNU, c28[1] - BD0, c28rms)
    )
    p2 = (
        lab(abs(nu - BNU) < abs(c28[0] - BNU) and abs(d0 - BD0) < abs(c28[1] - BD0))
        if N == 90
        else "not applicable (the c28 reference is the N = 90 run)"
    )
    p3 = lab(hist[-1]["rms"] < c28rms) if N == 90 else "not applicable"
    print("P2 %s; P3 %s; C4 %s" % (p2, p3, lab(conv)))
    print("READING q1705 done (labels by the script; no verdict word)")


if __name__ == "__main__":
    run()
