"""q1714 (10 Oct 2026, READING, no verdict word): the self-consistent edge pair of the exact clean block (iteration of q1698) with the bulk law through c26 (NG = 13), N = 90. Copy of q1698_pair_iteration_ng.py: the assert is widened to NG = 13, the result file is q1714_res_NG13_N90.json, the environment variables are Q1714_NG and Q1714_N, and the label block is replaced.
Predictions registered in the registration text of 11:59 (written before any NG = 13 output exists; the NG = 15 pair of q1705 had been read at 12:06, which gives p15 - p14 but says nothing directly on p13):
 P1 the one-order ratio (p14 - p13) / (p13 - p12) lies within 10 percent of 0.363 (q1701) in both parameters, with p12 and p14 from q1698_res_NG12_N90.json and q1698_res_NG14_N90.json;
 P2 p13 lies between p12 and p14 in both parameters;
 C4 every Newton solve of the run converges to a residual below 1e-60 within 8 updates.
Usage: py q1714_pair_iteration_ng13.py (about 6 minutes, one process)."""

import os, sys, json, time, importlib.util

sys.dont_write_bytecode = True
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
NG = int(os.environ.get("Q1714_NG", "13"))
N = int(os.environ.get("Q1714_N", "90"))
os.environ["Q1696_NG"] = str(NG)
assert NG in (13,) and N in (90,)
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
    json.dump(out, open(os.path.join(here, "q1714_res_NG%d_N%d.json" % (NG, N)), "w"), indent=1)
    print(
        "final pair (nu0, d0) = (%.12f, %.12f); differences to the births pair: %+.3e, %+.3e; combination minus the births value %+.3e"
        % (nu, d0, nu - BNU, d0 - BD0, comb(nu, d0) - BCOMB)
    )
    lab = lambda x: "holds" if x else "does not hold"
    p12 = json.load(open(os.path.join(here, "q1698_res_NG12_N90.json")))["final_pair"]
    p14 = json.load(open(os.path.join(here, "q1698_res_NG14_N90.json")))["final_pair"]
    p13 = [nu, d0]
    rr = [(p14[i] - p13[i]) / (p13[i] - p12[i]) for i in range(2)]
    print(
        "p13 minus p12: %+.3e, %+.3e; p14 minus p13: %+.3e, %+.3e; one-order ratios (p14-p13)/(p13-p12): %.4f (nu0), %.4f (d0)"
        % (p13[0] - p12[0], p13[1] - p12[1], p14[0] - p13[0], p14[1] - p13[1], rr[0], rr[1])
    )
    p1 = all(abs(r / 0.363 - 1) < 0.10 for r in rr)
    p2 = all(min(p12[i], p14[i]) < p13[i] < max(p12[i], p14[i]) for i in range(2))
    print("P1 %s; P2 %s; C4 %s" % (lab(p1), lab(p2), lab(conv)))
    print("READING q1714 done (labels by the script; no verdict word)")


if __name__ == "__main__":
    run()
