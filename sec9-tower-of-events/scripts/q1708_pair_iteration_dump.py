"""q1708 (10 Oct 2026, READING, no verdict word): copy of q1698_pair_iteration_ng.py that also saves the solved state of the last round (all z, 60 digits) and the pair used for it, to q1708_z_NG{NG}_N{N}.json, for the analysis q1709 of the deviation profile on the inner atoms. Nothing else is changed except the names of the environment variables (Q1708_NG, Q1708_N) and of the result file (q1708_res_NG{NG}_N{N}.json).
The predictions of q1698 for the labels are not repeated here (the run repeats q1698 with NG = 14 and N = 90, so its pair and rest rms must equal the q1698 values: control C0, below). Control C0: the final pair equals that of q1698_res_NG14_N90.json to 1e-12 in both parameters and the rest rms of the last round to 1 percent.
Usage: Q1708_NG=14 Q1708_N=90 py q1708_pair_iteration_dump.py (about 6 minutes, one process)."""

import os, sys, json, time, importlib.util

sys.dont_write_bytecode = True
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
NG = int(os.environ.get("Q1708_NG", "14"))
N = int(os.environ.get("Q1708_N", "90"))
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
    json.dump(
        dict(NG=NG, N=N, nu_used=nu_used, d0_used=d0_used, z=[t.str(60, radius=False) for t in zs]),
        open(os.path.join(here, "q1708_z_NG%d_N%d.json" % (NG, N)), "w"),
    )
    out = dict(
        NG=NG,
        N=N,
        hist=hist,
        final_pair=[nu, d0],
        newton_ok=conv,
        windows={"%d..%d" % k: v for k, v in extra.items()},
    )
    json.dump(out, open(os.path.join(here, "q1708_res_NG%d_N%d.json" % (NG, N)), "w"), indent=1)
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
    ref = json.load(open(os.path.join(here, "q1698_res_NG%d_N%d.json" % (NG, N))))
    c0 = (
        abs(nu - ref["final_pair"][0]) < 1e-12
        and abs(d0 - ref["final_pair"][1]) < 1e-12
        and abs(hist[-1]["rms"] / ref["hist"][-1]["rms"] - 1) < 0.01
    )
    print("C0 control against q1698: %s" % lab(c0))
    print("S1 %s; S2 %s; C1 %s; C2 %s; C3 %s; C4 %s" % (s1, s2, c1, c2, c3, lab(conv)))
    print("READING q1708 done (labels by the script; no verdict word)")


if __name__ == "__main__":
    run()
