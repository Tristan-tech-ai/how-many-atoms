"""q1680 (10 Oct 2026, READING, no verdict word, predictions written at 10:03): out-of-sample test of the matched truncation cells (NG, nmax) = (13, 14), (14, 14) and (14, 16) of q1677. The wall pair is refitted on K_b 83..130 (four linearisation rounds from the q1669 pair), frozen, and applied to the events K_b 131..135 and 137, which entered no fit. Differences a_pred - A only (BLIND RULE).
Predictions:
 P1 the out-of-sample rms of the six events is below 1e-12 in each of the three cells;
 P2 the best of the three cells beats the NG = 12, nmax = 18 value 1.897e-12 of q1663 by at least a factor 2;
 P3 the in-sample rms of the cell (14, 14) is below 1e-13.
For K_b 136 and 138 the script writes the three predictions a_pred to q1680_prereg_136_138.txt with the `ls` listing of the event directories of the same run and the bar |a_pred - A| < 2e-12 for the cell with the smallest out-of-sample rms among the three; the text is generated only if no event file exists.
Control: the cell (14, 18) reproduces the in-sample rms 1.5200e-13 of q1669 within 5 percent (not part of the three cells; computed once for the control).
Usage: py q1680_matched_cells_out_of_sample.py (from small_gap_q206/; about 1 minute)."""

import glob, json, os, re, time
import numpy as np
import sympy as sp
import mpmath as mp
import births_law as B

H = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(H, "..", "centre_scripts")
S20 = json.load(open(os.path.join(C, "d22_out_20.json")))
E = {n: float(S20["e%d" % n]) for n in range(2, 21)}
txt = open(os.path.join(H, "q1557_stdout_ORD28.txt"), encoding="utf-8").read()
f = {}
for n in (26, 28):
    m = re.search(r"^\s*f%d = (.*?)(?:\s{2,}= |$)" % n, txt, re.M)
    f[n] = float(mp.mpf(sp.N(sp.sympify(m.group(1).strip().replace("pi", "PI"), locals={"PI": sp.pi}), 30)))
FALL = (B.F + [-2.420924146814851e-10, -8.973800969707301e-11])[:13] + [f[26], f[28]]


def setlaw(NG):
    B.F = FALL[: NG + 1]
    B.NG = NG
    B.g = [1.0]
    for n in range(1, NG + 1):
        B.g.append(-sum(B.F[j] * B.g[n - j] for j in range(1, n + 1)))


Kin = [K for K in range(83, 131) if B.measured(K) is not None]
min_ = np.array([B.measured(K) for K in Kin])
Kout = [K for K in (131, 132, 133, 134, 135, 137) if B.measured(K) is not None]
h = 1e-7


def cell(NG, nmax):
    setlaw(NG)
    ex = [E[n] for n in range(6, nmax + 1)]
    nu, d0 = -0.0937736508347, 0.0011599160063
    for rnd in range(4):
        p0 = np.array([B.a_pred(K, nu, d0, ex) for K in Kin])
        M = np.stack(
            [
                [(B.a_pred(K, nu + h, d0, ex) - B.a_pred(K, nu - h, d0, ex)) / (2 * h) for K in Kin],
                [(B.a_pred(K, nu, d0 + h, ex) - B.a_pred(K, nu, d0 - h, ex)) / (2 * h) for K in Kin],
            ],
            1,
        )
        sol, *_ = np.linalg.lstsq(M, -(p0 - min_), rcond=None)
        nu, d0 = nu + sol[0], d0 + sol[1]
    rin = float(np.sqrt(((np.array([B.a_pred(K, nu, d0, ex) for K in Kin]) - min_) ** 2).mean()))
    out = {K: B.a_pred(K, nu, d0, ex) - B.measured(K) for K in Kout}
    ro = float(np.sqrt(np.mean([v**2 for v in out.values()])))
    return dict(
        pair=(nu, d0), rin=rin, rout=ro, out=out, pred={K: B.a_pred(K, nu, d0, ex) for K in (136, 138)}
    )


R = {c: cell(*c) for c in ((13, 14), (14, 14), (14, 16))}
for c, r in R.items():
    print(
        "cell NG %d nmax %d: pair (%.13f, %.13f), in-sample rms %.3e, out-of-sample rms %.3e; differences (1e-12) K_b 131..135, 137: %s"
        % (*c, *r["pair"], r["rin"], r["rout"], ", ".join("%+.2f" % (r["out"][K] * 1e12) for K in Kout))
    )
ctl18 = cell(14, 18)["rin"]
lab = lambda q: "holds" if q else "does not hold"
p1 = all(r["rout"] < 1e-12 for r in R.values())
best = min(R, key=lambda c: R[c]["rout"])
p2 = R[best]["rout"] < 1.897e-12 / 2
p3 = R[(14, 14)]["rin"] < 1e-13
ctl = abs(ctl18 / 1.5200e-13 - 1) < 0.05
print(
    "best cell by out-of-sample rms: NG %d nmax %d (%.3e); control (14, 18) in-sample rms %.4e"
    % (*best, R[best]["rout"], ctl18)
)
print(
    "P1 out-of-sample rms below 1e-12 in all three cells: %s; P2 best cell at least twice better than 1.897e-12: %s; P3 in-sample rms of (14, 14) below 1e-13: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(ctl))
)
listing = []
for dname in sorted(glob.glob(os.path.join(H, "fw_cap_par*"))) + [os.path.join(H, "fw_prod")]:
    for K in (136, 138):
        fn = os.path.join(dname, "event_Kb%d.json" % K)
        if os.path.exists(fn):
            listing.append("EXISTS %s" % fn)
if listing:
    print("event files exist for K_b 136 or 138: no registration text written:", listing)
else:
    t = (
        "q1680 registration written %s (clock); event files for K_b 136 and 138 in fw_cap_par*, fw_prod: none (no event_Kb136.json, no event_Kb138.json in any directory).\n"
        % time.strftime("%H:%M:%S")
    )
    t += (
        "Cells: "
        + "; ".join(
            "NG %d nmax %d (in-sample rms %.3e, out-of-sample rms %.3e)" % (*c, r["rin"], r["rout"])
            for c, r in R.items()
        )
        + ". Best by out-of-sample rms: NG %d nmax %d; bar |a_pred - A| < 2e-12 for that cell.\n" % best
    )
    for K in (136, 138):
        t += (
            "K_b %d: " % K
            + "; ".join("NG %d nmax %d a_pred = %.12f" % (*c, r["pred"][K]) for c, r in R.items())
            + "\n"
        )
    open(os.path.join(H, "q1680_prereg_136_138.txt"), "w").write(t)
    print(t, end="")
print("READING q1680 done (labels by the script; no verdict word)")
