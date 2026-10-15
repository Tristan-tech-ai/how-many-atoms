"""q1681 (10 Oct 2026, READING, no verdict word): (a) the matched cells (NG, nmax) = (13, 14), (14, 14), (14, 16) of q1680 against the new event K_b 138, which was delivered by chain 138b at 10:03:04 and merged by cap_cycle b39; the q1680 script refused to write a registration because the event file existed, and the cell predictions for K_b 138 were not compared with the event before this script was written. The cells and their refitted pairs come from K_b 83..130 and the choice of the cells was made after seeing K_b 131..137 and before K_b 138 was compared; the comparison for K_b 138 is therefore out of sample but NOT prospective for these cells. (The prospective registrations for K_b 138 of 09:08, 09:34 and 09:35 are evaluated by q1642 and q1668.) (b) a registration of the three cells for K_b 136, written only if no event file for K_b 136 exists.
Predictions for (a), written before the run from the trends of q1680 (differences at K_b 137 of -0.47, +0.07 and -0.26 in units of 1e-12 for the three cells):
 P1 the difference a_pred - A at K_b 138 for the cell (14, 14) lies in [-2e-13, +4e-13];
 P2 for the cell (13, 14) it lies in [-7e-13, -3e-13];
 P3 for the cell (14, 16) it lies in [-4e-13, -1.5e-13].
Registration for (b), written by the script if no K_b 136 file exists: the bars are |a_pred - A| < 5e-13 for the cell (14, 14), and the trend bars [-6e-13, -2e-13] for (13, 14) and [-3.5e-13, -1.2e-13] for (14, 16).
Control: the in-sample rms of the three cells reproduces q1680 (1.740e-13, 5.844e-14, 1.102e-13) within 2 percent.
Usage: py q1681_cells_k138_and_k136.py (from small_gap_q206/; about 1 minute)."""

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
    return dict(
        rin=rin,
        d138=(B.a_pred(138, nu, d0, ex) - B.measured(138)) if B.measured(138) is not None else None,
        p136=B.a_pred(136, nu, d0, ex),
        p138=B.a_pred(138, nu, d0, ex),
    )


R = {c: cell(*c) for c in ((13, 14), (14, 14), (14, 16))}
for c, r in R.items():
    print(
        "cell NG %d nmax %d: in-sample rms %.4e; K_b 138: a_pred - A = %s"
        % (*c, r["rin"], "%+.3e" % r["d138"] if r["d138"] is not None else "NO EVENT")
    )
lab = lambda q: "holds" if q else "does not hold"
d = {c: R[c]["d138"] for c in R}
p1 = -2e-13 <= d[(14, 14)] <= 4e-13
p2 = -7e-13 <= d[(13, 14)] <= -3e-13
p3 = -4e-13 <= d[(14, 16)] <= -1.5e-13
ctl = all(
    abs(R[c]["rin"] / v - 1) < 0.02
    for c, v in (((13, 14), 1.740e-13), ((14, 14), 5.844e-14), ((14, 16), 1.102e-13))
)
print(
    "P1 cell (14, 14) in [-2e-13, +4e-13]: %s; P2 cell (13, 14) in [-7e-13, -3e-13]: %s; P3 cell (14, 16) in [-4e-13, -1.5e-13]: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(ctl))
)
listing = []
for dname in sorted(glob.glob(os.path.join(H, "fw_cap_par*"))) + [os.path.join(H, "fw_prod")]:
    fn = os.path.join(dname, "event_Kb136.json")
    if os.path.exists(fn):
        listing.append("EXISTS %s" % fn)
if listing:
    print("event file for K_b 136 exists: no registration text written:", listing)
else:
    t = (
        "q1681 registration written %s (clock); event files for K_b 136 in fw_cap_par*, fw_prod: none (no event_Kb136.json in any directory); K_b 138 existed (delivered 10:03:04) and is not registered here.\n"
        % time.strftime("%H:%M:%S")
    )
    for c, r in R.items():
        t += "cell NG %d nmax %d: K_b 136 a_pred = %.12f\n" % (*c, r["p136"])
    t += "Bars: cell (14, 14) |a_pred - A| < 5e-13; cell (13, 14) a_pred - A in [-6e-13, -2e-13]; cell (14, 16) a_pred - A in [-3.5e-13, -1.2e-13].\n"
    open(os.path.join(H, "q1681_prereg_136.txt"), "w").write(t)
    print(t, end="")
print("READING q1681 done (labels by the script; no verdict word)")
