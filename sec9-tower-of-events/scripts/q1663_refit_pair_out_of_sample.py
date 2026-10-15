"""q1663 (10 Oct 2026, READING, no verdict word): the wall pair re-fitted to the 48 births K_b 83..130 in q1661 (shift (-2.8982e-10, +2.7012e-10) from the ladder pair, in-sample rms 9.524e-13) is frozen and applied to the events K_b 131..135, 137, which entered neither the fit nor the ladder. Amplitudes are loaded inside the script; only differences are printed.
Predictions written before the run:
 P1 every |a_pred - A_event| for K_b 131..135 and 137 is below 5e-12;
 P2 the rms of these six is below 3e-12;
 P3 the six differences have no common sign drift larger than 1e-12 per K_b (|slope| below 1e-12).
For K_b 136 and 138 the frozen prediction is written to q1663_prereg_136_138.txt with the `ls` listing of the event directories printed by the same run; bar |diff| < 5e-12 (registered before the events exist if the listing shows none).
Control: the frozen pair reproduces the in-sample rms 9.524e-13 of q1661 to 3 digits (recomputed here with the shifts printed there, rounded to 5 digits, so a tolerance of 1e-14 applies).
Usage: py q1663_refit_pair_out_of_sample.py (from small_gap_q206/; about 1 minute)."""

import glob, json, os, time
import numpy as np
import births_law as B

H = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(H, "..", "centre_scripts", "d22_out_18.json")))
E = {n: float(S["e%d" % n]) for n in range(2, 19)}
B.F = (B.F + [-2.420924146814851e-10, -8.973800969707301e-11])[:13]
B.NG = 12
B.g = [1.0]
for n in range(1, 13):
    B.g.append(-sum(B.F[j] * B.g[n - j] for j in range(1, n + 1)))
PAIR = (-0.0937736504642 - 2.8982e-10, 0.0011599156647 + 2.7012e-10)
ex = [E[n] for n in range(6, 19)]
Ks = [K for K in range(131, 139) if B.measured(K) is not None]
d = {K: B.a_pred(K, *PAIR, ex) - B.measured(K) for K in Ks}
for K in Ks:
    print("K_b %d: a_pred - A_event = %+.3e" % (K, d[K]))
v = np.array([d[K] for K in Ks])
rms = float(np.sqrt((v**2).mean()))
slope = float(np.polyfit(Ks, v, 1)[0])
ins = np.array([B.a_pred(K, *PAIR, ex) - B.measured(K) for K in range(83, 131) if B.measured(K) is not None])
rin = float(np.sqrt((ins**2).mean()))
print(
    "out of sample (%d events): rms %.3e, mean %+.3e, slope %+.3e per K_b; in-sample rms %.3e, mean %+.3e"
    % (len(Ks), rms, v.mean(), slope, rin, ins.mean())
)
lab = lambda q: "holds" if q else "does not hold"
print(
    "P1 every |diff| below 5e-12: %s; P2 rms below 3e-12: %s; P3 |slope| below 1e-12 per K_b: %s; control in-sample rms 9.524e-13: %s"
    % (
        lab(all(abs(x) < 5e-12 for x in v)),
        lab(rms < 3e-12),
        lab(abs(slope) < 1e-12),
        lab(abs(rin - 9.524e-13) < 1e-14),
    )
)
listing = []
for dname in sorted(glob.glob(os.path.join(H, "fw_cap_par*"))) + [os.path.join(H, "fw_prod")]:
    for K in (136, 138):
        f = os.path.join(dname, "event_Kb%d.json" % K)
        if os.path.exists(f):
            listing.append("EXISTS %s" % f)
txt = "q1663 registration written %s (clock); event files for K_b 136 and 138: %s\n" % (
    time.strftime("%H:%M:%S"),
    listing if listing else "none (no event_Kb136.json, no event_Kb138.json in any directory)",
)
for K in (136, 138):
    if B.measured(K) is None:
        txt += "K_b %d: a_pred (frozen refit pair, nmax 18) = %.12f; bar |a_pred - A_event| < 5e-12\n" % (
            K,
            B.a_pred(K, *PAIR, ex),
        )
open(os.path.join(H, "q1663_prereg_136_138.txt"), "w").write(txt)
print(txt, end="")
print("READING q1663 done (labels by the script; no verdict word)")
