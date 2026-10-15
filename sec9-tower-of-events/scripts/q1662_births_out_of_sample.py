"""q1662 (10 Oct 2026, READING, no verdict word): out-of-sample births K_b 131..137 with the stored series (centre_scripts/d22_out_18.json, nmax = 18), the bulk law through c24 and the c24 wall pair; none of these events entered the wall pair, the law or the series. The amplitudes are loaded inside the script and only differences are printed (BLIND RULE).
Predictions written before the run:
 P1 each measured event K_b 131..135 and 137 has |a_pred - A_event| below 3e-11;
 P2 the rms of the measured ones is below 2e-11;
 P3 |mean| of the differences is below 1e-11.
For the missing events K_b 136 and 138 the script writes the prediction a_pred (nmax = 18) into q1662_prereg_136_138.txt together with the `ls -l` listing of the event directories printed by the same run, and the bar |diff| < 3e-11 (registered before the events exist, if the listing shows no event_Kb136.json and no event_Kb138.json).
Control: for K_b 130 (inside the 48 births) the difference equals the one printed in q1660 for nmax = 18 (checked by recomputation of the rms over 83..130 = 8.996e-12 to 3 digits).
Usage: py q1662_births_out_of_sample.py (from small_gap_q206/; about 1 minute)."""

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
C24 = (-0.0937736504642, 0.0011599156647)
ex = [E[n] for n in range(6, 19)]
Ks = [K for K in range(131, 139) if B.measured(K) is not None]
d = {K: B.a_pred(K, *C24, ex) - B.measured(K) for K in Ks}
for K in Ks:
    print("K_b %d: a_pred - A_event = %+.3e" % (K, d[K]))
v = np.array(list(d.values()))
rms = float(np.sqrt((v**2).mean()))
inside = np.array(
    [B.a_pred(K, *C24, ex) - B.measured(K) for K in range(83, 131) if B.measured(K) is not None]
)
rin = float(np.sqrt((inside**2).mean()))
print(
    "events measured: %s; rms %.3e; mean %+.3e; control rms over K_b 83..130: %.3e" % (Ks, rms, v.mean(), rin)
)
lab = lambda q: "holds" if q else "does not hold"
print(
    "P1 every |diff| below 3e-11: %s; P2 rms below 2e-11: %s; P3 |mean| below 1e-11: %s; control rms(83..130) = 8.996e-12 to 3 digits: %s"
    % (
        lab(all(abs(x) < 3e-11 for x in v)),
        lab(rms < 2e-11),
        lab(abs(v.mean()) < 1e-11),
        lab(abs(rin - 8.996e-12) < 5e-15),
    )
)
miss = [K for K in (136, 138) if B.measured(K) is None]
listing = []
for dname in sorted(glob.glob(os.path.join(H, "fw_cap_par*"))) + [os.path.join(H, "fw_prod")]:
    for K in (136, 138):
        f = os.path.join(dname, "event_Kb%d.json" % K)
        if os.path.exists(f):
            listing.append(
                "EXISTS %s %s" % (f, time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(f))))
            )
txt = (
    "q1662 registration written %s (clock); event files for K_b 136 and 138 in fw_cap_par*, fw_prod: %s\n"
    % (
        time.strftime("%H:%M:%S"),
        listing if listing else "none (no event_Kb136.json, no event_Kb138.json in any directory)",
    )
)
for K in miss:
    txt += "K_b %d: a_pred (nmax 18, c24 law and pair) = %.12f; bar |a_pred - A_event| < 3e-11\n" % (
        K,
        B.a_pred(K, *C24, ex),
    )
open(os.path.join(H, "q1662_prereg_136_138.txt"), "w").write(txt)
print(txt, end="")
print("READING q1662 done (labels by the script; no verdict word)")
