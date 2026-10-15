"""q1642 (10 Oct 2026, READING with REGISTERED predictions, no verdict word): the parameter-free births model against the missing capacity events K_b 136 and 138.
Registered here, written 09:08 by date, BEFORE the events K_b 136 and 138 exist (the files fw_prod/event_Kb136.json and event_Kb138.json do not exist at writing time). The model has no fitted parameter: wall pair, bulk law, E_4, E_5, E_6 = e_6, e_7, e_8 (reading 126). Differences are predicted minus measured.
 R1 with the law through c20 and the c20 pair, the difference at K_b 136 lies between -1.2e-10 and -5e-11 and at K_b 138 between -1.4e-10 and -5e-11 (linear continuation of -4.4e-11 .. -8.5e-11 at K_b 130..137);
 R2 with the law through c24 and the c24 pair, the difference at K_b 136 lies between +1e-11 and +6e-11 and at K_b 138 between +1e-11 and +6e-11 (continuation of +4.8e-11 .. +2.9e-11);
 R3 the c24 model is closer to the measured amplitude than the c20 model at both events.
Labels per event that exists: R1, R2, R3 by the script, with the K_b 137 control (c24 +2.91e-11 and c20 -8.53e-11 must reproduce to 3 digits, else VOID). NO EVENT if none exists. Amplitudes are never printed.
Usage: py q1642_births_136_138.py (from small_gap_q206/)."""

import numpy as np
import births_law as B

F24 = B.F + [-2.420924146814851e-10, -8.973800969707301e-11]


def setlaw(NG):
    B.F = F24[: NG + 1]
    B.NG = NG
    B.g = [1.0]
    for n in range(1, NG + 1):
        B.g.append(-sum(B.F[j] * B.g[n - j] for j in range(1, n + 1)))


extra = [B.E_DER[6], B.E_DER[7], B.E_DER[8]]
C24 = (-0.0937736504642, 0.0011599156647)
setlaw(10)
c20 = B.diffs([137, 136, 138], *B.WALL, extra=extra)
setlaw(12)
c24 = B.diffs([137, 136, 138], *C24, extra=extra)
ctl = (
    137 in c20 and 137 in c24 and abs(c20[137] / -8.53e-11 - 1) < 2e-3 and abs(c24[137] / 2.91e-11 - 1) < 2e-3
)
print(
    "control K_b 137: c20 %+.2e (-8.53e-11), c24 %+.2e (+2.91e-11): %s"
    % (c20.get(137, float("nan")), c24.get(137, float("nan")), "ok" if ctl else "FAILED: VOID")
)
lab = lambda q: "holds" if q else "does not hold"
if ctl:
    ev = [K for K in (136, 138) if K in c20]
    if not ev:
        print("NO EVENT K_b 136 or 138 in fw_prod: no label")
    for K in ev:
        r1 = (-1.2e-10 if K == 136 else -1.4e-10) <= c20[K] <= -5e-11
        r2 = 1e-11 <= c24[K] <= 6e-11
        r3 = abs(c24[K]) < abs(c20[K])
        print(
            "K_b %d: c20 difference %+.2e, c24 difference %+.2e; R1 %s; R2 %s; R3 %s"
            % (K, c20[K], c24[K], lab(r1), lab(r2), lab(r3))
        )
print("READING q1642 done (labels by the script; no verdict word)")
