"""q1730 (10 Oct 2026, READING, POST HOC, no predictions; follows q1729): the limit of the births pair over the bulk order from its own steps, for comparison with the two ladder limits of q1729 (models A and B). Births pairs of the laws c24..c30 fitted on K_b 83..130 (q1664/q1665/q1667 outputs, same series order nmax = 18): NG = 12 (-0.0937736507540, 0.0011599159348), 13 (-0.0937736508135, 0.0011599159874), 14 (-0.0937736508344, 0.0011599160061), 15 (-0.0937736508428, 0.0011599160135). The tail after NG = 15 is summed with the last step ratio held constant (model C) and with the ratio rising by the last observed increment per order up to ratio 1 (model D). Written after q1729 was read; nothing here is a test."""

import numpy as np

P = {
    12: (-0.0937736507540, 0.0011599159348),
    13: (-0.0937736508135, 0.0011599159874),
    14: (-0.0937736508344, 0.0011599160061),
    15: (-0.0937736508428, 0.0011599160135),
}
L = {"A": (-0.0937736508471, 0.0011599160174), "B": (-0.0937736508483, 0.0011599160185)}
for i, n in enumerate(("nu0", "d0")):
    s = [P[ng][i] - P[ng - 1][i] for ng in (13, 14, 15)]
    r = [s[1] / s[0], s[2] / s[1]]
    inc = r[1] - r[0]
    tail_c = s[2] * r[1] / (1 - r[1])
    t, rr, tail_d = s[2], r[1], 0.0
    for _ in range(40):
        rr = min(rr + inc, 0.999)
        t = t * rr
        tail_d += t
        if abs(t) < 1e-18:
            break
    print(
        "%s: steps %s; ratios %.3f, %.3f; limit with constant ratio %.13f (tail %+.3e), with rising ratio %.13f (tail %+.3e)"
        % (
            n,
            ", ".join("%+.3e" % x for x in s),
            r[0],
            r[1],
            P[15][i] + tail_c,
            tail_c,
            P[15][i] + tail_d,
            tail_d,
        )
    )
    for m in ("A", "B"):
        print(
            "   ladder limit model %s %.13f: minus births limit (constant ratio) %+.2e, (rising ratio) %+.2e"
            % (m, L[m][i], L[m][i] - (P[15][i] + tail_c), L[m][i] - (P[15][i] + tail_d))
        )
print("READING q1730 done (post hoc; no labels)")
