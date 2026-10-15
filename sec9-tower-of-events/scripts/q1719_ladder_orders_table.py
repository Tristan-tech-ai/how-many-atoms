"""q1719 (10 Oct 2026, READING, POST HOC, no predictions; follows q1714 whose P1 did not hold): the self-consistent pairs of the exact N = 90 block at NG = 12, 13, 14, 15 (q1698_res_NG12_N90.json, q1714_res_NG13_N90.json, q1698_res_NG14_N90.json, q1705_res_NG15_N90.json), the one-order steps, their ratios, and the extrapolation to the limit with the last ratio held fixed, against the births pairs of the c28 and c30 refits (q1665, q1667). Written after the outputs of q1714 and q1705 had been read, so nothing here is a test."""

import json, os

H = os.path.dirname(os.path.abspath(__file__))
P = {
    ng: tuple(json.load(open(os.path.join(H, f)))["final_pair"])
    for ng, f in (
        (12, "q1698_res_NG12_N90.json"),
        (13, "q1714_res_NG13_N90.json"),
        (14, "q1698_res_NG14_N90.json"),
        (15, "q1705_res_NG15_N90.json"),
    )
}
B = {14: (-0.0937736508344, 0.0011599160061), 15: (-0.0937736508428, 0.0011599160135)}
for ng in (13, 14, 15):
    print(
        "NG %d minus NG %d: nu0 %+.3e, d0 %+.3e"
        % (ng, ng - 1, P[ng][0] - P[ng - 1][0], P[ng][1] - P[ng - 1][1])
    )
r = [(P[ng][i] - P[ng - 1][i]) / (P[ng - 1][i] - P[ng - 2][i]) for i in range(2) for ng in (14, 15)]
print(
    "one-order ratios of successive steps: NG 14/13: nu0 %.4f, d0 %.4f; NG 15/14: nu0 %.4f, d0 %.4f"
    % (r[0], r[2], r[1], r[3])
)
for i, n in enumerate(("nu0", "d0")):
    rr = (P[15][i] - P[14][i]) / (P[14][i] - P[13][i])
    lim = P[15][i] + (P[15][i] - P[14][i]) * rr / (1 - rr)
    print(
        "%s: limit with the last ratio %.4f fixed: %.13f; minus the c30 births pair %+.3e; c30 ladder pair minus c30 births pair %+.3e"
        % (n, rr, lim, lim - B[15][i], P[15][i] - B[15][i])
    )
print(
    "ladder minus births, NG = 14 (c28 births pair of q1665): nu0 %+.3e, d0 %+.3e; NG = 15 (c30 births pair): nu0 %+.3e, d0 %+.3e"
    % (P[14][0] - B[14][0], P[14][1] - B[14][1], P[15][0] - B[15][0], P[15][1] - B[15][1])
)
print("READING q1719 done (post hoc; no labels)")
