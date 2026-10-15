"""q1701 (10 Oct 2026, READING, no verdict word): geometric extrapolation over the bulk order of the self-consistent edge pair of the exact clean N = 90 block (window 30..60) from the laws c20 (q1610_res.json), c24 (q1698_res_NG12_N90.json) and c28 (q1698_res_NG14_N90.json), and comparison with the births pair (-0.0937736508372, +0.0011599160085).
NOT BLIND: the three pairs and their differences were read (reading 162 and 164) before the predictions below were written; the extrapolation step itself had not been computed.
Predictions:
 P1 the ratio of the successive differences (c28 minus c24 over c24 minus c20) is the same in nu0 and d0 to within 5 percent of each other;
 P2 the extrapolated pair p_inf = p28 + (p28 - p24) r / (1 - r), with r the ratio of P1 per parameter, differs from the births pair by less than 3e-11 in each parameter;
 P3 the extrapolated pair is closer to the births pair than the c28 pair in both parameters.
Control: a synthetic sequence L + A r^k with r = 0.13 is extrapolated to its limit within 1e-15.
Usage: py q1701_ladder_pair_extrapolation.py (seconds)."""

import json, os

H = os.path.dirname(os.path.abspath(__file__))
B = (-0.0937736508372, 0.0011599160085)


def pair(f):
    return tuple(json.load(open(os.path.join(H, f)))["final_pair"])


def extrap(p20, p24, p28):
    out, rs = [], []
    for i in range(2):
        d1 = p24[i] - p20[i]
        d2 = p28[i] - p24[i]
        r = d2 / d1
        rs.append(r)
        out.append(p28[i] + d2 * r / (1 - r))
    return out, rs


L, A, r = 1.2345, 0.7, 0.13
syn = [L + A * r**k for k in (1, 2, 3)]
lim, _ = extrap(*[(v, 2 * v + 1) for v in syn])
ctl = abs(lim[0] - L) < 1e-15 and abs(lim[1] - (2 * L + 1)) < 1e-15
p20, p24, p28 = pair("q1610_res.json"), pair("q1698_res_NG12_N90.json"), pair("q1698_res_NG14_N90.json")
pinf, rs = extrap(p20, p24, p28)
print("pairs: c20 (%.12f, %.12f); c24 (%.12f, %.12f); c28 (%.12f, %.12f)" % (*p20, *p24, *p28))
print(
    "successive differences c24-c20: %+.3e, %+.3e; c28-c24: %+.3e, %+.3e; ratios r: %.4f (nu0), %.4f (d0)"
    % (p24[0] - p20[0], p24[1] - p20[1], p28[0] - p24[0], p28[1] - p24[1], rs[0], rs[1])
)
print(
    "extrapolated pair (%.12f, %.12f); minus births pair: %+.3e, %+.3e; c28 pair minus births: %+.3e, %+.3e"
    % (pinf[0], pinf[1], pinf[0] - B[0], pinf[1] - B[1], p28[0] - B[0], p28[1] - B[1])
)
lab = lambda q: "holds" if q else "does not hold"
p1 = abs(rs[0] - rs[1]) / max(abs(rs[0]), abs(rs[1])) < 0.05
p2 = all(abs(pinf[i] - B[i]) < 3e-11 for i in range(2))
p3 = all(abs(pinf[i] - B[i]) < abs(p28[i] - B[i]) for i in range(2))
print("P1 %s; P2 %s; P3 %s; control: %s" % (lab(p1), lab(p2), lab(p3), lab(ctl)))
print("READING q1701 done (labels by the script; no verdict word)")
