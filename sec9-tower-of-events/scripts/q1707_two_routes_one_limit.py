"""q1707 (10 Oct 2026, READING, no verdict word): the births pair (refit on K_b 83..130 with the stored series, wall pair per bulk order NG, outputs q1664_out.txt for NG = 10, 11 and q1667_out.txt for NG = 12 to 15) against the ladder pair (self-consistent pair of the exact N = 90 block, window 30..60: q1610_res.json for NG = 10, q1698_res_NG12_N90.json, q1698_res_NG14_N90.json, and q1705_res_NG15_N90.json when it exists).
NOT BLIND: the ladder pairs and the births pairs of NG = 10 and 12 were seen (readings 164, 165 and the output q1664) before the predictions below were written; the table of differences was not computed.
Predictions:
 P1 the difference ladder minus births in nu0 falls in absolute value at every step NG = 10 -> 12 -> 14, and the ratio of successive differences lies between 0.1 and 0.3 at each step;
 P2 the same holds in d0, with the opposite sign of the difference;
 P3 the difference at NG = 14 is below 1e-10 in absolute value in both parameters.
Control: the table built twice from the files is identical, and the births pairs of NG = 12 agree between q1664_out.txt and q1667_out.txt to 1e-13.
Usage: py q1707_two_routes_one_limit.py (seconds)."""

import json, os, re

H = os.path.dirname(os.path.abspath(__file__))


def births(fn):
    out = {}
    for l in open(os.path.join(H, fn), encoding="utf-8"):
        m = re.match(r"NG = (\d+)[^:]*:? ?\(?[^()]*?pair \((-?[0-9.]+), ([0-9.]+)\)", l) or re.match(
            r"NG = (\d+): pair \((-?[0-9.]+), ([0-9.]+)\)", l
        )
        if m:
            out[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return out


def ladder(fn):
    p = os.path.join(H, fn)
    return tuple(json.load(open(p))["final_pair"]) if os.path.exists(p) else None


def table():
    b = births("q1664_out.txt")
    b.update(births("q1667_out.txt"))
    b12 = births("q1664_out.txt")[12]
    L = {
        10: ladder("q1610_res.json"),
        12: ladder("q1698_res_NG12_N90.json"),
        14: ladder("q1698_res_NG14_N90.json"),
        15: ladder("q1705_res_NG15_N90.json"),
    }
    rows = [(ng, L[ng][0] - b[ng][0], L[ng][1] - b[ng][1]) for ng in (10, 12, 14, 15) if L[ng] is not None]
    return b, rows, b12


b, rows, b12 = table()
b2, rows2, _ = table()
ctl = rows == rows2 and abs(b12[0] - b[12][0]) < 1e-13 and abs(b12[1] - b[12][1]) < 1e-13
print("births pairs per NG: " + "; ".join("NG %d (%.10f, %.10f)" % (k, *b[k]) for k in sorted(b)))
for ng, dn, dd in rows:
    print("NG = %2d: ladder minus births: nu0 %+.3e, d0 %+.3e" % (ng, dn, dd))
d = {ng: (dn, dd) for ng, dn, dd in rows}
rn = [abs(d[b_][0]) / abs(d[a][0]) for a, b_ in ((10, 12), (12, 14))]
rd = [abs(d[b_][1]) / abs(d[a][1]) for a, b_ in ((10, 12), (12, 14))]
print("ratios of successive differences (10 -> 12, 12 -> 14): nu0 %.3f, %.3f; d0 %.3f, %.3f" % (*rn, *rd))
if 15 in d:
    print(
        "NG = 14 -> 15: ratio nu0 %.3f, d0 %.3f (not part of the predictions)"
        % (abs(d[15][0]) / abs(d[14][0]), abs(d[15][1]) / abs(d[14][1]))
    )
lab = lambda q: "holds" if q else "does not hold"
p1 = abs(d[12][0]) < abs(d[10][0]) and abs(d[14][0]) < abs(d[12][0]) and all(0.1 <= r <= 0.3 for r in rn)
p2 = (
    abs(d[12][1]) < abs(d[10][1])
    and abs(d[14][1]) < abs(d[12][1])
    and all(0.1 <= r <= 0.3 for r in rd)
    and all(d[k][0] * d[k][1] < 0 for k in (10, 12, 14))
)
p3 = abs(d[14][0]) < 1e-10 and abs(d[14][1]) < 1e-10
print("P1 %s; P2 %s; P3 %s; control: %s" % (lab(p1), lab(p2), lab(p3), lab(ctl)))
print("READING q1707 done (labels by the script; no verdict word)")
