"""q1608_check (10 Oct 2026, READING, no verdict word): applies P1 to P4 of the docstring of q1608_wide_window_clean.py. Usage: py q1608_check.py"""

import json, os
from mpmath import mp, mpf, nstr

mp.dps = 80
Ns = [N for N in (30, 60, 90, 120) if os.path.exists("q1608_res_N%d.json" % N)]
R = {N: json.load(open("q1608_res_N%d.json" % N)) for N in Ns}
Q = {N: json.load(open("q1602_res_N%d.json" % N)) for N in Ns}
need = {30: 1e-30, 60: 1e-30, 90: 1e-30, 120: 1e-35}
ok1 = True
ok2 = True
ok4 = True
for N in Ns:
    r = R[N]
    narrow = mpf(json.load(open("q1601_z_N%d.json" % N))[0])
    wide = mpf(r["d1_50"])
    dd = abs(wide - narrow)
    passing = [x for x in r["rows"] if x["inside"]]
    print(
        "N = %3d (window 25): ||C||_inf %.3e; largest component of C F %.3e (window 15: %.3e); first depth differs from the window-15 value by %s; passes for r = %s; norm/r %.3e; 2C control inside %s"
        % (
            N,
            r["Cnorm_inf"],
            r["CFmax"],
            Q[N]["CFmax"],
            nstr(dd, 4),
            ", ".join("%.0e" % x["r"] for x in passing),
            r["rows"][0]["q"] / r["rows"][0]["r"],
            r["ctrl_2C"]["inside"],
        )
    )
    ok1 = ok1 and dd < mpf("1e-47")
    ok2 = ok2 and any(abs(x["r"] - need[N]) < need[N] * 1e-3 and x["inside"] for x in r["rows"])
    ok4 = ok4 and not r["ctrl_2C"]["inside"]
p3 = (R[120]["CFmax"] < 1e-58) if 120 in R else None
Etrap = 5.2e-119
tail = 9.8e-136
for N in Ns:
    print(
        "   N = %3d: continuum radius = largest component of C F + ||C||_inf (2 E + tail) = %.3e + %.3e"
        % (N, R[N]["CFmax"], R[N]["Cnorm_inf"] * (2 * Etrap + tail))
    )


def lab(x):
    return "pending" if x is None else ("holds" if x else "does not hold")


print(
    "P1 %s; P2 %s; P3 %s; P4 %s"
    % (
        lab(ok1 if len(Ns) == 4 else None),
        lab(ok2 if len(Ns) == 4 else None),
        lab(p3),
        lab(ok4 if len(Ns) == 4 else None),
    )
)
print("READING q1608 done (labels by the script; no verdict word)")
