"""q1700 (10 Oct 2026, READING, no verdict word; task Q99/Q100, registration centre_scripts/q1700_prereg_e11.txt of 11:54:39): comparison of the exact closed form e_11 of the symbolic order-11 run (centre_scripts/q1634_out_11.json, log q1634_se_11.log) with the stored numerical e_11 of 8 Oct (d22_out_18.json).
Predictions (copied from the registration, written before the run was started):
 P1 the closed form of e_11 evaluated at 90 digits equals the stored e_11 to a relative difference below 1e-50;
 P2 the line 'completed eta_10: far-field constant minus e_10 = 0' prints 0;
 P3 the total time lies between 25000 and 40000 s and the peak working set between 1.0 and 2.2 GB.
Control: the order-10 data (q1634_out_10.json, q1634_se_10.log) go through the same function with n = 10 and give P1 and P2 true; the P3 bars are not applied to the control. Selftest (--selftest): (a) the control; (b) a stored value with its 60th digit changed is detected with a relative difference between 1e-60 and 1e-50; (c) a log with the completion line changed to '= 5' makes P2 fail; (d) a missing output file prints VOID and exits with code 2.
Usage: py q1700_e11_check.py [--selftest] (from small_gap_q206/; seconds). Without the file q1634_out_11.json the real run prints 'VOID: output file missing' and exits with code 2.
"""

import json, os, re, sys
import sympy as sp
import mpmath as mp

mp.mp.dps = 90
H = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(H, "..", "centre_scripts")


def closed(D, n):
    return mp.mpf(sp.N(sp.sympify(D["e%d" % n].replace("pi", "PI"), locals={"PI": sp.pi}), 90))


def evaluate(n, dpath, lpath, stored, log_text=None):
    D = json.load(open(dpath))
    en = closed(D, n)
    sn = mp.mpf(stored["e%d" % n])
    rel = abs(en - sn) / abs(en)
    log = log_text if log_text is not None else open(lpath, encoding="utf-8").read()
    m = re.search(r"completed eta_%d: far-field constant minus e_%d = (\S+)" % (n - 1, n - 1), log)
    stamps = re.findall(r"\[(\d+) s\]", log)
    pk = re.findall(r"PEAK_WORKING_SET_MB (\d+)", log)
    return dict(
        e=en,
        s=sn,
        rel=rel,
        completion=(m.group(1) if m else None),
        total=(int(stamps[-1]) if stamps else None),
        peak=(int(pk[-1]) if pk else None),
    )


def selftest():
    S = json.load(open(os.path.join(C, "d22_out_18.json")))
    d10 = os.path.join(C, "q1634_out_10.json")
    l10 = os.path.join(C, "q1634_se_10.log")
    r = evaluate(10, d10, l10, S)
    a = r["rel"] < mp.mpf(10) ** -50 and r["completion"] == "0"
    bad = dict(S)
    bad["e10"] = mp.nstr(mp.mpf(S["e10"]) * (1 + mp.mpf(10) ** -59), 85)
    rb = evaluate(10, d10, l10, bad)["rel"]
    b = mp.mpf(10) ** -60 < rb < mp.mpf(10) ** -50
    txt = open(l10, encoding="utf-8").read()
    c = (
        evaluate(
            10,
            d10,
            l10,
            S,
            log_text=re.sub(r"(completed eta_9: far-field constant minus e_9 = )0", r"\g<1>5", txt),
        )["completion"]
        != "0"
    )
    try:
        evaluate(10, os.path.join(C, "q1634_out_NOFILE.json"), l10, S)
        dd = False
    except FileNotFoundError:
        dd = True
    lab = lambda q: "holds" if q else "does not hold"
    print(
        "SELFTEST q1700: (a) control order 10 %s (relative difference %s); (b) changed 60th digit detected %s (%s); (c) changed completion line detected %s; (d) missing file raises %s"
        % (lab(a), mp.nstr(r["rel"], 4), lab(b), mp.nstr(rb, 4), lab(c), lab(dd))
    )
    print("SELFTEST " + ("PASS" if (a and b and c and dd) else "FAIL"))
    sys.exit(0 if (a and b and c and dd) else 1)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    d11 = os.path.join(C, "q1634_out_11.json")
    if not os.path.exists(d11):
        print("VOID: output file missing (%s)" % d11)
        sys.exit(2)
    S = json.load(open(os.path.join(C, "d22_out_18.json")))
    r = evaluate(11, d11, os.path.join(C, "q1634_se_11.log"), S)
    print(
        "e_11 closed form %s, stored %s; relative difference %s"
        % (mp.nstr(r["e"], 22), mp.nstr(r["s"], 22), mp.nstr(r["rel"], 4))
    )
    print(
        "completion line of e_10 prints: %s; total time %s s; peak working set %s MB"
        % (r["completion"], r["total"], r["peak"])
    )
    lab = lambda q: "holds" if q else "does not hold"
    p1 = r["rel"] < mp.mpf(10) ** -50
    p2 = r["completion"] == "0"
    p3 = (
        r["total"] is not None
        and r["peak"] is not None
        and 25000 <= r["total"] <= 40000
        and 1000 <= r["peak"] <= 2200
    )
    print(
        "P1 closed form equals the stored value to 1e-50: %s; P2 completion line prints 0: %s; P3 time 25000 to 40000 s and peak 1.0 to 2.2 GB: %s"
        % (lab(p1), lab(p2), lab(p3))
    )
    print("READING q1700 done (labels by the script; no verdict word)")
