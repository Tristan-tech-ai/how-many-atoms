"""q1695 (10 Oct 2026, READING, no verdict word, predictions written at 11:19, written before the result file was read): comparison of the exact closed form e_10 of the symbolic order-10 run (centre_scripts/q1634_out_10.json, finished before a restart) with the stored numerical e_10 of centre_scripts/d22_out_18.json (8 Oct), and the completion line for e_9 in centre_scripts/q1634_se_10.log.
Predictions:
 P1 the closed form of e_10 evaluated at 90 digits equals the stored e_10 to a relative difference below 1e-50;
 P2 the line 'completed eta_9: far-field constant minus e_9 = 0' prints 0;
 P3 the ratio e_10/e_9 of the closed forms equals 51.0357 to 4 digits.
Control: the same comparison for e_9 (closed form of q1634_out_9.json against the stored value) gives a relative difference below 1e-50, and a copy of the stored e_10 with its 60th digit changed is detected as a difference above 1e-60 and below 1e-50.
Usage: py q1695_e10_check.py (from small_gap_q206/; seconds)."""

import json, os, re
import sympy as sp
import mpmath as mp

mp.mp.dps = 90
H = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(H, "..", "centre_scripts")
D10 = json.load(open(os.path.join(C, "q1634_out_10.json")))
D9 = json.load(open(os.path.join(C, "q1634_out_9.json")))
S = json.load(open(os.path.join(C, "d22_out_18.json")))


def closed(D, n):
    return mp.mpf(sp.N(sp.sympify(D["e%d" % n].replace("pi", "PI"), locals={"PI": sp.pi}), 90))


e10, e9 = closed(D10, 10), closed(D10, 9)
s10, s9 = mp.mpf(S["e10"]), mp.mpf(S["e9"])
r10 = abs(e10 - s10) / abs(e10)
r9 = abs(closed(D9, 9) - s9) / abs(closed(D9, 9))
r9b = abs(e9 - s9) / abs(e9)
log = open(os.path.join(C, "q1634_se_10.log"), encoding="utf-8").read()
m = re.search(r"completed eta_9: far-field constant minus e_9 = (\S+)", log)
bad = s10 * (1 + mp.mpf(10) ** -59)
rb = abs(bad - e10) / abs(e10)
print(
    "e_10 closed form %s, stored %s; relative difference %s"
    % (mp.nstr(e10, 22), mp.nstr(s10, 22), mp.nstr(r10, 4))
)
print(
    "e_9 from the order-10 run against stored: %s; e_9 of the order-9 run against stored: %s"
    % (mp.nstr(r9b, 4), mp.nstr(r9, 4))
)
print(
    "completion line of e_9 in q1634_se_10.log prints: %s; ratio e_10/e_9 = %s; control (change of 1e-59): %s"
    % (m.group(1) if m else "NOT FOUND", mp.nstr(e10 / e9, 7), mp.nstr(rb, 4))
)
lab = lambda q: "holds" if q else "does not hold"
p1 = r10 < mp.mpf(10) ** -50
p2 = bool(m) and m.group(1) == "0"
p3 = abs(e10 / e9 - mp.mpf("51.0357")) < mp.mpf("0.00005")
ctl = r9 < mp.mpf(10) ** -50 and mp.mpf(10) ** -60 < rb < mp.mpf(10) ** -50
print(
    "P1 closed form equals the stored value to 1e-50: %s; P2 completion line prints 0: %s; P3 ratio 51.0357 to 4 digits: %s; control: %s"
    % (lab(p1), lab(p2), lab(p3), lab(ctl))
)
print("READING q1695 done (labels by the script; no verdict word)")
