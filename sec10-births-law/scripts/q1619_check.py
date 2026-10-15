"""q1619_check.py (10 Oct 2026): labels P1 to P4 of q1619_e6_lean.py (predictions in its docstring, written before the first run). Reads q1619_old_4.log, q1619_lean_4.log, q1619_out_5.json, d20_out_5.json, q1619_lean_6.log, q1619_out_6.json. READING, no verdict word."""

import json, re, os

H = os.path.dirname(os.path.abspath(__file__))
lab = lambda p: "holds" if p else "does not hold"
peak = lambda f: float(re.findall(r"PEAK_WORKING_SET_MB (\d+)", open(os.path.join(H, f)).read())[-1])
a = json.load(open(os.path.join(H, "d20_out_5.json")))
b = json.load(open(os.path.join(H, "q1619_out_5.json")))
p1 = all(
    a["e%d" % n] == b["e%d" % n] and abs(a["e%d_num" % n] / b["e%d_num" % n] - 1) < 1e-12 for n in range(2, 6)
)
po, pl = peak("q1619_old_4.log"), peak("q1619_lean_4.log")
p2 = pl < 0.4 * po
print("control e_2..e_5 against d20_out_5.json: strings equal and numbers to 1e-12:", p1)
print("NMAX = 4 peak working set: old %.0f MB, lean %.0f MB, ratio %.2f" % (po, pl, pl / po))
t = lambda f: re.findall(r"\[(\d+) s\]", open(os.path.join(H, f)).read())[-1]
print(
    "NMAX = 4 time: old %s s, lean %s s; NMAX = 5 lean time %s s, peak %.0f MB (old NMAX = 5: 1206 s and more in the logs of 1 Oct)"
    % (t("q1619_old_4.log"), t("q1619_lean_4.log"), t("q1619_lean_5.log"), peak("q1619_lean_5.log"))
)
f6 = os.path.join(H, "q1619_out_6.json")
if os.path.exists(f6):
    c = json.load(open(f6))
    e6 = c["e6_num"]
    p6 = peak("q1619_lean_6.log")
    p3 = p6 < 8000
    p4 = -1.5e6 <= e6 <= -1e5
    print("NMAX = 6: e_6 = %s = %.10g; time %s s; peak %.0f MB" % (c["e6"], e6, t("q1619_lean_6.log"), p6))
    print(
        "P1 control exact: %s; P2 NMAX 4 peak below 40 percent of the old: %s; P3 NMAX 6 below 8 GB: %s; P4 e_6 in [-1.5e+06, -1e+05]: %s"
        % (lab(p1), lab(p2), lab(p3), lab(p4))
    )
else:
    print("q1619_out_6.json does not exist yet: P3 and P4 open; P1 %s; P2 %s" % (lab(p1), lab(p2)))
print("READING q1619 check done (labels by the script; no verdict word)")
