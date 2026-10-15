"""q1713 (10 Oct 2026, READING, no verdict word): the births model of q1671 (bulk law through c28, stored series to order 18, wall pair of q1669 at nmax = 18) against the lower births K_b 61..82 with the stored tower amplitudes and with the amplitudes re-measured by q1712 (q1712_res.json) where they exist.
Prediction P3 (registered in the registration text of 12:14 and in the docstring of q1712 before q1712 produced its second output): after replacing the stored amplitudes by the re-measured ones the largest |a_pred - A| over K_b 61..82 falls below 1.055e-11, the stored value (q1671).
Control: with no replacement the table equals the differences printed in q1671_out.txt (largest difference of the two tables below 0.06e-12 for the rounded values).
Usage: py q1713_lower_births_remeasured_model.py (seconds; needs q1712_res.json)."""

import os, sys, json, io, re, contextlib

sys.dont_write_bytecode = True
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
src = open(os.path.join(here, "q1671_lower_k_births.py"), encoding="utf-8").read()
src = src[: src.index("d = {}")]
ns = {"__file__": os.path.join(here, "q1671_lower_k_births.py"), "__name__": "q1671ns"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(src, ns)
B, PAIR, ex, meas = ns["B"], ns["PAIR"], ns["ex"], ns["meas"]
R = json.load(open(os.path.join(here, "q1712_res.json")))
new = {int(K): float(v["new"]) for K, v in R.items() if v.get("status") == "ok"}
old, upd = {}, {}
for K in range(61, 83):
    m = meas(K)
    if m is None:
        continue
    old[K] = B.a_pred(K, *PAIR, ex) - m
    upd[K] = B.a_pred(K, *PAIR, ex) - (new[K] if K in new else m)
q = [
    float(x)
    for x in re.findall(
        r"\d+ ([+-][0-9]+\.[0-9])",
        open(os.path.join(here, "q1671_out.txt"), encoding="utf-8").read().splitlines()[1].split(": ", 1)[1],
    )
]
ctl = len(q) == len(old) and max(abs(a * 1e12 - b) for a, b in zip((old[K] for K in sorted(old)), q)) < 0.06
print("re-measured events used: %s" % sorted(new))
print("K_b   stored-amplitude difference   re-measured-amplitude difference   (a_pred - A, 1e-12)")
for K in sorted(old):
    print(
        "%3d   %+8.1f   %+8.1f%s" % (K, old[K] * 1e12, upd[K] * 1e12, "   (re-measured)" if K in new else "")
    )
mo, mu = max(abs(v) for v in old.values()), max(abs(v) for v in upd.values())
lo = [abs(upd[K]) for K in upd if K <= 69]
hi = [abs(upd[K]) for K in upd if K >= 70]
print(
    "largest |a_pred - A|: stored %.3e, with the re-measured amplitudes %.3e; K_b 61..69 now %.3e, K_b 70..82 now %.3e"
    % (mo, mu, max(lo), max(hi))
)
lab = lambda q_: "holds" if q_ else "does not hold"
print("P3 largest size below 1.055e-11: %s; control: %s" % (lab(mu < 1.055e-11), lab(ctl)))
print("READING q1713 done (labels by the script; no verdict word)")
