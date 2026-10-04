"""Registered inner-ring tests of paper 1 (Table "Registered losses of inner rings" and its supplement list).

Reads every row of the supplement table (paper_onelaw/supplement_inner_rings.tex) that has a registered band and an observed value,
recomputes the verdict as band_lo <= observed <= band_hi, checks it against the printed verdict, checks that each band end and each
observed value occurs verbatim in the registration log (PREREG_Q32.md), and counts PASS / registered per problem and ring type.
Writes inner_rings_check.json. Reads only.
"""
import os, re, json

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(HERE)
SUP = open(os.path.join(TW, "paper_onelaw", "supplement_inner_rings.tex"), encoding="utf-8").read()
LOG = open(os.path.join(TW, "PREREG_Q32.md"), encoding="utf-8", errors="replace").read()
ROW = re.compile(r"^(capacity|LFP|NPMLE)[^&]*& (pair|quartet)[^&]*& \$([\d.]+)\$--\$([\d.]+)\$ & \$([\d.]+)\$ & (PASS|FAIL)", re.M)

rows, counts = [], {}
for m in ROW.finditer(SUP):
    prob, ring, lo, hi, obs, printed = m.groups()
    ok = float(lo) <= float(obs) <= float(hi)
    in_log = all(s in LOG for s in (lo, hi, obs.rstrip("0") if obs.endswith("0") else obs))
    rows.append({"problem": prob, "ring": ring, "lo": lo, "hi": hi, "observed": obs, "printed": printed,
                 "recomputed": "PASS" if ok else "FAIL", "agrees": (printed == "PASS") == ok, "numbers_in_log": in_log})
    c = counts.setdefault(prob, {}).setdefault(ring, [0, 0])
    c[0] += int(ok)
    c[1] += 1
out = {"rows": rows, "registered": len(rows), "passed": sum(1 for r in rows if r["recomputed"] == "PASS"),
       "all_verdicts_agree": all(r["agrees"] for r in rows), "all_numbers_in_log": all(r["numbers_in_log"] for r in rows),
       "counts": counts}
out["failed"] = out["registered"] - out["passed"]
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
for r in rows:
    if not r["agrees"] or not r["numbers_in_log"]:
        print("CHECK:", r)
json.dump(out, open(os.path.join(HERE, "inner_rings_check.json"), "w", encoding="utf-8"), indent=1)
