"""selftest_q1446 (9 Oct 2026): controls for centre_scripts/q1446_margin_law_verdict.py with the real main(); expected verdicts are computed here by an independent loop, and the registered table is recomputed here from the training snapshot by plain normal equations (no numpy lstsq) as the control against an independent form. Cases: rows at the table; all rows at +0.99 and -0.99 bar; one row at 1.01 bar; rows at the exponent-1 null; a missing row; a PN row below one; FAIL together with a missing row; an edited table, an edited bar, an edited training value, an unreadable loader, a failed row control in the live file."""

import contextlib, io, json, os, sys, tempfile

sys.path.insert(0, "centre_scripts")
sys.argv = ["x"]
import q1446_margin_law_verdict as G

tab = json.load(open("centre_scripts/q1446_margin_table.json"))
trn = json.load(open("centre_scripts/q1446_training_rows.json"))
T = list(range(109, 115))
Q = ("PN", "Re")
bar = lambda K, q: tab["bar"]["even" if K % 2 == 0 else "odd"][q]
pred = lambda K, q: tab["pred"][str(K)][q]
null = lambda K, q: tab["null"][str(K)][q]
base = lambda: {K: {q: pred(K, q) for q in Q} for K in T}


def expect(m):
    parts = []
    for item in ("M1", "M2", "M3"):
        v = "PASS"
        miss = False
        for K in T:
            if m.get(K) is None:
                miss = True
                continue
            for q in Q:
                x = m[K][q]
                if item == "M1" and not abs(x - pred(K, q)) < bar(K, q):
                    v = "FAIL"
                if item == "M2" and not x > 1.0:
                    v = "FAIL"
                if item == "M3" and not abs(x - pred(K, q)) < abs(x - null(K, q)):
                    v = "FAIL"
        if v == "PASS" and miss:
            v = "INCOMPLETE"
        parts.append(f"{item} {v}")
    return "VERDICT q1446: " + "; ".join(parts)


def run(loader, train=None, table=None):
    b = io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(b):
            G.main(loader, train, table)
    except SystemExit as e:
        code = e.code
    ls = [l for l in b.getvalue().splitlines() if l.startswith("VERDICT")]
    return code, (ls[-1] if ls else "NO VERDICT LINE")


def shifted(K, q, d):
    m = base()
    m[K][q] += d
    return m


cases = [
    ("rows at the table", base),
    ("all rows at +0.99 bar", lambda: {K: {q: pred(K, q) + 0.99 * bar(K, q) for q in Q} for K in T}),
    ("all rows at -0.99 bar", lambda: {K: {q: pred(K, q) - 0.99 * bar(K, q) for q in Q} for K in T}),
    ("one row at 1.01 bar (odd, PN)", lambda: shifted(111, "PN", 1.01 * bar(111, "PN"))),
    ("one row at -1.01 bar (even, Re)", lambda: shifted(112, "Re", -1.01 * bar(112, "Re"))),
    ("rows at the exponent-1 null", lambda: {K: {q: null(K, q) for q in Q} for K in T}),
    ("row 112 missing", lambda: {K: v for K, v in base().items() if K != 112}),
    ("PN of row 113 below one", lambda: shifted(113, "PN", 1.0 - pred(113, "PN") - 1e-3)),
    ("Re of row 110 exactly one", lambda: shifted(110, "Re", 1.0 - pred(110, "Re"))),
    (
        "FAIL together with a missing row",
        lambda: {K: v for K, v in shifted(110, "PN", 3 * bar(110, "PN")).items() if K != 114},
    ),
    ("no row measured", lambda: {}),
    (
        "rows at the midpoint of table and null",
        lambda: {K: {q: 0.5 * (pred(K, q) + null(K, q)) for q in Q} for K in T},
    ),
]
ok = 0
tot = 0
for name, mk in cases:
    d = mk()
    code, line = run(lambda d=d: d)
    exp = expect(d)
    g = code == 0 and line == exp
    ok += g
    tot += 1
    print(f"{'ok ' if g else 'BAD'} {name}: exit {code}, {line} (expected {exp})")


def indep(rows, par, q, p, K):
    s = [(r["K"], r[q]) for r in rows if r["parity"] == par]
    n = len(s)
    sx = sum(k**-p for k, _ in s)
    sxx = sum(k ** (-2 * p) for k, _ in s)
    sy = sum(v for _, v in s)
    sxy = sum(k**-p * v for k, v in s)
    a1 = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a0 = (sy - a1 * sx) / n
    return a0 + a1 * K**-p


worst = max(
    abs(indep(trn["rows"], "even" if K % 2 == 0 else "odd", q, 2 / 3, K) - pred(K, q)) for K in T for q in Q
)
worstn = max(
    abs(indep(trn["rows"], "even" if K % 2 == 0 else "odd", q, 1.0, K) - null(K, q)) for K in T for q in Q
)
g = worst < 1e-9 and worstn < 1e-9
ok += g
tot += 1
print(
    f"{'ok ' if g else 'BAD'} independent normal-equation recomputation of the table: largest difference {worst:.1e} (law) and {worstn:.1e} (null)"
)


def edited(kind):
    tmp = tempfile.mkdtemp()
    jt = json.load(open("centre_scripts/q1446_margin_table.json"))
    jr = json.load(open("centre_scripts/q1446_training_rows.json"))
    if kind == "pred":
        jt["pred"]["109"]["PN"] += 1e-9
    elif kind == "bar":
        jt["bar"]["odd"]["PN"] *= 2
    elif kind == "train":
        jr["rows"][3]["PN"] += 1e-9
    tp = os.path.join(tmp, "t.json")
    rp = os.path.join(tmp, "r.json")
    json.dump(jt, open(tp, "w"))
    json.dump(jr, open(rp, "w"))
    return rp, tp


for name, kind in (
    ("edited prediction in the table", "pred"),
    ("edited bar in the table", "bar"),
    ("edited training value", "train"),
):
    rp, tp = edited(kind)
    code, line = run(lambda: base(), rp, tp)
    g = code == 2 and line == "VERDICT q1446: VOID"
    ok += g
    tot += 1
    print(f"{'ok ' if g else 'BAD'} {name}: exit {code}, {line}")


def boom():
    raise IOError("unreadable")


code, line = run(boom)
g = code == 2 and line == "VERDICT q1446: VOID"
ok += g
tot += 1
print(f"{'ok ' if g else 'BAD'} unreadable live file: exit {code}, {line}")
tmp = tempfile.mkdtemp()
lp = os.path.join(tmp, "rows.json")
rows = [{"K": 109, "parity": "odd", "PN": 1.05, "Re": 1.07, "control_ok": False}]
json.dump(rows, open(lp, "w"))
old = G.LIVE
G.LIVE = lp
code, line = run(None)
G.LIVE = old
g = code == 2 and line == "VERDICT q1446: VOID"
ok += g
tot += 1
print(f"{'ok ' if g else 'BAD'} live row with a failed control: exit {code}, {line}")
print(f"selftest q1446: {ok} of {tot} cases as expected")
print("SELFTEST q1446:", "PASS" if ok == tot else "FAIL")
sys.exit(0 if ok == tot else 1)
