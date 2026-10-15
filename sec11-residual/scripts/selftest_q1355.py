"""selftest_q1355 (8 Oct 2026, amendment 2122): negative and positive controls for the READOUT of the registered blind test q1355_hardwall_global_series_blind.py, written and run BEFORE the test events exist. Run: TIGHT_DEG=17 py tight_run.py ../gate_selftest/selftest_q1355.py (from temporal_within/centre_scripts/).
Cases (each calls the real read() with the amplitude reader H.locr patched, or an input file edited and restored): perfect amplitudes A = a_med + prediction Da 1e-9 -> PASS; all A shifted by +1e-10 -> FAIL; all A shifted by -1e-10 -> FAIL; ten events of k = 0.15 unlocated (22 of 32 left) -> INCOMPLETE; one single event of k = 0.4 shifted by +1e-10 and the rest perfect -> FAIL (one bad event is enough); edited prediction table -> VOID (exit 2); edited training-events file -> VOID (exit 2). Last line: SELFTEST q1355: PASS only if all seven behave as stated and both files are restored byte for byte.
"""

import io, contextlib, hashlib, json, os, sys

sys.argv = ["x"]
b = io.StringIO()
with contextlib.redirect_stdout(b):
    import q1355_hardwall_global_series_blind as T
H = T.H
orig = H.locr
js = json.load(open("q1355_predictions.json"))
coef = np_coef = js["coef"]
import numpy as np


def name_of(f):  # c61_u1345_hw015_K33 / c61_u1355_hw015_K20 -> (k, K)
    p = f.split("_")
    tag = p[2]
    K = int(p[3][1:])
    return {v: k for k, v in T.TAGS.items()}[tag], K


def perfect(f):
    k, K = name_of(f)
    e = js[str(k)][str(K)]
    return e["a_med"] + e["pred"] * T.Da(e["v2"]) * 1e-9


ALL = lambda f: True


def run():
    bb = io.StringIO()
    try:
        with contextlib.redirect_stdout(bb):
            T.read()
        code = 0
    except SystemExit as e:
        code = e.code
    out = bb.getvalue()
    ls = [l for l in out.splitlines() if l.startswith("VERDICT") or "VOID" in l]
    return code, (ls[-1] if ls else "NO VERDICT LINE")


def sha(f):
    return hashlib.sha256(open(f, "rb").read()).hexdigest()


before = {f: sha(f) for f in ("q1355_predictions.json", "q1352_events.json")}
res = {}
H.locr = perfect
res["perfect"] = run()
H.locr = lambda f: perfect(f) + 1e-10
res["shift +1e-10"] = run()
H.locr = lambda f: perfect(f) - 1e-10
res["shift -1e-10"] = run()
drop = {T.evfile("hw015", K) for K in range(14, 24)}
H.locr = lambda f: None if f in drop else perfect(f)
res["ten missing"] = run()
bad = T.evfile("hw04", 40)
H.locr = lambda f: perfect(f) + (1e-10 if f == bad else 0.0)
res["one bad event"] = run()
H.locr = orig


def tamper(fn, edit):
    raw = open(fn, "rb").read()
    try:
        if fn.endswith("predictions.json"):
            j = json.loads(raw.decode())
            edit(j)
            open(fn, "w", encoding="utf-8").write(json.dumps(j, indent=1))
        else:
            open(fn, "wb").write(raw + b" ")
        return run()
    finally:
        open(fn, "wb").write(raw)


res["edited table"] = tamper(
    "q1355_predictions.json", lambda j: j["0.15"]["20"].__setitem__("pred", j["0.15"]["20"]["pred"] + 0.1)
)
res["edited events"] = tamper("q1352_events.json", None)
for k, v in res.items():
    print(f"case {k}: exit {v[0]}, last line: {v[1][:110]}")
want = {
    "perfect": "PASS",
    "shift +1e-10": "FAIL",
    "shift -1e-10": "FAIL",
    "ten missing": "INCOMPLETE",
    "one bad event": "FAIL",
}
ok = (
    all(res[k][1].startswith("VERDICT") and (f"class: {w}" in res[k][1]) for k, w in want.items())
    and res["edited table"][0] == 2
    and res["edited events"][0] == 2
)
restored = all(sha(f) == h for f, h in before.items())
print(f"files restored byte for byte: {restored}")
print(f"SELFTEST q1355: {'PASS' if (ok and restored) else 'FAIL'}")
