"""selftest_q1412 (9 Oct 2026): negative and positive controls for the gate q1412_instrument_equivalence_gate.py, written and run BEFORE any event above K_b 82
is measured with the new instrument. Each case builds a temporary tree (reference files of the tower and event files of the walker), points the real main() at it
and records the exit code and the VERDICT line. Run: py gate_selftest/selftest_q1412.py  (from temporal_within/); output gate_selftest/selftest_q1412_out.txt.
"""

import contextlib, io, json, os, shutil, sys, tempfile

sys.path.insert(0, "centre_scripts")
sys.argv = ["x"]
import q1412_instrument_equivalence_gate as T

REFV = {62: 42.26402683378489, 82: 52.47472053143398, 63: 42.792398545313695, 81: 51.98024360768851}


def build(offsets=None, missing=(), wrongK=None, edit_ref=None):
    tmp = tempfile.mkdtemp()
    arch = os.path.join(tmp, "scratch")
    base = os.path.join(tmp, "small_gap_q206")
    os.makedirs(arch)
    os.makedirs(base)
    for K, (f, key, kind) in T.REFS.items():
        d = {key: REFV[K]}
        if edit_ref == K:
            d = {"other": 1.0}
        json.dump(d, open(os.path.join(arch, f), "w"))
    for K in T.EVENT_DIRS:
        if K in missing:
            continue
        os.makedirs(os.path.join(base, T.EVENT_DIRS[K]), exist_ok=True)
        val = REFV[K] + (offsets or {}).get(K, 0.0)
        json.dump(
            {"K_b": (wrongK if wrongK is not None and K == 63 else K), "A_event": repr(val)},
            open(os.path.join(base, T.EVENT_DIRS[K], f"event_Kb{K}.json"), "w"),
        )
    return tmp


def run(tmp, args=()):
    T.ROOT = tmp
    T.ARCH = os.path.join(tmp, "scratch")
    sys.argv = ["x", *args]
    b = io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(b):
            T.main()
    except SystemExit as e:
        code = e.code
    ls = [l for l in b.getvalue().splitlines() if l.startswith("VERDICT")]
    shutil.rmtree(tmp, ignore_errors=True)
    return code, (ls[-1] if ls else "NO VERDICT LINE")


cases = [
    ("perfect", build(), (0, "VERDICT q1412: PASS")),
    ("all shifted +3e-10", build({K: 3e-10 for K in REFV}), (1, "VERDICT q1412: FAIL")),
    ("birth K_b 62 shifted 0.9e-10", build({62: 0.9e-10}), (0, "VERDICT q1412: PASS")),
    ("birth K_b 62 shifted 1.2e-10", build({62: 1.2e-10}), (1, "VERDICT q1412: FAIL")),
    ("birth K_b 82 shifted -1.2e-10", build({82: -1.2e-10}), (1, "VERDICT q1412: FAIL")),
    ("split K_b 63 shifted 1.5e-10", build({63: 1.5e-10}), (0, "VERDICT q1412: PASS")),
    ("split K_b 63 shifted 2.5e-10", build({63: 2.5e-10}), (1, "VERDICT q1412: FAIL")),
    ("split K_b 81 shifted 2.5e-10", build({81: 2.5e-10}), (1, "VERDICT q1412: FAIL")),
    ("K_b 82 missing", build(missing=(82,)), (3, "VERDICT q1412: INCOMPLETE")),
    ("K_b 81 missing (not required)", build(missing=(81,)), (0, "VERDICT q1412: PASS")),
    ("reference file edited", build(edit_ref=62), (2, "VERDICT q1412: VOID")),
    ("event file of another K_b", build(wrongK=99), (2, "VERDICT q1412: VOID")),
    ("registration run", None, (0, "")),
]
res = []
allok = True
for name, tmp, (ecode, eline) in cases:
    if tmp is None:
        tmp = build()
        code, line = run(tmp, ("--registration-run",))
        ok = code == ecode
    else:
        code, line = run(tmp)
        ok = code == ecode and line == eline
    allok = allok and ok
    res.append((name, code, line, ok))
    print(f"{name:34s}: exit {code}, {line or '(controls only)'} -> {'OK' if ok else 'WRONG'}")
print("SELFTEST q1412:", "PASS" if allok else "FAIL", f"({sum(1 for r in res if r[3])} of {len(res)} cases)")
sys.exit(0 if allok else 1)
