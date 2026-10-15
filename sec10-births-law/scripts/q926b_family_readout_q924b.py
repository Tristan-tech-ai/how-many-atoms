"""q926 (7 Oct 2026): text copy of q902k for the outputs of q924 (files q924_K<K>_A<A>_out.txt, last line starting VERDICT q924); registered in the amendment that registers q924. Original header of q902k: family readout for any stored capacity family (1685).
Reads the last line of scratch/q901d_K<K>_A<A>_out.txt for every state of the list file (lines 'A file') and prints a verdict:
  PASS = all outputs exist and every last line is 'VERDICT q901d ...: PASS;'
  FAIL = all outputs are readable (PASS or FAIL lines) and at least one is FAIL
  VOID = any output missing, empty, unreadable or VOID (instrument defect; no verdict on the family)
Usage: py q902k_family_readout.py K LISTFILE [DIR]     (from temporal_within/; default DIR scratch)
       py q902k_family_readout.py --selftest            (controls on synthetic directories with 13 states; exit 1 if any fails)
Exit code: 0 PASS, 1 FAIL, 2 VOID."""

import os, sys, tempfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))


def read_one(d, K, a):
    p = os.path.join(d, f"q924b_K{K}_A{a}_out.txt")
    if not os.path.exists(p):
        return "MISSING", "no file"
    lines = [t for t in open(p, encoding="utf-8", errors="replace").read().splitlines() if t.strip()]
    if not lines:
        return "UNREADABLE", "empty file"
    last = lines[-1]
    if last.startswith("VERDICT q924b") and ": PASS;" in last:
        return "PASS", last
    if last.startswith("VERDICT q924b") and ": FAIL;" in last:
        return "FAIL", last
    if last.startswith("VOID"):
        return "VOID", last
    return "UNREADABLE", last[:120]


def family(d, K, states):
    if not states:
        return "VOID", {"EMPTY": 1}, []
    res = [(a,) + read_one(d, K, a) for a in states]
    cnt = {}
    for _, k, _ in res:
        cnt[k] = cnt.get(k, 0) + 1
    if cnt.get("MISSING", 0) or cnt.get("UNREADABLE", 0) or cnt.get("VOID", 0):
        v = "VOID"
    elif cnt.get("FAIL", 0):
        v = "FAIL"
    elif cnt.get("PASS", 0) == len(states):
        v = "PASS"
    else:
        v = "VOID"
    return v, cnt, res


def selftest():
    states = ["35.%04d" % (5000 + 100 * i) for i in range(13)]
    K = 50

    def mk(spec):
        d = tempfile.mkdtemp(prefix="q926_self_", dir=HERE)
        for a in states:
            kind = spec.get(a, "PASS")
            if kind == "MISSING":
                continue
            line = {
                "PASS": "VERDICT q924b f.json: PASS; K1 True",
                "FAIL": "VERDICT q924b f.json: FAIL; K2 False",
                "VOID": "VOID q924b f.json: instrument controls failed",
                "JUNK": "Traceback",
            }[kind]
            open(os.path.join(d, f"q924b_K{K}_A{a}_out.txt"), "w").write("[0s] x\n" + line + "\n")
        return d

    cases = [
        ("S1 all PASS gives PASS", {}, "PASS"),
        ("S2 one FAIL gives FAIL", {states[3]: "FAIL"}, "FAIL"),
        ("S3 one missing gives VOID", {states[0]: "MISSING"}, "VOID"),
        ("S4 one VOID gives VOID", {states[12]: "VOID"}, "VOID"),
        ("S5 one junk last line gives VOID", {states[5]: "JUNK"}, "VOID"),
        ("S6 FAIL and VOID together give VOID", {states[3]: "FAIL", states[12]: "VOID"}, "VOID"),
    ]
    bad = 0
    for name, spec, want in cases:
        d = mk(spec)
        v, cnt, _ = family(d, K, states)
        shutil.rmtree(d, ignore_errors=True)
        ok = v == want
        bad += 0 if ok else 1
        print(("PASS  " if ok else "FAIL  ") + name + f"  got {v}")
    d = tempfile.mkdtemp(prefix="q926_self_", dir=HERE)
    v, cnt, _ = family(d, K, states)
    shutil.rmtree(d, ignore_errors=True)
    ok = v == "VOID"
    bad += 0 if ok else 1
    print(("PASS  " if ok else "FAIL  ") + f"S7 empty directory gives VOID  got {v}")
    v, cnt, _ = family(HERE, K, [])
    ok = v == "VOID"
    bad += 0 if ok else 1
    print(("PASS  " if ok else "FAIL  ") + f"S8 empty state list gives VOID  got {v}")
    print(f"SELFTEST q926: {8 - bad} of 8 pass")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        selftest()
    K = int(sys.argv[1])
    lst = sys.argv[2]
    d = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "scratch")
    states = [l.split()[0] for l in open(lst).read().splitlines() if l.strip()]
    v, cnt, res = family(d, K, states)
    for a, k, line in res:
        print(f"A {a}: {k}  {line[:230]}")
    print(f"FAMILY VERDICT q926b K {K}: {v}; counts {cnt}; states expected {len(states)}")
    sys.exit({"PASS": 0, "FAIL": 1, "VOID": 2}[v])
