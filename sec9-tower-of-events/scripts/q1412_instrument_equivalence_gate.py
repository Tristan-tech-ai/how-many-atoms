"""q1412 (9 Oct 2026, GATE, written and selftested BEFORE any event above K_b 82 is measured with the new instrument): does the arb walker q1411
(small_gap_q206/, 192 bits, trapezoid step 0.05, grid half-length A + 9) measure the same event amplitudes as the tower driver of amendments 920 to 2051
on events whose tower values exist? Events and references: births K_b 62 and 82 (tower files tower_K63_birth_out.json and tower_K83_birth_out.json, key A_cross),
splits K_b 63 and 81 (tower_K64_As.json and tower_K82_As.json, key A_s). The walker writes small_gap_q206/<dir>/event_Kb<K>.json with the key A_event.
Bars (derived from the items they protect, not from the data): the smallest registered item bar is D1 with 0.3e-9 (amendment 1968); the instrument error must be
below a third of it, so births: |A_walker - A_tower| < 1e-10; splits: < 2e-10, the larger bar because the tower's split locator stopped at |D''(0)| < 1e-22
(termination residual up to 0.11e-9 at K_b 81, amendments 2001 to 2005). Required: K_b 62, 63 and 82 (K_b 81 is used when its event file exists).
Usage: py q1412_instrument_equivalence_gate.py [--registration-run]   (from temporal_within/centre_scripts/)
Output: one line per event; the last line is VERDICT q1412: PASS | FAIL | INCOMPLETE (exit 0, 1, 3); an input that cannot be read gives VOID (exit 2).
"""

import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = os.path.join(ROOT, "scratch")
EVENT_DIRS = {62: "fw_ev_h005", 82: "fw_birth82", 63: "fw_split63", 81: "fw_split81"}
REFS = {
    62: ("tower_K63_birth_out.json", "A_cross", "birth"),
    82: ("tower_K83_birth_out.json", "A_cross", "birth"),
    63: ("tower_K64_As.json", "A_s", "split"),
    81: ("tower_K82_As.json", "A_s", "split"),
}
BAR = {"birth": 1e-10, "split": 2e-10}
REQUIRED = (62, 63, 82)


def load_ref(K):
    f, key, kind = REFS[K]
    return float(json.load(open(os.path.join(ARCH, f)))[key]), kind


def load_event(K, base):
    f = os.path.join(base, EVENT_DIRS[K], f"event_Kb{K}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    if d.get("K_b") != K:
        raise ValueError(f"event file of K_b {K} holds K_b {d.get('K_b')}")
    return float(d["A_event"])


def judge(measured, refs, required=REQUIRED):
    """measured, refs: dict K -> (value, kind) / value; returns (lines, verdict)"""
    lines = []
    bad = False
    missing = False
    for K in sorted(set(required) | set(k for k in measured if measured[k] is not None)):
        m = measured.get(K)
        r = refs.get(K)
        if m is None:
            if K in required:
                missing = True
                lines.append(f"K_b {K}: event missing")
            continue
        val, kind = r
        d = m - val
        ok = abs(d) < BAR[kind]
        bad = bad or not ok
        lines.append(
            f"K_b {K} ({kind}): walker {m:.12f}, tower {val:.12f}, difference {d:+.3e}, bar {BAR[kind]:.0e}: {ok}"
        )
    verdict = "FAIL" if bad else ("INCOMPLETE" if missing else "PASS")
    return lines, verdict


def main():
    base = os.path.join(ROOT, "small_gap_q206")
    try:
        refs = {K: load_ref(K) for K in REFS}
        measured = {K: load_event(K, base) for K in REFS}
    except Exception as e:
        print("VOID: input unreadable:", repr(e))
        print("VERDICT q1412: VOID")
        sys.exit(2)
    # registration run: the judging function on synthetic inputs (the selftest covers corrupted inputs in more detail)
    perfect = {K: refs[K][0] for K in REFS}
    _, v1 = judge(perfect, refs)
    _, v2 = judge({K: refs[K][0] + 3e-10 for K in REFS}, refs)
    _, v3 = judge({K: (None if K == 63 else refs[K][0]) for K in REFS}, refs)
    ctl = v1 == "PASS" and v2 == "FAIL" and v3 == "INCOMPLETE"
    print(f"controls: perfect -> {v1}, all shifted by 3e-10 -> {v2}, K_b 63 missing -> {v3}: {ctl}")
    if not ctl:
        print("VERDICT q1412: VOID")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("controls only")
        sys.exit(0)
    lines, verdict = judge(measured, refs)
    for l in lines:
        print(l)
    print(f"VERDICT q1412: {verdict}")
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[verdict])


if __name__ == "__main__":
    main()
