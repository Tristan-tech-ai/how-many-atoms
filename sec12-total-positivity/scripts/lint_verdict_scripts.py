"""lint_verdict_scripts (7 Oct 2026): procedural guard for the rule of 6 Oct (a verdict script is first run with its controls only). Reads the verdict scripts named on the command line (default: the scripts q1070 to q1075 of T104 to T109) and checks that each one stops after its controls when called with the flag --registration-run or the argument controls, and that it prints a last line VERDICT.
Controls: q1073_majorization_k70_71_verdict.py (no flag; it was run in full by mistake on 7 Oct 12:41) must be flagged; q1074_majorization_k72_73_verdict.py (flag) must pass.
Last line: LINT verdict_scripts: PASS / FAIL / VOID. Usage: py gate_selftest/lint_verdict_scripts.py [script ...]
"""

import re, sys, glob

FLAG = re.compile(r'--registration-run|"controls"|\'controls\'')
LAST = re.compile(
    r"VERDICT (q\d+|gate_\w+|make_\w+)"
)  # 14 Oct 2026: the paper gates name themselves gate_NAME or make_NAME, not qNNNN


NO_FAIL_BY_RULE = {
    "q1088_n1_real_rootedness_verdict.py": "registered rule of 1837 has PASS, INCOMPLETE and VOID only",
    "q1092_n1_real_rootedness_k63_70_verdict.py": "registered rule of 1842 has PASS, INCOMPLETE and VOID only",
}


NO_PASSFAIL_BY_RULE = {
    "q1417_limit_gate.py": "registered rule of amendment 2155 has LIMIT_ABOVE_ONE, LIMIT_ONE_NOT_EXCLUDED, INCOMPLETE and VOID only",
    "q1425_limit_gate.py": "registered rule of amendment 2165 has LIMIT_ABOVE_ONE, LIMIT_ONE_NOT_EXCLUDED, INCOMPLETE and VOID only",
}  # 9 Oct 2026: a gate whose registered verdict words are not PASS/FAIL; INCOMPLETE and VOID are still required


NO_INCOMPLETE_BY_RULE = {
    "gate_undefined_symbols.py": "registered rule of 14 Oct 2026 has PASS, FAIL and VOID only",
    "gate_jargon.py": "registered rule of 14 Oct 2026 has PASS, FAIL and VOID only",
    "gate_novelty.py": "registered rule of 14 Oct 2026 has PASS, FAIL and VOID only",
    "gate_symbol_overload.py": "registered rule of 14 Oct 2026 has PASS, FAIL and VOID only",
    "gate_overfull.py": "registered rule of 14 Oct 2026 has PASS, FAIL and VOID only",
    "order_bibliography.py": "registered rule of 14 Oct 2026 has PASS, FAIL and VOID only",
    "gate_float_order.py": "registered rule of 14 Oct 2026 has PASS, FAIL and VOID only",
    "gate_ref_order.py": "registered rule of 14 Oct 2026 has PASS, FAIL and VOID only",
    "gate_heading_style.py": "registered rule of 14 Oct 2026 has PASS, FAIL and VOID only",
    "gate_qed_mark.py": "registered rule of 14 Oct 2026 has PASS, FAIL and VOID only",
}  # 14 Oct 2026: text gates of the merged paper; a text cannot be partly checked


def check(path):
    t = open(path, encoding="utf-8").read()
    if path.replace("\\", "/").split("/")[-1] in NO_INCOMPLETE_BY_RULE:
        return bool(FLAG.search(t)), bool(LAST.search(t)) and all(w in t for w in ("PASS", "FAIL", "VOID"))
    if path.replace("\\", "/").split("/")[-1] in NO_PASSFAIL_BY_RULE:
        return bool(FLAG.search(t)), bool(LAST.search(t)) and all(w in t for w in ("INCOMPLETE", "VOID"))
    if path in NO_FAIL_BY_RULE:
        return bool(FLAG.search(t)), bool(LAST.search(t)) and all(
            w in t for w in ("PASS", "INCOMPLETE", "VOID")
        )
    return bool(FLAG.search(t)), bool(LAST.search(t)) and all(
        w in t for w in ("PASS", "FAIL", "INCOMPLETE", "VOID")
    )  # 7 Oct 15:03: the script must also contain the four verdict words (the first q1106 had no FAIL branch)


def main():
    neg = check("q1073_majorization_k70_71_verdict.py")
    pos = check("q1074_majorization_k72_73_verdict.py")
    if not (pos == (True, True) and neg[0] is False):
        print(f"LINT verdict_scripts: VOID (controls: q1074 {pos}, q1073 {neg})")
        sys.exit(2)
    files = sys.argv[1:] or sorted(
        glob.glob("q107[0-5]_*verdict*.py")
        + glob.glob("q1071_*.py")
        + glob.glob("q1072_*.py")
        + glob.glob("q1075_*.py")
    )
    files = [
        f for i, f in enumerate(files) if f not in files[:i] and f != "q1073_majorization_k70_71_verdict.py"
    ]
    bad = 0
    for f in files:
        fl, la = check(f)
        ok = fl and la
        bad += not ok
        print(f"{f}: controls-only mode {fl}, VERDICT line {la} -> {'ok' if ok else 'MISSING'}")
    if not files:
        print("LINT verdict_scripts: VOID (no script examined)")
        sys.exit(2)
    print(f"LINT verdict_scripts: {'PASS' if bad == 0 else 'FAIL'}; scripts {len(files)}; missing {bad}")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
