"""q1072 (7 Oct 2026): verdict script of T106 (registered in the amendment that follows 1822; a copy of q1065 for the odd family K 71): the sign regularity of T101b at five states of the family K 71 (m = 34): the first, the quarter (index 26), the median (index 52), the three-quarter (index 78) and the last (index 103) of the 104 OPTIMAL states of tower/K71 (none of them was processed by q1058 to q1065).
Matrix, classes, reference signs (those of the control state K 65, A = 43.5803), windows (first row index and first column index at least 6, band |i - j| <= k + 2, orders 1 to 4) and the controls PC, NC1 and NC2 are those of q1065 (T102).
Verdict: PASS if at every one of the 5 states every class of the orders 1 to 4 has all its minors of the reference sign and no minor is undecided; FAIL if the states are processed and some minor has the other sign or is undecided; VOID if a state cannot be processed or a control misbehaves.
Reading only (no criterion): the minority counts of the orders 5 and 6 at each state.
Usage: py q1072_sign_regularity_k71_verdict.py controls | py q1072_sign_regularity_k71_verdict.py
Last line: VERDICT q1072 T106: PASS / FAIL / VOID / INCOMPLETE."""

import sys, os
import q1058_extended_jacobian_sign_classes as Q58

IMIN = 6
CONTROL = "tower/K65/results_q220_gauss_K65_A43.5803.json"


def reference(cls):
    ref = {}
    for key, (p, n, u) in cls.items():
        if key[0] > 4:
            continue
        if u != 0 or (p > 0 and n > 0) or (p == 0 and n == 0):
            return None
        ref[key] = 1 if p > 0 else -1
    return ref


def judge(cls, ref):
    bad = 0
    undec = 0
    for key, sg in ref.items():
        p, n, u = cls.get(key, [0, 0, 0])
        undec += u
        bad += n if sg > 0 else p
        if key not in cls:
            bad += 1
    return bad, undec


def main():
    m, M = Q58.extended_matrix(CONTROL)
    cls = Q58.signs_by_class(m, M, 6, IMIN)
    ref = reference(cls)
    if ref is None:
        print("VERDICT q1072 T106: VOID (the control state does not have pure classes at orders 1 to 4)")
        sys.exit(2)
    pc = judge(cls, ref)
    k1 = sorted(ref)[0]
    ref1 = dict(ref)
    ref1[k1] = -ref[k1]
    nc1 = judge(cls, ref1)
    M2 = [list(r) for r in M]
    M2[10], M2[12] = M2[12], M2[10]
    cls2 = Q58.signs_by_class(m, M2, 4, IMIN)
    nc2 = judge(cls2, ref)
    print(
        f"control {CONTROL}: m {m}, classes {len(ref)}, PC violations {pc[0]}, undecided {pc[1]}; NC1 (class {k1} inverted) violations {nc1[0]}; NC2 (rows 10 and 12 exchanged) violations {nc2[0]}",
        flush=True,
    )
    if not (pc == (0, 0) and nc1[0] > 0 and nc2[0] > 0):
        print("VERDICT q1072 T106: VOID (a control misbehaves)")
        sys.exit(2)
    if len(sys.argv) > 1 and sys.argv[1] == "controls":
        print("VERDICT q1072 T106: INCOMPLETE (controls run only)")
        return
    paths = [
        "tower/K71/results_q220_gauss_K71_A46.4359.json",
        "tower/K71/results_q220_gauss_K71_A46.5589.json",
        "tower/K71/results_q220_gauss_K71_A46.6889.json",
        "tower/K71/results_q220_gauss_K71_A46.8189.json",
        "tower/K71/results_q220_gauss_K71_A46.9439.json",
    ]
    nfail = 0
    for p in paths:
        try:
            m, M = Q58.extended_matrix(p)
            cls = Q58.signs_by_class(m, M, 6, IMIN)
        except Exception as e:
            print(f"VERDICT q1072 T106: VOID (state {p} not processed: {e})")
            sys.exit(2)
        bad, undec = judge(cls, ref)
        nfail += bad > 0 or undec > 0
        minor56 = sum(min(c[0], c[1]) for k, c in cls.items() if k[0] >= 5)
        print(
            f"{os.path.basename(p)}: m {m}, violations (orders 1-4) {bad}, undecided {undec}, minority minors at orders 5 and 6 {minor56}, {'PASS' if bad == 0 and undec == 0 else 'FAIL'}",
            flush=True,
        )
    print(f"states 5, states failing {nfail}")
    print("VERDICT q1072 T106: " + ("PASS" if nfail == 0 else "FAIL") + f"; states 5; failing {nfail}")


if __name__ == "__main__":
    main()
