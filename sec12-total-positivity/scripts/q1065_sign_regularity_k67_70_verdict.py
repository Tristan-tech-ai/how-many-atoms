"""q1065 (7 Oct 2026): verdict script of T102 (registered in the amendment that follows 1815): the sign regularity of T101b at the first, the median and the last state of the families K 67, K 68, K 69 and K 70 (12 states, none of them in the list of T101b): sign regularity of the extended Jacobian of the capacity KKT system outside the nearest-neighbour band.
Matrix and classes as in q1058 (rows D'_1, D_1, ..., D'_m, D_m, D_A, columns u_1, w_1, ..., u_m, w_m, we; interior windows of k consecutive rows and columns, k = 1 to 4, whose first row index and first column index are at least 6 (the windows that start at one of the first three atoms are excluded); position above if j - i > k + 2, below if i - j > k + 2; class = (k, type of the first row d or v, type of the first column u or w, position)).
At the control state (K 65, A = 43.5803) every class of the orders 1 to 4 is pure (all minors of one certain sign); the sign of each class at the control is the reference sign. Verdict: PASS if at every one of the 12 states every class of the orders 1 to 4 has all its minors of the reference sign and no undecided minor; FAIL if the 12 states are processed and some minor has another sign or is undecided; VOID if a state cannot be polished or read, the list does not have 12 states, or a control misbehaves.
Reading only (no criterion): the minority counts of the orders 5 and 6 at each state.
Controls: PC (the control state judged against its own signs) passes; NC1 (the signs of one class inverted) fails; NC2 (the minors of the control with two rows of the matrix exchanged) fails.
Usage: py q1065_sign_regularity_k67_70_verdict.py controls | py q1065_sign_regularity_k67_70_verdict.py
Last line: VERDICT q1065 T102: PASS / FAIL / VOID / INCOMPLETE."""

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
        print("VERDICT q1065 T102: VOID (the control state does not have pure classes at orders 1 to 4)")
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
        print("VERDICT q1065 T102: VOID (a control misbehaves)")
        sys.exit(2)
    if len(sys.argv) > 1 and sys.argv[1] == "controls":
        print("VERDICT q1065 T102: INCOMPLETE (controls run only)")
        return
    paths = [
        "tower/K67/results_q220_gauss_K67_A44.3683.json",
        "tower/K67/results_q220_gauss_K67_A44.6248.json",
        "tower/K67/results_q220_gauss_K67_A44.8848.json",
        "tower/K68/results_q220_gauss_K68_A44.8898.json",
        "tower/K68/results_q220_gauss_K68_A45.1325.json",
        "tower/K68/results_q220_gauss_K68_A45.4025.json",
        "tower/K69/results_q220_gauss_K69_A45.4075.json",
        "tower/K69/results_q220_gauss_K69_A45.6625.json",
        "tower/K69/results_q220_gauss_K69_A45.9175.json",
        "tower/K70/results_q220_gauss_K70_A45.9225.json",
        "tower/K70/results_q220_gauss_K70_A46.1559.json",
        "tower/K70/results_q220_gauss_K70_A46.4309.json",
    ]
    nfail = 0
    for p in paths:
        try:
            m, M = Q58.extended_matrix(p)
            cls = Q58.signs_by_class(m, M, 6, IMIN)
        except Exception as e:
            print(f"VERDICT q1065 T102: VOID (state {p} not processed: {e})")
            sys.exit(2)
        bad, undec = judge(cls, ref)
        nfail += bad > 0 or undec > 0
        minor56 = sum(min(c[0], c[1]) for k, c in cls.items() if k[0] >= 5)
        print(
            f"{os.path.basename(p)}: m {m}, violations (orders 1-4) {bad}, undecided {undec}, minority minors at orders 5 and 6 {minor56}, {'PASS' if bad == 0 and undec == 0 else 'FAIL'}",
            flush=True,
        )
    print(f"states 12, states failing {nfail}")
    print("VERDICT q1065 T102: " + ("PASS" if nfail == 0 else "FAIL") + f"; states 12; failing {nfail}")


if __name__ == "__main__":
    main()
