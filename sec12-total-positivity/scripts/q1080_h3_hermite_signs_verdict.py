"""q1080 (7 Oct 2026): verdict script of T112 (registered in the amendment that follows 1830): the signs of the consecutive minors of the extended Jacobian of the KKT system outside the nearest-neighbour band (hypothesis H3, amendments 1808 to 1817, 1824) are predicted by the Gram structure of paper_births/H2_PROOF_NOTE.md for the classes whose row window and column window are Hermite patterns.
Derivation. At a stationary configuration the extended Jacobian is minus one half of a positive row scaling times the Gram matrix G of the functions f = (psi_1, phi_1, psi_2, phi_2, ..., psi_m, phi_m, phi_A) in the weight 1/Q, plus the diagonal term D'' on the entries (psi_p, psi_p). Rows and columns have the same function list in the same interleaved order (d = derivative function, v or w = value function). A window of k consecutive rows and k consecutive columns outside the band |i - j| <= k + 2 contains no diagonal entry, so its minor is (-1/2)^k times a positive factor times a minor of G. The window is a Hermite pattern (a derivative function only together with the value function of its node) iff it starts with d and k is even, or starts with a value function and k is odd. Reordering a window from the interleaved order (psi_p, phi_p) to the natural Hermite order (phi_p, psi_p) costs a sign -1 per complete pair. A minor of G with Hermite patterns for rows and columns is positive (H2_PROOF_NOTE.md, Steps 1 and 2). So for such classes the sign of the minor is (-1)^k (-1)^(pr + pc), where pr and pc are the numbers of complete pairs in the row window and the column window (k/2 for a window starting with d, (k - 1)/2 for a window starting with a value function). For the predicted classes pr = pc, so the predicted sign is (-1)^k.
Classes: (k, first row type, first column type, above or below) for k = 1..4 as in q1058; windows start at row and column indices of at least 6 (as in T101b). Exactly one first row type and one first column type are Hermite for each k, so 8 of the 32 classes are predicted.
Items: P1 at every listed state and at the synthetic configurations, every minor of each of the 8 predicted classes has the predicted sign and none is undecided.
Verdict (last line): PASS if P1 holds at all 6 listed states; FAIL if a processed state violates P1; VOID if a state cannot be processed, a control does not behave or an exception occurs.
Controls: PC three synthetic configurations (random positions, masses and no KKT condition, m = 10 to 12, matrix -G in the interleaved order, no diagonal term) must satisfy P1; NC1 the prediction +1 for every predicted class (the factor (-1)^k dropped) must fail on the synthetic matrices; NC2 the prediction with the global sign reversed must fail.
States: tower/K65 A = 43.5803 and tower/K71 A = 46.4359, 46.5589, 46.6889, 46.8189 and 46.9439.
Usage: py q1080_h3_hermite_signs_verdict.py [--registration-run] (controls only with the flag)"""

import random, sys, os
from flint import arb, arb_mat, ctx
import q1058_extended_jacobian_sign_classes as Q58
import q920_cap_flint as F

IMIN = 6
STATES = ["tower/K65/results_q220_gauss_K65_A43.5803.json"] + [
    f"tower/K71/results_q220_gauss_K71_A{a}.json"
    for a in ("46.4359", "46.5589", "46.6889", "46.8189", "46.9439")
]


def hermite_start(k, t, first_is_derivative):
    """t is the first type; True if the window is a Hermite pattern, with the number of complete pairs."""
    if first_is_derivative:
        return (k % 2 == 0), k // 2
    return (k % 2 == 1), (k - 1) // 2


def predicted(key, with_pairs=True, flip=False):
    k, rt, ct, pos = key
    rh, pr = hermite_start(k, rt, rt == "d")
    ch, pc = hermite_start(k, ct, ct == "u")
    if not (rh and ch):
        return None
    s = (
        (-1) ** k * (-1) ** (pr + pc) if with_pairs else 1
    )  # with_pairs False is the control NC1: the factor (-1)^k is dropped (the pair factor is +1 for every predicted class because pr = pc)
    return -s if flip else s


def check(cls, with_pairs=True, flip=False):
    """returns (classes checked, violating minors, undecided minors)."""
    nck = bad = und = 0
    for key, (p, n, u) in cls.items():
        if key[0] > 4:
            continue
        s = predicted(key, with_pairs, flip)
        if s is None:
            continue
        nck += 1
        und += u
        bad += n if s > 0 else p
    return nck, bad, und


def synthetic(seed):
    ctx.prec = 1000
    F.L2P = (arb.pi() * 2).log() / 2
    rng = random.Random(seed)
    m = rng.randint(10, 12)
    u = [rng.uniform(0.3, 1.0)]
    for _ in range(m - 1):
        u.append(u[-1] + rng.uniform(0.5, 1.5))
    A = u[-1] + rng.uniform(0.5, 1.5)
    w = [rng.uniform(0.2, 1.0) for _ in range(m)]
    we = rng.uniform(0.2, 1.0)
    tot = 2 * sum(w) + 2 * we
    w = [x / tot for x in w]
    we /= tot
    cap = F.Cap(repr(A), m, False, grid_A=repr(A + 30.0))
    uu = [arb(x) for x in u]
    ww = [arb(x) for x in w]
    we_ = arb(we)
    rows, (SA, LA, PA), (SmA, LmA, PmA), cen, Qn, lQ = cap.tables(uu, ww, we_, None)
    N = cap.N
    Hh = arb(float(F.G.H))
    invQ = [1 / q for q in Qn]
    fs = []
    for p in range(m):
        fs.append([ww[p] * (rows[p][2][i] * rows[p][0][i] - rows[p][5][i] * rows[p][3][i]) for i in range(N)])
        fs.append([rows[p][2][i] + rows[p][5][i] for i in range(N)])
    fs.append([PA[i] + PmA[i] for i in range(N)])
    n = len(fs)
    Mx = [
        [-(Hh * sum((fs[a][i] * fs[b][i] * invQ[i] for i in range(N)), arb(0))) for b in range(n)]
        for a in range(n)
    ]
    ctx.prec = 400
    return m, [[x.mid() for x in r] for r in Mx]


def main():
    reg = "--registration-run" in sys.argv
    syn = [synthetic(20261007 + k) for k in range(3)]
    cl = [Q58.signs_by_class(m, M, 4, IMIN) for m, M in syn]
    pc = all(check(c)[0] == 8 and check(c)[1] == 0 and check(c)[2] == 0 for c in cl)
    nc1 = all(check(c, with_pairs=False)[1] > 0 for c in cl)
    nc2 = all(check(c, flip=True)[1] > 0 for c in cl)
    print(
        f"controls on {len(cl)} synthetic matrices (m {[m for m, _ in syn]}): PC {pc} (classes checked {[check(c)[0] for c in cl]}); NC1 (factor (-1)^k dropped) fails {nc1}; NC2 (global sign reversed) fails {nc2}",
        flush=True,
    )
    if not (pc and nc1 and nc2):
        print("VERDICT q1080 T112: VOID (a control does not behave)")
        sys.exit(2)
    if reg:
        print("VERDICT q1080 T112: INCOMPLETE (controls run only)")
        sys.exit(3)
    nfail = 0
    for p in STATES:
        try:
            m, M = Q58.extended_matrix(p)
            cls = Q58.signs_by_class(m, M, 4, IMIN)
        except Exception as e:
            print(f"VERDICT q1080 T112: VOID (state {p} not processed: {e})")
            sys.exit(2)
        nck, bad, und = check(cls)
        ok = nck == 8 and bad == 0 and und == 0
        nfail += not ok
        print(
            f"{os.path.basename(p)}: m {m}, predicted classes checked {nck}, violating minors {bad}, undecided {und}, {'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    print(f"VERDICT q1080 T112: {'PASS' if nfail == 0 else 'FAIL'}; states {len(STATES)}; failing {nfail}")
    sys.exit(0 if nfail == 0 else 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VERDICT q1080 T112: VOID (exception)")
        sys.exit(2)
