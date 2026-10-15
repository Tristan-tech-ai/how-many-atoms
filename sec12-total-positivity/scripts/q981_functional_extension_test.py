"""q981 (7 Oct 2026): verdict script of T80 (amendment 1756): is the extension X = [T | v] strictly totally positive on the rows of the immune functional (results_q584.json) and of the NPMLE functional (results_q601.json), as on the capacity states?
Construction as in q880/q881/q886 and q980: re-polish a stored row at 200 bits (IMP.polish), Jacobian by central differences (step 1e-30), dF/da by central differences in a with the steps 1e-20 and 1e-24, masses eliminated, S = A - B Dm^-1 C, b_eff = b1 - B Dm^-1 b2, T = -E S E, v = sigma E b_eff with sigma = (-1)^(J+1) (J positions), minors at 800 bits from the midpoints.
Rows: every 40th row of the file starting at index 20 (immune 612 rows, NPMLE 544 rows; selected by code). Items per row: X1 all entries of v certainly positive for both steps of dF/da; X2 all J(J + 1) initial minors of X certainly positive for both steps; X3 the J maximal minors with v last certainly positive; X4 the opposite orientation fails X1;
X5 the relative difference of b_eff between the two steps is below 1e-20 (instrument check).
Verdict (last line, one per functional run): PASS if X1 to X5 hold at every selected row; FAIL if a processed row fails X1 to X4; VOID if X5 fails, a row does not polish (residual above 1e-40) or a control does not behave.
Controls (synthetic, 800 bits, J = 10): PC Gaussian kernel with the continuation column passes X1 to X3; NC1 the same with the sign of v[3] flipped fails X2; NC2 alternating v fails X1 and X2.
Usage: py q981_functional_extension_test.py immune|npmle [--controls-only]   (the environment variable Q981_DRY_ROW=N processes the single row N for an instrument check and prints no verdict)   (from temporal_within/)
"""

import json, sys
import numpy as np
from flint import arb, arb_mat, ctx
import q579_immune_mp as M
import q973_extended_tp_test_hi as Q


def _gs_npmle(self, p, v, c, ys):
    P = self.P(p, v, c)
    Cn = sum(self.Q)
    W = [q / (2 * Pi) for q, Pi in zip(self.Q, P)]
    out = []
    for y in ys:
        g0 = g1 = g2 = arb(0)
        for x, Wi in zip(self.x, W):
            dm, dp = x - y, x + y
            em, ep = M.f(dm), M.f(dp)
            g0 += Wi * (em + ep)
            g1 += Wi * (dm * em - dp * ep)
            g2 += Wi * ((dm * dm - 1) * em + (dp * dp - 1) * ep)
        out.append((g0 / Cn, g1 / Cn, g2 / Cn))
    return out, Cn


def controls(J=10):
    ctx.prec = 800
    K = [[(arb(-1) / 10 * ((p - q) ** 2)).exp() for q in range(J)] for p in range(J)]
    v = [(arb(-1) / 10 * ((p - J) ** 2)).exp() for p in range(J)]
    pc = Q.gates(K, v, J)
    v1 = list(v)
    v1[3] = -v1[3]
    n1 = Q.gates(K, v1, J)
    n2 = Q.gates(K, [v[p] * (1 if p % 2 == 0 else -1) for p in range(J)], J)
    c = {
        "PC": pc["X1"] and pc["X2"] and pc["X3"],
        "NC1": not n1["X2"],
        "NC2": (not n2["X1"]) and (not n2["X2"]),
    }
    ctx.prec = 200
    return c


def row_items(I, r, u0=None):
    ctx.prec = 200
    a = I.a if hasattr(I, "a") else None
    return None


def main():
    which = sys.argv[1]
    c = controls()
    print("controls:", c, flush=True)
    if not all(c.values()):
        print(f"VERDICT q981 T80 {which}: VOID (a control does not behave)")
        sys.exit(2)
    if "--controls-only" in sys.argv:
        print(f"VERDICT q981 T80 {which}: INCOMPLETE (no row processed)")
        sys.exit(3)
    if which == "npmle":
        M.IMP.gs = _gs_npmle
    rows = json.load(open("results_q584.json" if which == "immune" else "results_q601.json"))["rows"]
    sel = list(range(20, len(rows), 40))
    allok = True
    HJ = arb("1e-30")
    import os

    dry = os.environ.get("Q981_DRY_ROW")
    if dry:
        sel = [
            int(dry)
        ]  # instrument check on a row outside the selection; the output carries the label DRY RUN and no verdict
    for i in sel:
        ctx.prec = 200
        r = rows[i]
        a = arb(float(r["a"]))
        centre = bool(r["centre"])
        I = M.IMP(a, 9)
        pp, vv, res = I.polish([float(t) for t in r["p"]], [float(t) for t in r["v"]], centre)
        rr = abs(float(res.mid())) if hasattr(res, "mid") else abs(float(res))
        J = len(pp)
        u = list(pp) + (list(vv) if centre else list(vv[:-1]))
        n = len(u)
        m = n - J
        Jm = arb_mat(n, n)
        for k in range(n):
            up = list(u)
            um = list(u)
            up[k] += HJ
            um[k] -= HJ
            Fp = I.F(up, J, centre)
            Fm = I.F(um, J, centre)
            for q in range(n):
                Jm[q, k] = ((Fp[q] - Fm[q]) / (2 * HJ)).mid()
        A = arb_mat(J, J)
        B = arb_mat(J, m)
        C = arb_mat(m, J)
        Dm = arb_mat(m, m)
        for p in range(J):
            for q in range(J):
                A[p, q] = Jm[p, q]
            for q in range(m):
                B[p, q] = Jm[p, J + q]
        for p in range(m):
            for q in range(J):
                C[p, q] = Jm[J + p, q]
            for q in range(m):
                Dm[p, q] = Jm[J + p, J + q]
        Di = Dm.inv()
        S = A - B * Di * C
        be = {}
        for lab, HA in (("20", arb("1e-20")), ("24", arb("1e-24"))):
            Fp = M.IMP(a + HA, 9).F(u, J, centre)
            Fm = M.IMP(a - HA, 9).F(u, J, centre)
            b = [((Fp[k] - Fm[k]) / (2 * HA)).mid() for k in range(n)]
            b1 = arb_mat([[b[k]] for k in range(J)])
            b2 = arb_mat([[b[J + k]] for k in range(m)])
            be[lab] = b1 - B * Di * b2
        dev = max(abs(((be["20"][k, 0] - be["24"][k, 0]) / be["20"][k, 0]).mid()) for k in range(J))
        x5 = bool(dev < arb("1e-20"))
        ctx.prec = 800
        sg = 1 if J % 2 == 1 else -1
        Tm = [[S[p, q].mid() * (-1 if (p + q) % 2 == 0 else 1) for q in range(J)] for p in range(J)]
        its = []
        for lab in ("20", "24"):
            vv_ = [sg * be[lab][p, 0].mid() * (1 if p % 2 == 0 else -1) for p in range(J)]
            its.append(Q.gates(Tm, vv_, J))
        x1 = all(g["X1"] for g in its)
        x2 = all(g["X2"] and g["count"] == J * (J + 1) for g in its)
        x3 = all(g["X3"] for g in its)
        opp = [-sg * be["20"][p, 0].mid() * (1 if p % 2 == 0 else -1) for p in range(J)]
        x4 = not all(t > 0 for t in opp)
        ok = x1 and x2 and x3 and x4 and x5
        allok = allok and ok
        print(
            f"row {i}: a {float(a.mid()):.6f}, K {int(r['K'])}, centre {centre}, J {J}, polish residual {rr:.2e}: X1 {x1}, X2 {x2} ({its[0]['bad']} and {its[1]['bad']} not certainly positive), X3 {x3}, X4 {x4}, X5 {x5} (b_eff difference {float(dev.mid()):.2e})",
            flush=True,
        )
        if not x5 or rr > 1e-40:
            print(f"VERDICT q981 T80 {which}: VOID (instrument check at row {i})")
            sys.exit(2)
    if dry:
        print("END DRY RUN (no verdict)")
        sys.exit(0)
    print(f"VERDICT q981 T80 {which}: {'PASS' if allok else 'FAIL'}; rows {len(sel)}")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("VERDICT q981 T80: VOID (exception)")
        sys.exit(2)
