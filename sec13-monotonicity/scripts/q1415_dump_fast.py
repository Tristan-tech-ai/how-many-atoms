"""q1415 (9 Oct 2026, INSTRUMENT, no item, no verdict): fast version of q1388_dump_jb.py. py q1415_dump_fast.py STATEFILE A_TARGET LABEL OUTFILE: load a state written by q1411 (or a tower state), re-solve it at A_TARGET (band evaluation, step 0.02, tolerance 1e-45; the discretisation of the record), compute J and b = dF/dA (central difference 1e-25 at fixed grid) and write the dump in the format of q1388 (J and b as 45-digit strings) so that q1389, q1393, q1395, q1398, q1400 read it unchanged.
Control: with STATEFILE = the K = 62 state of q1388 (A = 42.258175) and A_TARGET = the same A, the dump agrees with q1388_jb_K62_end.json to 1e-15 relative per entry of J (max over the row maximum) and in b. The bar is 1e-15 and not smaller because the q1388 dump was taken at the stored state, which solves the system at the double nearest to A (offset up to 3.5e-15), while q1415 re-solves at the decimal A; the first version of this control had the bar 1e-30, failed with 4.4e-19, and the bar was corrected on that ground.
"""

import sys, json, time, os

DIGITS = int(
    os.environ.get("Q1415_DIGITS", "56")
)  # digits of the strings in the dump (45 as in q1388; 56 for the repair of amendment 2159)
from flint import arb
from q1411_fast_walk import Walker, load, S


def dump(statefile, A_target, label, outfile, H=0.02):
    W = Walker(192, "fw_dump", H=H)
    F = W.F
    A0, u, w, we, wc = load(statefile, F)
    A = arb(A_target)
    if abs(float((A - A0).mid())) > 1e-12:
        m = len(u)
        odd = wc is not None
        dz, J = W.tangent(A0, u, w, we, wc)
        v0 = W.pack(u, w, we, wc)
        vp = [a + (A - A0) * d for a, d in zip(v0, dz)]
        u, w, we, wc = W.unpack(vp, m, odd)
    u, w, we, wc, res, k = F.newton(A, u, w, we, wc, tol="1e-45", steps=40)
    if not res < arb("1e-40"):
        raise RuntimeError(f"newton did not converge: {float(res)}")
    y = F.grid(A)
    r0, J = F.residual_and_jacobian(A, u, w, we, wc, y=y)
    dl = arb("1e-25")
    rp, _ = F.residual_and_jacobian(A + dl, u, w, we, wc, want_J=False, y=y)
    rm, _ = F.residual_and_jacobian(A - dl, u, w, we, wc, want_J=False, y=y)
    b = [(a - c) / (2 * dl) for a, c in zip(rp, rm)]
    m = len(u)
    odd = wc is not None
    n = J.nrows()
    K = 2 * m + 2 + (1 if odd else 0)
    rec = dict(
        K=K,
        label=label,
        A=S(A, DIGITS),
        m=m,
        odd=odd,
        u=[S(t, DIGITS) for t in u],
        w=[S(t, DIGITS) for t in w],
        we=S(we, DIGITS),
        wc=(None if wc is None else S(wc, DIGITS)),
        r0max=f"{float(res):.3e}",
        b=[S(t, DIGITS) for t in b],
        J=[[S(J[a, c], DIGITS) for c in range(n)] for a in range(n)],
    )
    json.dump(rec, open(outfile, "w"))
    return rec


if __name__ == "__main__":
    if sys.argv[1] == "control":
        t0 = time.time()
        ref = json.load(open("q1388_jb_K62_end.json"))
        import tempfile, os

        sf = os.path.join("..", "capacity_even62_family", "results_q220_gauss_K62_A42.2582.json")
        rec = dump(sf, ref["A"], "end", "q1415_control_dump.json")
        n = len(rec["J"])
        worst = 0.0
        wb = 0.0
        for a in range(n):
            rowmax = max(abs(float(arb(ref["J"][a][c]).mid())) for c in range(n))
            for c in range(n):
                worst = max(worst, abs(float((arb(rec["J"][a][c]) - arb(ref["J"][a][c])).mid())) / rowmax)
            bm = max(abs(float(arb(t).mid())) for t in ref["b"])
            wb = max(wb, abs(float((arb(rec["b"][a]) - arb(ref["b"][a])).mid())) / bm)
        ok = worst < 1e-15 and wb < 1e-15
        print(
            f"control q1415: K {rec['K']}, J worst {worst:.2e}, b worst {wb:.2e}, {time.time() - t0:.0f}s: {'OK' if ok else 'FAILED'}"
        )
        sys.exit(0 if ok else 2)
    rec = dump(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    print("written", sys.argv[4], "K", rec["K"], "A", rec["A"][:12], "residual", rec["r0max"])
