"""q1294 diagnostic of control c5 (8 Oct 2026, no item): per exponent and event the difference quintic - septic a_med relative to R_pred (the registration pass printed only the maximum, 0.00934, and VOID)."""

import io, contextlib, os, sys

os.environ["E_FILE"] = "d22_out_from_log.json"
sys.argv = ["x"]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import q1294_closure_linear_k_blind as M
Q, Z, W = M.Q, M.Z, M.W
for k in M.STAGE:
    N5, _ = M.nh(k, 5)
    N7, _ = M.nh(k, 7)
    N9, _ = M.nh(k, 9)
    row = []
    for K in M.TARGET:
        am = Q.root(N5, K)
        v2 = Z.v2f(N5, am)
        rp = M.rpred(k, v2)
        row.append((K, abs(am - Q.root(N7, K)) * 1e9 / rp, abs(am - Q.root(N9, K)) * 1e9 / rp))
    print(f"k {k}: " + "; ".join(f"{K}: {a:.4f}/{b:.4f}" for K, a, b in row))
    print(
        f"   max q5-q7 {max(r[1] for r in row):.5f} at K_b {max(row, key=lambda r: r[1])[0]}; max q5-q9 {max(r[2] for r in row):.5f} at K_b {max(row, key=lambda r: r[2])[0]}"
    )
