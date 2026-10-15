"""q1416 (9 Oct 2026, READING tool, no verdict): for a completed family K of the production run (small_gap_q206/fw_prod) take the state at the stage f0.97 (97 percent of the way in A from the previous event to the end event of the family), dump J and b with q1415 (step 0.02), and compute in basis D1 of q1389 the flow of the innermost atom, P/N, the edge-only ratio R_e (six edge terms), rho = -c_1/E, the cancellation ratio and the control (sum of contributions against the stored-system flow, bar 1e-30 times the cancellation ratio). One row is appended to q1416_rows.json (a list) unless the family K is already in it.
Usage: py q1416_stage_row.py K PREV_EVENT_KB   (the family K starts at the event K_b = K - 1 and ends at the event K_b = K; both event_Kb<n>.json files must exist in fw_prod or, for K_b <= 82, in the dirs of the controls).
"""

import sys, os, json, glob, subprocess
from mpmath import mp, mpf, matrix, lu_solve

mp.dps = (
    int(os.environ.get("Q1415_DIGITS", "56")) - 5
)  # 51 digits for dumps with 56 digits (amendment 2159); rows K 83-91 were made with 45-digit dumps analysed at 40 digits, controls met


def event_A(Kb):
    for d in ("fw_prod", "fw_birth82", "fw_split81", "fw_split63", "fw_ev_h005"):
        f = os.path.join(d, f"event_Kb{Kb}.json")
        if os.path.exists(f):
            return mpf(json.load(open(f))["A_event"])
    raise FileNotFoundError(f"event K_b {Kb}")


def analyse(fn):
    d = json.load(open(fn))
    m = d["m"]
    odd = d["odd"]
    n = 2 * m + 1 + (1 if odd else 0)
    idx = (2 * m + 1) if odd else 0
    J = matrix(n, n)
    for a in range(n):
        for c in range(n):
            J[a, c] = mpf(d["J"][a][c])
    b = [mpf(t) for t in d["b"]]
    ref = lu_solve(J, matrix([-t for t in b]))[idx]
    T = matrix(n, n)
    for a in range(n):
        T[a, a] = 1
    for a in range(1, m):
        T[a, 0] = -1
    if odd:
        T[2 * m + 1, 0] = -1
    Jd = T * J
    bd = T * matrix(b)
    e = matrix(n, 1)
    e[idx] = 1
    x = lu_solve(Jd.T, e)
    c = [-x[k] * bd[k] for k in range(n)]
    tot = sum(c)
    S = sum(abs(t) for t in c)
    P = sum(t for t in c if t > 0)
    N = -sum(t for t in c if t < 0)
    rel = abs(tot - ref) / abs(ref)
    ctl = rel < mpf("1e-30") * S / abs(tot)
    six = [c[j] for j in (m - 3, m - 2, m - 1)] + [c[m + j] for j in (m - 3, m - 2, m - 1)]
    Re = sum(t for t in six if t > 0) / -sum(t for t in six if t < 0)
    E = ref - c[0]
    return dict(
        K=d["K"],
        parity="odd" if odd else "even",
        A=d["A"][:14],
        flow=float(ref),
        PN=float(P / N),
        Re=float(Re),
        rho=float(-c[0] / E),
        cancel=float(S / abs(tot)),
        control_rel=float(rel),
        control_ok=bool(ctl),
    )


if __name__ == "__main__":
    K = int(sys.argv[1])
    if K >= 100 and not (
        os.path.exists("q1417_verdict.txt") and "VERDICT q1417: LIMIT" in open("q1417_verdict.txt").read()
    ):
        raise SystemExit(
            "registration of amendment 2162: no row above K = 99 before the verdict of q1417 is saved in q1417_verdict.txt (first run after the row K = 99)"
        )
    if K >= 108 and not (
        os.path.exists("q1425_verdict.txt") and "VERDICT q1425: LIMIT" in open("q1425_verdict.txt").read()
    ):
        raise SystemExit(
            "registration of amendment 2165: no row above K = 107 before the verdict of q1425 is saved in q1425_verdict.txt (first run after the rows K = 106 and K = 107)"
        )
    a0 = event_A(K - 1)
    a1 = event_A(K)
    At = a0 + mpf("0.97") * (a1 - a0)
    fs = sorted(
        glob.glob(
            os.path.join(os.environ.get("Q1416_STATES", "fw_prod"), f"results_q1411_gauss_K{K}_A*.json")
        ),
        key=lambda f: abs(float(f.split("_A")[1][:-5]) - float(At)),
    )
    if not fs:
        raise SystemExit(f"no states of K {K}")
    out = os.environ.get("Q1416_DUMP", f"q1415_jb_K{K}_f0.97.json")
    r = subprocess.run(
        [sys.executable, "q1415_dump_fast.py", fs[0], mp.nstr(At, 30), "f0.97", out],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit("dump failed")
    row = analyse(out)
    row.update(A_target=float(At), family_start=float(a0), family_end=float(a1))
    ROWSFILE = os.environ.get("Q1416_ROWSFILE", "q1416_rows.json")
    rows = json.load(open(ROWSFILE)) if os.path.exists(ROWSFILE) else []
    if any(x["K"] == K for x in rows):
        print("K", K, "already in q1416_rows.json; row not appended")
    else:
        rows.append(row)
        json.dump(rows, open(ROWSFILE, "w"), indent=1)
    print(json.dumps(row))
