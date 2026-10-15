"""q1417 (9 Oct 2026, GATE, registered in amendment 2155 BEFORE any family above K = 84 has been decomposed): does the ratio P/N of the innermost flow (and the edge-only
ratio R_e) of the capacity families, taken at the stage f0.97 for K = 62 up to K_max, prefer a limit above 1 for K -> infinity, or is a limit of exactly 1 not excluded?
Data: K = 62..82 from the dumps q1388_jb_K<K>_f0.97.json (analysed by the function of q1416), K >= 83 from q1416_rows.json. Even and odd K are fitted separately.
Fits (q1403): forced 1 + a K^-p (a, p free) and free L + a K^-p (all free), least squares. Rule registered in advance: with K_max >= 98 for both parities (the largest even K below 99 is 98), the verdict
LIMIT_ABOVE_ONE needs, for each of the four sequences (P/N and R_e, even and odd), rms(forced) >= 5 rms(free) AND free L > 1.01; otherwise LIMIT_ONE_NOT_EXCLUDED.
With K_max below 98 (either parity) the verdict is INCOMPLETE. Controls (they must pass or the verdict is VOID): on the same K grid a synthetic 1 + 3 K^-0.5 with Gaussian noise 1e-5 is fitted by
the forced fit with rms < 3e-5 and judged not above one, and a synthetic 1.04 + K^-0.7 with the same noise is rejected by the forced fit (rms(forced) >= 5 rms(free) and
rms(forced) > 3e-5): the instrument must be able to tell the two cases on the actual K grid.
Usage: py q1417_limit_gate.py [--registration-run]  (from temporal_within/small_gap_q206/). The last line is VERDICT q1417: ...; exit 0 (verdict issued), 3 (INCOMPLETE), 2 (VOID).
"""

import json, os, sys
import numpy as np
from scipy.optimize import least_squares


def forced(K, Y, L=1.0):
    best = None
    for p0 in (0.1, 0.3, 0.6, 1.0, 2.0):
        r = least_squares(lambda q: L + q[0] * K ** (-q[1]) - Y, [(Y[0] - L) * K[0] ** p0, p0])
        rms = float(np.sqrt(np.mean(r.fun**2)))
        if best is None or rms < best[0]:
            best = (rms, r.x[0], r.x[1])
    return best


def free(K, Y):
    best = None
    for p0 in (0.3, 0.7, 1.5):
        for L0 in (0.9 * Y.min(), 1.0, Y.min() - 0.01):
            r = least_squares(
                lambda q: q[0] + q[1] * K ** (-abs(q[2])) - Y, [L0, (Y[0] - L0) * K[0] ** p0, p0]
            )
            rms = float(np.sqrt(np.mean(r.fun**2)))
            if best is None or rms < best[0]:
                best = (rms, r.x[0], r.x[1], abs(r.x[2]))
    return best


def judge_seq(K, Y):
    f1 = forced(K, Y)
    f2 = free(K, Y)
    ratio = f1[0] / max(f2[0], 1e-300)
    return dict(
        rms_forced=f1[0],
        rms_free=f2[0],
        ratio=ratio,
        L=f2[1],
        p_free=f2[3],
        above=(ratio >= 5 and f2[1] > 1.01),
    )


def judge(rows, kmax_required=98):
    """rows: list of dicts with K, parity, PN, Re. returns (lines, verdict)"""
    lines = []
    incomplete = False
    allabove = True
    for par in ("even", "odd"):
        sel = sorted([r for r in rows if r["parity"] == par], key=lambda r: r["K"])
        K = np.array([r["K"] for r in sel], float)
        if len(sel) < 8 or K.max() < kmax_required:
            incomplete = True
            lines.append(
                f"{par}: K up to {int(K.max()) if len(K) else 0}, {len(sel)} rows: below the registered K_max {kmax_required}"
            )
            continue
        for nm in ("PN", "Re"):
            j = judge_seq(K, np.array([r[nm] for r in sel]))
            allabove = allabove and j["above"]
            lines.append(
                f"{par} {nm}: K {int(K.min())}..{int(K.max())}, rms forced {j['rms_forced']:.2e}, free {j['rms_free']:.2e}, ratio {j['ratio']:.1f}, free L {j['L']:.5f} (p {j['p_free']:.2f}) -> above one: {j['above']}"
            )
    if incomplete:
        return lines, "INCOMPLETE"
    return lines, ("LIMIT_ABOVE_ONE" if allabove else "LIMIT_ONE_NOT_EXCLUDED")


def controls(Kgrid):
    ok = True
    msgs = []
    for par_off in (0, 1):
        K = np.array([k for k in Kgrid if k % 2 == par_off], float)
        if len(K) < 8:
            continue
        rng = np.random.default_rng(7 + par_off)
        a = 1 + 3 / K**0.5 + rng.normal(0, 1e-5, len(K))
        fa = forced(K, a)
        ja = judge_seq(K, a)
        b = 1.04 + 1 / K**0.7 + rng.normal(0, 1e-5, len(K))
        jb = judge_seq(K, b)
        c1 = fa[0] < 3e-5 and not ja["above"]
        c2 = jb["ratio"] >= 5 and jb["rms_forced"] > 3e-5 and jb["above"]
        msgs.append(
            f"parity {par_off}: K {int(K.min())}..{int(K.max())}: synthetic limit 1: forced rms {fa[0]:.1e}, judged not above one: {c1}; synthetic limit 1.04: ratio {jb['ratio']:.1f}, forced rms {jb['rms_forced']:.1e}, judged above one: {c2}"
        )
        ok = ok and c1 and c2
    return ok, msgs


def load_rows():
    sys.path.insert(0, ".")
    import q1416_stage_row as Q

    cache = "q1417_rows_old.json"
    if os.path.exists(cache):
        old = json.load(open(cache))
    else:
        old = []
        for K in range(62, 83):
            r = Q.analyse(f"q1388_jb_K{K}_f0.97.json")
            old.append(
                dict(K=r["K"], parity=r["parity"], PN=r["PN"], Re=r["Re"], rho=r["rho"], flow=r["flow"])
            )
        json.dump(old, open(cache, "w"))
    new = json.load(open("q1416_rows.json")) if os.path.exists("q1416_rows.json") else []
    return old + [
        dict(K=r["K"], parity=r["parity"], PN=r["PN"], Re=r["Re"], rho=r["rho"], flow=r["flow"])
        for r in new
        if r["K"] >= 83 and r.get("control_ok", False)
    ]


def main():
    try:
        rows = load_rows()
    except Exception as e:
        print("VOID: input unreadable:", repr(e))
        print("VERDICT q1417: VOID")
        sys.exit(2)
    Kgrid = sorted(set(r["K"] for r in rows))
    Kgrid = list(range(62, max(Kgrid) + 1)) if Kgrid else []
    ok, msgs = controls(Kgrid if Kgrid and max(Kgrid) >= 99 else list(range(62, 100)))
    for m in msgs:
        print("control:", m)
    if not ok:
        print("VERDICT q1417: VOID")
        sys.exit(2)
    if "--registration-run" in sys.argv:
        print("controls only")
        sys.exit(0)
    lines, verdict = judge(rows)
    for l in lines:
        print(l)
    print(f"VERDICT q1417: {verdict}")
    sys.exit(3 if verdict == "INCOMPLETE" else 0)


if __name__ == "__main__":
    main()
