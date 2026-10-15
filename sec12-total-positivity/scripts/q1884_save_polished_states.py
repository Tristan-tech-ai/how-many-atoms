r"""q1884_save_polished_states.py (14 Oct 2026, task F8, reading 355): writes the polished state that q1882 uses as the centre of its box, for
every state of q1843_pr1b_states.txt, so that the statement "within 1e-110 of the polished state" of Proposition prop:exact can be checked by a
reader. The polish is the one of q1882 (and q1836, q1857): the stored state, 600 bits, the grid of step 0.01 from -A-40, Newton to 1e-135 with
at most 14 steps. Output: small_gap_q206/q1884_polished/<state>.json with A, K, the positions, the masses (60 significant digits each) and the
final residual, and a sha256 of the values. A reading with no verdict word; resumable (a state whose file exists is skipped); --shard i/n.
Registered before the first run: P1 every state polishes (ok True) with a residual below 1e-130; P2 at the control K = 40, A = 29.85 the first
position agrees with q1882's polished state to the printed digits (q1882 prints no positions, so P2 compares with a second polish in the same run
and checks determinism to 1e-150).
Usage: py small_gap_q206/q1884_save_polished_states.py [--shard i/n]   (from temporal_within/)"""

import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
SG = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(SG)
sys.path.insert(0, TW)
os.chdir(TW)
from flint import arb, ctx  # noqa: E402
import q920_cap_flint as F  # noqa: E402
import q921_cap_tools as T  # noqa: E402

OUT = os.path.join(SG, "q1884_polished")


def polish(state):
    ctx.prec = 600
    F.L2P = (arb.pi() * 2).log() / 2
    A_str, u, w, we, wc, d = F.load_state(state)
    m = len(u)
    odd = wc is not None
    hd0, h0 = F.HD, F.G.H
    F.HD = F.HD / 2
    F.G.H = F.G.H / 2
    try:
        cap = F.Cap(A_str, m, odd, grid_A=repr(float(A_str) + 31))
        v = [t.mid() for t in (u + w + [we] + ([wc] if odd else []))]
        v0, res, ok = T.newton_full(cap, v, lambda *x: None, tol=arb("1e-135"), maxit=14)
    finally:
        F.HD, F.G.H = hd0, h0
    return A_str, m, odd, v0, F.maxabs(res), ok


def main():
    os.makedirs(OUT, exist_ok=True)
    i, n = (0, 1)
    if "--shard" in sys.argv:
        i, n = (int(x) for x in sys.argv[sys.argv.index("--shard") + 1].split("/"))
    states = [
        l.strip() for l in open(os.path.join(SG, "q1843_pr1b_states.txt"), encoding="utf-8") if l.strip()
    ]
    for k, s in enumerate(states):
        if k % n != i:
            continue
        o = os.path.join(OUT, s.replace("/", "__").replace("\\", "__"))
        if os.path.exists(o):
            continue
        A_str, m, odd, v0, res, ok = polish(s)
        vals = [t.mid().str(60, radius=False) for t in v0]
        rec = {
            "state": s,
            "A": A_str,
            "positive_atoms": m,
            "center_atom": odd,
            "positions": vals[:m],
            "masses": vals[m : 2 * m],
            "wall_mass": vals[2 * m],
            "center_mass": vals[2 * m + 1] if odd else None,
            "residual": float(res),
            "ok": bool(ok),
            "sha256": hashlib.sha256("\n".join(vals).encode()).hexdigest(),
        }
        if s.endswith("results_q206_gauss_K40_A29.85.json"):
            _, _, _, v1, _, _ = polish(s)
            rec["determinism_max_diff"] = float(max(abs(a - b) for a, b in zip(v0, v1)).mid())
        json.dump(rec, open(o, "w", encoding="utf-8"), indent=1)
        print(s, "ok", ok, "residual %.3e" % float(res), flush=True)
    files = [f for f in os.listdir(OUT) if f.endswith(".json")]
    recs = [json.load(open(os.path.join(OUT, f), encoding="utf-8")) for f in files]
    p1 = len(recs) == len(states) and all(r["ok"] and r["residual"] < 1e-130 for r in recs)
    ctl = [r for r in recs if "determinism_max_diff" in r]
    p2 = bool(ctl) and ctl[0]["determinism_max_diff"] < 1e-150
    print(
        "states %d of %d written; P1 %s; P2 %s"
        % (
            len(recs),
            len(states),
            "held" if p1 else "not yet/did not hold",
            "held" if p2 else "not yet/did not hold",
        )
    )
    print(
        "READING (no verdict): polished centres of the boxes of q1882, written to small_gap_q206/q1884_polished/."
    )


if __name__ == "__main__":
    main()
