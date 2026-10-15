"""q1482 (9 Oct 2026, INSTRUMENT, no item, no verdict): the verified GPU walker of the GPU work (gpu_lab/gpu_walker_verified.py: mixed Newton with polish to the 192-bit residual floor, arb residual of every accepted state,
arb re-evaluation of the signal at both ends of every final bracket) run along the capacity family from the last stored state of family K0 through the following events up to A_end. It imports those modules read-only.
Validated on the 11 stored events K_b 104 to 114 (A_event equal to the stored value to the 5e-19 resolution, PROOF_TRACKS addendum 2 to reading 19). Blind rule: the whole output of the walker (stdout and files) goes to OUTDIR; never read the log or the event files by eye;
count events with ls | grep -c event_Kb and compare to the chain events by script (difference only).
Usage: gpu_lab/venv/Scripts/python.exe q1482_gpu_walk_capacity.py K0 A_END OUTDIR [--test]   (K0 = 114 starts from the last stored state of family 114 below the stored event K_b 114; restart: if OUTDIR holds states the walk resumes from the
latest state of the highest family)."""

import sys, os, json, glob, io, contextlib, time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(os.path.dirname(HERE), "gpu_lab")
sys.path.insert(0, os.path.join(LAB, "src"))
sys.path.insert(0, LAB)
from flint import arb, ctx
import q1411_fast_walk as QW
from gpu_walker_verified import VerifiedGPUWalker
import gpu_walker as GW

K0 = int(sys.argv[1])
AEND = float(sys.argv[2])
OUT = os.path.abspath(sys.argv[3])
os.makedirs(OUT, exist_ok=True)
PROD = os.path.join(HERE, "fw_prod")
DIRS = [PROD] + [os.path.join(HERE, f"fw_cap_par_chain{c}") for c in (104, 106, 108, 110, 112, 114)]
DIRS += [
    os.path.join(HERE, x) for x in os.environ.get("GW_EXTRA_DIRS", "").split(",") if x
]  # added 10 Oct 10:4x: extra directories with stored states (for K0 above 114); empty by default, behaviour unchanged


def states(d, K):
    return sorted(
        (
            (float(f.split("_A")[1][:-5]), f)
            for f in glob.glob(os.path.join(d, f"results_q1411_gauss_K{K}_A*.json"))
        )
    )


mine = []
for f in glob.glob(os.path.join(OUT, "results_q1411_gauss_K*_A*.json")):
    mine.append((int(f.split("_K")[1].split("_A")[0]), float(f.split("_A")[1][:-5]), f))
if mine:
    mine.sort()
    sf = mine[-1][2]
    print("resume from", os.path.basename(sf))
else:
    ev = None
    for d in DIRS:
        f = os.path.join(d, f"event_Kb{K0}.json")
        if os.path.exists(f):
            ev = json.load(open(f))
            break
    assert ev is not None, "stored event of the start family not found"
    c = [x for d in DIRS for x in states(d, K0) if x[0] <= float(ev["A_event"]) - 0.003]
    c.sort()
    sf = c[-1][1]
print("start state", os.path.basename(sf), "A_end", AEND, "out", OUT, flush=True)
ctx.prec = 192
GW.CNT.update(dict(J_builds=0, res_only=0, tJ=0.0, tres=0.0, solves=0, tsolve=0.0))
t0 = time.perf_counter()
buf = open(os.path.join(OUT, "walk.log"), "a", buffering=1)
with contextlib.redirect_stdout(buf):
    W = VerifiedGPUWalker(
        192,
        OUT,
        H=0.05,
        polish=True,
        compare_arb_update=False,
        polish_residual="arb",
        backend=os.environ.get("GW_BACKEND", "dd"),
    )
    A, u, w, we, wc = QW.load(sf, W.F)
    u, w, we, wc, res, k = W.solve_at(A, u, w, we, wc)
    st = (u, w, we, wc)
    Acur = A
    chunk = 0
    while float(Acur) < AEND - 1e-9:
        out = W.run(Acur, st, min(AEND, float(Acur) + 0.25), hmax=0.03)
        W.save_verify()
        chunk += 1  # verify.json after every chunk of 0.25 in A (events of the chunk included)
        if out[0] != "end":
            break
        _, Acur, st = out
        print(
            f"chunk {chunk} done, events so far {len(glob.glob(os.path.join(OUT, 'event_Kb*.json')))}",
            file=sys.__stdout__,
            flush=True,
        )
    W.save_verify()
wall = time.perf_counter() - t0
nev = len(glob.glob(os.path.join(OUT, "event_Kb*.json")))
print(
    f"done: wall {wall:.0f} s, events written {nev}, newton calls {W.V['newton_calls']}, fails {W.V['newton_fail']}, max arb residual {W.V['max_arb_res']:.1e} (counts only; amplitudes are not printed)",
    flush=True,
)
