"""q922 (7 Oct 2026): first state of the K 61 family of the capacity at the A of a stored K 60 state, by the born-centre continuation of q921 (instrument for T61, recorded in an amendment).
From the base K 60 state (centre signal positive, c = 0) the centre mass c is raised to the root of G0(c) = D(0) - D(A); the full odd state is then polished by Newton on all unknowns, checked with q206's signals (centre and off-atom signals not positive, atoms ordered, masses positive)
and saved as results_q220_gauss_K61_A<A>.json in the output directory (format of q220_walk_A.py). Last line 'RESULT q922: <verdict> ...'.
Usage: py q922_born_centre_run.py BASE_K60_STATE.json OUT_DIR"""

import os, sys, json, time
from flint import arb
from mpmath import mp, mpf, log, pi
import q920_cap_flint as F
import q921_cap_tools as T
import q206_gauss_mp as G

mp.dps = 40
G.L2P = log(2 * pi) / 2
t0 = time.time()
say = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
f, out_dir = sys.argv[1], sys.argv[2]
A_str, u, w, we, wc, d = F.load_state(f)
if wc is not None:
    sys.exit("base state must be a K 60 state without centre atom")
m = len(u)
capE = F.Cap(A_str, m, False)
capO = F.Cap(A_str, m, True)
say(
    f"base {f}: A = {A_str}, m = {m}, stored centre signal {d.get('centre_signal')}; odd Cap grid {capO.N} points"
)
v0 = u + w + [we]
res0, _, _ = capE.res_jac(v0, False, False)
say(f"base residual {F.maxabs(res0)}")
# the base state satisfies the K 60 rows to its stored residual; polish them in ball arithmetic at c = 0 first
vb, g0, ok = T.newton_fixed_c(capO, v0, arb(0), say)
say(f"c = 0 polished: {ok}, G0(0) = {g0.str(8, radius=False)}")
if not ok or g0 <= 0:
    sys.exit("base is not a K 60 solution with positive centre signal")
v, c = T.born_centre(capO, vb, say)
vfull = v + [c]
vfull, res, ok = T.newton_full(capO, vfull, say)
say(
    f"full odd Newton: converged {ok}, max |res| {F.maxabs(res)}, wc = {vfull[2 * m + 1].str(10, radius=False)}"
)
mu = [mpf(t.str(45, radius=False)) for t in vfull[:m]]
mw = [mpf(t.str(45, radius=False)) for t in vfull[m : 2 * m]]
mwe = mpf(vfull[2 * m].str(45, radius=False))
mwc = mpf(vfull[2 * m + 1].str(45, radius=False))
y = G.grid(mpf(A_str))
cen, off, where, C = G.signals(mpf(A_str), y, mu, mw, mwe, mwc)
ordered = all(mu[i] < mu[i + 1] for i in range(m - 1))
wmin = min(list(mw) + [mwe, mwc])
rmax = F.maxabs(res)
optimal = ok and ordered and wmin > 0 and cen <= 0 and off <= 0 and rmax < arb("1e-20")
verdict = (
    "OPTIMAL"
    if optimal
    else ("not converged" if not ok else ("centre positive" if cen > 0 else "off-atom positive"))
)
say(
    f"K {2 * m + 3}: wc {mp.nstr(mwc, 8)}, centre signal {mp.nstr(cen, 5)}, off-atom max {mp.nstr(off, 5)}, ordered {ordered}, min w {mp.nstr(wmin, 5)} -> {verdict}"
)
os.makedirs(out_dir, exist_ok=True)
F.dump_state(
    os.path.join(out_dir, f"results_q220_gauss_K{2 * m + 3}_A{float(mpf(A_str)):.4f}.json"),
    A_str,
    capO,
    vfull,
    res,
    verdict,
    {"centre_signal": float(cen), "off_atom_max": float(off), "w_min": float(wmin), "ordered": ordered},
)
print(
    f"RESULT q922: {verdict} K {2 * m + 3} A {A_str} wc {mp.nstr(mwc, 8)} residual {F.maxabs(res)}",
    flush=True,
)
sys.exit(0 if optimal else 1)
