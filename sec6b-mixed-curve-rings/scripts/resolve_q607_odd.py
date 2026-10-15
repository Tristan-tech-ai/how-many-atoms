"""resolve_q607_odd.py (14 Oct 2026, task KKTO, reading, no verdict): re-solve the odd-K_b located NPMLE interval states of amendment 2040
with their centre mass, which the stored records (centre_scripts/q607_u1207_K*.json) do not hold.

Instrument: the Chain class of centre_scripts/c35_chain_k.py (q607 text, KEXP 1 = the NPMLE weight), NF = K // 2 (every positive atom
free) plus the centre atom, 200 bits, 24-node Gauss-Legendre panels of width 0.5; the record used is the last solved record before the
transition (largest a below a_t). Seed: the stored positions and masses, centre mass = the innermost stored mass; C fixed at g(p_1) of
the seed (the conditions are invariant under the mass scaling, as in q607). Registered in task KKTO before the run: the re-solve
converges; the side positions agree with the stored ones within 1e-10; the side masses agree with the stored ones up to one common factor
(relative spread of the ratio below 1e-10); the centre mass is positive. Output gate_truth/kkt_scans/q607_u1207_K<K>_full.json (a, p,
m, m0, C, residual) for gate_truth/kkt_scan_q607.py.
Usage (from temporal_within/centre_scripts): py ../gate_truth/resolve_q607_odd.py K [K ...]"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.getcwd())
from flint import arb, ctx  # noqa: E402
import c35_chain_k as Q  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kkt_scans")


def main(argv):
    os.makedirs(OUT, exist_ok=True)
    for ks in argv:
        K = int(ks)
        d = json.load(open("q607_u1207_K%d.json" % K))
        at = d.get("a_t")
        rec = max([h for h in d["hist"] if "p" in h and (at is None or h["a"] < at)], key=lambda h: h["a"])
        a, p0, m0s = float(rec["a"]), [float(x) for x in rec["p"]], [float(x) for x in rec["m"]]
        NF = K // 2
        t0 = time.time()
        ch = Q.Chain(K, NF, 200)
        ch.tol_abs = arb(math.exp(-4 * math.pi**2 * Q.rho_of(a) ** 2) * 1e-7)
        ch.setup(a)
        u = [arb(x) for x in p0] + [arb(x) for x in m0s] + [arb(m0s[0])]
        ch.Cfix = arb(0)
        o = ch.evaluate(u, jac=False)
        ch.Cfix = arb(o["E"][0].mid())
        u, r, it, o = ch.solve(u)
        p = [float(t.mid()) for t in u[:NF]]
        m = [float(t.mid()) for t in u[NF : 2 * NF]]
        mc = float(u[2 * NF].mid())
        dp = max(abs(x - y) for x, y in zip(p, p0))
        ratio = [x / y for x, y in zip(m, m0s)]
        spread = (max(ratio) - min(ratio)) / (sum(ratio) / len(ratio))
        ok = dp < 1e-10 and spread < 1e-10 and mc > 0
        print(
            "K %d (a %.4f): residual %.2e in %d steps; max |p - p_stored| %.1e; mass-ratio spread %.1e; centre mass %+.6e; %s [%.0f s]"
            % (K, a, r, it, dp, spread, mc, "as registered" if ok else "NOT as registered", time.time() - t0),
            flush=True,
        )
        json.dump(
            {
                "K": K,
                "a": a,
                "p": p,
                "m": m,
                "m0": mc,
                "C": float(ch.Cfix.mid()),
                "residual": r,
                "dp": dp,
                "mass_ratio_spread": spread,
                "by": "gate_truth/resolve_q607_odd.py",
            },
            open(os.path.join(OUT, "q607_u1207_K%d_full.json" % K), "w"),
            indent=1,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
