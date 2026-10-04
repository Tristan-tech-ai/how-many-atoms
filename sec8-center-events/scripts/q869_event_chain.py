"""q869: chain of LFP centre events in dimension d by sequential q868 tracks (1579).
Start: the e2 (lift-off) threshold state {centre w0, wall} at m2. After a split/lift-off the centre becomes a shell at 0.25 carrying its
weight (kind birth next); after a birth a centre of weight 0.003 is added (kind split next). Prints each event; usage:
py q869_event_chain.py d m2 n_events [span, default 1.4]
"""

import sys, json
import numpy as np
from q868_lfp_events_dimd import track, two

d = int(sys.argv[1])
m2 = float(sys.argv[2])
n_ev = int(sys.argv[3])
SPAN = float(sys.argv[4]) if len(sys.argv) > 4 else 1.4
LIFT, DM = 0.25, 0.02
SW0, SDM = 0.003, 0.01
if (
    len(sys.argv) > 5
):  # start from a given validated event state: {"m", "a", "w", "w0" (null if none), "next": birth|split}
    S0 = json.loads(sys.argv[5])
    m = S0["m"]
    a = list(S0["a"])
    ww = list(S0["w"])
    w0 = S0["w0"]
    kind = S0["next"]
    LIFT = S0.get("lift", 0.25)
    DM = S0.get("dm", 0.02)
    SW0 = S0.get("sw0", 0.003)
    SDM = S0.get("sdm", 0.01)
else:
    at, w = two(d, m2)
    m = m2
    a = []
    ww = []
    w0 = w[0]
    kind = "birth"
events = []
for i in range(n_ev):
    if kind == "birth":
        a_g = [LIFT] + list(a)
        w_g = [w0] + list(ww)
        w0_g = None
        m0 = m + DM
    else:
        a_g = list(a)
        w_g = list(ww)
        w0_g = SW0
        m0 = m + SDM
    root, st = track(d, m0, m0 + SPAN, a_g, w_g, w0_g, kind, 0.025)
    if root is None:
        print(f"d {d}: no {kind} crossing after m {m:.6f}; chain stopped", flush=True)
        break
    sol, atv, wt, res = st[1], st[2], st[3], st[4]
    if (
        min(atv) < 0 or min(wt) < -1e-12 or max(wt) > 1 or res > 1e-8
    ):  # validity guard (added 04:4x after the 2D divergence)
        print(
            f"d {d}: INVALID {kind} state at m {root:.6f} (min radius {min(atv):.3g}, weights {min(wt):.3g}..{max(wt):.3g}, "
            f"residual {res:.1e}); chain stopped",
            flush=True,
        )
        break
    events.append((kind, root))
    print(
        f"CHAIN d {d} e{i + 3} {kind}: m {root:.8f}; support {np.round(atv, 6)}; weights {np.round(wt, 6)}; residual {res:.1e}",
        flush=True,
    )
    m = root
    if kind == "birth":  # state: shells (no centre) + wall
        a = list(atv[:-1])
        ww = list(wt[:-1])
        kind = "split"
    else:  # state: centre + shells + wall
        w0 = wt[0]
        a = list(atv[1:-1])
        ww = list(wt[1:-1])
        kind = "birth"
print("EVENTS", json.dumps([(k, round(r, 8)) for k, r in events]))
