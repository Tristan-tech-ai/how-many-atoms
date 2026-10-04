"""I_steps (03 Oct 2026): continuation of one ring by plain Newton (G_core, float64) through a list of rm values, with the scan validity
read at each; writes states.jsonl lines (G_core Ring json + rm + validity) for I_centre_scan.
Usage: py I_steps.py RP SEED.json "rm1,rm2,..." OUT.jsonl"""

import sys, json
import numpy as np
from G_core import Curve, Ring

rp = float(sys.argv[1])
d = json.load(open(sys.argv[2]))
rms = [float(x) for x in sys.argv[3].split(",")]
out = sys.argv[4]
cv = Curve(rp)
ring = Ring.from_json(d)
with open(out, "w") as fh:
    for rm in rms:
        ring, nr = cv.newton(ring, rm)
        sc = cv.scan(ring, rm)
        birth = sc.get("birth", None) if isinstance(sc, dict) else None
        st = ring.to_json(rm, {"rp": rp, "residual": float(nr)})
        if isinstance(sc, dict):
            for k in ("birthval", "birtht", "d2", "margins"):
                if k in sc:
                    st[k] = sc[k] if not isinstance(sc[k], np.ndarray) else sc[k].tolist()
        fh.write(json.dumps(st, default=float) + "\n")
        fh.flush()
        d2 = st.get("d2")
        bv = st.get("birthval")
        print(
            f"rm {rm}: residual {nr:.1e}; K {ring.K}; birthval {bv}; d2max {max(d2) if d2 else None}",
            flush=True,
        )
