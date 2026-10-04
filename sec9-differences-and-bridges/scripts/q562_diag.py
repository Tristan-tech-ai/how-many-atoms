"""Post hoc diagnostic (label M, after the sealed file was opened): I at the certified transitions near the failed cases."""

import sys, json
import numpy as np
import q562_rdlib as R

mode = sys.argv[1]
a = float(sys.argv[2])
K, ok, ys, ws, Dn, I = R.solve(a, mode)
print(
    json.dumps(
        {
            "mode": mode,
            "a": a,
            "K": K,
            "ok": ok,
            "I": I,
            "minw": float(min(ws)),
            "ys": np.round(ys, 4).tolist(),
        }
    ),
    flush=True,
)
