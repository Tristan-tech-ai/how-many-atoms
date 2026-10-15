"""q1616b (10 Oct 2026, READING, post hoc, written after q1616 was read, no prediction registered, no verdict word): the log-log slope of the response columns of E_4, E_5 and E_6 over K_b 83..130 and their pairwise correlation, at the c20 block pair. Usage: py q1616b_post_hoc.py (about 30 s)."""

import io, contextlib, os
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(H, "q1616_E4_E5.py"), "__name__": "q1616ns"}
src = open(os.path.join(H, "q1616_E4_E5.py"), encoding="utf-8").read()
src = src[: src.index("out = {}")]
with contextlib.redirect_stdout(io.StringIO()):
    exec(src, ns)
offs, scale, Ks = ns["offs"], ns["scale"], np.array(ns["Ks"], dtype=float)
pair = (-0.093773648256, 0.001159913641)
cols = []
for j in range(3):
    up = [0.0] * 3
    dn = [0.0] * 3
    up[j] = scale[j]
    dn[j] = -scale[j]
    cols.append((offs(pair, up) - offs(pair, dn)) / (2 * scale[j]))
for j, c in enumerate(cols):
    s = np.polyfit(np.log(Ks), np.log(np.abs(c)), 1)[0]
    print(
        "column E_%d: sign %s, log-log slope against K_b %.3f, value at K_b 83 %.3e, at K_b 130 %.3e"
        % (4 + j, "+" if c[0] > 0 else "-", s, c[0], c[-1])
    )
C = np.corrcoef(np.array(cols))
print("correlations: E_4 E_5 %.5f, E_4 E_6 %.5f, E_5 E_6 %.5f" % (C[0, 1], C[0, 2], C[1, 2]))
print("READING q1616b done (post hoc; no verdict word)")
