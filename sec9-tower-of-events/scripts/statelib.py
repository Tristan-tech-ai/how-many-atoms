"""state helpers: load a stored q1411 state; make a SYNTHETIC state of another size K (for timing / evaluation-equivalence tests only: not a solution of the system)."""

import json, numpy as np
from flint import arb


def load_state(fn):
    d = json.load(open(fn))
    d = d[-1] if isinstance(d, list) else d
    return (
        arb(d["A"]),
        [arb(x) for x in d["u"]],
        [arb(x) for x in d["w"]],
        arb(d["we"]),
        (None if d.get("wc") in (None, "None") else arb(d["wc"])),
        d,
    )


def synth_state(base_fn, K):
    """profile of the stored state resampled to K atoms: positions u/A and masses as functions of the relative index, A scaled as K^(3/4)."""
    A0, u0, w0, we0, wc0, d = load_state(base_fn)
    m0 = len(u0)
    K0 = d["K"]
    odd = K % 2 == 1
    m = (K - 3) // 2 if odd else (K - 2) // 2
    A = arb(float(A0.mid()) * (K / K0) ** 0.75)
    s0 = (np.arange(m0) + 0.5) / m0
    s1 = (np.arange(m) + 0.5) / m
    # depth from the wall d = A - u, resampled in the variable j (count from the wall) near the wall, and u/A in the bulk
    u0f = np.array([float(t.mid()) for t in u0])
    w0f = np.array([float(t.mid()) for t in w0])
    A0f = float(A0.mid())
    # positions: u/A as a function of relative index (monotone), edge atoms keep their depth by blending
    uu = np.interp(s1, s0, u0f / A0f) * float(A.mid())
    depth0 = A0f - u0f[::-1]
    depth_new = np.interp(np.arange(m), np.arange(m0), depth0)  # first atoms from the wall keep depth
    uu_sorted = np.sort(uu)
    for j in range(min(6, m)):
        uu_sorted[m - 1 - j] = float(A.mid()) - depth_new[j]
    uu_sorted = np.sort(uu_sorted)
    ww = np.interp(s1, s0, w0f)
    wwe = float(we0.mid())
    wwc = float(wc0.mid()) if wc0 is not None else None
    tot = 2 * (ww.sum() + wwe) + (wwc if odd and wwc is not None else 0.0)
    ww = ww / tot
    wwe = wwe / tot
    wwc = wwc / tot if odd else None
    if odd and wwc is None:
        wwc = 0.5 * ww[0]
    return (
        A,
        [arb(float(t)) for t in uu_sorted],
        [arb(float(t)) for t in ww],
        arb(wwe),
        (arb(wwc) if odd else None),
        dict(K=K, synthetic=True),
    )
