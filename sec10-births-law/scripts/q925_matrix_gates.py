"""q925 (7 Oct 2026): matrix gates of q901c (checkerboard sign, inverse negative, total positivity to order 3 all/sampled and orders 4-5 sampled, in ball arithmetic at 400 and 1600 bits), text copy of the functions of q901c_capacity_ift.py without its mpmath and q206 parts; `say` prints."""

import itertools
import numpy as np
from flint import arb, arb_mat, ctx

say = lambda *a: print(*a, flush=True)


def sign_gate(Sm, m):
    bad = 0
    for p in range(m):
        for q in range(m):
            x = Sm[p][q]
            ok = (x < 0) if (p + q) % 2 == 0 else (x > 0)
            if not ok:
                bad += 1
    return bad == 0, bad


def inv_gate(Sm, m):
    M = arb_mat(m, m)
    for p in range(m):
        for q in range(m):
            M[p, q] = Sm[p][q]
    for prec in (400, 1600, 6400):
        ctx.prec = prec
        try:
            Si = M.inv()
            ok = all(Si[p, q] < 0 for p in range(m) for q in range(m))
            ctx.prec = 400
            return ok
        except Exception:
            continue
    ctx.prec = 400
    return False


def det_pos(T, rs, cs):
    k = len(rs)
    for prec in (400, 1600):
        ctx.prec = prec
        Mx = arb_mat(k, k)
        for x, ri in enumerate(rs):
            for yy, ci in enumerate(cs):
                Mx[x, yy] = T[ri][ci]
        dd = Mx.det()
        if dd > 0:
            ctx.prec = 400
            return True, dd
        if dd < 0:
            ctx.prec = 400
            return False, dd
    ctx.prec = 400
    return False, dd


def tp_gate(T, m, label, early, high=False):
    """returns (ok, counts, failures); orders 1,2 all and 3 sampled 20000 (high=False); orders 4,5 sampled 5000 (high=True)."""
    rng = np.random.default_rng(20261006 + (1000 if high else 0))
    counts = {}
    fails = []
    bad = 0
    plan = [(4, 5000), (5, 5000)] if high else [(1, None), (2, None), (3, 20000)]
    for k, ns in plan:
        cnt = 0
        if ns is None:
            idx = list(itertools.combinations(range(m), k))
            pairs = ((list(rs), list(cs)) for rs in idx for cs in idx)
        else:
            pairs = (
                (
                    sorted(rng.choice(m, k, replace=False).tolist()),
                    sorted(rng.choice(m, k, replace=False).tolist()),
                )
                for _ in range(ns)
            )
        for rs, cs in pairs:
            cnt += 1
            ok, dd = det_pos(T, rs, cs)
            if not ok:
                bad += 1
                if len(fails) < 8:
                    fails.append((k, rs, cs, dd.mid().str(5)))
                if early:
                    counts[k] = cnt
                    return False, counts, fails
        counts[k] = cnt
    if sum(counts.values()) == 0:
        raise RuntimeError("zero minors examined")
    return bad == 0, counts, fails


def matrix_gates(Sm, m, label, early):
    T = [[Sm[p][q] * (-1 if (p + q) % 2 == 0 else 1) for q in range(m)] for p in range(m)]
    sg = sign_gate(Sm, m)
    iv = inv_gate(Sm, m)
    say(f"[{label}] C-sign violations {sg[1]}; C-inv {iv}")
    t1 = tp_gate(T, m, label, early)
    say(f"[{label}] TP orders 1-3: minors examined {t1[1]}, failures {0 if t1[0] else 'some'} {t1[2][:3]}")
    t2 = tp_gate(T, m, label, early, high=True)
    say(f"[{label}] TPh orders 4-5: minors examined {t2[1]}, failures {0 if t2[0] else 'some'} {t2[2][:3]}")
    if sum(t1[1].values()) == 0 or sum(t2[1].values()) == 0:
        raise RuntimeError("zero minors examined")
    return {"C_sign": sg[0], "C_inv": iv, "TP": t1[0], "TPh": t2[0]}
