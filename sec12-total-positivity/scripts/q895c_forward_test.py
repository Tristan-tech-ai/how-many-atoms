"""q895c (7 Oct 2026): text copy of q895b with the repair F4c of amendment 1710: the item F4-first is examined only for a family whose starting event is bracketed to a width of at most 1e-3 in a (the first row of a family lies within one chain step of the event only then); otherwise it is reported as not examined and counted. All other items unchanged. Text of q895b follows: q895b (6 Oct 2026): text copy of q895 with a loud exit (code 2, line VOID) when a chain has no new rows, no pair for F1 or no step for F2 (the original printed PASS on zero rows); criteria F1-F5 unchanged. Original header: F1-F5, the blind forward test of the structure of amendments 1643-1662 on rows that did not exist when it was registered (amendment 1663).
New rows come from the extension runs q584x_extend.py (immune, results_q584_ext.json, rows with a > 29.5845) and q601x_extend.py (NPMLE, results_q601_ext.json, rows with a > 26.5623), both started 21:48.
Families are found on the whole extended chain; the immune K 39 family (stored up to a = 29.5845) continues into the new rows. "Completed" means that a later family exists. Definitions as in q877 (edge index, A2), q891 (r = c/m_ref),
q892 (u, w) and q893 (mu_j, mu_bar = mean of mu_j over j = 3..8; cell = half the distance of the neighbours, the centre atom's cell is p_1).
F1: A2 on every pair of consecutive rows that involves a new row: x_j(a') > x_j(a) for every atom j of the earlier row.
F2: inside every family, over the steps that involve a new row, the innermost mass (c with a centre atom, the mass v_1 of the innermost pair otherwise) increases strictly.
F3: for every family with a centre atom and new rows, r = c/m_ref increases strictly over the new steps, and at the last row of every completed such family r lies in [0.999, 1.0).
F4: at the last row of every completed new family u (centre families) or w (no-centre families) lies in [0.98, 1.02]; at the first row of every new family, u lies in [0.50, 0.60] (centre) and w is below 0.5 (no centre).
F5: at the last row of every completed new family |c/(mu_bar p_1) - 1| <= 1e-3 (centre) or |mu_1/mu_bar - 1| <= 1e-3 (no centre).
PASS: F1-F5 all hold in both chains. FAIL lists each violated item with its family. If a chain has no completed new family, F3-F5 report 'no completed family' for that chain and the verdict covers only the items that exist.
Usage: py q895_forward_test.py   (from temporal_within/)"""

import json, sys
import numpy as np

CFG = [("immune", "results_q584_ext.json", 29.5845), ("NPMLE", "results_q601_ext.json", 26.5623)]
allok = True
for name, f, a0 in CFG:
    dd = json.load(open(f))
    rows = dd["rows"]
    events = dd["events"]
    n_not_examined = 0
    A = [float(r["a"]) for r in rows]
    new0 = next((i for i, a in enumerate(A) if a > a0), None)
    print(
        f"[{name}] rows {len(rows)}, new rows from index {new0}: {len(rows) - new0 if new0 is not None else 0}"
    )
    if new0 is None or len(rows) - new0 <= 0:
        print(f"VOID q895b: chain {name} has no new rows")
        sys.exit(2)
    fam = []
    cur = None
    for i, r in enumerate(rows):
        key = (int(r["K"]), bool(r["centre"]))
        if key != cur:
            fam.append([key, i, i])
            cur = key
        else:
            fam[-1][2] = i
    P = [np.array([float(t) for t in r["p"]]) for r in rows]
    V = [np.array([float(t) for t in r["v"]]) for r in rows]
    # F1
    viol = 0
    pairs = 0
    for i in range(max(new0 - 1, 0), len(rows) - 1):
        n = len(P[i])
        pairs += 1
        for j in range(1, n + 1):
            if j <= len(P[i + 1]) and not (P[i + 1][len(P[i + 1]) - j] > P[i][n - j]):
                viol += 1
    if pairs == 0:
        print(f"VOID q895b: chain {name} has no pair for F1")
        sys.exit(2)
    nsteps_chain = 0
    ncompleted = 0
    f1 = viol == 0
    print(f"   F1 A2: pairs {pairs}, violations {viol} -> {'PASS' if f1 else 'FAIL'}")
    ok = f1
    for fi, (key, i0, i1) in enumerate(fam):
        if i1 < new0:
            continue
        centre = key[1]
        completed = fi < len(fam) - 1
        s0 = max(i0, new0 - 1)

        def inner(i):
            return (1.0 - V[i].sum()) if centre else V[i][0]

        steps = [(inner(i + 1) > inner(i)) for i in range(s0, i1)]
        f2 = all(steps) if steps else True
        nsteps_chain += len(steps)
        ncompleted += 1 if completed else 0

        def ratio(i):
            p = P[i]
            v = V[i]
            X = np.vstack([np.ones(6), p[:6] ** 2, p[:6] ** 4]).T
            coef, *_ = np.linalg.lstsq(X, v[:6] / 2, rcond=None)
            return (1.0 - v.sum()) / coef[0]

        msg = f"   family K {key[0]} centre {str(centre):5s} rows {i0}-{i1} ({'completed' if completed else 'not completed'}): F2 {'PASS' if f2 else 'FAIL'} ({sum(steps)} of {len(steps)} steps)"
        okf = f2
        if centre:
            rs = [ratio(i) for i in range(s0, i1 + 1)]
            inc = all(b > a for a, b in zip(rs, rs[1:]))
            f3 = inc and (not completed or 0.999 <= rs[-1] < 1.0)
            msg += f"; F3 r {rs[0]:.4f} -> {rs[-1]:.6f} increasing {inc} -> {'PASS' if f3 else 'FAIL'}"
            okf = okf and f3

        def uw(i):
            p = P[i]
            s = p[1] - p[0]
            return (p[0] / s) if centre else (2 * p[0] / s)

        def mu_stats(i):
            p = P[i]
            v = V[i]
            J = len(p)
            m = v / 2
            c = 1 - v.sum()
            xs = np.concatenate(([0.0] if centre else [-p[0]], p))
            mu = np.array([m[j] / ((xs[j + 2] - xs[j]) / 2) for j in range(J - 1)])
            mub = mu[2:8].mean()
            return (c / (mub * p[0])) if centre else (mu[0] / mub)

        if completed:
            last = uw(i1)
            f4l = 0.98 <= last <= 1.02
            f5 = abs(mu_stats(i1) - 1) <= 1e-3
            msg += f"; F4 last {'u' if centre else 'w'} {last:.4f} -> {'PASS' if f4l else 'FAIL'}; F5 last law ratio {mu_stats(i1):.5f} -> {'PASS' if f5 else 'FAIL'}"
            okf = okf and f4l and f5
        if i0 >= new0:
            evs = [e for e in events if e["K_before"] == key[0] - 1 and e["a_hi"] <= A[i0] + 1e-9]
            if not evs:
                print(f"VOID q895c: no event with K_before {key[0] - 1} before row {i0} of chain {name}")
                sys.exit(2)
            wid = evs[-1]["a_hi"] - evs[-1]["a_lo"]
            if wid > 1e-3:
                msg += f"; F4 first not examined (event bracket width {wid:.3e})"
                n_not_examined += 1
            else:
                first = uw(i0)
                f4f = (0.50 <= first <= 0.60) if centre else (first < 0.5)
                msg += (
                    f"; F4 first {first:.3f} -> {'PASS' if f4f else 'FAIL'} (event bracket width {wid:.2e})"
                )
                okf = okf and f4f
        print(msg + f" => {'PASS' if okf else 'FAIL'}")
        ok = ok and okf
    if nsteps_chain == 0:
        print(f"VOID q895b: chain {name} has no step for F2")
        sys.exit(2)
    print(
        f"   [{name}] families with new rows examined, completed among them: {ncompleted}, F2 steps examined: {nsteps_chain}, F4-first not examined: {n_not_examined}"
    )
    allok = allok and ok
print("FORWARD TEST VERDICT (q895c):", "PASS" if allok else "FAIL")
