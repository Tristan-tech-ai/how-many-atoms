"""q1063 (7 Oct 2026): copy of q944 with an adaptive step in the downward walk of the K + 1 branch (amendment 1813; the first run of q944 for K 71 stopped with VOID q937: K 61 corrector failed at A 46.430872). Original header of q944: generic text copy of q937 for any birth K -> K + 1 (tower, T66): arguments ODD_BASE.json (first odd state, at the amplitude of EVEN_HI), EVEN_HI.json (first even state with the centre signal positive), EVEN_LO.json (last optimal even state), OUT.json, GRID. Text of q937: direction of the birth K 60 -> K 61 on the capacity (T67, registered in amendment 1714 before any state was computed).
All states on the frozen grid 42.0 in 200-bit ball arithmetic.
Part K61: from the first K 61 state (capacity_odd, A = 41.21, wc = 0.0028265) the K 61 branch is followed DOWN in A by step_in_A with the steps -0.0025 until wc is negative, then the root of wc(A) is found by secant steps with step_in_A to |wc| < 1e-12; the slope dwc/dA at the root comes from the last two states.
Part K60: from the stored K 60 state at A = 41.21 (centre signal positive) and the stored one at A = 41.20, the root A_b0 of the centre signal D(0) - C along the K 60 branch is found by secant steps with step_in_A to |D(0) - C| < 1e-20.
Usage: py q944_birth_generic.py ODD_BASE EVEN_HI EVEN_LO OUT.json GRID"""

import json, sys, time
from flint import arb
import q920_cap_flint as F
import q921_cap_tools as T
from q920_cap_flint import Cap, maxabs

t0 = time.time()
say = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
ODD_BASE, EVEN_HI, EVEN_LO, OUT, GRID = sys.argv[1:6]
TOL = arb("1e-38")


class CapB(Cap):
    """amendment 1717: feasibility without the sign condition on the centre mass (the K 61 branch crosses wc = 0)"""

    def feasible(self, v):
        m = self.m
        u = v[:m]
        w = v[m : 2 * m]
        we = v[2 * m]
        return (
            all(u[i] < u[i + 1] for i in range(m - 1))
            and u[0] > 0
            and u[-1] < self.A
            and all(t > 0 for t in w)
            and we > 0
        )


def load(path, odd):
    A, u, w, we, wc, d = F.load_state(path)
    cap = (CapB if odd else Cap)(A, len(u), odd, grid_A=GRID)
    v = [t.mid() for t in (u + w + [we] + ([wc] if odd else []))]
    v, res, ok = T.newton_full(cap, v, say, tol=TOL)
    if not ok:
        sys.exit(f"VOID q937: {path} does not polish on the grid {GRID}")
    return cap, v


def wc_of(v, m):
    return v[2 * m + 1]


def centre_of(cap, v):
    return cap.signals(v)[0]


out = {"K61": [], "K60": []}
cap, v = load(ODD_BASE, True)
m = cap.m
out["K61"].append({"A": float(cap.A.mid()), "wc": float(wc_of(v, m).mid())})
say(f"K 61 base A {cap.A.str(10, radius=False)} wc {wc_of(v, m).str(10, radius=False)}")
states = [(float(cap.A.mid()), wc_of(v, m), cap, v)]
A0_base = float(cap.A.mid())
neg = 0  # amendment 1728: two further states below the first state with wc < 0, so that at least four states are traced
A_cur = A0_base
stp = 0.0025  # q1063 (7 Oct 2026, amendment 1813): the step is halved after a failed corrector (minimum 1e-4) and kept after a success; everything else is the text of q944
for k in range(1, 400):
    A = round(A_cur - stp, 6)
    capn, vn, rn, ok = T.step_in_A(states[-1][2], states[-1][3], f"{A:.6f}", say)
    if not ok:
        stp /= 2
        if stp < 1e-4:
            sys.exit(f"VOID q937: K 61 corrector failed at A {A} with the step below 1e-4")
        say(f"K 61 corrector failed at A {A:.6f}; step -> {stp}")
        continue
    A_cur = A
    states.append((A, wc_of(vn, m), capn, vn))
    out["K61"].append({"A": A, "wc": float(wc_of(vn, m).mid())})
    say(f"K 61 A {A:.6f}: wc {wc_of(vn, m).str(10, radius=False)}")
    if wc_of(vn, m) < 0:
        neg += 1
    if neg >= 3:
        break
else:
    sys.exit("VOID q937: wc did not change sign within 399 attempts")
i_neg = next(i for i, st_ in enumerate(states) if st_[1] < 0)
(A1, w1, c1, v1), (A0, w0, c0, v0) = states[i_neg - 1], states[i_neg]
for it in range(12):
    An = float(A1) - float(w1.mid()) * (float(A1) - float(A0)) / float((w1 - w0).mid())
    cn, vn, rn, ok = T.step_in_A(c1, v1, f"{An:.15f}", say)
    if not ok:
        sys.exit("VOID q937: corrector failed in the secant for the wc root")
    wn = wc_of(vn, m)
    out["K61"].append({"A": An, "wc": float(wn.mid()), "secant": True})
    say(f"K 61 secant {it}: A {An:.9f}, wc {wn.str(8, radius=False)}")
    A0, w0, c0, v0, A1, w1, c1, v1 = A1, w1, c1, v1, An, wn, cn, vn
    if abs(wn) < arb("1e-12"):
        break
else:
    sys.exit("VOID q937: secant for wc did not reach 1e-12")
out["A_cross"] = A1
out["wc_at_cross"] = float(w1.mid())
out["slope_wc"] = float(((w1 - w0) / (arb(A1) - arb(A0))).mid()) if A1 != A0 else None
say(f"K 61 root of wc: A {A1:.9f}, slope dwc/dA from the last two states {out['slope_wc']}")
cap60, v60 = load(EVEN_HI, False)
c60 = [(float(cap60.A.mid()), centre_of(cap60, v60), cap60, v60)]
cap60b, v60b = load(EVEN_LO, False)
c60.append((float(cap60b.A.mid()), centre_of(cap60b, v60b), cap60b, v60b))
say(
    f"K 60 centre signals: A {c60[0][0]} {c60[0][1].str(6, radius=False)}, A {c60[1][0]} {c60[1][1].str(6, radius=False)}"
)
out["K60"] = [
    {"A": c60[0][0], "centre": float(c60[0][1].mid())},
    {"A": c60[1][0], "centre": float(c60[1][1].mid())},
]
(Aa, ga, ca, va), (Ab, gb, cb, vb) = c60
for it in range(12):
    An = Aa - float(ga.mid()) * (Aa - Ab) / float((ga - gb).mid())
    cn, vn, rn, ok = T.step_in_A(ca, va, f"{An:.15f}", say)
    if not ok:
        sys.exit("VOID q937: corrector failed in the secant for A_b0")
    gn = centre_of(cn, vn)
    out["K60"].append({"A": An, "centre": float(gn.mid()), "secant": True})
    say(f"K 60 secant {it}: A {An:.9f}, centre signal {gn.str(8, radius=False)}")
    Ab, gb, cb, vb, Aa, ga, ca, va = Aa, ga, ca, va, An, gn, cn, vn
    if abs(gn) < arb("1e-20"):
        break
else:
    sys.exit("VOID q937: secant for A_b0 did not reach 1e-20")
out["A_b0"] = Aa
json.dump(out, open(OUT, "w"), indent=1)
print(
    f"RESULT q944: A_cross {out['A_cross']:.9f}, A_b0 {out['A_b0']:.9f}, slope dwc/dA {out['slope_wc']}",
    flush=True,
)
