"""q1073 (7 Oct 2026): verdict script of T107 (registered in the amendment that follows 1822; a copy of q1003 of T87): the majorization property p(A2) majorized by p(A1) for A1 < A2 (Barletta and Dytso, arXiv 2609.10039, Section 5) on the families K 70 and K 71 of the tower, with the last OPTIMAL state of K 69 as the start of the sequence.
p(A) is the vector of the probability masses of the optimal input (2 w_j at +-u_j counted as two atoms of mass w_j, two atoms of mass we, one atom of mass wc for odd K), divided by its sum, in non-increasing order and padded with zeros; tolerance 1e-40 on the partial sums, 50-digit arithmetic.
Verdict (last line): PASS if the families K 70 and K 71 are complete (each has an OPTIMAL state followed by a state with the verdict "centre positive"), the number of consecutive pairs is at least 200 and no pair violates; FAIL if the families are complete and a pair violates or the pairs are fewer than 200; INCOMPLETE if a family is missing or incomplete; VOID if a file is unreadable, a mass sum differs from 1 by more than 1e-30 or a control misbehaves.
Controls on the stored family K 66 (known): PC the K 66 states in increasing A must pass; NC1 the same states in decreasing A must violate; NC2 the K 66 states in a fixed pseudo-random order (seed 20261007) must violate. Usage: py q1073_majorization_k70_71_verdict.py
"""

import glob, json, sys
from mpmath import mp, mpf

mp.dps = 50
TOL = mpf(10) ** -40


def clean(v):
    return mpf(str(v).replace("np.float64(", "").rstrip(")"))


def masses(j):
    w = [clean(t) for t in j["w"]]
    we = clean(j["we"])
    wc = j.get("wc")
    wc = None if wc is None else clean(wc)
    mm = [x for x in w for _ in (0, 1)] + [we, we] + ([wc] if wc is not None else [])
    tot = sum(mm)
    if abs(tot - 1) > mpf(10) ** -30:
        raise ValueError(f"mass sum {tot}")
    return sorted([x / tot for x in mm], reverse=True)


def load(d, K):
    st = []
    nonopt = []
    for f in glob.glob(d + f"/results_q220_gauss_K{K}_A*.json"):
        j = json.load(open(f))
        a = float(clean(j["A"]))
        if str(j["verdict"]) == "OPTIMAL":
            st.append((a, K, masses(j)))
        else:
            nonopt.append((a, str(j["verdict"])))
    st.sort()
    nonopt.sort()
    return st, nonopt


def complete(st, nonopt):
    return bool(st) and any(a > st[-1][0] and v == "centre positive" for a, v in nonopt)


def violations(states):
    v = 0
    worst = mpf(0)
    for (a1, k1, m1), (a2, k2, m2) in zip(states, states[1:]):
        n = max(len(m1), len(m2))
        p1 = m1 + [mpf(0)] * (n - len(m1))
        p2 = m2 + [mpf(0)] * (n - len(m2))
        s1 = s2 = mpf(0)
        mx = mpf(0)
        for r in range(n):
            s1 += p1[r]
            s2 += p2[r]
            mx = max(mx, s2 - s1)
        if mx > TOL:
            v += 1
        worst = max(worst, mx)
    return v, worst


def main():
    try:
        k66, _ = load("tower/K66", 66)
    except Exception as e:
        print(f"VERDICT q1073 T107: VOID (unreadable K 66 files: {e})")
        sys.exit(2)
    pc = violations(k66)[0] == 0
    n1 = violations(list(reversed(k66)))[0] > 0
    import random

    sh = list(k66)
    random.Random(20261007).shuffle(sh)
    n2 = violations(sh)[0] > 0
    c = {"PC": pc, "NC1": n1, "NC2": n2}
    print("controls:", c)
    if not all(c.values()):
        print("VERDICT q1073 T107: VOID (a control does not behave)")
        sys.exit(2)
    fams = {}
    ok = True
    for K, d in ((70, "tower/K70"), (71, "tower/K71")):
        try:
            st, no = load(d, K)
        except Exception as e:
            print(f"VERDICT q1073 T107: VOID (unreadable {d}: {e})")
            sys.exit(2)
        fams[K] = (st, complete(st, no))
        print(f"   K {K}: optimal states {len(st)}, complete {fams[K][1]}")
    if not all(f[1] for f in fams.values()):
        print("VERDICT q1073 T107: INCOMPLETE (a family is missing or not complete)")
        sys.exit(3)
    k69, _ = load("tower/K69", 69)
    allst = sorted(k69[-1:] + fams[70][0] + fams[71][0])
    v, worst = violations(allst)
    pairs = len(allst) - 1
    print(f"pairs {pairs}, pairs with a violation {v}, largest increase of a partial sum {mp.nstr(worst, 4)}")
    res = "PASS" if (pairs >= 200 and v == 0) else "FAIL"
    print(f"VERDICT q1073 T107: {res}")
    sys.exit(0 if res == "PASS" else 1)


if __name__ == "__main__":
    main()
