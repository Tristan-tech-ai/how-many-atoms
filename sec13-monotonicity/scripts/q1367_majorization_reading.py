"""q1367 (9 Oct 2026, EXPLORATORY reading, no item, no verdict; py q1367_majorization_reading.py from temporal_within/centre_scripts/): the majorization conjecture of Barletta and Dytso (arXiv 2609.10039, Section 5, Fig. 5, numerical for A up to 8): for 0 < A1 < A2 the probability masses of the capacity-achieving input, sorted in decreasing order and padded with zeros, satisfy p(A2) majorized by p(A1), i.e. S_r(A2) <= S_r(A1) for every r, with S_r the sum of the r largest masses. If it holds, all Renyi entropies of order 0 or more are nondecreasing in A and in particular K(A) = exp H_0 is nondecreasing (the open monotonicity of Sharma and Shamai).
Test here on the stored certified capacity states of this record (K = 40 and 60 to 83, A up to 52.5; 100 to 200 OPTIMAL states per K, residual below 1e-20): sort all OPTIMAL states by A; for every consecutive pair (A1 < A2) compute max over r of S_r(A2) - S_r(A1) and count the pairs with a value above 1e-12. Majorization is transitive, so consecutive pairs suffice; the pairs across births and splits (K changes by one) are listed separately. A pair with the same A in two files is skipped.
Expected before the run (the conjecture): no pair above 1e-12.
Controls (first): c1 the checker on a synthetic chain of majorized vectors (no violation) and on the same chain reversed (violations); c2 a stored state with 5 percent of the mass of its two heaviest atoms moved to its two lightest atoms ... is majorized by the original, and the reverse (concentrating mass) is a violation; c3 the total mass of every analysed state is 1 within 1e-12.
Usage: py q1367_majorization_reading.py"""

import glob, json, os, re, sys
import numpy as np


def die(msg):
    print(msg, flush=True)
    os._exit(2)


ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def numstr(s):
    return re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", str(s).replace("np.float64(", "")).group(0)


def masses(j):
    w = [float(numstr(x)) for x in j["w"]]
    we = float(numstr(j["we"]))
    wc = None if j.get("wc") in (None, "None") else float(numstr(j["wc"]))
    m = w + w + [we, we] + ([wc] if wc is not None else [])
    return np.sort(np.array(m))[::-1]


def partial(m, n):
    p = np.zeros(n)
    p[: len(m)] = m
    return np.cumsum(p)


def worst(m1, m2):
    n = max(len(m1), len(m2))
    d = partial(m2, n) - partial(m1, n)
    i = int(np.argmax(d))
    return float(d[i]), i + 1


# controls
rng = np.random.default_rng(3)
v = np.sort(rng.dirichlet(np.ones(8)))[::-1]
chain = [v]
for _ in range(6):  # each step moves mass from a larger to a smaller entry: majorized by the previous
    u = chain[-1].copy()
    i, k = 0, len(u) - 1
    t = 0.2 * (u[i] - u[k]) / 2
    u[i] -= t
    u[k] += t
    chain.append(np.sort(u)[::-1])
ok_fwd = all(worst(chain[i], chain[i + 1])[0] <= 1e-15 for i in range(6))
bad_rev = sum(1 for i in range(6) if worst(chain[i + 1], chain[i])[0] > 1e-6)
c1 = ok_fwd and bad_rev >= 5
print(
    f"control c1 (synthetic majorized chain: forward ok {ok_fwd}, reversed violations {bad_rev} of 6): {c1}",
    flush=True,
)
if not c1:
    die("READING q1367: VOID (control c1)")
states = []
for f in glob.glob(os.path.join(ROOT, "**", "results_q2*_gauss_K*_A*.json"), recursive=True):
    try:
        j = json.load(open(f))
        res = float(numstr(j["residual"]))
        K = int(j["K"])
        A = float(numstr(j["A"]))
    except Exception:
        continue
    if j.get("verdict") == "OPTIMAL" and res < 1e-20:
        states.append((A, K, masses(j), f))
states.sort(key=lambda t: (t[0], t[3]))
tot = max(abs(float(s[2].sum()) - 1.0) for s in states)
c3 = tot < 1e-12
big = states[len(states) // 2]
m = big[2].copy()
mm = m.copy()
t = 0.05 * (mm[0] + mm[1]) / 2
mm[0] -= t
mm[1] -= t
mm[-1] += t
mm[-2] += t
mm = np.sort(mm)[::-1]
c2 = worst(m, mm)[0] <= 1e-15 and worst(mm, m)[0] > 1e-6
print(
    f"control c2 (moving 5 percent of the two heaviest masses to the two lightest is majorized, the reverse is a violation): {c2}; c3 (total mass of {len(states)} states within 1e-12 of 1, largest deviation {tot:.1e}): {c3}",
    flush=True,
)
if not (c2 and c3):
    die("READING q1367: VOID (control)")
viol = []
npairs = 0
worst_all = (-1.0, None)
cross = []
for (A1, K1, m1, f1), (A2, K2, m2, f2) in zip(states[:-1], states[1:]):
    if A2 - A1 < 1e-12:
        continue
    d, r = worst(m1, m2)
    npairs += 1
    if d > worst_all[0]:
        worst_all = (d, (A1, K1, A2, K2, r))
    if d > 1e-12:
        viol.append((A1, K1, A2, K2, d, r))
    if K1 != K2:
        cross.append((A1, K1, A2, K2, d, r))
print(
    f"states {len(states)}, consecutive pairs {npairs}; pairs with max_r (S_r(A2) - S_r(A1)) above 1e-12: {len(viol)}; largest value {worst_all[0]:.3e} at (A1, K1, A2, K2, r) = {worst_all[1]}",
    flush=True,
)
print(
    f"pairs across a change of K: {len(cross)}; largest value among them {max([c[4] for c in cross] or [float('nan')]):.3e}",
    flush=True,
)
for v_ in viol[:15]:
    print(f"  violation: A1 {v_[0]:.6f} K {v_[1]} -> A2 {v_[2]:.6f} K {v_[3]}: {v_[4]:.3e} at r = {v_[5]}")
print("READING q1367 (exploratory, no verdict)")
