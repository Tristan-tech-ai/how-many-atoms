"""Recompute the form-S (Szego / copying) loss points of the rp 3 ellipse rings K = 6..16 and their misses against the solved events.

Inputs (read only): agent G's formal-density outputs temporal_within/agents/G_work/formal/<chain>/c_eps<eps>.txt (line "CVEC c_2,c_4,..."),
the chains and eps pairs agent G used (G_family.md table: M3r9 1.5/1.6, M4r12 1.3/1.35, M5r12 1.20/1.22, M6r12 1.12/1.13,
M7r12b 1.05/1.055, M8r12 0.99/1.00), and the solved event brackets in rm from G_work/c3a_part1.log, c3b.log, c3c.log (eps = 3 - rm).
Method: Verblunsky coefficients alpha_n = -Phi_{n+1}(0) of f(t) dt/2pi, f = 1 + sum c_2k cos 2kt, by Toeplitz solves (a text copy of the
phi() of agents/G_copytest.py, itself a copy of the record's q806); the K-ring (tau = +1, X atom) is lost where alpha_(K-1) = +1,
located by the straight line through the two eps points (as agent G did). Miss = solved eps / form eps - 1.
Writes only rp3_form_loss.json in this folder.
"""
import os, re, json
from mpmath import mp, mpf, matrix, lu_solve

mp.dps = 50
HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(os.path.dirname(HERE))
GW = os.path.join(TW, "agents", "G_work")
CASES = [(6, "M3r9", "1.5", "1.6", "c3a_part1.log", "X1G"), (8, "M4r12", "1.3", "1.35", "c3b.log", "XY1G"),
         (10, "M5r12", "1.2", "1.22", "c3b.log", "X2G"), (12, "M6r12", "1.12", "1.13", "c3b.log", "XY2G"),
         (14, "M7r12b", "1.05", "1.055", "c3c.log", "X3G"), (16, "M8r12", "0.99", "1.0", "c3c.log", "XY3G")]


def cvec(chain, eps):
    txt = open(os.path.join(GW, "formal", chain, f"c_eps{eps}.txt"), encoding="utf-8").read()
    return [mpf(x) for x in re.search(r"CVEC (\S+)", txt).group(1).split(",")]


def alpha(cv, n_max):
    c = {0: mpf(2)}
    for i, x in enumerate(cv):
        c[2 * (i + 1)] = x
    mu = lambda m: c.get(abs(m), mpf(0)) / 2 if abs(m) % 2 == 0 else mpf(0)

    def phi(n):
        if n == 0:
            return [mpf(1)]
        A = matrix(n, n); rhs = matrix(n, 1)
        for j in range(n):
            for k in range(n):
                A[j, k] = mu(j - k)
            rhs[j] = -mu(j - n)
        a = lu_solve(A, rhs)
        return [a[k] for k in range(n)] + [mpf(1)]
    return [-phi(n + 1)[0] for n in range(n_max)]


def main():
    out = {}
    for K, chain, e1, e2, log, lab in CASES:
        a1 = alpha(cvec(chain, e1), K)[K - 1]
        a2 = alpha(cvec(chain, e2), K)[K - 1]
        x1, x2 = mpf(e1), mpf(e2)
        form = x1 + (1 - a1) * (x2 - x1) / (a2 - a1)
        t = open(os.path.join(GW, log), encoding="utf-8").read()
        m = re.search(r"EVENT %s \(K %d\) lost in rm \((\d\.\d+), (\d\.\d+)\)" % (re.escape(lab), K), t)
        rm_mid = (float(m.group(1)) + float(m.group(2))) / 2
        solved = 3 - rm_mid
        lead = 100 * (solved / float(form) - 1)
        out[f"K{K}"] = dict(chain=chain, eps_pair=[e1, e2], alpha_at_pair=[float(a1), float(a2)], form_loss_eps=float(form),
                            solved_rm_bracket=[float(m.group(1)), float(m.group(2))], solved_eps=solved, miss_pct=lead)
        print(f"K {K:2d} {chain:7s} alpha_{K-1}({e1}) = {float(a1):.6f}, ({e2}) = {float(a2):.6f}; form loss {float(form):.6f}; "
              f"solved {solved:.7f}; miss {lead:+.4f} %")
    json.dump(out, open(os.path.join(HERE, "rp3_form_loss.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
