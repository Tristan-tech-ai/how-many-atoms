r"""q1887_npmle_birth_300nodes_verdict.py (14 Oct 2026): is the NPMLE center
birth on the filled ellipse (rp 2.1) and on the mixed curve (lam 0.2, rp 2.1), source N(0, 4 I), within 3e-7 of its registered value
when the ring's Gauss-Hermite quadrature has 300 nodes instead of 200?

Background (amendment 1447): the formal-density prediction is converged in its own quadrature to 8 digits; the ring's center excess moved
by 0.76 % between 120 and 200 nodes for the wide source, and no run above 200 nodes exists. The paper prints "within 3e-7" for all six
registered births.

Method. For each case and each of the two amplitudes rm bracketing the birth, the K = 20 rd ring is re-solved by D_solver.py newton
(harmonic mode, from the stored 200-node state, NGH 300, PREC 192, FUNC rd, SRC_S 2, RP 2.1, LAM as stored), and its center excess
D(0) - C is evaluated by I_rd_centre.py at NGH 300, PREC 192. The birth is the zero of the linear interpolant of the two excesses.
Criteria (registered 14 Oct 2026 about 21:4x, before any 300-node output exists):
 V1 every Newton solve ends with a residual below 1e-25 and every excess is a finite number (else VOID);
 B1 ellipse: |zero_300 - 1.7416352| <= 3e-7;   B2 mixed: |zero_300 - 1.7363460| <= 3e-7;
 the excesses at the two amplitudes have opposite signs in each case (else VOID: the bracket does not hold the zero).
Verdict (last line): VERDICT q1887: PASS if V1, B1, B2 hold; FAIL if V1 holds and B1 or B2 fails; VOID otherwise; INCOMPLETE if a case
is missing. Printed for information: zero_200 from the stored 200-node excesses and |zero_300 - zero_200|.
Usage (from temporal_within): py small_gap_q206/q1887_npmle_birth_300nodes_verdict.py [--registration-run] [--selftest]
"""
import json
import os
import re
import subprocess
import sys

TW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AG = os.path.join(TW, "runs")
CASES = [
    ("ellipse", "0", "1.7416352", ["1.7416", "1.7417"], ["D_newton_harm_rm1.7416_rd21_K20.json", "D_newton_harm_rm1.7417_rd21_K20.json"]),
    ("mixed", "0.2", "1.7363460", ["1.7362", "1.7364"], ["D_newton_harm_rm1.7362_rdmx_K20.json", "D_newton_harm_rm1.7364_rdmx_K20.json"]),
]
TOL = 3e-7


def zero(r1, e1, r2, e2):
    return r1 - e1 * (r2 - r1) / (e2 - e1)


def judge(results):
    """results: {case: [(rm, excess, resid), (rm, excess, resid)]} -> (verdict, lines)"""
    lines = []
    for name, lam, reg, rms, _ in CASES:
        if name not in results or len(results[name]) != 2:
            return "INCOMPLETE", lines + ["case %s missing" % name]
    ok = True
    for name, lam, reg, rms, _ in CASES:
        (r1, e1, s1), (r2, e2, s2) = results[name]
        if any(x is None for x in (e1, e2, s1, s2)) or max(s1, s2) > 1e-25:
            return "VOID", lines + ["case %s: Newton residual or excess missing (V1)" % name]
        if e1 * e2 >= 0:
            return "VOID", lines + ["case %s: excesses %+.3e and %+.3e do not bracket a zero" % (name, e1, e2)]
        z = zero(r1, e1, r2, e2)
        d = abs(z - float(reg))
        lines.append("case %s: zero_300 %.8f, registered %s, |difference| %.2e (bound 3e-7): %s" % (name, z, reg, d, d <= TOL))
        ok &= d <= TOL
    return ("PASS" if ok else "FAIL"), lines


def selftest():
    good = {"ellipse": [(1.7416, -2.3e-5, 1e-30), (1.7417, 4.2e-5, 1e-30)], "mixed": [(1.7362, -9.25e-5, 1e-30), (1.7364, 3.40e-5, 1e-30)]}
    shifted = {"ellipse": [(1.7416, -2.3e-5, 1e-30), (1.7417, 1.0e-5, 1e-30)], "mixed": good["mixed"]}
    nonconv = {"ellipse": [(1.7416, -2.3e-5, 1e-20), (1.7417, 4.2e-5, 1e-30)], "mixed": good["mixed"]}
    nobracket = {"ellipse": [(1.7416, 2.3e-5, 1e-30), (1.7417, 4.2e-5, 1e-30)], "mixed": good["mixed"]}
    missing = {"ellipse": good["ellipse"]}
    got = [judge(good)[0], judge(shifted)[0], judge(nonconv)[0], judge(nobracket)[0], judge(missing)[0]]
    ok = got == ["PASS", "FAIL", "VOID", "VOID", "INCOMPLETE"]
    print("SELFTEST:", "PASS" if ok else "FAIL", got)
    return ok


def run_case(name, lam, rms, states):
    out = []
    env = dict(os.environ, NGH="300", PREC="192", FUNC="rd", SRC_S="2", RP="2.1", LAM=lam)
    for rm, st in zip(rms, states):
        p = subprocess.run([sys.executable, "D_solver.py", "newton", os.path.join("D_work", st), "-", rm, "harm", "30"], cwd=AG, env=env,
                           capture_output=True, text=True)
        txt = p.stdout + p.stderr
        res = [float(x) for x in re.findall(r"  it \d+: residual ([0-9.]+e[-+]\d+)", txt)]   # D_solver prints "  it N: residual X"
        new = sorted([f for f in os.listdir(os.path.join(AG, "D_work")) if f.startswith("D_newton_harm_rm%s_ngh300_p192" % rm)],
                     key=lambda f: os.path.getmtime(os.path.join(AG, "D_work", f)))
        print("  %s rm %s: solver exit %d; last residuals %s; state %s" % (name, rm, p.returncode, res[-3:], new[-1:] if new else None), flush=True)
        if p.returncode != 0 or not new:
            out.append((float(rm), None, None))
            continue
        q = subprocess.run([sys.executable, "I_rd_centre.py", os.path.join("D_work", new[-1]), "300", "192"], cwd=AG, capture_output=True, text=True)
        m = re.search(r"D\(0\) - C = ([+-][0-9.]+e[-+]\d+)", q.stdout)
        print("  %s rm %s: %s" % (name, rm, q.stdout.strip().splitlines()[0] if q.stdout.strip() else q.stderr[-200:]), flush=True)
        out.append((float(rm), float(m.group(1)) if m else None, res[-1] if res else None))
    return out


def main(argv):
    if "--selftest" in argv:
        return 0 if selftest() else 2
    if "--registration-run" in argv:
        ok = selftest()
        print("VERDICT q1887: INCOMPLETE (controls only)")
        return 0 if ok else 2
    if not selftest():
        print("VERDICT q1887: VOID (selftest failed)")
        return 2
    results = {}
    for name, lam, reg, rms, states in CASES:
        results[name] = run_case(name, lam, rms, states)
    v, lines = judge(results)
    for l in lines:
        print(l)
    print("VERDICT q1887: %s" % v)
    return 0 if v == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
