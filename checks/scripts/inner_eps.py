"""Inner eccentricity eps_i = (A - B)/(A + B) of the inner formal-density curve at the observed inner-pair losses (Table V of paper 1).

harm(), the row pattern and the cubic fit through the four logged rows around each rm are TEXT COPIES of
temporal_within/centre_scripts/q841_harm_at.py (the script behind 1477's 'E - 1 ... eps 0.048 / 0.100'). Inputs (read only):
  LFP rp 2.8:  centre_scripts/q833_lfp28_inner.log at the observed loss rm 2.246310 (1477)
  rd  rp 3.1:  centre_scripts/q833_rd31_inner.log  at the observed loss rm 2.498436 (1477)
  cap rp 4.3:  centre_scripts/q822_rp43_MI1.log    at the observed loss rm 3.29254  (1439)
Writes only inner_eps.json in this folder.
"""
import os, re, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "centre_scripts")
CASES = {"lfp_rp2.8": ("q833_lfp28_inner.log", 2.246310), "rd_rp3.1": ("q833_rd31_inner.log", 2.498436),
         "cap_rp4.3": ("q822_rp43_MI1.log", 3.29254)}
pat = re.compile(r"rm ([0-9.]+): .*inner radius a_0..: (\[[^\]]*\]); inner d_1..: (\[[^\]]*\]).*residual ([0-9.e+-]+)")


def harm(a, d, K):
    th = np.linspace(0, 2*np.pi, 200001)[:-1]
    rho = sum(ak*np.cos(2*k*th) for k, ak in enumerate(a)); g = 1 + sum(dk*np.cos(2*(k + 1)*th) for k, dk in enumerate(d))
    x, y = rho*np.cos(th), rho*np.sin(th); A, B = rho[0], rho[len(th)//4]
    s = np.unwrap(np.arctan2(y/B, x/A))
    return 2*np.mean(np.cos(K*s)*g), 2*np.mean(np.cos(K*th)*g), (A - B)/(A + B)


def main():
    out = {}
    for key, (log, rm) in CASES.items():
        rows = {}
        for line in open(os.path.join(CS, log), encoding="utf-8", errors="replace"):
            m = pat.search(line)
            if m:
                rows[float(m.group(1))] = harm(eval(m.group(2)), eval(m.group(3)), 2)
        xs = np.array(sorted(rows))
        i = int(np.searchsorted(xs, rm)); idx = [j for j in range(i - 2, i + 2) if 0 <= j < len(xs)]
        h = np.polyval(np.polyfit(xs[idx] - rm, [rows[xs[j]][0] for j in idx], len(idx) - 1), 0.0)
        ep = np.polyval(np.polyfit(xs[idx] - rm, [rows[xs[j]][2] for j in idx], len(idx) - 1), 0.0)
        out[key] = dict(log=log, rm=rm, rows_used=[float(xs[j]) for j in idx], eccentric_harmonic=float(h), E_minus_1=float(h/2 - 1),
                        eps_i=float(ep))
        print(f"{key}: {log} rm {rm}: rows {[float(xs[j]) for j in idx]}; harmonic {h:.6f}, E - 1 = {h/2 - 1:+.4e}, eps_i {ep:.6f}")
    json.dump(out, open(os.path.join(HERE, "inner_eps.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
