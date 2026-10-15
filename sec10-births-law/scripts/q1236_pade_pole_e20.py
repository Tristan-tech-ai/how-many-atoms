"""q1236 (8 Oct 2026, item E3 of amendment 1994): the diagonal Pade [9/9] of the Borel series built as in q1220 from e_2 .. e_20 (d22_out_20.json, inner series of 19 coefficients, weight Gamma(n + 3/2)); item E3: the smallest positive real pole y of the denominator has 1/y in [4.47, 4.58].
Control (first): the same function applied to e_2 .. e_18 with [8/8] returns the poles 1/y = 4.42976 and 4.08335 of q1220 within 1e-4.
Readings without item: all real positive poles with their 1/y, the complex poles nearest to the real axis, and the [9/9] poles of a 17-coefficient [8/8] for comparison.
Usage: py q1236_pade_pole_e20.py  (from temporal_within/centre_scripts/)"""

import json, re, sys
import mpmath as mp

mp.mp.dps = 80
B = mp.mpf("1.5")


def poles(fn, nmax, L):
    E = {
        int(k[1:]): mp.mpf(v)
        for k, v in json.load(open(fn)).items()
        if re.fullmatch(r"e\d+", k) and k != "e1"
    }
    c = [E[k + 2] / mp.gamma(k + 2 + B) for k in range(nmax - 1)]
    p, q = mp.pade(c, L, L)
    rt = mp.polyroots(list(reversed(q)), maxsteps=400, extraprec=400)
    return rt


def main():
    rt18 = poles("d22_out_18.json", 18, 8)
    real18 = sorted([mp.re(z) for z in rt18 if abs(mp.im(z)) < mp.mpf(10) ** -30 and mp.re(z) > 0])
    c = len(real18) >= 2 and abs(1 / real18[0] - 4.42976) < 1e-4 and abs(1 / real18[1] - 4.08335) < 1e-4
    print(
        "control: [8/8] with e_2..e_18, real positive poles 1/y: "
        + ", ".join(mp.nstr(1 / y, 8) for y in real18)
        + f"; reproduces q1220: {c}"
    )
    if not c:
        print("READING q1236: VOID (control)")
        sys.exit(2)
    rt20 = poles("d22_out_20.json", 20, 9)
    real20 = sorted([mp.re(z) for z in rt20 if abs(mp.im(z)) < mp.mpf(10) ** -30 and mp.re(z) > 0])
    print(
        "[9/9] with e_2..e_20, real positive poles y (1/y): "
        + ", ".join(f"{mp.nstr(y, 8)} ({mp.nstr(1 / y, 8)})" for y in real20)
    )
    cx = sorted([z for z in rt20 if abs(mp.im(z)) >= mp.mpf(10) ** -30], key=lambda z: abs(mp.im(z)))[:4]
    print("complex poles nearest to the real axis (1/y): " + "; ".join(mp.nstr(1 / z, 6) for z in cx))
    e3 = bool(real20) and 4.47 <= float(1 / real20[0]) <= 4.58
    print(f"smallest positive real pole 1/y = {mp.nstr(1 / real20[0], 8)}; item E3 ([4.47, 4.58]): {e3}")
    print("READING q1236: E3 " + str(e3) + " (no verdict)")


if __name__ == "__main__":
    main()
