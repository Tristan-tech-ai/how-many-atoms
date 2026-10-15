r"""q1877_pr2d_MA_limit_reading.py (14 Oct 2026, task PR2d, claim B3): numerical check of the lemma used by the dual upper bound,
M_A - 2A -> 2 int (G - 1{t<0}) dt = 2(sum nu - g), where M_A = int G(y - A) G(-y - A) dy and G is the test profile of q1876
(g = 4.361045; atoms (0, 3.49413), (-2.68983, 2.25076), (-4.361045, 0.62400)).
Proof of the lemma (written before the run): with R = G - H, H = 1{t < 0},
  G(y - A) G(-y - A) - 1{|y| < A} = R(y - A) H(-y - A) + H(y - A) R(-y - A) + R(y - A) R(-y - A).
The first two terms integrate to int_{t > -2A} R each. In the third, max(|y - A|, |y + A|) >= A, so its integral is at most
2 sup_{|u| >= A} |R(u)| int |R|. R is integrable and |R(u)| -> 0 as |u| -> inf (Gaussian tails on both sides), so M_A - 2A -> 2 int R,
and int R = sum nu - g because int (Phi(-g - t) - 1{t < 0}) dt = -g.
A reading with no verdict word; it writes nothing.
Registered before the first run (mpmath, 30 digits):
 P1 2 int R evaluated by quadrature equals 2(sum nu - g) = 4.01569 to 1e-20;
 P2 |M_A - 2A - 4.01569| < 1e-10 at A = 10, 20 and 40.
Usage: py small_gap_q206/q1877_pr2d_MA_limit_reading.py   (from temporal_within/)"""

import mpmath as mp

mp.mp.dps = 30
g = mp.mpf("4.361045")
at = [
    (mp.mpf("0"), mp.mpf("3.49413")),
    (mp.mpf("-2.68983"), mp.mpf("2.25076")),
    (mp.mpf("-4.361045"), mp.mpf("0.62400")),
]
phi = lambda t: mp.exp(-t * t / 2) / mp.sqrt(2 * mp.pi)
G = lambda t: mp.ncdf(-g - t) + sum(n * phi(t - s) for s, n in at)
dual = 2 * (sum(n for _, n in at) - g)
R = lambda t: G(t) - (1 if t < 0 else 0)
iR = mp.quad(R, [-mp.inf, -40, -10, -g, 0, 10, mp.inf])
p1 = abs(2 * iR - dual) < mp.mpf("1e-20")
rows = []
for A in (10, 20, 40):
    A = mp.mpf(A)
    f = lambda y: G(y - A) * G(-y - A)
    M = mp.quad(
        f, sorted(set([-A - 30, -A - 8, -A, -A + 8, -A + 30, mp.mpf(0), A - 30, A - 8, A, A + 8, A + 30]))
    )
    rows.append((A, M - 2 * A - dual))
p2 = all(abs(d) < mp.mpf("1e-10") for _, d in rows)
print("2 int R = %s; 2(sum nu - g) = %s" % (mp.nstr(2 * iR, 25), mp.nstr(dual, 25)))
for A, d in rows:
    print("A = %s: M_A - 2A - 2(sum nu - g) = %s" % (mp.nstr(A, 3), mp.nstr(d, 5)))
print("P1 %s; P2 %s" % tuple("held" if p else "did not hold" for p in (p1, p2)))
print(
    "READING (no verdict): the limit lemma of the dual upper bound checked by quadrature; the script writes nothing."
)
