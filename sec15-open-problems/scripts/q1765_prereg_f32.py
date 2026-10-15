"""q1765 (10 Oct 2026, REGISTRATION, no verdict word): prospective predictions for f32 = F_16, the next bulk coefficient, from the exact structure of readings 205 (F_j = 8^-j sum_k c_k(j) u^(j-k), u = 1/pi^2, with c_1(j) = -j(j-1)(6j^3 + j^2 - 19j + 16)/90 and c_2(j) = j(j-1)(j-2)(j-3) R(j)/340200). The exact run py q1557_f12_symbolic.py 32 (PID in the registration file) does not have an output yet; F_16 was never used to find the structure (j <= 15 only).
Predictions: (R1) the coefficient of pi^-32 in f32 is exactly 1/8^16; (R2) the coefficient of pi^-30 times 8^16 equals c_1(16); (R3) the coefficient of pi^-28 times 8^16 equals c_2(16) from the k = 2 formula; (R4) the lowest power of 1/pi in f32 is pi^-16 (u^8) and f32 has no lower power; (R5) as a consequence the logarithm coefficient L_16 has the top coefficient 1/(16 x 8^16) and its u^15 coefficient times 16 x 8^16 equals -(15 x 16 x 17)(3 x 256 - 128 + 6)/9 (the level formula of q1764).
Usage: py q1765_prereg_f32.py (writes q1765_prereg_f32.txt with the values and the sha256 of this script)."""

import os, hashlib, datetime
import sympy as sp

H = os.path.dirname(os.path.abspath(__file__))
j = sp.Integer(16)
c1 = -j * (j - 1) * (6 * j**3 + j**2 - 19 * j + 16) / 90
c2 = (
    j
    * (j - 1)
    * (j - 2)
    * (j - 3)
    * (118 * j**6 - 1092 * j**5 + 1135 * j**4 + 14790 * j**3 - 46973 * j**2 + 27054 * j + 22212)
    / 340200
)
l1 = -(j - 1) * j * (j + 1) * (3 * j**2 - 8 * j + 6) / 9
me = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
txt = (
    "Registration q1765 at %s on %s by code (f32 = F_16 does not exist yet; the run py q1557_f12_symbolic.py 32 starts after this file is written). Predictions: R1 the coefficient of pi^-32 is 1/%d; R2 the coefficient of pi^-30 times 8^16 is %s; R3 the coefficient of pi^-28 times 8^16 is %s; R4 the lowest power is pi^-16 and no lower power occurs; R5 the logarithm coefficient L_16 has the u^15 coefficient times 16 x 8^16 equal to %s. Rule and formulas in the docstring of small_gap_q206/q1765_prereg_f32.py (sha256 %s). Evaluate with q1766 when q1557_stdout_ORD32.txt contains the line for f32.\n"
    % (datetime.datetime.now().strftime("%H:%M:%S"), datetime.date.today().isoformat(), 8**16, c1, c2, l1, me)
)
open(os.path.join(H, "q1765_prereg_f32.txt"), "w", encoding="utf-8").write(txt)
print(txt)
