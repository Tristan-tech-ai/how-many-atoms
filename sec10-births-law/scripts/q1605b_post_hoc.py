"""q1605b (10 Oct 2026, READING, post hoc, no prediction registered, no verdict word): the first depth of the clean 180-atom block as d_1(120) plus the shift after the second Newton update of q1605, the ratio to the earlier shifts and a rough remainder estimate by the ratio pattern. Usage: py q1605b_post_hoc.py"""

import json
from mpmath import mp, mpf, nstr

mp.dps = 70
d120 = mpf(json.load(open("q1601_z_N120.json"))[0])
sh = mpf("1.6849438e-31")
d180 = d120 + sh
print("d_1(120) = %s" % nstr(d120, 55))
print(
    "d_1(180) = d_1(120) + %s = %s (the digits beyond the 38th decimal are limited by the 8-digit shift)"
    % (nstr(sh, 8), nstr(d180, 55))
)
shifts = [mpf("7.73310e-12"), mpf("1.81232e-18"), mpf("1.53487e-23"), mpf("8.3076e-28"), sh]
labels = ["10 -> 30", "30 -> 60", "60 -> 90", "90 -> 120", "120 -> 180"]
for i in range(1, 5):
    print("shift %s / shift %s = %s" % (labels[i - 1], labels[i], nstr(shifts[i - 1] / shifts[i], 4)))
r = shifts[3] / shifts[4]
print(
    "if the next ratio (180 -> 240 against 120 -> 180) is not below 1e+03, the shift beyond 180 atoms is below %s"
    % nstr(sh / mpf(1000), 3)
)
print("READING q1605b done (post hoc; no verdict word)")
