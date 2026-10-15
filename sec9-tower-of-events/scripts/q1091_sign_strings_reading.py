"""q1091 (7 Oct 2026): READING (no verdict): the full sign strings of r = S 1 + b_eff and of b_eff at the states given on the command line, with the number of adjacent equal signs (0 means perfect alternation) and the signs at the outermost atom. Uses state_items of q1090.
Usage: py q1091_sign_strings_reading.py STATE.json [STATE2.json ...]"""

import sys, os
import q1090_translation_residual_sign_verdict_v2 as Q

for p in sys.argv[1:]:
    r = Q.state_items(p)
    if r is None:
        print("VOID q1091: state does not polish", p)
        sys.exit(2)
    A_str, m, odd, g = r
    f = lambda s: "".join("+" if x > 0 else ("-" if x < 0 else "0") for x in s)
    adj = lambda s: sum(1 for a, b in zip(s, s[1:]) if a == b)
    print(
        f"READING q1091: {os.path.basename(p)}: m {m}\n  sign r : {f(g['sr'])}  (adjacent equal pairs {adj(g['sr'])}; outermost atom {g['sr'][-1]:+d})\n  sign b : {f(g['sb'])}  (adjacent equal pairs {adj(g['sb'])}; outermost atom {g['sb'][-1]:+d})",
        flush=True,
    )
print("END q1091 READING (no verdict)")
