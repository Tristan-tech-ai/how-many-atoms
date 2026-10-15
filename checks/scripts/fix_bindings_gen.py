"""one-off repair of make_bindings_unified_new.py (10 Oct 2026): powers of ten need 10**(-N), one constant binding replaced by a computed one."""
p = "make_bindings_unified_new.py"
t = open(p, encoding="utf-8").read()
n0 = t.count('LB(r"10^{-')
t = t.replace('LB(r"10^{-', 'PW(r"10^{-')
t = t.replace('def TB(printed, ctx, path, rx, why, post="", kind=None):',
              'def PW(printed, ctx, amend, rx, why, post="", kind=None):\n    B(printed, ctx, "10**(-L(%d, r\'%s\'))" % (amend, rx), why, kind)\ndef TB(printed, ctx, path, rx, why, post="", kind=None):', 1)
t = t.replace('r"stored states of K 40 and K 26" and r"of K (40) and K 26\\)"', 'r"of K (40) and K 26\\)"')
t = t.replace('"amendment 1919: u_end rho^2 / C for K 73 to 79 is within 0.52 percent of one", post="*0+0.5")', '"amendment 1919: u_end rho^2 / C for K 73 to 79 is within 0.52 percent of one", post="*(-100)+100")')
open(p, "w", encoding="utf-8", newline="\n").write(t)
print("PW replacements:", n0, "| PW def:", "def PW(" in t, "| cheat removed:", "*0+0.5" not in t, "| clean 40:", 'and r"of K' not in t)
