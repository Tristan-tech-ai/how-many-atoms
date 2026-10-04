"""d10_bulk_law: the bulk law rho^2 rho' = s0 S(rho) of the universal (linearised) problem, to O(rho^-8), from the continuum
first-harmonic phase condition arg Ghat = pi, evaluated EXACTLY on the saddle line (no gradient expansion).

Model. Depth coordinate d (density increasing), N' = rho, rho' = s0 S(rho)/rho^2, S = 1 + a2/rho^2 + a4/rho^4 + a6/rho^6 + a8/rho^8.
Ghat(d0) = int psi(u) exp(2 pi i [N(d0 + u) - N(d0)]) du, psi = N(0, 2). Contour shift u = i v + t (v = 4 pi rho0):
    arg Ghat(d0) = arg int psi(t) exp(2 pi i [N(d0 + iv + t) - N(d0) - rho0 (iv + t)]) dt.
N(d0 + w) - N(d0) is summed from the Taylor series of rho(d0 + s), generated from the ODE (radius ~ d0 >> v for rho0 >= 1).
The condition arg Ghat = pi (mod 2 pi) must hold at every rho0; it fixes a2, a4, a6, a8 order by order.
Checks: a2 = 1/(8 pi^2), a4 = 1/(64 pi^4) - 1/(96 pi^2) (the record's O(rho^-4) law) must come out.
Usage: py d10_bulk_law.py"""
import mpmath as mp, json, sys
mp.mp.dps = 40
pi = mp.pi; s0 = 1/(16*pi**2)
A2 = 1/(8*pi**2); A4 = 1/(64*pi**4) - 1/(96*pi**2)

def taylor_rho(rho0, a, nmax):
    """Taylor coefficients r_n of rho(d0 + s) for rho' = s0 (y + a2 y^2 + a4 y^3 + a6 y^4 + a8 y^5), y = 1/rho^2."""
    a2, a4, a6, a8 = a
    r = [mp.mpf(rho0)]; inv = [1/r[0]]; y = [inv[0]**2]
    P = {1: y}                                   # P[k] = y^k series
    for k in range(2, 6): P[k] = [y[0]**k]
    coef = {1: 1, 2: a2, 3: a4, 4: a6, 5: a8}
    for n in range(0, nmax):
        fn = s0*sum(coef[k]*P[k][n] for k in range(1, 6))
        r.append(fn/(n + 1))
        m = n + 1                                # extend inv, y, P[k] to order m
        inv.append(-sum(r[k]*inv[m - k] for k in range(1, m + 1))/r[0])
        y.append(sum(inv[k]*inv[m - k] for k in range(0, m + 1)))
        P[1] = y
        for k in range(2, 6):
            P[k].append(sum(P[k - 1][j]*y[m - j] for j in range(0, m + 1)))
    return r

def phase(rho0, a, nmax=90, L=18):
    rho0 = mp.mpf(rho0); v = 4*pi*rho0; r = taylor_rho(rho0, a, nmax)
    def dN(w):   # N(d0 + w) - N(d0) - rho0 w
        s = mp.mpc(0); wp = w
        for n in range(1, nmax + 1):
            wp = wp*w; s += r[n]*wp/(n + 1)
        return s
    f = lambda t: mp.exp(-t**2/4)/(2*mp.sqrt(pi))*mp.expj(2*pi*dN(mp.mpc(t, v)))
    G = mp.quad(f, mp.linspace(-L, L, 37))
    ph = mp.arg(G)
    if ph < 0: ph += 2*pi
    return ph - pi, abs(G), r

if __name__ == "__main__":
    out = {}
    print("check of the record's law: Phi(rho) = arg Ghat - pi with (a2, a4) exact and a6 = a8 = 0")
    for rho0 in [1.0, 1.5, 2.0, 3.0, 4.0]:
        ph, g, r = phase(rho0, (A2, A4, 0, 0))
        ph2, _, _ = phase(rho0, (A2, 0, 0, 0))
        ph0, _, _ = phase(rho0, (0, 0, 0, 0))
        print(f"  rho {rho0}: Phi(a2,a4) = {mp.nstr(ph, 8)}  Phi rho^6 = {mp.nstr(ph*rho0**6, 8)} | a4 = 0: Phi rho^4 = {mp.nstr(ph2*rho0**4, 8)}"
              f" (pred pi*a4 = {mp.nstr(pi*A4, 6)}) | a2 = 0: Phi rho^2 = {mp.nstr(ph0*rho0**2, 6)} (pred pi a2 = {mp.nstr(pi*A2, 6)})  |G| {mp.nstr(g, 5)}", flush=True)
    # solve for a6 at each rho (a8 = 0), then for (a6, a8) jointly from two rho values
    print("\na6 from Phi(rho) = 0 with a8 = 0, rho by rho:")
    a6s = {}
    for rho0 in [2.0, 3.0, 4.0, 6.0, 8.0]:
        f = lambda a6: phase(rho0, (A2, A4, a6, 0))[0]
        a6r = mp.findroot(f, (mp.mpf(0), mp.mpf('1e-4')), solver='secant', tol=mp.mpf(10)**-30)
        a6s[rho0] = a6r
        print(f"  rho {rho0}: a6 = {mp.nstr(a6r, 12)}", flush=True)
    # Richardson in 1/rho^2 (a6(rho) = a6 + a8'/rho^2 + ...)
    ks = sorted(a6s)
    for i in range(len(ks) - 1):
        r1, r2 = ks[i], ks[i + 1]
        e = (a6s[r2]*r2**2 - a6s[r1]*r1**2)/(r2**2 - r1**2)
        print(f"  Richardson ({r1},{r2}): a6 = {mp.nstr(e, 12)}")
    print("\njoint (a6, a8) from two rho values:")
    for (r1, r2) in [(3.0, 4.0), (4.0, 6.0), (6.0, 8.0)]:
        F = lambda a6, a8: [phase(r1, (A2, A4, a6, a8))[0], phase(r2, (A2, A4, a6, a8))[0]]
        sol = mp.findroot(F, (a6s[r2], mp.mpf(0)), tol=mp.mpf(10)**-28)
        print(f"  ({r1},{r2}): a6 = {mp.nstr(sol[0], 12)}, a8 = {mp.nstr(sol[1], 12)}", flush=True)
        out[f"{r1},{r2}"] = [str(sol[0]), str(sol[1])]
    json.dump(out, open("d10_bulk_law_out.json", "w"), indent=1)
