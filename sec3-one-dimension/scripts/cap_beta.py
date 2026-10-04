"""beta(A) = 1/c* - 2A = e^{h(Y)} - 2A on stored Gaussian capacity states (pred_cap_beta.txt). Read only; writes nothing.
Also 2A w_A J'(A) against I(p) - 1, h(Y) against -log c*, and the centre ripple p(0)/c* - 1."""

import json, os, sys, mpmath as mp

mp.mp.dps = int(os.environ.get("DPS", "30"))
h = mp.mpf(os.environ.get("QSTEP", "0.5"))


def grid(a, b):
    n = max(2, int((b - a) / h) + 1)
    return mp.linspace(a, b, n)


for f in sys.argv[1:]:
    d = json.load(open(f))
    A = mp.mpf(str(d["A"]).replace("np.float64(", "").rstrip(")"))
    u = [mp.mpf(s) for s in d["u"]]
    w = [mp.mpf(s) for s in d["w"]]
    we = mp.mpf(d["we"])
    wc = None if d.get("wc") is None else mp.mpf(d["wc"])
    xs = [-A] + [-x for x in u[::-1]] + ([mp.mpf(0)] if wc is not None else []) + u + [A]
    ws = [we] + w[::-1] + ([wc] if wc is not None else []) + w + [we]
    p = lambda y: mp.fsum(wi * mp.npdf(y - x) for x, wi in zip(xs, ws))
    p1 = lambda y: mp.fsum(-wi * (y - x) * mp.npdf(y - x) for x, wi in zip(xs, ws))
    J = lambda x: mp.quad(lambda y: mp.npdf(y - x) * mp.log(p(y)), grid(x - 12, x + 12))
    Jp = lambda x: mp.quad(lambda y: mp.npdf(y - x) * p1(y) / p(y), grid(x - 12, x + 12))
    cst = mp.e ** J(u[0])
    beta = 1 / cst - 2 * A
    lo, hi = -A - 12, A + 12
    I = mp.quad(lambda y: p1(y) ** 2 / p(y), grid(lo, hi))
    H = -mp.quad(lambda y: p(y) * mp.log(p(y)), grid(lo, hi))
    JA = Jp(A)
    vir = 2 * A * we * JA - (I - 1)
    eps = p(mp.mpf(0)) / cst - 1
    print(
        "%s A=%s K=%d res=%s %s | beta=%s | 2A w_A J'(A)-(I-1)=%s | h(Y)+log c*=%s | eps=%s | w_A J'(A)/c*+1=%s | J'(u1)=%s"
        % (
            f,
            mp.nstr(A, 10),
            len(xs),
            d.get("residual"),
            str(d.get("verdict"))[:14],
            mp.nstr(beta, 12),
            mp.nstr(vir, 3),
            mp.nstr(H + mp.log(cst), 3),
            mp.nstr(eps, 4),
            mp.nstr(we * JA / cst + 1, 4),
            mp.nstr(Jp(u[0]), 3),
        ),
        flush=True,
    )
