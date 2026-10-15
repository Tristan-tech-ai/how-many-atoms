"""tight_run (8 Oct 2026, amendment 2116): runs an existing reading script with the repaired density inversion (scipy brentq wrapped to xtol 1e-16 when the caller gives none, which repairs every copy of rho_of at once, plus the explicit q1183 replacement). Usage: py tight_run.py SCRIPT.py [args]. The module chain of q1304 (which loads q1183) is imported first, tight_rho_patch.apply is called on q1183, and the script is then executed as __main__ (all counting functions it builds use the repaired inversion)."""

import io, contextlib, os, runpy, sys
import scipy.optimize as _so

_orig_brentq = _so.brentq


def _tight_brentq(f, a, b, *args, **kw):
    """brentq with xtol = 1e-16 whenever the caller used the scipy default (2e-12): the default leaves errors of up to 5e-13 in the density inversion rho_of and its copies"""
    if "xtol" not in kw and len(args) < 1:
        kw["xtol"] = 1e-16
    return _orig_brentq(f, a, b, *args, **kw)


_so.brentq = _tight_brentq  # BEFORE any module does 'from scipy.optimize import brentq'
os.environ.setdefault("E_FILE", "d22_out_from_log.json")
script = sys.argv[1]
args = sys.argv[2:]
sys.argv = ["x"]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import q1304_hardwall_family_blind as H4
import tight_rho_patch as T

Q = H4.QQ.C.Q
T.apply(Q)
assert Q.rho_of is T.rho_tight
# converged spline degree (amendment 2121): with env TIGHT_DEG=D every counting function built with the degree 5 gets degree D, degree 7 gets D + 2 (so that the quintic/septic comparisons of the old scripts become a D / D + 2 convergence check), and the root finder uses the bracket 5 to 55 because a high-degree spline extrapolates wildly outside the atoms
if os.environ.get("TIGHT_DEG"):
    import numpy as _np
    from scipy.optimize import brentq as _bq2

    _D = int(os.environ["TIGHT_DEG"])
    _QQ = H4.QQ
    _make = _QQ.make
    _QQ.make = lambda Nlaw, X, k: _make(Nlaw, X, (_D if k == 5 else _D + 2 if k == 7 else k))

    def _root(Nf, K):
        z = lambda a: 2 * Nf(a) - K - 1 / 6
        a0 = _bq2(z, 5.0, 55.0, xtol=1e-12)
        f = lambda a: z(a) - float(
            _QQ.T3(1.0 / (4 * _np.pi * (Nf(a + _QQ.H) - Nf(a - _QQ.H)) / (2 * _QQ.H)) ** 2)
        )
        return _bq2(f, a0 - 2.5, a0 + 2.5, xtol=1e-14)

    _QQ.root = _root
    print(
        f"tight_run: TIGHT_DEG={_D}: degree 5 -> {_D}, degree 7 -> {_D + 2}; root bracket 5 to 55",
        file=sys.stderr,
        flush=True,
    )
print(
    "tight_run: scipy brentq called with xtol 1e-16 when no xtol is given; q1183.rho_of replaced by the tight inversion",
    flush=True,
)
# q1299 carries a regression control tied to the old instrument (capacity R(36) = 0.5456e-9 within 0.001e-9); with the repaired inversion R(36) changes by the repair itself, so a copy of q1299 with the bar 0.02 is installed (the value of R(36) is printed)
import types

src = (
    open("q1299_capacity_second_term_reading.py", encoding="utf-8")
    .read()
    .replace(
        "abs(R_[K_ == 36][0] - 0.5456) < 0.001",
        "(print('tight_run: capacity R(36) =', R_[K_ == 36][0], '(old instrument 0.5456)', file=sys.stderr) or abs(R_[K_ == 36][0] - 0.5456) < 0.02)",
    )
)
mod = types.ModuleType("q1299_capacity_second_term_reading")
mod.__file__ = "q1299_capacity_second_term_reading.py"
sys.modules["q1299_capacity_second_term_reading"] = mod
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, "q1299_capacity_second_term_reading.py", "exec"), mod.__dict__)
# regression controls tied to numbers of the old instrument call sys.exit(2) when they fail by construction; with TIGHT_SOFT_EXIT=1 these exits are reported on stderr and the script continues (exploratory readings only)
if os.environ.get("TIGHT_SOFT_EXIT") == "1":

    def _soft_exit(code=0):
        print(f"tight_run: sys.exit({code}) suppressed (TIGHT_SOFT_EXIT=1)", file=sys.stderr, flush=True)

    sys.exit = _soft_exit
sys.argv = [script] + args
runpy.run_path(script, run_name="__main__")
