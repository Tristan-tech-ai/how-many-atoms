"""G_trackE (03 Oct 2026): the tracker (D_track.cmd_track, imported, unchanged) with reads at both vertices at every
stored state: D - C at t = 0 and t = pi/2, D - C on 0..5 deg and 85..90 deg in 0.05-deg steps (written to FINEDIR/fine_rm<rm>.json),
the maxima of both fine windows with their angles, and the curvature of D - C at every atom (finite difference, step DD rad, default
1e-3). The tracker's validity test is unchanged (scan max over NSCAN + 1 points <= TOLV, masses > 0); the event is read afterwards from
the stored quantities. Generalises G_track22.py (one vertex).
Usage (env as D_solver: FUNC, RP, PREC, FDH, TOLR, TOLV, BTOL, NSCAN, STEPMAX, MAXIT, EVSCAN, CHORD, REUSEJ, PRED, NPROC; plus FINEDIR, DD):
  py G_trackE.py OUTDIR RM_END STEP STATE_1[,STATE_2]"""

import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flint import arb
import D_solver as DS
import D_track as DT

f2 = DS.f2


def fine_reads(S, v, rm):
    P = S.P
    pi = P.pi
    lo = [pi * (arb(i) * arb("0.05")) / 180 for i in range(101)]
    hi = [pi * (arb(85) + arb(i) * arb("0.05")) / 180 for i in range(101)]
    fl = [f2(x) for x in P.kkt_many(v, rm, lo)]
    fh = [f2(x) for x in P.kkt_many(v, rm, hi)]
    kl = max(range(101), key=lambda i: fl[i])
    kh = max(range(101), key=lambda i: fh[i])
    dd = arb(os.environ.get("DD", "1e-3"))
    d2 = []
    for t in P.reps(v[: S.nG]):
        tt = t if f2(t) > 1e-12 else arb(0)
        a = P.kkt(v, rm, tt - dd) if f2(tt) > 0 else P.kkt(v, rm, tt + dd)
        b = P.kkt(v, rm, tt + dd) if f2(tt) < 1.5707 else a
        d2.append(f2((a + b - 2 * P.kkt(v, rm, tt)) / (dd * dd)))
    out = {
        "DC_t0": fl[0],
        "DC_t90": fh[-1],
        "fine0_5_max": [round(0.05 * kl, 2), fl[kl]],
        "fine85_90_max": [round(85 + 0.05 * kh, 2), fh[kh]],
        "d2_atoms": d2,
    }
    fd = os.environ.get("FINEDIR")
    if fd:
        os.makedirs(fd, exist_ok=True)
        rs = rm.mid().str(20, radius=False)
        json.dump(
            {
                "rm": rs,
                "deg_lo": [round(0.05 * i, 2) for i in range(101)],
                "DC_lo": fl,
                "deg_hi": [round(85 + 0.05 * i, 2) for i in range(101)],
                "DC_hi": fh,
                "d2_atoms": d2,
            },
            open(os.path.join(fd, f"fine_rm{rs}.json"), "w"),
        )
    return out


DT.fine_reads = fine_reads

if __name__ == "__main__":
    DT.cmd_track(sys.argv[1:])
