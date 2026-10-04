"""G_track22 (03 Oct 2026): the tracker (D_track.cmd_track, imported, unchanged) with richer reads at every stored state
(coordinator's rule after the K 20 tracker missed a 1e-21 bump): D - C on 85..90 deg in 0.05-deg steps (101 points, written to
FINEDIR/fine_rm<rm>.json), D - C at exactly 90 deg, the maximum of the fine reads and its angle, and the curvature of D - C at every
atom (finite difference 1e-3 rad). The tracker's own validity test is unchanged (scan max over NSCAN + 1 points <= TOLV, masses > 0);
the event is read afterwards from the stored quantities.
Usage (env as D_solver: FUNC, RP, PREC, FDH, TOLR, TOLV, BTOL, NSCAN, STEPMAX, MAXIT, EVSCAN; plus FINEDIR):
  py G_track22.py OUTDIR RM_END STEP STATE_1[,STATE_2]"""

import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flint import arb
import D_solver as DS
import D_track as DT

f2 = DS.f2


def fine_reads(S, v, rm):
    P = S.P
    pi = P.pi
    ts = [pi * (arb(85) + arb(i) * arb("0.05")) / 180 for i in range(101)]
    fv = [f2(x) for x in P.kkt_many(v, rm, ts)]
    k = max(range(101), key=lambda i: fv[i])
    dd = arb("1e-3")
    d2 = []
    for t in P.reps(v[: S.nG]):
        tt = t if f2(t) > 1e-12 else arb(0)
        a = P.kkt(v, rm, tt - dd) if f2(tt) > 0 else P.kkt(v, rm, tt + dd)
        b = P.kkt(v, rm, tt + dd) if f2(tt) < 1.5707 else a
        d2.append(f2((a + b - 2 * P.kkt(v, rm, tt)) / (dd * dd)))
    out = {"DC_t90": fv[-1], "fine85_90_max": [round(85 + 0.05 * k, 2), fv[k]], "d2_atoms": d2}
    fd = os.environ.get("FINEDIR")
    if fd:
        os.makedirs(fd, exist_ok=True)
        rs = rm.mid().str(15, radius=False)
        json.dump(
            {"rm": rs, "deg": [round(85 + 0.05 * i, 2) for i in range(101)], "DC": fv, "d2_atoms": d2},
            open(os.path.join(fd, f"fine_rm{rs}.json"), "w"),
        )
    return out


DT.fine_reads = fine_reads

if __name__ == "__main__":
    DT.cmd_track(sys.argv[1:])
