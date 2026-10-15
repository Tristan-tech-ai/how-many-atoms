"""c33: the capacity system entirely in 320-bit fixed point on the GPU (integer units only): residual, Jacobian (GEMMs G1 = RQ C, G2 = RSQ C in fixed point), LU solve in fixed point.
No double-double, no mixed iteration: the Jacobian is as accurate as the residual (unit 2^-288), so Newton converges quadratically to the 320-bit floor and there is no condition-number limit
below about 1e50. Interface of the walker (GPUF-like): FXF.residual_and_jacobian(A, u, w, we, wc, want_J, y) -> (residual as arb list, FXMat with .solve(arb_mat)).
Control: py c33_fx_full.py STATEFILE [H]  (J against arb 192 bits, solve against arb solve, timings; and K = 60 .. 200 synthetic sizes).
"""

import sys, os, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, HERE)
import numpy as np, cupy as cp
from flint import arb, arb_mat, fmpz, ctx
import c30_fx_residual as C30
from c30_fx_residual import to_limbs, from_limbs, NLIMB

NL = NLIMB
FXF_SRC = r"""
__device__ __forceinline__ fx fx_sar(const fx& a, int n){ bool s=fx_negp(a); fx m=s?fx_neg(a):a; fx r=fx_shift(m,-n); return s?fx_neg(r):r; }
__device__ fx fx_exp(const fx& x, const fx& ln2){
  double xd=fx_to_double(x);
  if (xd < -200.0 || xd > 20.0) return fx_zero();
  int k=(int)nearbyint(xd*1.4426950408889634);
  fx kl=fx_mul_small(ln2,(u32)(k<0?-k:k)); fx r=(k<0)? fx_add(x,kl) : fx_sub(x,kl); fx rs=fx_sar(r,8);
  fx one=fx_zero(); one.v[9]=1; fx term=one, sum=one;
  for(int j=1;j<=30;j++){ term=fx_div_small(fx_mul(term,rs),(u32)j); sum=fx_add(sum,term); }
  #pragma unroll
  for(int q=0;q<8;q++) sum=fx_mul(sum,sum);
  return fx_shift(sum,k);
}
__device__ fx fx_recip1(const fx& d){      // d in [1,2): reciprocal by Newton from a double estimate
  double dd=fx_to_double(d); fx x=fx_from_double(1.0/dd); fx two=fx_zero(); two.v[9]=2;
  #pragma unroll
  for(int it=0;it<3;it++){ x=fx_mul(x, fx_sub(two, fx_mul(d,x))); }
  return x; }
__device__ __forceinline__ int fx_lead(const fx& m){ int top=NL-1; while(top>0 && m.v[top]==0) top--; return 32*top + (31 - __clz(m.v[top])); }   // magnitude, nonzero
__device__ fx fx_sshift(const fx& a, int n){ bool s=fx_negp(a); fx m=s?fx_neg(a):a; fx r=fx_shift(m,n); return s?fx_neg(r):r; }              // signed shift: n>0 left, n<0 right
__device__ bool fx_iszero(const fx& a){ bool z=true;
  #pragma unroll
  for(int i=0;i<NL;i++) z = z && (a.v[i]==0); return z; }
__device__ fx fx_div(const fx& a, const fx& b){        // a / b, result must fit in 31 integer bits
  if (fx_iszero(a)) return fx_zero();
  bool sb=fx_negp(b); fx bm=sb?fx_neg(b):b; int bl=fx_lead(bm); int k=bl-288; fx bn=fx_shift(bm, 288-bl); fx r=fx_recip1(bn);
  fx q=fx_mul(a,r); q=fx_sshift(q,-k); return sb?fx_neg(q):q; }
__device__ bool fx_abs_gt(const fx& a, const fx& b){ fx x=fx_abs(a), y=fx_abs(b);
  for(int i=NL-1;i>=0;i--){ if(x.v[i]!=y.v[i]) return x.v[i]>y.v[i]; } return false; }

// ---- kernel rows by fx_exp: P[r][i], D[r][i] = y_i - x_r on the band
extern "C" __global__ void k_fx_rows_exp(const u32* XR, const int* i0, const int* i1, const u32* Y0, const u32* HF, const u32* L2PP, const u32* LN2, int nrows, int N, u32* P, u32* D){
  long tid=(long)blockIdx.x*blockDim.x+threadIdx.x; if (tid >= (long)nrows*N) return; int r=tid/N; int i=tid-(long)r*N;
  if (i < i0[r] || i >= i1[r]) { st_fx(P+tid*NL, fx_zero()); st_fx(D+tid*NL, fx_zero()); return; }
  fx y=fx_add(ld_fx(Y0), fx_mul_small(ld_fx(HF),(u32)i)); fx d=fx_sub(y, ld_fx(XR + r*NL)); fx L=fx_neg(fx_add(fx_shr1(fx_mul(d,d)), ld_fx(L2PP)));
  st_fx(P+tid*NL, fx_exp(L, ld_fx(LN2))); st_fx(D+tid*NL, d); }
// 1/Q_i as (mantissa in (0.5,1], exponent): invQ = rq * 2^(-k)
extern "C" __global__ void k_fx_qinv(const u32* Q, int N, u32* RQ, int* KQ){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if (i>=N) return; fx q=ld_fx(Q+(long)i*NL); int bl=fx_lead(q); int k=bl-288; fx qn=fx_shift(q,288-bl); st_fx(RQ+(long)i*NL, fx_recip1(qn)); KQ[i]=k; }
// RQ[a][i] = P[row_a][i] / Q_i ; RS[a][i] = RQ * D[row_a][i]  (a < m)
extern "C" __global__ void k_fx_RQ(const u32* P, const u32* D, const int* rowmap, int nrq, int m, int N, const u32* RINV, const int* KQ, u32* RQo, u32* RSo){
  long tid=(long)blockIdx.x*blockDim.x+threadIdx.x; if (tid >= (long)nrq*N) return; int a=tid/N; int i=tid-(long)a*N; long src=((long)rowmap[a]*N+i);
  fx p=ld_fx(P+src*NL); fx q=fx_sshift(fx_mul(p, ld_fx(RINV+(long)i*NL)), -KQ[i]); st_fx(RQo+tid*NL, q);
  if (a < m) st_fx(RSo+tid*NL, fx_mul(q, ld_fx(D+src*NL))); }
// C[i][c]
extern "C" __global__ void k_fx_C(const u32* P, const u32* D, const u32* WT, int m, int ncol, int N, int rowA, int rowmA, int rowc, u32* Co){
  long tid=(long)blockIdx.x*blockDim.x+threadIdx.x; if (tid >= (long)N*ncol) return; int i=tid/ncol; int c=tid-(long)i*ncol; fx v=fx_zero();
  if (c < m){ long a=(long)c*N+i, b=(long)(m+c)*N+i; fx wk=ld_fx(WT+c*NL);
    fx t1=fx_mul(fx_mul(wk, ld_fx(P+a*NL)), ld_fx(D+a*NL)), t2=fx_mul(fx_mul(wk, ld_fx(P+b*NL)), ld_fx(D+b*NL)); v=fx_sub(t1,t2); }
  else if (c < 2*m){ int k=c-m; v=fx_add(ld_fx(P+((long)k*N+i)*NL), ld_fx(P+((long)(m+k)*N+i)*NL)); }
  else if (c == 2*m){ v=fx_add(ld_fx(P+((long)rowA*N+i)*NL), ld_fx(P+((long)rowmA*N+i)*NL)); }
  else v=ld_fx(P+((long)rowc*N+i)*NL);
  st_fx(Co+tid*NL, v); }
// fixed-point GEMM  G (M x ncol) = X (M x N) * C (N x ncol), 8x8 tiles
extern "C" __global__ void k_fx_gemm(const u32* X, const u32* Cm, int M, int ncol, int N, u32* G){
  __shared__ fx sX[8][9]; __shared__ fx sC[8][9];
  int tx=threadIdx.x, ty=threadIdx.y; int row=blockIdx.y*8+ty, col=blockIdx.x*8+tx; fx acc=fx_zero();
  for (int k0=0;k0<N;k0+=8){ int kx=k0+tx, ky=k0+ty;
    sX[ty][tx] = (row<M && kx<N) ? ld_fx(X+((long)row*N+kx)*NL) : fx_zero();
    sC[ty][tx] = (col<ncol && ky<N) ? ld_fx(Cm+((long)ky*ncol+col)*NL) : fx_zero();
    __syncthreads();
    #pragma unroll
    for (int k=0;k<8;k++) acc=fx_add(acc, fx_mul(sX[ty][k], sC[k][tx]));
    __syncthreads(); }
  if (row<M && col<ncol) st_fx(G+((long)row*ncol+col)*NL, acc); }
// assemble J (n x n) from G1, G2, sums S (per eval row: t0,t1,t2 = raw sums; times H applied here)
extern "C" __global__ void k_fx_assemble(const u32* G1, const u32* G2, const u32* S, const u32* HF, int m, int ncol, int odd, u32* J){
  int n=ncol; long tid=(long)blockIdx.x*blockDim.x+threadIdx.x; if (tid >= (long)n*n) return; int a=tid/n, b=tid-(long)a*n; fx H=ld_fx(HF); fx v=fx_zero();
  if (a < m){ fx g=fx_sub(ld_fx(G1+((long)m*ncol+b)*NL), ld_fx(G1+((long)a*ncol+b)*NL)); v=fx_mul(g,H);
    if (b==a) v=fx_add(v, fx_mul(ld_fx(S+((long)a*3+1)*NL), H)); }
  else if (a < 2*m){ int j=a-m; v=fx_neg(fx_mul(ld_fx(G2+((long)j*ncol+b)*NL), H)); if (b==j) v=fx_add(v, fx_mul(ld_fx(S+((long)j*3+2)*NL), H)); }
  else if (a==2*m){ fx two=fx_zero(); two.v[9]=2; fx one=fx_zero(); one.v[9]=1; if (b>=m && b<2*m) v=two; else if (b==2*m) v=two; else if (odd && b==2*m+1) v=one; }
  else { v=fx_mul(fx_sub(ld_fx(G1+((long)m*ncol+b)*NL), ld_fx(G1+((long)(m+1)*ncol+b)*NL)), H); }
  st_fx(J+tid*NL, v); }
// LU with partial pivoting + solve in fixed point (one block). A (n x n, overwritten), b (n, overwritten with x)
extern "C" __global__ void k_fx_lu_solve(u32* A, u32* B, int n){
  __shared__ int best[256]; __shared__ int prow;
  for (int k=0;k<n;k++){
    int bi=k; fx bv=fx_zero(); bool have=false;
    for (int i=k+threadIdx.x;i<n;i+=blockDim.x){ fx v=ld_fx(A+((long)i*n+k)*NL); if (!have || fx_abs_gt(v,bv)){ bv=v; bi=i; have=true; } }
    best[threadIdx.x]= have? bi : -1; __syncthreads();
    if (threadIdx.x==0){ int bb=-1; fx bvv=fx_zero(); for(int t=0;t<blockDim.x;t++){ int c=best[t]; if(c<0) continue; fx v=ld_fx(A+((long)c*n+k)*NL); if (bb<0 || fx_abs_gt(v,bvv)){ bb=c; bvv=v; } } prow=bb; }
    __syncthreads(); int p=prow;
    if (p!=k){ for(int j=threadIdx.x;j<n;j+=blockDim.x){ fx t=ld_fx(A+((long)k*n+j)*NL), s=ld_fx(A+((long)p*n+j)*NL); st_fx(A+((long)k*n+j)*NL, s); st_fx(A+((long)p*n+j)*NL, t); }
      if (threadIdx.x==0){ fx t=ld_fx(B+k*NL), s=ld_fx(B+p*NL); st_fx(B+k*NL, s); st_fx(B+p*NL, t); } }
    __syncthreads();
    fx akk=ld_fx(A+((long)k*n+k)*NL);
    for (int i=k+1+threadIdx.x;i<n;i+=blockDim.x){ fx l=fx_div(ld_fx(A+((long)i*n+k)*NL), akk); st_fx(A+((long)i*n+k)*NL, l); }
    __syncthreads();
    for (long idx=threadIdx.x; idx<(long)(n-k-1)*(n-k-1); idx+=blockDim.x){ int i=k+1+idx/(n-k-1), j=k+1+idx%(n-k-1);
      fx l=ld_fx(A+((long)i*n+k)*NL), u=ld_fx(A+((long)k*n+j)*NL), v=ld_fx(A+((long)i*n+j)*NL); st_fx(A+((long)i*n+j)*NL, fx_sub(v, fx_mul(l,u))); }
    __syncthreads();
    if (threadIdx.x==0){ fx bk=ld_fx(B+k*NL); for(int i=k+1;i<n;i++){ fx l=ld_fx(A+((long)i*n+k)*NL); st_fx(B+i*NL, fx_sub(ld_fx(B+i*NL), fx_mul(l,bk))); } }
    __syncthreads(); }
  if (threadIdx.x==0){
    for (int i=n-1;i>=0;i--){ fx s=ld_fx(B+i*NL); for(int j=i+1;j<n;j++) s=fx_sub(s, fx_mul(ld_fx(A+((long)i*n+j)*NL), ld_fx(B+j*NL))); st_fx(B+i*NL, fx_div(s, ld_fx(A+((long)i*n+i)*NL))); } } }
"""

_mod = None


def module():
    global _mod
    if _mod is None:
        _mod = cp.RawModule(
            code=C30.FX_SRC + FXF_SRC, options=("--std=c++17", "-arch=sm_120", "--fmad=false")
        )
    return _mod


class FXMat:
    def __init__(self, F, Jdev, n):
        self.F = F
        self.J = Jdev
        self.n = n

    def nrows(self):
        return self.n

    def solve_limbs(self, rhs_limbs):
        A = self.J.copy()
        B = cp.asarray(rhs_limbs)
        self.F.k_lu((1,), (256,), (A, B, np.int32(self.n)))
        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(B)

    def solve(self, Bm):
        n = self.n
        rhs = np.stack([to_limbs(Bm[i, 0]) for i in range(n)])
        x = self.solve_limbs(rhs)

        class _S:
            def __init__(s, a):
                s.a = a

            def __getitem__(s, ij):
                return s.a[ij[0]]

        return _S([from_limbs(x[i]) for i in range(n)])


class FXF:
    WB = 19.5

    def __init__(self, Hf=0.05, pad=9):
        self.Hf = Hf
        self.pad = pad
        self.mod = module()
        old = ctx.prec
        ctx.prec = 360
        H = arb(Hf)
        self.Harb = H
        self.L2P = (2 * arb.pi()).log() / 2
        ln2 = arb(2).log()
        tab = np.zeros((256, NL), dtype=np.uint32)
        for t in range(256):
            tab[t] = to_limbs((1 + arb(2 * t + 1) / 512).log())
        self.d_tab = cp.asarray(tab)
        self.d_ln2 = cp.asarray(to_limbs(ln2))
        self.d_L2P = cp.asarray(to_limbs(self.L2P))
        self.d_H = cp.asarray(to_limbs(H))
        ctx.prec = old
        g = self.mod.get_function
        self.k_rows = g("k_fx_rows_exp")
        self.k_Q = g("k_fxQ")
        self.k_sums = g("k_fxsums")
        self.k_qinv = g("k_fx_qinv")
        self.k_RQ = g("k_fx_RQ")
        self.k_C = g("k_fx_C")
        self.k_gemm = g("k_fx_gemm")
        self.k_asm = g("k_fx_assemble")
        self.k_lu = g("k_fx_lu_solve")

    def run(self, A, u, w, we, wc, y0, N, want_J=True, timing=None):
        sync = cp.cuda.Stream.null.synchronize
        T = {}
        t_all = time.perf_counter()
        m = len(u)
        odd = wc is not None
        Hf = self.Hf
        y0f = float(y0.mid())
        t0 = time.perf_counter()
        xs = list(u) + [-t for t in u] + [A, -A, arb(0)]
        ws_ = list(w) + list(w) + [we, we, (wc if odd else arb(0))]
        nrows = len(xs)
        i0 = np.zeros(nrows, dtype=np.int32)
        i1 = np.zeros(nrows, dtype=np.int32)
        XR = np.zeros((nrows, NL), dtype=np.uint32)
        WT = np.zeros((nrows, NL), dtype=np.uint32)
        for r, x in enumerate(xs):
            xf = float(x.mid())
            i0[r] = max(0, int((xf - self.WB - y0f) / Hf))
            i1[r] = min(N, int((xf + self.WB - y0f) / Hf) + 2)
            XR[r] = to_limbs(x)
            WT[r] = to_limbs(ws_[r])
        Y0 = to_limbs(y0)
        T["host_prep"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        d_i0, d_i1, d_XR, d_WT, d_Y0 = (
            cp.asarray(i0),
            cp.asarray(i1),
            cp.asarray(XR),
            cp.asarray(WT),
            cp.asarray(Y0),
        )
        P = cp.zeros((nrows, N, NL), dtype=cp.uint32)
        D = cp.zeros((nrows, N, NL), dtype=cp.uint32)
        Q = cp.zeros((N, NL), dtype=cp.uint32)
        LQ = cp.zeros((N, NL), dtype=cp.uint32)
        sync()
        T["h2d"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.k_rows(
            ((nrows * N + 127) // 128,),
            (128,),
            (d_XR, d_i0, d_i1, d_Y0, self.d_H, self.d_L2P, self.d_ln2, np.int32(nrows), np.int32(N), P, D),
        )
        sync()
        T["rows (fx_exp)"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.k_Q(
            ((N + 63) // 64,), (64,), (P, d_WT, np.int32(nrows), np.int32(N), self.d_tab, self.d_ln2, Q, LQ)
        )
        sync()
        T["Q, log Q"] = time.perf_counter() - t0
        evalrows = np.array(list(range(m)) + [2 * m, 2 * m + 2], dtype=np.int32)
        d_ev = cp.asarray(evalrows)
        S = cp.zeros((len(evalrows), 3, NL), dtype=cp.uint32)
        t0 = time.perf_counter()
        self.k_sums(
            (len(evalrows),),
            (128,),
            (P, LQ, d_ev, d_i0, d_i1, d_XR, d_Y0, self.d_H, self.d_L2P, np.int32(N), S),
        )
        sync()
        T["sums"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        Sh = cp.asnumpy(S)
        Hfa = arb(Hf)
        val = lambda a, q: from_limbs(Sh[a, q]) * Hfa
        out = dict(
            D=[val(j, 0) for j in range(m)],
            Dp=[val(j, 1) for j in range(m)],
            DA=val(m, 0),
            centre=(val(m + 1, 0), val(m + 1, 1), val(m + 1, 2)),
        )
        T["host_convert"] = time.perf_counter() - t0
        Jdev = None
        if want_J:
            ncol = 2 * m + 1 + (1 if odd else 0)
            rowA, rowmA, rowc = 2 * m, 2 * m + 1, 2 * m + 2
            evalmap = np.array(list(range(m)) + [rowA] + ([rowc] if odd else []), dtype=np.int32)
            nrq = len(evalmap)
            d_map = cp.asarray(evalmap)
            RINV = cp.zeros((N, NL), dtype=cp.uint32)
            KQ = cp.zeros(N, dtype=cp.int32)
            RQ = cp.zeros((nrq, N, NL), dtype=cp.uint32)
            RS = cp.zeros((m, N, NL), dtype=cp.uint32)
            C = cp.zeros((N, ncol, NL), dtype=cp.uint32)
            t0 = time.perf_counter()
            self.k_qinv(((N + 63) // 64,), (64,), (Q, np.int32(N), RINV, KQ))
            self.k_RQ(
                ((nrq * N + 127) // 128,),
                (128,),
                (P, D, d_map, np.int32(nrq), np.int32(m), np.int32(N), RINV, KQ, RQ, RS),
            )
            self.k_C(
                ((N * ncol + 127) // 128,),
                (128,),
                (
                    P,
                    D,
                    d_WT,
                    np.int32(m),
                    np.int32(ncol),
                    np.int32(N),
                    np.int32(rowA),
                    np.int32(rowmA),
                    np.int32(rowc),
                    C,
                ),
            )
            sync()
            T["RQ, RSQ, C"] = time.perf_counter() - t0
            G1 = cp.zeros((nrq, ncol, NL), dtype=cp.uint32)
            G2 = cp.zeros((m, ncol, NL), dtype=cp.uint32)
            t0 = time.perf_counter()
            self.k_gemm(
                ((ncol + 7) // 8, (nrq + 7) // 8),
                (8, 8),
                (RQ, C, np.int32(nrq), np.int32(ncol), np.int32(N), G1),
            )
            self.k_gemm(
                ((ncol + 7) // 8, (m + 7) // 8), (8, 8), (RS, C, np.int32(m), np.int32(ncol), np.int32(N), G2)
            )
            sync()
            T["fx GEMM G1,G2"] = time.perf_counter() - t0
            Jdev = cp.zeros((ncol, ncol, NL), dtype=cp.uint32)
            t0 = time.perf_counter()
            self.k_asm(
                ((ncol * ncol + 127) // 128,),
                (128,),
                (G1, G2, S, self.d_H, np.int32(m), np.int32(ncol), np.int32(1 if odd else 0), Jdev),
            )
            sync()
            T["assemble"] = time.perf_counter() - t0
        T["total"] = time.perf_counter() - t_all
        if timing is not None:
            timing.update(T)
        return out, Jdev

    def residual_and_jacobian(self, A, u, w, we, wc, want_J=True, y=None, timing=None):
        if y is None:
            raise ValueError("grid required")
        N = len(y)
        odd = wc is not None
        m = len(u)
        s, Jdev = self.run(A, u, w, we, wc, y[0], N, want_J=want_J, timing=timing)
        res = (
            [s["D"][j] - s["DA"] for j in range(m)]
            + s["Dp"]
            + [2 * (sum(w, arb(0)) + we) + (wc if odd else 0) - 1]
        )
        if odd:
            res.append(s["centre"][0] - s["DA"])
        self.last_sums = s
        return res, (FXMat(self, Jdev, 2 * m + 1 + (1 if odd else 0)) if want_J else None)


if __name__ == "__main__":
    from q1414_fast_gauss_band import FastGB
    import statelib

    base = os.path.join(HERE, "..", "small_gap_q206", "fw_prod", "results_q1411_gauss_K101_A61.606866.json")
    ctx.prec = 192
    F = FastGB(192, H=0.05)
    X = FXF(Hf=0.05)
    for K in [60, 101, 140, 200]:
        A, u, w, we, wc, d = statelib.synth_state(base, K) if K != 101 else statelib.load_state(base)
        A = arb(A.mid())
        y = F.grid(A)
        N = len(y)
        t0 = time.perf_counter()
        r_ref, J_ref = F.residual_and_jacobian(A, u, w, we, wc, y=y)
        t_arb = time.perf_counter() - t0
        for rep in range(2):
            T = {}
            r_fx, Jm = X.residual_and_jacobian(A, u, w, we, wc, y=y, timing=T)
        n = Jm.n
        Jh = cp.asnumpy(Jm.J)
        worst = 0.0
        for a in range(n):
            rowmax = max(abs(float(J_ref[a, c].mid())) for c in range(n))
            for c in range(n):
                worst = max(worst, abs(float((from_limbs(Jh[a, c]) - J_ref[a, c]).mid())) / rowmax)
        dres = max(abs(float((a - b).mid())) for a, b in zip(r_fx, r_ref))
        rhs = arb_mat([[-t.mid()] for t in r_ref])
        t0 = time.perf_counter()
        x = Jm.solve(rhs)
        t_solve = time.perf_counter() - t0
        xa = J_ref.solve(rhs)
        xmax = max(abs(float(xa[i, 0].mid())) for i in range(n))
        sd = max(abs(float((x[i, 0] - xa[i, 0]).mid())) for i in range(n)) / xmax
        print(
            f"K={K} (n={n}, N={N}): arb J {t_arb:.2f} s | fx pipeline total {T['total']*1e3:.1f} ms (rows {T['rows (fx_exp)']*1e3:.1f}, GEMM {T['fx GEMM G1,G2']*1e3:.1f}, RQ/C {T['RQ, RSQ, C']*1e3:.1f}); residual differs from arb by {dres:.1e}; J differs from arb by {worst:.1e} of the row max; fx LU solve {t_solve*1e3:.1f} ms, relative difference to the arb solve {sd:.1e} (|r| max {max(abs(float(t.mid())) for t in r_ref):.1e})",
            flush=True,
        )
