"""c10: double-double (dd, about 104 bits) GPU implementation of the capacity Jacobian build of q1414 FastGB (band version, trapezoid grid) and of a dd LU solve.

Everything the arb code does for J is done on the GPU in dd arithmetic on fp64 CUDA cores (cupy RawModule, NVRTC, sm_120):
  rows: P = exp(-(y - x)^2/2 - log sqrt(2 pi)) on the band of every atom row, Q = sum w P, 1/Q, log Q;
  residual sums D, D', D'' ; matrices RQ, RSQ, C ; products G1 = RQ C, G2 = RSQ C (dd GEMM) ; assembly of J ; LU solve.
Inputs come from an arb state (positions etc. converted to dd). Output: residual (dd) and J (dd) as numpy hi/lo arrays.
Usage as module: from c10_dd_jacobian import DDJac. Self-test: py c10_dd_jacobian.py STATEFILE [H]
"""

import sys, os, time, json
import numpy as np
import cupy as cp
import mpmath as mp

mp.mp.prec = 300


def dd_const(x):
    hi = float(x)
    lo = float(x - mp.mpf(hi))
    return hi, lo


LN2 = dd_const(mp.log(2))
L2P = dd_const(mp.log(2 * mp.pi) / 2)

CUDA_SRC = r"""
struct dd { double hi, lo; };
__device__ __forceinline__ dd dd_make(double h, double l){ dd r; r.hi=h; r.lo=l; return r; }
__device__ __forceinline__ dd dd_neg(dd a){ return dd_make(-a.hi, -a.lo); }
__device__ __forceinline__ dd two_sum(double a, double b){ double s=a+b; double bb=s-a; double e=(a-(s-bb))+(b-bb); return dd_make(s,e); }
__device__ __forceinline__ dd quick_two_sum(double a, double b){ double s=a+b; double e=b-(s-a); return dd_make(s,e); }
__device__ __forceinline__ dd two_prod(double a, double b){ double p=a*b; double e=fma(a,b,-p); return dd_make(p,e); }
__device__ __forceinline__ dd dd_add(dd a, dd b){
  dd s=two_sum(a.hi,b.hi); dd t=two_sum(a.lo,b.lo); s.lo+=t.hi; s=quick_two_sum(s.hi,s.lo); s.lo+=t.lo; s=quick_two_sum(s.hi,s.lo); return s; }
__device__ __forceinline__ dd dd_sub(dd a, dd b){ return dd_add(a, dd_make(-b.hi,-b.lo)); }
__device__ __forceinline__ dd dd_add_d(dd a, double b){ dd s=two_sum(a.hi,b); s.lo+=a.lo; return quick_two_sum(s.hi,s.lo); }
__device__ __forceinline__ dd dd_mul(dd a, dd b){ dd p=two_prod(a.hi,b.hi); p.lo += (a.hi*b.lo + a.lo*b.hi); return quick_two_sum(p.hi,p.lo); }
__device__ __forceinline__ dd dd_mul_d(dd a, double b){ dd p=two_prod(a.hi,b); p.lo += a.lo*b; return quick_two_sum(p.hi,p.lo); }
__device__ __forceinline__ dd dd_mul_pwr2(dd a, double b){ return dd_make(a.hi*b, a.lo*b); }
__device__ __forceinline__ dd dd_fma(dd acc, dd a, dd b){ return dd_add(acc, dd_mul(a,b)); }
__device__ __forceinline__ dd dd_div(dd a, dd b){
  double q1=a.hi/b.hi; dd r=dd_sub(a, dd_mul_d(b,q1)); double q2=r.hi/b.hi; r=dd_sub(r, dd_mul_d(b,q2)); double q3=r.hi/b.hi;
  dd q=quick_two_sum(q1,q2); return dd_add_d(q,q3); }
__device__ __forceinline__ dd dd_inv(dd b){ return dd_div(dd_make(1.0,0.0), b); }
__device__ __forceinline__ dd dd_ldexp(dd a, int e){ return dd_make(ldexp(a.hi,e), ldexp(a.lo,e)); }

#define LN2_HI @LN2H@
#define LN2_LO @LN2L@
#define L2P_HI @L2PH@
#define L2P_LO @L2PL@
// inverse factorials 1/3! ... 1/17! as dd (computed on host)
__constant__ double INVF[2*15] = { @INVF@ };

__device__ dd dd_exp(dd a){
  // QD-style: a = k ln2 + r, r/512, Taylor for exp(r)-1, square 9 times
  if (a.hi < -700.0) return dd_make(0.0,0.0);
  double k = nearbyint(a.hi / LN2_HI);
  dd r = dd_sub(a, dd_mul_d(dd_make(LN2_HI,LN2_LO), k));
  r = dd_mul_pwr2(r, 1.0/512.0);
  dd p = dd_mul(r,r);
  dd s = dd_add(r, dd_mul_pwr2(p, 0.5));
  p = dd_mul(p,r);
  dd t = dd_mul(p, dd_make(INVF[0],INVF[1]));
  int i=0;
  do { s = dd_add(s,t); p = dd_mul(p,r); ++i; t = dd_mul(p, dd_make(INVF[2*i],INVF[2*i+1])); } while (fabs(t.hi) > 1e-36*1.0/512.0*0.0 + 1e-37 && i < 14);
  s = dd_add(s,t);
  #pragma unroll
  for (int q=0;q<9;q++) s = dd_add(dd_mul_pwr2(s,2.0), dd_mul(s,s));
  s = dd_add_d(s, 1.0);
  return dd_ldexp(s, (int)k);
}
__device__ dd dd_log(dd a){
  double x = log(a.hi);
  dd e = dd_exp(dd_make(-x,0.0));
  dd r = dd_mul(a, e);               // a*exp(-x)
  r = dd_add_d(r, -1.0);
  return dd_add_d(r, x);             // x + a exp(-x) - 1
}
__device__ __forceinline__ dd ld(const double* hi, const double* lo, long i){ return dd_make(hi[i], lo[i]); }
__device__ __forceinline__ void st(double* hi, double* lo, long i, dd v){ hi[i]=v.hi; lo[i]=v.lo; }

// ---- K1: kernel rows. grid: (row r, i). Row descriptors: x (dd), band [i0,i1)
extern "C" __global__ void k_rows(const double* xh, const double* xl, const int* i0, const int* i1, int nrows, int N,
    double y0h, double y0l, double Hf, double* Ph, double* Pl, double* Lh, double* Ll, double* Dh, double* Dl){
  long tid = (long)blockIdx.x*blockDim.x + threadIdx.x; long tot = (long)nrows*N; if (tid>=tot) return;
  int r = tid / N; int i = tid - (long)r*N;
  if (i < i0[r] || i >= i1[r]) { Ph[tid]=0; Pl[tid]=0; Lh[tid]=0; Ll[tid]=0; Dh[tid]=0; Dl[tid]=0; return; }
  dd y = dd_add(dd_make(y0h,y0l), two_prod(Hf,(double)i));
  dd d = dd_sub(y, dd_make(xh[r],xl[r]));
  dd L = dd_sub(dd_mul_pwr2(dd_neg(dd_mul(d,d)), 0.5), dd_make(L2P_HI,L2P_LO));
  dd P = dd_exp(L);
  Ph[tid]=P.hi; Pl[tid]=P.lo; Lh[tid]=L.hi; Ll[tid]=L.lo; Dh[tid]=d.hi; Dl[tid]=d.lo;
}
// ---- K2: Q_i = sum_r wt_r P[r,i]; Qinv; lQ
extern "C" __global__ void k_Q(const double* Ph, const double* Pl, const double* wh, const double* wl, int nrows, int N,
    double* Qh, double* Ql, double* Qih, double* Qil, double* lQh, double* lQl){
  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i>=N) return;
  dd q = dd_make(0,0);
  for (int r=0;r<nrows;r++){ long idx=(long)r*N+i; q = dd_add(q, dd_mul(dd_make(wh[r],wl[r]), dd_make(Ph[idx],Pl[idx]))); }
  dd qi = dd_inv(q); dd lq = dd_log(q);
  Qh[i]=q.hi; Ql[i]=q.lo; Qih[i]=qi.hi; Qil[i]=qi.lo; lQh[i]=lq.hi; lQl[i]=lq.lo;
}
// ---- K3: matrices RQ (nrq x N), RSQ (m x N): rows of P (plus rows idx map) times Qinv (and times d)
extern "C" __global__ void k_RQ(const double* Ph, const double* Pl, const double* Dh, const double* Dl, const int* rowmap, int nrq, int m, int N,
    const double* Qih, const double* Qil, double* RQh, double* RQl, double* RSh, double* RSl){
  long tid = (long)blockIdx.x*blockDim.x + threadIdx.x; long tot=(long)nrq*N; if (tid>=tot) return;
  int a = tid / N; int i = tid - (long)a*N; long src = (long)rowmap[a]*N + i;
  dd P = dd_make(Ph[src],Pl[src]); dd qi = dd_make(Qih[i],Qil[i]); dd pq = dd_mul(P,qi);
  RQh[tid]=pq.hi; RQl[tid]=pq.lo;
  if (a < m){ dd s = dd_mul(pq, dd_make(Dh[src],Dl[src])); RSh[tid]=s.hi; RSl[tid]=s.lo; }
}
// ---- K3b: C matrix (N x ncol). rows index map: plus rows pr[k], minus rows mr[k] for k<m; wall rows wa, wm; centre row rc (or -1)
extern "C" __global__ void k_C(const double* Ph, const double* Pl, const double* Dh, const double* Dl, const double* wh, const double* wl,
    int m, int ncol, int N, int rowA, int rowmA, int rowc, double* Ch, double* Cl){
  long tid=(long)blockIdx.x*blockDim.x + threadIdx.x; long tot=(long)N*ncol; if (tid>=tot) return;
  int i = tid / ncol; int c = tid - (long)i*ncol; dd v = dd_make(0,0);
  if (c < m){            // dQ/du_c : atom c plus row = c, minus row = m + c  (rows are laid out: plus rows 0..m-1, minus rows m..2m-1)
    long a=(long)c*N+i, b=(long)(m+c)*N+i; dd wk=dd_make(wh[c],wl[c]);
    dd t1=dd_mul(dd_mul(wk, dd_make(Ph[a],Pl[a])), dd_make(Dh[a],Dl[a]));
    dd t2=dd_mul(dd_mul(wk, dd_make(Ph[b],Pl[b])), dd_make(Dh[b],Dl[b]));
    v = dd_sub(t1,t2);
  } else if (c < 2*m){   // dQ/dw_k
    int k=c-m; long a=(long)k*N+i, b=(long)(m+k)*N+i; v = dd_add(dd_make(Ph[a],Pl[a]), dd_make(Ph[b],Pl[b]));
  } else if (c == 2*m){  // dQ/dwe
    long a=(long)rowA*N+i, b=(long)rowmA*N+i; v = dd_add(dd_make(Ph[a],Pl[a]), dd_make(Ph[b],Pl[b]));
  } else {               // centre
    long a=(long)rowc*N+i; v = dd_make(Ph[a],Pl[a]);
  }
  Ch[tid]=v.hi; Cl[tid]=v.lo;
}
// ---- K4: dd GEMM  G (M x ncol) = X (M x N) * C (N x ncol), tile 16x16, K tile 16
extern "C" __global__ void k_gemm(const double* Xh, const double* Xl, const double* Ch, const double* Cl, int M, int ncol, int N, double* Gh, double* Gl){
  __shared__ double sXh[16][17], sXl[16][17], sCh[16][17], sCl[16][17];
  int tx=threadIdx.x, ty=threadIdx.y; int row=blockIdx.y*16+ty, col=blockIdx.x*16+tx; dd acc=dd_make(0,0);
  for (int k0=0;k0<N;k0+=16){
    int kx=k0+tx, ky=k0+ty;
    if (row<M && kx<N){ sXh[ty][tx]=Xh[(long)row*N+kx]; sXl[ty][tx]=Xl[(long)row*N+kx]; } else { sXh[ty][tx]=0; sXl[ty][tx]=0; }
    if (col<ncol && ky<N){ sCh[ty][tx]=Ch[(long)ky*ncol+col]; sCl[ty][tx]=Cl[(long)ky*ncol+col]; } else { sCh[ty][tx]=0; sCl[ty][tx]=0; }
    __syncthreads();
    #pragma unroll 4
    for (int k=0;k<16;k++){ dd a=dd_make(sXh[ty][k],sXl[ty][k]); dd b=dd_make(sCh[k][tx],sCl[k][tx]); acc=dd_add(acc, dd_mul(a,b)); }
    __syncthreads();
  }
  if (row<M && col<ncol){ Gh[(long)row*ncol+col]=acc.hi; Gl[(long)row*ncol+col]=acc.lo; }
}
// ---- K5: residual sums per atom row (plus rows 0..m-1) and wall row: D, D', D''. one block per row, reduction in dd
extern "C" __global__ void k_sums(const double* Ph, const double* Pl, const double* Lh, const double* Ll, const double* Dh, const double* Dl,
    const double* lQh, const double* lQl, const int* rowmap, int N, double Hf, double* Sh, double* Sl, int wantpp){
  // output S[row*3 + {0,1,2}] = H * sum P (L-q), H * sum P s (L-q+1), H * sum P (s^2 (L-q+2) - (L-q+1))
  int a = blockIdx.x; long base = (long)rowmap[a]*N; __shared__ dd red[3][256];
  dd s0=dd_make(0,0), s1=dd_make(0,0), s2=dd_make(0,0);
  for (int i=threadIdx.x;i<N;i+=blockDim.x){
    dd P=dd_make(Ph[base+i],Pl[base+i]);
    if (P.hi==0.0) continue;
    dd lq=dd_make(lQh[i],lQl[i]); dd Lm=dd_sub(dd_make(Lh[base+i],Ll[base+i]), lq); dd s=dd_make(Dh[base+i],Dl[base+i]);
    s0=dd_add(s0, dd_mul(P,Lm));
    s1=dd_add(s1, dd_mul(dd_mul(P,s), dd_add_d(Lm,1.0)));
    if (wantpp){ dd s2t = dd_sub(dd_mul(dd_mul(s,s), dd_add_d(Lm,2.0)), dd_add_d(Lm,1.0)); s2=dd_add(s2, dd_mul(P,s2t)); }
  }
  red[0][threadIdx.x]=s0; red[1][threadIdx.x]=s1; red[2][threadIdx.x]=s2; __syncthreads();
  for (int off=blockDim.x/2; off>0; off>>=1){
    if (threadIdx.x<off){ for(int q=0;q<3;q++) red[q][threadIdx.x]=dd_add(red[q][threadIdx.x], red[q][threadIdx.x+off]); }
    __syncthreads();
  }
  if (threadIdx.x==0){ for(int q=0;q<3;q++){ dd v=dd_mul_d(red[q][0], Hf); Sh[a*3+q]=v.hi; Sl[a*3+q]=v.lo; } }
}
// ---- K6: assemble J (n x n) and residual on the device
extern "C" __global__ void k_assemble(const double* G1h, const double* G1l, const double* G2h, const double* G2l, const double* Sh, const double* Sl,
    int m, int ncol, int odd, double Hf, double nrh, double nrl, double* Jh, double* Jl, double* rh, double* rl){
  int n = ncol; long tid=(long)blockIdx.x*blockDim.x + threadIdx.x; if (tid >= (long)n*n + n) return;
  if (tid >= (long)n*n){   // residual entries
    int a = (int)(tid - (long)n*n); dd v = dd_make(0,0);
    dd DA = dd_make(Sh[m*3], Sl[m*3]);
    if (a < m) v = dd_sub(dd_make(Sh[a*3],Sl[a*3]), DA);
    else if (a < 2*m) v = dd_make(Sh[(a-m)*3+1], Sl[(a-m)*3+1]);
    else if (a == 2*m) v = dd_make(nrh, nrl);
    else v = dd_sub(dd_make(Sh[(m+1)*3],Sl[(m+1)*3]), DA);
    rh[a]=v.hi; rl[a]=v.lo; return;
  }
  int a = tid / n, b = tid - (long)a*n; dd v = dd_make(0,0);
  if (a < m){
    dd g = dd_sub(dd_make(G1h[(long)m*ncol+b],G1l[(long)m*ncol+b]), dd_make(G1h[(long)a*ncol+b],G1l[(long)a*ncol+b])); v = dd_mul_d(g, Hf);
    if (b == a) v = dd_add(v, dd_make(Sh[a*3+1],Sl[a*3+1]));
  } else if (a < 2*m){
    int j = a - m; dd g = dd_make(G2h[(long)j*ncol+b],G2l[(long)j*ncol+b]); v = dd_neg(dd_mul_d(g, Hf));
    if (b == j) v = dd_add(v, dd_make(Sh[j*3+2],Sl[j*3+2]));
  } else if (a == 2*m){
    if (b >= m && b < 2*m) v = dd_make(2.0,0.0); else if (b == 2*m) v = dd_make(2.0,0.0); else if (odd && b == 2*m+1) v = dd_make(1.0,0.0);
  } else {
    dd g = dd_sub(dd_make(G1h[(long)m*ncol+b],G1l[(long)m*ncol+b]), dd_make(G1h[(long)(m+1)*ncol+b],G1l[(long)(m+1)*ncol+b])); v = dd_mul_d(g, Hf);
  }
  Jh[tid]=v.hi; Jl[tid]=v.lo;
}
// ---- K7: LU with partial pivoting + solve, single block. A (n x n, row major) overwritten; b overwritten by x.
extern "C" __global__ void k_lu_solve(double* Ah, double* Al, double* bh, double* bl, int n, int* piv){
  __shared__ double mx[256]; __shared__ int mi[256]; __shared__ int prow;
  for (int k=0;k<n;k++){
    double best=-1.0; int bi=k;
    for (int i=k+threadIdx.x;i<n;i+=blockDim.x){ double v=fabs(Ah[(long)i*n+k]); if (v>best){best=v;bi=i;} }
    mx[threadIdx.x]=best; mi[threadIdx.x]=bi; __syncthreads();
    for (int off=blockDim.x/2;off>0;off>>=1){ if (threadIdx.x<off && mx[threadIdx.x+off]>mx[threadIdx.x]){ mx[threadIdx.x]=mx[threadIdx.x+off]; mi[threadIdx.x]=mi[threadIdx.x+off]; } __syncthreads(); }
    if (threadIdx.x==0){ prow=mi[0]; piv[k]=prow; }
    __syncthreads();
    int p=prow;
    if (p!=k){ for (int j=threadIdx.x;j<n;j+=blockDim.x){ double th=Ah[(long)k*n+j], tl=Al[(long)k*n+j]; Ah[(long)k*n+j]=Ah[(long)p*n+j]; Al[(long)k*n+j]=Al[(long)p*n+j]; Ah[(long)p*n+j]=th; Al[(long)p*n+j]=tl; }
               if (threadIdx.x==0){ double th=bh[k], tl=bl[k]; bh[k]=bh[p]; bl[k]=bl[p]; bh[p]=th; bl[p]=tl; } }
    __syncthreads();
    dd akk=dd_make(Ah[(long)k*n+k],Al[(long)k*n+k]); dd rk=dd_inv(akk);
    for (int i=k+1+threadIdx.x;i<n;i+=blockDim.x){
      dd l=dd_mul(dd_make(Ah[(long)i*n+k],Al[(long)i*n+k]), rk); Ah[(long)i*n+k]=l.hi; Al[(long)i*n+k]=l.lo;
    }
    __syncthreads();
    for (long idx=threadIdx.x; idx<(long)(n-k-1)*(n-k-1); idx+=blockDim.x){
      int i=k+1+idx/(n-k-1), j=k+1+idx%(n-k-1);
      dd l=dd_make(Ah[(long)i*n+k],Al[(long)i*n+k]); dd u=dd_make(Ah[(long)k*n+j],Al[(long)k*n+j]); dd v=dd_make(Ah[(long)i*n+j],Al[(long)i*n+j]);
      v=dd_sub(v, dd_mul(l,u)); Ah[(long)i*n+j]=v.hi; Al[(long)i*n+j]=v.lo;
    }
    // apply to b (forward elimination) by thread 0 after sync
    __syncthreads();
    if (threadIdx.x==0){ dd bk=dd_make(bh[k],bl[k]); for (int i=k+1;i<n;i++){ dd l=dd_make(Ah[(long)i*n+k],Al[(long)i*n+k]); dd bi_=dd_sub(dd_make(bh[i],bl[i]), dd_mul(l,bk)); bh[i]=bi_.hi; bl[i]=bi_.lo; } }
    __syncthreads();
  }
  if (threadIdx.x==0){
    for (int i=n-1;i>=0;i--){ dd s=dd_make(bh[i],bl[i]); for (int j=i+1;j<n;j++) s=dd_sub(s, dd_mul(dd_make(Ah[(long)i*n+j],Al[(long)i*n+j]), dd_make(bh[j],bl[j]))); s=dd_div(s, dd_make(Ah[(long)i*n+i],Al[(long)i*n+i])); bh[i]=s.hi; bl[i]=s.lo; }
  }
}
"""


def _invf_table():
    out = []
    fct = mp.mpf(1)
    for k in range(1, 18):
        fct *= k
        if k >= 3:
            h, l = dd_const(1 / fct)
            out += [h, l]
    return ", ".join(f"{v:.17g}" for v in out[:30])


def build_module():
    src = (
        CUDA_SRC.replace("@LN2H@", f"{LN2[0]:.17g}")
        .replace("@LN2L@", f"{LN2[1]:.17g}")
        .replace("@L2PH@", f"{L2P[0]:.17g}")
        .replace("@L2PL@", f"{L2P[1]:.17g}")
        .replace("@INVF@", _invf_table())
    )
    return cp.RawModule(code=src, options=("--std=c++17", "-arch=sm_120", "--fmad=false"))


def arb_to_dd(x):
    from flint import arb

    hi = float(x.mid())
    lo = float((x - arb(hi)).mid())
    return hi, lo


class DDJac:
    WB = 19.5

    def __init__(self, Hf=0.05, pad=9):
        self.mod = build_module()
        self.Hf = Hf
        self.pad = pad
        self.k = {
            n: self.mod.get_function(n)
            for n in ("k_rows", "k_Q", "k_RQ", "k_C", "k_gemm", "k_sums", "k_lu_solve", "k_assemble")
        }

    def build(self, A, u, w, we, wc, y0, N, timing=None):
        """A, u[], w[], we, wc (arb or None), y0 = first grid point (arb), N grid size. Returns (res_dd (n,2), J_dd (n,n,2)) numpy."""
        sync = cp.cuda.Stream.null.synchronize
        t_all = time.perf_counter()
        T = {}
        m = len(u)
        odd = wc is not None
        ncol = 2 * m + 1 + (1 if odd else 0)
        n = ncol
        Hf = self.Hf
        y0f = float(y0.mid())
        y0h, y0l = arb_to_dd(y0)
        # rows: plus rows 0..m-1 (x = +u), minus rows m..2m-1 (x = -u), wall plus 2m, wall minus 2m+1, centre 2m+2 (odd)
        xs = (
            [arb_to_dd(t) for t in u]
            + [arb_to_dd(-t) for t in u]
            + [arb_to_dd(A), arb_to_dd(-A)]
            + ([arb_to_dd(A * 0)] if odd else [])
        )
        wts = (
            [arb_to_dd(t) for t in w] * 2 + [arb_to_dd(we), arb_to_dd(we)] + ([arb_to_dd(wc)] if odd else [])
        )
        nrows = len(xs)
        i0 = np.zeros(nrows, dtype=np.int32)
        i1 = np.zeros(nrows, dtype=np.int32)
        xfl = (
            [float(t.mid()) for t in u]
            + [-float(t.mid()) for t in u]
            + [float(A.mid()), -float(A.mid())]
            + ([0.0] if odd else [])
        )
        for r, xf in enumerate(xfl):
            i0[r] = max(0, int((xf - self.WB - y0f) / Hf))
            i1[r] = min(N, int((xf + self.WB - y0f) / Hf) + 2)
        t0 = time.perf_counter()
        xh = cp.asarray(np.array([a for a, b in xs]))
        xl = cp.asarray(np.array([b for a, b in xs]))
        wh = cp.asarray(np.array([a for a, b in wts]))
        wl = cp.asarray(np.array([b for a, b in wts]))
        di0 = cp.asarray(i0)
        di1 = cp.asarray(i1)
        f64 = lambda *s: cp.zeros(s, dtype=cp.float64)
        Ph, Pl, Lh, Ll, Dh, Dl = [f64(nrows, N) for _ in range(6)]
        sync()
        T["h2d"] = time.perf_counter() - t0

        def launch(name, total, args, block=256):
            self.k[name](((total + block - 1) // block,), (block,), args)

        t0 = time.perf_counter()
        launch(
            "k_rows",
            nrows * N,
            (
                xh,
                xl,
                di0,
                di1,
                np.int32(nrows),
                np.int32(N),
                np.float64(y0h),
                np.float64(y0l),
                np.float64(Hf),
                Ph,
                Pl,
                Lh,
                Ll,
                Dh,
                Dl,
            ),
        )
        sync()
        T["rows(exp)"] = time.perf_counter() - t0
        Qh, Ql, Qih, Qil, lQh, lQl = [f64(N) for _ in range(6)]
        t0 = time.perf_counter()
        launch("k_Q", N, (Ph, Pl, wh, wl, np.int32(nrows), np.int32(N), Qh, Ql, Qih, Qil, lQh, lQl))
        sync()
        T["Q,1/Q,logQ"] = time.perf_counter() - t0
        rowA, rowmA, rowc = 2 * m, 2 * m + 1, (2 * m + 2 if odd else -1)
        evalmap = np.array(list(range(m)) + [rowA] + ([rowc] if odd else []), dtype=np.int32)
        nrq = len(evalmap)
        drm = cp.asarray(evalmap)
        RQh, RQl = f64(nrq, N), f64(nrq, N)
        RSh, RSl = f64(m, N), f64(m, N)
        Ch, Cl = f64(N, ncol), f64(N, ncol)
        t0 = time.perf_counter()
        launch(
            "k_RQ",
            nrq * N,
            (Ph, Pl, Dh, Dl, drm, np.int32(nrq), np.int32(m), np.int32(N), Qih, Qil, RQh, RQl, RSh, RSl),
        )
        launch(
            "k_C",
            N * ncol,
            (
                Ph,
                Pl,
                Dh,
                Dl,
                wh,
                wl,
                np.int32(m),
                np.int32(ncol),
                np.int32(N),
                np.int32(rowA),
                np.int32(rowmA),
                np.int32(max(rowc, 0)),
                Ch,
                Cl,
            ),
        )
        sync()
        T["build RQ,RSQ,C"] = time.perf_counter() - t0
        G1h, G1l = f64(nrq, ncol), f64(nrq, ncol)
        G2h, G2l = f64(m, ncol), f64(m, ncol)
        t0 = time.perf_counter()
        for Xh, Xl, M, Gh, Gl in ((RQh, RQl, nrq, G1h, G1l), (RSh, RSl, m, G2h, G2l)):
            self.k["k_gemm"](
                ((ncol + 15) // 16, (M + 15) // 16),
                (16, 16),
                (Xh, Xl, Ch, Cl, np.int32(M), np.int32(ncol), np.int32(N), Gh, Gl),
            )
        sync()
        T["dd GEMM G1,G2"] = time.perf_counter() - t0
        # sums for plus rows + wall row (+ centre row for the residual only)
        sumrows = np.array(list(range(m)) + [rowA] + ([rowc] if odd else []), dtype=np.int32)
        dsr = cp.asarray(sumrows)
        Sh, Sl = f64(len(sumrows) * 3), f64(len(sumrows) * 3)
        t0 = time.perf_counter()
        self.k["k_sums"](
            (len(sumrows),),
            (256,),
            (Ph, Pl, Lh, Ll, Dh, Dl, lQh, lQl, dsr, np.int32(N), np.float64(Hf), Sh, Sl, np.int32(1)),
        )
        sync()
        T["sums D,D',D''"] = time.perf_counter() - t0
        # normalisation residual 2 (sum w + we) + wc - 1 in arb (exact, host, O(m))
        from flint import arb as _arb

        nr = 2 * (sum(w, _arb(0)) + we) + (wc if odd else 0) - 1
        nrh, nrl = arb_to_dd(nr)
        Jh, Jl = f64(n, n), f64(n, n)
        rh, rl = f64(n), f64(n)
        t0 = time.perf_counter()
        launch(
            "k_assemble",
            n * n + n,
            (
                G1h,
                G1l,
                G2h,
                G2l,
                Sh,
                Sl,
                np.int32(m),
                np.int32(ncol),
                np.int32(1 if odd else 0),
                np.float64(Hf),
                np.float64(nrh),
                np.float64(nrl),
                Jh,
                Jl,
                rh,
                rl,
            ),
        )
        sync()
        T["assemble J (GPU)"] = time.perf_counter() - t0
        self.dev = dict(Jh=Jh, Jl=Jl, rh=rh, rl=rl)
        t0 = time.perf_counter()
        res = np.stack([cp.asnumpy(rh), cp.asnumpy(rl)], axis=1)
        J = np.stack([cp.asnumpy(Jh), cp.asnumpy(Jl)], axis=2)
        T["d2h"] = time.perf_counter() - t0
        T["total"] = time.perf_counter() - t_all
        if timing is not None:
            timing.update(T)
        return res, J

    # --- double-double helpers on numpy (host) for the small final assembly
    @staticmethod
    def _two_sum(a, b):
        s = a + b
        bb = s - a
        e = (a - (s - bb)) + (b - bb)
        return s, e

    def _dd_add(self, a, b):
        s, e = self._two_sum(a[0], b[0])
        t, f = self._two_sum(a[1], b[1])
        e += t
        s2 = s + e
        e2 = e - (s2 - s)
        e2 += f
        s3 = s2 + e2
        e3 = e2 - (s3 - s2)
        return s3, e3

    def _assemble(self, G1, G2, S, m, odd, ncol, Hf, w, we, wc):
        # use exact rational arithmetic via python Fractions is slow; use mpmath at 120 bits on the O(n^2) entries (cheap enough: 10^4 operations)
        mp.mp.prec = 140
        n = ncol
        H = mp.mpf(Hf)
        g1 = lambda a, b: mp.mpf(G1[0][a, b]) + mp.mpf(G1[1][a, b])
        g2 = lambda a, b: mp.mpf(G2[0][a, b]) + mp.mpf(G2[1][a, b])
        Sv = lambda a, q: mp.mpf(S[0][a * 3 + q]) + mp.mpf(S[1][a * 3 + q])
        J = [[mp.mpf(0)] * n for _ in range(n)]
        for j in range(m):
            for k in range(ncol):
                J[j][k] = -H * g1(j, k) + H * g1(m, k) + (Sv(j, 1) if k == j else 0)
                J[m + j][k] = -H * g2(j, k) + (Sv(j, 2) if k == j else 0)
        for k in range(m):
            J[2 * m][m + k] = mp.mpf(2)
        J[2 * m][2 * m] = mp.mpf(2)
        if odd:
            J[2 * m][2 * m + 1] = mp.mpf(1)
            for k in range(ncol):
                J[2 * m + 1][k] = -H * g1(m + 1, k) + H * g1(m, k)
        DA = Sv(m, 0)
        res = [Sv(j, 0) - DA for j in range(m)] + [Sv(j, 1) for j in range(m)]
        wsum = sum((mp.mpf(arb_to_dd(t)[0]) + mp.mpf(arb_to_dd(t)[1]) for t in w))
        wev = mp.mpf(arb_to_dd(we)[0]) + mp.mpf(arb_to_dd(we)[1])
        wcv = (mp.mpf(arb_to_dd(wc)[0]) + mp.mpf(arb_to_dd(wc)[1])) if odd else 0
        res.append(2 * (wsum + wev) + wcv - 1)
        if odd:
            res.append(Sv(m + 1, 0) - DA)
        # note: the J built here is rounded to a 140-bit mp float; convert to dd pairs
        Jd = np.zeros((n, n, 2))
        for a in range(n):
            for b in range(n):
                x = J[a][b]
                hi = float(x)
                Jd[a, b, 0] = hi
                Jd[a, b, 1] = float(x - mp.mpf(hi))
        rd = np.zeros((n, 2))
        for a in range(n):
            x = res[a]
            hi = float(x)
            rd[a, 0] = hi
            rd[a, 1] = float(x - mp.mpf(hi))
        return rd, Jd

    def solve(self, Jd, b):
        """dd LU solve J x = b on the GPU. Jd (n,n,2), b (n,2) numpy; returns x (n,2)."""
        n = Jd.shape[0]
        Ah = cp.asarray(np.ascontiguousarray(Jd[:, :, 0]))
        Al = cp.asarray(np.ascontiguousarray(Jd[:, :, 1]))
        bh = cp.asarray(np.ascontiguousarray(b[:, 0]))
        bl = cp.asarray(np.ascontiguousarray(b[:, 1]))
        piv = cp.zeros(n, dtype=cp.int32)
        self.k["k_lu_solve"]((1,), (256,), (Ah, Al, bh, bl, np.int32(n), piv))
        cp.cuda.Stream.null.synchronize()
        return np.stack([cp.asnumpy(bh), cp.asnumpy(bl)], axis=1)


if __name__ == "__main__":
    from flint import arb, arb_mat, ctx

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    from q1414_fast_gauss_band import FastGB

    fn = sys.argv[1]
    Hs = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    ctx.prec = 192
    F = FastGB(192, H=Hs)
    d = json.load(open(fn))
    A = arb(d["A"])
    u = [arb(x) for x in d["u"]]
    w = [arb(x) for x in d["w"]]
    we = arb(d["we"])
    wc = None if d.get("wc") in (None, "None") else arb(d["wc"])
    A = arb(A.mid())
    y = F.grid(A)
    N = len(y)
    t0 = time.perf_counter()
    r_ref, J_ref = F.residual_and_jacobian(A, u, w, we, wc, y=y)
    t_arb = time.perf_counter() - t0
    G = DDJac(Hf=Hs)
    T = {}
    for rep in range(3):
        T = {}
        res, J = G.build(A, u, w, we, wc, y[0], N, timing=T)
    n = J.shape[0]
    print(f"state K={d['K']} n={n} N={N}; arb J build {t_arb:.3f} s; GPU dd build total {T['total']:.4f} s")
    for k, v in T.items():
        print(f"   {k:20s} {v*1e3:9.2f} ms")
    # compare J
    to_arb = lambda hi, lo: arb(float(hi)) + arb(float(lo))
    worst = 0.0
    wr = 0.0
    for a in range(n):
        rowmax = max(abs(float(J_ref[a, c].mid())) for c in range(n))
        for c in range(n):
            dj = abs(float((to_arb(J[a, c, 0], J[a, c, 1]) - J_ref[a, c]).mid())) / rowmax
            worst = max(worst, dj)
    for a in range(n):
        wr = max(wr, abs(float((to_arb(res[a, 0], res[a, 1]) - r_ref[a]).mid())))
    print(
        f"max |J_dd - J_arb| / row max = {worst:.2e};  max |r_dd - r_arb| = {wr:.2e}  (residual size {float(max(abs(t.mid()) for t in r_ref)):.2e})"
    )
