"""c30: multi-limb FIXED-POINT residual of the capacity system (q1414 FastGB.residual_and_jacobian, want_J=False) on the GPU.
Format: 10 limbs of 32 bits, two's complement, value = integer / 2^288 (32 integer bits, 288 fractional bits, 320 bits in all). Integer-only CUDA code (IMAD), no fp64.
Kernel rows: the Gaussian kernel row P(y_i - x) is produced by a ratio recurrence (P(d+H) = P(d) exp(-H d - H^2/2), the ratio itself multiplied by exp(-H^2) per step), seeded
with three arb exponentials per row (computed on the CPU at 320 bits), starting at the grid point nearest to the atom and going outward both ways.
Then: Q_i = sum_r w_r P_r(i); log Q_i by exponent split + table (256 entries) + atanh series (16 terms, reciprocal by Newton); sums D, D', D'' per evaluation row.
Absolute accuracy is limited by the 2^-288 unit (1e-87), the seeds are 320-bit.  Use: from c30_fx_residual import FXRes ; see __main__ for the control against FastGB (arb, 192 bits).
"""

import sys, os, time, json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import numpy as np
import cupy as cp
from flint import arb, fmpz, ctx

NLIMB = 10
FRAC = 288

FX_SRC = r"""
typedef unsigned int u32; typedef unsigned long long u64;
#define NL 10
struct fx { u32 v[NL]; };
__device__ __forceinline__ fx fx_zero(){ fx r;
  #pragma unroll
  for(int i=0;i<NL;i++) r.v[i]=0; return r; }
__device__ __forceinline__ bool fx_negp(const fx& a){ return (a.v[NL-1] >> 31) != 0; }
__device__ __forceinline__ fx fx_add(const fx& a, const fx& b){ fx r; u64 c=0;
  #pragma unroll
  for(int i=0;i<NL;i++){ u64 t=(u64)a.v[i]+b.v[i]+c; r.v[i]=(u32)t; c=t>>32; } return r; }
__device__ __forceinline__ fx fx_neg(const fx& a){ fx r; u64 c=1;
  #pragma unroll
  for(int i=0;i<NL;i++){ u64 t=(u64)(~a.v[i])+c; r.v[i]=(u32)t; c=t>>32; } return r; }
__device__ __forceinline__ fx fx_sub(const fx& a, const fx& b){ return fx_add(a, fx_neg(b)); }
__device__ __forceinline__ fx fx_abs(const fx& a){ return fx_negp(a) ? fx_neg(a) : a; }
__device__ __forceinline__ fx fx_mulpos(const fx& a, const fx& b){
  u32 r[2*NL];
  #pragma unroll
  for(int i=0;i<2*NL;i++) r[i]=0;
  #pragma unroll
  for(int i=0;i<NL;i++){
    u64 carry=0;
    #pragma unroll
    for(int j=0;j<NL;j++){
      if (i+j >= 6){ u64 t=(u64)a.v[i]*b.v[j] + r[i+j] + carry; r[i+j]=(u32)t; carry=t>>32; }
    }
    r[i+NL]=(u32)carry;
  }
  fx o;
  #pragma unroll
  for(int k=0;k<NL;k++) o.v[k]=r[k+NL-1];
  return o; }
__device__ __forceinline__ fx fx_mul(const fx& a, const fx& b){
  bool sa=fx_negp(a), sb=fx_negp(b); fx p=fx_mulpos(sa?fx_neg(a):a, sb?fx_neg(b):b); return (sa!=sb)?fx_neg(p):p; }
__device__ __forceinline__ fx fx_mul_small(const fx& a, u32 k){   // exact (sign-magnitude)
  bool s=fx_negp(a); fx m=s?fx_neg(a):a; fx r; u64 c=0;
  #pragma unroll
  for(int i=0;i<NL;i++){ u64 t=(u64)m.v[i]*k + c; r.v[i]=(u32)t; c=t>>32; }
  return s?fx_neg(r):r; }
__device__ __forceinline__ fx fx_div_small(const fx& a, u32 k){
  bool s=fx_negp(a); fx m=s?fx_neg(a):a; fx r; u64 rem=0;
  #pragma unroll
  for(int i=NL-1;i>=0;i--){ u64 cur=(rem<<32)|m.v[i]; r.v[i]=(u32)(cur/k); rem=cur%k; }
  return s?fx_neg(r):r; }
__device__ __forceinline__ fx fx_shr1(const fx& a){ fx r;   // arithmetic >> 1 of a nonnegative
  #pragma unroll
  for(int i=0;i<NL-1;i++) r.v[i]=(a.v[i]>>1)|(a.v[i+1]<<31);
  r.v[NL-1]=a.v[NL-1]>>1; return r; }
__device__ __forceinline__ fx fx_shl1(const fx& a){ fx r;
  #pragma unroll
  for(int i=NL-1;i>0;i--) r.v[i]=(a.v[i]<<1)|(a.v[i-1]>>31);
  r.v[0]=a.v[0]<<1; return r; }
__device__ fx fx_shift(const fx& a, int n){   // magnitude shift: n>0 left, n<0 right (nonnegative input)
  fx r=fx_zero();
  if (n>=0){ int ls=n>>5, bs=n&31;
    for(int i=NL-1;i>=0;i--){ int s=i-ls; u32 lo=(s>=0)?a.v[s]:0; u32 lo2=(s-1>=0)?a.v[s-1]:0; r.v[i]= bs? ((lo<<bs)|(lo2>>(32-bs))) : lo; } }
  else { int m=-n; int ls=m>>5, bs=m&31;
    for(int i=0;i<NL;i++){ int s=i+ls; u32 hi=(s<NL)?a.v[s]:0; u32 hi2=(s+1<NL)?a.v[s+1]:0; r.v[i]= bs? ((hi>>bs)|(hi2<<(32-bs))) : hi; } }
  return r; }
__device__ double fx_to_double(const fx& a){
  bool s=fx_negp(a); fx m=s?fx_neg(a):a; int top=NL-1; while(top>0 && m.v[top]==0) top--;
  double d=0.0; int low=top-2; if(low<0) low=0;
  for(int i=top;i>=low;i--) d = d*4294967296.0 + (double)m.v[i];
  d = ldexp(d, 32*(low-(NL-1))); return s?-d:d; }
__device__ fx fx_from_double(double x){
  fx r=fx_zero(); bool s=x<0; x=fabs(x); if (x==0.0) return r;
  int ex; double f=frexp(x,&ex); u64 mant=(u64)ldexp(f,53); int p=288+ex-53;
  for(int part=0;part<2;part++){ u32 piece = part? (u32)(mant>>32) : (u32)mant; int pp=p+32*part; if(pp<0) continue;
    int li=pp>>5, bs=pp&31; if(li<NL) r.v[li] |= piece<<bs; if(bs && li+1<NL) r.v[li+1] |= piece>>(32-bs); }
  return s?fx_neg(r):r; }
__device__ __forceinline__ fx ld_fx(const u32* p){ fx r;
  #pragma unroll
  for(int i=0;i<NL;i++) r.v[i]=p[i]; return r; }
__device__ __forceinline__ void st_fx(u32* p, const fx& a){
  #pragma unroll
  for(int i=0;i<NL;i++) p[i]=a.v[i]; }

// log of a positive fixed-point number
__device__ fx fx_log(const fx& Q, const u32* tab, const fx& ln2){
  int top=NL-1; while(top>0 && Q.v[top]==0) top--;
  int b = 32*top + (31 - __clz(Q.v[top])); int k = b - 288;
  fx M = fx_shift(Q, 288-b);
  int t = (M.v[8] >> 24) & 255;
  fx at = fx_zero(); at.v[9]=1; at.v[8]=((u32)(2*t+1))<<23;
  fx num=fx_sub(M,at), den=fx_add(M,at);
  double dd=fx_to_double(den); fx x=fx_from_double(1.0/dd);
  fx two=fx_zero(); two.v[9]=2;
  #pragma unroll
  for(int it=0;it<3;it++){ x = fx_mul(x, fx_sub(two, fx_mul(den,x))); }
  fx z=fx_mul(num,x), z2=fx_mul(z,z), term=z, sum=z;
  for(int j=1;j<=16;j++){ term=fx_mul(term,z2); sum=fx_add(sum, fx_div_small(term,2*j+1)); }
  fx r = fx_add(ld_fx(tab + t*NL), fx_shl1(sum));
  if (k!=0){ fx kl=fx_mul_small(ln2, (u32)(k<0?-k:k)); r = (k<0)? fx_sub(r,kl) : fx_add(r,kl); }
  return r; }

// ---- rows by ratio recurrence. one thread per row. seeds: ic, Pc, rr, rl (all fx), band [i0,i1)
extern "C" __global__ void k_fxrows(const int* ic, const int* i0, const int* i1, const u32* Pc, const u32* RR, const u32* RL, const u32* Cc, int nrows, int N, u32* P){
  int r = blockIdx.x*blockDim.x+threadIdx.x; if (r>=nrows) return;
  fx c = ld_fx(Cc); u32* Pr = P + (long)r*N*NL;
  for (int i=0;i<N;i++){ fx z=fx_zero(); st_fx(Pr + (long)i*NL, z); }
  int i_c=ic[r]; fx p=ld_fx(Pc + r*NL); st_fx(Pr + (long)i_c*NL, p);
  fx rr=ld_fx(RR + r*NL); fx pp=p;
  for (int i=i_c+1;i<i1[r];i++){ pp=fx_mul(pp,rr); st_fx(Pr + (long)i*NL, pp); rr=fx_mul(rr,c); }
  fx rl=ld_fx(RL + r*NL); pp=p;
  for (int i=i_c-1;i>=i0[r];i--){ pp=fx_mul(pp,rl); st_fx(Pr + (long)i*NL, pp); rl=fx_mul(rl,c); }
}
// ---- Q, log Q per column
extern "C" __global__ void k_fxQ(const u32* P, const u32* Wt, int nrows, int N, const u32* tab, const u32* LN2, u32* Q, u32* LQ){
  int i = blockIdx.x*blockDim.x+threadIdx.x; if (i>=N) return;
  fx q=fx_zero();
  for (int r=0;r<nrows;r++){ fx p=ld_fx(P + ((long)r*N+i)*NL); bool nz=false;
    #pragma unroll
    for(int k=0;k<NL;k++) nz = nz || (p.v[k]!=0);
    if (nz) q = fx_add(q, fx_mul(ld_fx(Wt + r*NL), p)); }
  st_fx(Q + (long)i*NL, q);
  fx ln2=ld_fx(LN2); fx l=fx_log(q, tab, ln2); st_fx(LQ + (long)i*NL, l);
}
// ---- sums per evaluation row (one block per row): t0 = sum P (L - lQ), t1 = sum P d (L - lQ + 1), t2 = sum P (d^2 (L - lQ + 2) - (L - lQ + 1))
extern "C" __global__ void k_fxsums(const u32* P, const u32* LQ, const int* rowidx, const int* i0, const int* i1, const u32* XR, const u32* Y0, const u32* HF,
    const u32* L2PP, int N, u32* S){
  __shared__ fx red[3][128];
  int a = blockIdx.x; int r = rowidx[a]; fx x=ld_fx(XR + r*NL), y0=ld_fx(Y0), H=ld_fx(HF), L2P=ld_fx(L2PP);
  fx one=fx_zero(); one.v[9]=1; fx two=fx_zero(); two.v[9]=2;
  fx t0=fx_zero(), t1=fx_zero(), t2=fx_zero();
  for (int i=i0[r]+threadIdx.x; i<i1[r]; i+=blockDim.x){
    fx p=ld_fx(P + ((long)r*N+i)*NL); fx y=fx_add(y0, fx_mul_small(H,(u32)i)); fx d=fx_sub(y,x);
    fx d2=fx_mul(d,d); fx L=fx_neg(fx_add(fx_shr1(d2), L2P)); fx lm=fx_sub(L, ld_fx(LQ + (long)i*NL));
    fx l1=fx_add(lm,one), l2=fx_add(lm,two);
    t0=fx_add(t0, fx_mul(p,lm)); t1=fx_add(t1, fx_mul(fx_mul(p,d), l1));
    t2=fx_add(t2, fx_mul(p, fx_sub(fx_mul(d2,l2), l1)));
  }
  red[0][threadIdx.x]=t0; red[1][threadIdx.x]=t1; red[2][threadIdx.x]=t2; __syncthreads();
  for(int off=blockDim.x/2; off>0; off>>=1){
    if (threadIdx.x<off){ for(int q=0;q<3;q++) red[q][threadIdx.x]=fx_add(red[q][threadIdx.x], red[q][threadIdx.x+off]); }
    __syncthreads(); }
  if (threadIdx.x==0){ for(int q=0;q<3;q++) st_fx(S + ((long)a*3+q)*NL, red[q][0]); }
}
"""

_mod = None


def module():
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=FX_SRC, options=("--std=c++17", "-arch=sm_120", "--fmad=false"))
    return _mod


def to_limbs(x):
    """arb (or fmpz-castable) -> 10 little-endian uint32 limbs of the two's complement fixed-point value (x * 2^288)"""
    m, e = x.mid().man_exp()
    m = int(m)
    e = int(e)
    sh = e + FRAC
    v = (m << sh) if sh >= 0 else (m >> (-sh)) if m >= 0 else -((-m) >> (-sh))
    v %= 1 << (32 * NLIMB)
    return np.frombuffer(v.to_bytes(4 * NLIMB, "little"), dtype=np.uint32)


def from_limbs(limbs):
    v = int.from_bytes(np.asarray(limbs, dtype=np.uint32).tobytes(), "little")
    if v >= 1 << (32 * NLIMB - 1):
        v -= 1 << (32 * NLIMB)
    return arb(fmpz(v)) * arb(2) ** (-FRAC)


class FXRes:
    WB = 19.5

    def __init__(self, Hf=0.05, pad=9):
        self.Hf = Hf
        self.pad = pad
        self.mod = module()
        old = ctx.prec
        ctx.prec = 360
        H = arb(Hf)
        self.Harb = H
        self.c = (-(H * H)).exp()
        self.L2P = (2 * arb.pi()).log() / 2
        ln2 = arb(2).log()
        tab = np.zeros((256, NLIMB), dtype=np.uint32)
        for t in range(256):
            tab[t] = to_limbs((1 + arb(2 * t + 1) / 512).log())
        self.d_tab = cp.asarray(tab)
        self.d_ln2 = cp.asarray(to_limbs(ln2))
        self.d_c = cp.asarray(to_limbs(self.c))
        self.d_L2P = cp.asarray(to_limbs(self.L2P))
        self.d_H = cp.asarray(to_limbs(H))
        ctx.prec = old
        self.k_rows = self.mod.get_function("k_fxrows")
        self.k_Q = self.mod.get_function("k_fxQ")
        self.k_sums = self.mod.get_function("k_fxsums")

    def sums(self, A, u, w, we, wc, y0, N, timing=None, extra_center=True):
        """returns dict(D=[...], Dp=[...], DA, centre=(t0,t1,t2) arb sums times H) as arb numbers; rows: plus 0..m-1, minus m..2m-1, wall 2m, 2m+1, centre 2m+2 (weight wc or 0)"""
        sync = cp.cuda.Stream.null.synchronize
        T = {}
        t_all = time.perf_counter()
        m = len(u)
        odd = wc is not None
        Hf = self.Hf
        old = ctx.prec
        ctx.prec = 360
        t0 = time.perf_counter()
        xs = list(u) + [-t for t in u] + [A, -A, arb(0)]
        ws_ = list(w) + list(w) + [we, we, (wc if odd else arb(0))]
        nrows = len(xs)
        y0f = float(y0.mid())
        ic = np.zeros(nrows, dtype=np.int32)
        i0 = np.zeros(nrows, dtype=np.int32)
        i1 = np.zeros(nrows, dtype=np.int32)
        Pc = np.zeros((nrows, NLIMB), dtype=np.uint32)
        RR = np.zeros_like(Pc)
        RL = np.zeros_like(Pc)
        XR = np.zeros_like(Pc)
        WT = np.zeros_like(Pc)
        H = self.Harb
        L2P = self.L2P
        for r, x in enumerate(xs):
            xf = float(x.mid())
            i0[r] = max(0, int((xf - self.WB - y0f) / Hf))
            i1[r] = min(N, int((xf + self.WB - y0f) / Hf) + 2)
            k = int(round((xf - y0f) / Hf))
            k = min(max(k, i0[r]), i1[r] - 1)
            ic[r] = k
            dc = (y0 + H * k) - x
            Pc[r] = to_limbs((-(dc * dc) / 2 - L2P).exp())
            RR[r] = to_limbs((-(H * dc) - H * H / 2).exp())
            RL[r] = to_limbs(((H * dc) - H * H / 2).exp())
            XR[r] = to_limbs(x)
            WT[r] = to_limbs(ws_[r])
        Y0 = to_limbs(y0)
        ctx.prec = old
        T["host_seeds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        d_ic, d_i0, d_i1 = cp.asarray(ic), cp.asarray(i0), cp.asarray(i1)
        d_Pc, d_RR, d_RL, d_XR, d_WT, d_Y0 = [cp.asarray(a) for a in (Pc, RR, RL, XR, WT, Y0)]
        P = cp.zeros((nrows, N, NLIMB), dtype=cp.uint32)
        Q = cp.zeros((N, NLIMB), dtype=cp.uint32)
        LQ = cp.zeros((N, NLIMB), dtype=cp.uint32)
        sync()
        T["h2d"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.k_rows(
            ((nrows + 63) // 64,),
            (64,),
            (d_ic, d_i0, d_i1, d_Pc, d_RR, d_RL, self.d_c, np.int32(nrows), np.int32(N), P),
        )
        sync()
        T["rows (recurrence)"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.k_Q(
            ((N + 63) // 64,), (64,), (P, d_WT, np.int32(nrows), np.int32(N), self.d_tab, self.d_ln2, Q, LQ)
        )
        sync()
        T["Q, log Q"] = time.perf_counter() - t0
        evalrows = np.array(list(range(m)) + [2 * m, 2 * m + 2], dtype=np.int32)
        d_ev = cp.asarray(evalrows)
        S = cp.zeros((len(evalrows), 3, NLIMB), dtype=cp.uint32)
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
        T["d2h"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        Hfa = arb(Hf)
        val = lambda a, q: from_limbs(Sh[a, q]) * Hfa
        out = dict(
            D=[val(j, 0) for j in range(m)],
            Dp=[val(j, 1) for j in range(m)],
            Dpp=[val(j, 2) for j in range(m)],
            DA=val(m, 0),
            centre=(val(m + 1, 0), val(m + 1, 1), val(m + 1, 2)),
        )
        T["host_convert"] = time.perf_counter() - t0
        T["total"] = time.perf_counter() - t_all
        if timing is not None:
            timing.update(T)
        self.last_P = P
        self.last_LQ = LQ
        self.last_Q = Q
        return out

    def residual(self, A, u, w, we, wc, y0, N, timing=None):
        s = self.sums(A, u, w, we, wc, y0, N, timing=timing)
        m = len(u)
        odd = wc is not None
        res = (
            [s["D"][j] - s["DA"] for j in range(m)]
            + s["Dp"]
            + [2 * (sum(w, arb(0)) + we) + (wc if odd else 0) - 1]
        )
        if odd:
            res.append(s["centre"][0] - s["DA"])
        return res, s


if __name__ == "__main__":
    from q1414_fast_gauss_band import FastGB

    fn = sys.argv[1]
    Hs = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    ctx.prec = 192
    F = FastGB(192, H=Hs)
    R = FXRes(Hf=Hs)
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
    r_ref, _ = F.residual_and_jacobian(A, u, w, we, wc, want_J=False, y=y)
    t_arb = time.perf_counter() - t0
    for rep in range(3):
        T = {}
        r_fx, s = R.residual(A, u, w, we, wc, y[0], N, timing=T)
    dif = max(abs(float((a - b).mid())) for a, b in zip(r_fx, r_ref))
    big = max(abs(float(a.mid())) for a in r_ref)
    # mixed-size check for the D, D' values themselves: compare with a direct arb band evaluation is the residual itself; also print |r|
    print(
        f"K={d['K']} m={len(u)} N={N}: arb residual {t_arb*1e3:.1f} ms;  GPU fixed-point residual total {T['total']*1e3:.1f} ms"
    )
    for k, v in T.items():
        print(f"   {k:20s} {v*1e3:9.2f} ms")
    print(f"max |r_fx - r_arb| = {dif:.2e}  (max |r| = {big:.2e})")
    # same with a perturbed state (residual of O(1e-3)) to test the evaluation itself
    u2 = [t * (1 + arb("1e-3")) for t in u]
    r_ref2, _ = F.residual_and_jacobian(A, u2, w, we, wc, want_J=False, y=y)
    r_fx2, _ = R.residual(A, u2, w, we, wc, y[0], N)
    dif2 = max(abs(float((a - b).mid())) for a, b in zip(r_fx2, r_ref2))
    big2 = max(abs(float(a.mid())) for a in r_ref2)
    print(f"perturbed state: max |r_fx - r_arb| = {dif2:.2e}  (max |r| = {big2:.2e})")
