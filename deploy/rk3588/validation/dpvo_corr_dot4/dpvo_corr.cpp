/* Derived from DPVO MIT forward correlation semantics. See dpvo_corr.h/LICENSE.
 * Explicit FP32, no fast-math/FMA: summation order can differ from CUDA/Torch,
 * so equivalence is tested with fixed rtol=1e-4/atol=1e-5, not byte equality. */
#include "dpvo_corr.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "DPVO correlation forbids fast-math/finite-only compilation"
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__aarch64__) && !defined(DPVO_V27_DISABLE_NEON)
#include <arm_neon.h>
#define DPVO_V27_NEON 1
#else
#define DPVO_V27_NEON 0
#endif

namespace {
using Clock = std::chrono::steady_clock;
constexpr uint64_t kMemoryLimit = 512ULL * 1024 * 1024;
constexpr uint32_t kABI = 1;
struct Context {
    int64_t b,p,v,c,ph,pw,h,w,h4,w4;
    int32_t threads;
    std::vector<float> patches, level1, level4;
    uint64_t bytes() const { return (patches.size()+level1.size()+level4.size())*sizeof(float); }
};
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}
uint64_t product(std::initializer_list<int64_t> dimensions) {
    uint64_t result = 1;
    for (int64_t dimension : dimensions) {
        if (dimension < 0 || dimension > INT32_MAX)
            throw std::invalid_argument("dimension outside [0,INT32_MAX]");
        if (dimension && result > UINT64_MAX / static_cast<uint64_t>(dimension))
            throw std::invalid_argument("dimension product overflow");
        result *= static_cast<uint64_t>(dimension);
    }
    if (result > kMemoryLimit/sizeof(float))
        throw std::invalid_argument("buffer exceeds 512MiB safety budget");
    return result;
}
void message(char *error, size_t capacity, const char *value) {
    if (error && capacity) {
        const size_t size = std::min(capacity-1, std::strlen(value));
        std::memcpy(error,value,size); error[size]='\0';
    }
}
void stats_init(dpvo_corr_v27_stats *stats, int32_t threads=0) {
    if (stats) { std::memset(stats,0,sizeof(*stats)); stats->abi_version=kABI;
                 stats->threads=threads; stats->neon=DPVO_V27_NEON; }
}
void finite_input(const float *value, uint64_t count, const char *name) {
    if (count && !value) throw std::invalid_argument("NULL input pointer");
    for (uint64_t i=0;i<count;++i)
        if (!std::isfinite(value[i])) throw std::invalid_argument(name);
}
void layout(const float *source, float *dest, int64_t frames, int64_t channels,
            int64_t spatial, int32_t threads) {
    const int64_t cells=frames*spatial;
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t cell=0;cell<cells;++cell) {
        const int64_t frame=cell/spatial, pixel=cell%spatial;
        for (int64_t channel=0;channel<channels;++channel)
            dest[cell*channels+channel]=source[(frame*channels+channel)*spatial+pixel];
    }
}
float dot(const float *first, const float *second, int64_t channels) {
    #if DPVO_V27_NEON
    float32x4_t sum=vdupq_n_f32(0.0f);
    int64_t channel=0;
    for (;channel+4<=channels;channel+=4)
        sum=vaddq_f32(sum,vmulq_f32(vld1q_f32(first+channel),vld1q_f32(second+channel)));
    float parts[4]; vst1q_f32(parts,sum);
    float result=((parts[0]+parts[1])+parts[2])+parts[3];
    for (;channel<channels;++channel) result += first[channel]*second[channel];
    return result;
    #else
    float result=0.0f;
    for (int64_t channel=0;channel<channels;++channel) result += first[channel]*second[channel];
    return result;
    #endif
}
bool overlaps(const void *a, uint64_t a_bytes, const void *b, uint64_t b_bytes) {
    if (!a_bytes || !b_bytes) return false;
    const uintptr_t aa=reinterpret_cast<uintptr_t>(a), bb=reinterpret_cast<uintptr_t>(b);
    if (aa > UINTPTR_MAX-a_bytes || bb > UINTPTR_MAX-b_bytes)
        throw std::invalid_argument("pointer range overflow");
    return aa < bb+b_bytes && bb < aa+a_bytes;
}
}

extern "C" uint32_t dpvo_corr_v27_abi_version() { return kABI; }
extern "C" uint64_t dpvo_corr_v27_stats_size() { return sizeof(dpvo_corr_v27_stats); }
extern "C" const char *dpvo_corr_v27_build_info() {
    #if DPVO_V27_NEON
    return "FP32;AArch64-NEON-mul-add;no-fast-math;requires-ffp-contract-off;OpenMP-explicit-num-threads";
    #else
    return "FP32;generic-scalar;no-fast-math;requires-ffp-contract-off;OpenMP-explicit-num-threads";
    #endif
}

extern "C" int dpvo_corr_v27_prepare(
    const float *gmap, uint64_t ng, const float *fmap, uint64_t nf,
    int64_t b,int64_t p,int64_t v,int64_t c,int64_t ph,int64_t pw,int64_t h,int64_t w,
    int32_t threads,void **handle,dpvo_corr_v27_stats *stats,char *error,size_t capacity) {
    const auto begin=Clock::now(); stats_init(stats,threads); message(error,capacity,"");
    try {
        if (!handle || *handle) throw std::invalid_argument("prepare requires NULL destination context");
        if (std::min({b,p,v,c,ph,pw,h,w})<1 || c%8 || h<4 || w<4)
            throw std::invalid_argument("positive dimensions, C divisible by8, H/W>=4 required");
        if (threads<1 || threads>64) throw std::invalid_argument("threads must be in [1,64]");
        #ifndef _OPENMP
        if (threads != 1) throw std::invalid_argument("library built without OpenMP; threads must be1");
        #endif
        const uint64_t gsize=product({b,p,c,ph,pw}), fsize=product({b,v,c,h,w});
        const uint64_t psize=product({b,v,c,h/4,w/4});
        if (gsize != ng || fsize != nf) throw std::invalid_argument("feature buffer length mismatch");
        if ((gsize+fsize+psize)*sizeof(float)>kMemoryLimit)
            throw std::invalid_argument("context exceeds 512MiB safety budget");
        auto scan=Clock::now(); finite_input(gmap,ng,"gmap contains nonfinite");
        finite_input(fmap,nf,"fmap contains nonfinite");
        if (stats) stats->scan_seconds=elapsed(scan);
        auto ctx=std::make_unique<Context>();
        ctx->b=b;ctx->p=p;ctx->v=v;ctx->c=c;ctx->ph=ph;ctx->pw=pw;
        ctx->h=h;ctx->w=w;ctx->h4=h/4;ctx->w4=w/4;ctx->threads=threads;
        ctx->patches.resize(gsize);ctx->level1.resize(fsize);ctx->level4.resize(psize);
        auto copy=Clock::now();
        layout(gmap,ctx->patches.data(),b*p,c,ph*pw,threads);
        layout(fmap,ctx->level1.data(),b*v,c,h*w,threads);
        if (stats) stats->layout_seconds=elapsed(copy);
        auto pool=Clock::now(); const int64_t pool_cells=b*v*ctx->h4*ctx->w4;
        int invalid=0;
        #pragma omp parallel for schedule(static) num_threads(threads) reduction(|:invalid)
        for (int64_t cell=0;cell<pool_cells;++cell) {
            const int64_t x=cell%ctx->w4,y=(cell/ctx->w4)%ctx->h4,frame=cell/(ctx->w4*ctx->h4);
            for (int64_t channel=0;channel<c;++channel) {
                float sum=0.0f;
                for (int64_t sy=0;sy<4;++sy) for (int64_t sx=0;sx<4;++sx)
                    sum += ctx->level1[((frame*h+y*4+sy)*w+x*4+sx)*c+channel];
                const float result=sum/16.0f;
                ctx->level4[cell*c+channel]=result;
                if (!std::isfinite(result)) invalid=1;
            }
        }
        if (stats) stats->pool_seconds=elapsed(pool);
        if (invalid) throw std::invalid_argument("FP32 pooling overflow/nonfinite result");
        if (stats) stats->context_bytes=ctx->bytes();
        *handle=ctx.release();
        if (stats) stats->total_seconds=elapsed(begin);
        return 0;
    } catch(const std::exception &failure) { message(error,capacity,failure.what()); }
      catch(...) { message(error,capacity,"unknown prepare failure"); }
    if (stats) stats->total_seconds=elapsed(begin);
    return 1;
}

extern "C" int dpvo_corr_v27_correlate(
    const void *handle,const float *coords,uint64_t nc,const int64_t *kk,const int64_t *jj,
    uint64_t ni,int64_t e,int32_t radius,float *output,uint64_t no,
    dpvo_corr_v27_stats *stats,char *error,size_t capacity) {
    const auto begin=Clock::now(); stats_init(stats); message(error,capacity,"");
    try {
        if (!handle) throw std::invalid_argument("closed/NULL correlation context");
        const auto &ctx=*static_cast<const Context*>(handle);
        if (stats) { stats->threads=ctx.threads; stats->context_bytes=ctx.bytes(); }
        if (radius!=3) throw std::invalid_argument("only original radius3 supported");
        const uint64_t expected_nc=product({ctx.b,e,2,ctx.ph,ctx.pw});
        const uint64_t expected_no=product({ctx.b,e,7,7,ctx.ph,ctx.pw,2});
        if (nc!=expected_nc || no!=expected_no || ni!=static_cast<uint64_t>(e))
            throw std::invalid_argument("coordinate/index/output buffer length mismatch");
        if ((e && (!kk || !jj)) || (no && !output)) throw std::invalid_argument("NULL buffer");
        if (overlaps(output,no*4,coords,nc*4) || overlaps(output,no*4,kk,ni*8) || overlaps(output,no*4,jj,ni*8))
            throw std::invalid_argument("output may not overlap request buffers");
        const auto scan=Clock::now();
        finite_input(coords,nc,"coords contains nonfinite");
        for(uint64_t i=0;i<nc;++i) if(std::fabs(coords[i])>=1073741824.0f)
            throw std::invalid_argument("coords exceed safe int32 sampling range");
        for(int64_t i=0;i<e;++i) if(kk[i]<0 || kk[i]>=ctx.p || jj[i]<0 || jj[i]>=ctx.v)
            throw std::invalid_argument("patch/view index out of range");
        if(stats) { stats->scan_seconds=elapsed(scan); stats->output_elements=no; }
        const auto kernel=Clock::now();
        const int64_t pp=ctx.ph*ctx.pw, tasks=ctx.b*e*pp;
        int invalid=0;
        #pragma omp parallel for schedule(static) num_threads(ctx.threads) reduction(|:invalid)
        for(int64_t task=0;task<tasks;++task) {
            const int64_t pixel=task%pp, edge=(task/pp)%e, batch=task/(pp*e);
            const float *patch=ctx.patches.data()+((batch*ctx.p+kk[edge])*pp+pixel)*ctx.c;
            const float cx=coords[(batch*e+edge)*2*pp+pixel];
            const float cy=coords[(batch*e+edge)*2*pp+pp+pixel];
            for(int64_t level=0;level<2;++level) {
                const float factor=level==0 ? 1.0f : 4.0f;
                const float x=cx/factor,y=cy/factor,fx=std::floor(x),fy=std::floor(y);
                const int64_t ix=static_cast<int64_t>(fx),iy=static_cast<int64_t>(fy);
                const float dx=x-fx,dy=y-fy;
                const int64_t h=level==0 ? ctx.h : ctx.h4, w=level==0 ? ctx.w : ctx.w4;
                const float *target=(level==0 ? ctx.level1.data() : ctx.level4.data());
                float raw[8][8];
                for(int64_t oy=0;oy<8;++oy) for(int64_t ox=0;ox<8;++ox) {
                    const int64_t tx=ix+ox-3,ty=iy+oy-3;
                    float value=0.0f;
                    if(tx>=0 && tx<w && ty>=0 && ty<h)
                        value=dot(patch,target+(((batch*ctx.v+jj[edge])*h+ty)*w+tx)*ctx.c,ctx.c);
                    raw[oy][ox]=value;
                    if(!std::isfinite(value)) invalid=1;
                }
                for(int64_t ox=0;ox<7;++ox) for(int64_t oy=0;oy<7;++oy) {
                    float value=((1.0f-dx)*(1.0f-dy))*raw[oy][ox];
                    value += (dx*(1.0f-dy))*raw[oy][ox+1];
                    value += ((1.0f-dx)*dy)*raw[oy+1][ox];
                    value += (dx*dy)*raw[oy+1][ox+1];
                    const int64_t index=((((batch*e+edge)*7+ox)*7+oy)*pp+pixel)*2+level;
                    output[index]=value;
                    if(!std::isfinite(value)) invalid=1;
                }
            }
        }
        if(stats) { stats->kernel_seconds=elapsed(kernel); stats->total_seconds=elapsed(begin); }
        if(invalid) throw std::invalid_argument("FP32 correlation overflow/nonfinite result");
        return 0;
    } catch(const std::exception &failure) { message(error,capacity,failure.what()); }
      catch(...) { message(error,capacity,"unknown correlation failure"); }
    if(stats) stats->total_seconds=elapsed(begin);
    return 1;
}

extern "C" int dpvo_corr_v27_destroy(void **handle,char *error,size_t capacity) {
    message(error,capacity,"");
    if(!handle) { message(error,capacity,"NULL context pointer destination"); return 1; }
    delete static_cast<Context*>(*handle); *handle=nullptr; return 0;
}
