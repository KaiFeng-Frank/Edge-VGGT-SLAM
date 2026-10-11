/* DPVO MIT research: v45 derives the frozen v36 per-frame correlation cache.
 * v36 source SHA: a04ab5a4b86cfd189ef194893b9b07c06742e21141056cc94a120b42c9d2899d.
 * v45 additionally computes four valid horizontal targets with independent
 * NEON accumulators, preserving each output's original FP32 operation order.
 * Preparation, pooling, frame ownership, guards and bilinear output are unchanged.
 * Build with -fno-fast-math -ffp-contract=off. Frozen included v27 parent SHA:
 * 3cf0c9cf79754fff996efc03c2ca2b162733f10b2b9aa580dee263e5df5b9990. */
#include "dpvo_frame_corr.h"
#include "dpvo_corr.cpp"


/* v45 candidate: independent horizontal accumulators; no cross-output reduction.
 * Each accumulator has the original channel order and lane reduction.
 * The scalar architecture deliberately retains the frozen scalar dot. */
namespace {
void horizontal_dot4_same_order(const float *first, const float *second,
                               int64_t channels, float result[4]) {
    #if DPVO_V27_NEON
    float32x4_t a0=vdupq_n_f32(0.0f), a1=vdupq_n_f32(0.0f);
    float32x4_t a2=vdupq_n_f32(0.0f), a3=vdupq_n_f32(0.0f);
    int64_t channel=0;
    for (;channel+4<=channels;channel+=4) {
        const float32x4_t patch=vld1q_f32(first+channel);
        a0=vaddq_f32(a0,vmulq_f32(patch,vld1q_f32(second+channel)));
        a1=vaddq_f32(a1,vmulq_f32(patch,vld1q_f32(second+channels+channel)));
        a2=vaddq_f32(a2,vmulq_f32(patch,vld1q_f32(second+2*channels+channel)));
        a3=vaddq_f32(a3,vmulq_f32(patch,vld1q_f32(second+3*channels+channel)));
    }
    const float32x4_t sums[4]={a0,a1,a2,a3};
    for (int q=0;q<4;++q) {
        float parts[4]; vst1q_f32(parts,sums[q]);
        float total=((parts[0]+parts[1])+parts[2])+parts[3];
        for (int64_t c=channel;c<channels;++c)
            total += first[c]*second[q*channels+c];
        result[q]=total;
    }
    #else
    for (int q=0;q<4;++q)
        result[q]=dot(first,second+q*channels,channels);
    #endif
}
}

extern "C" const char *dpvo_corr_v45_math_variant() {
    return "horizontal4_independent_original_channel_lane_order_no_FMA";
}

extern "C" uint32_t dpvo_corr_v36_abi_version() { return 1; }
extern "C" uint64_t dpvo_corr_v36_stats_size() { return sizeof(dpvo_corr_v27_stats); }
extern "C" const char *dpvo_corr_v36_build_info() { return dpvo_corr_v27_build_info(); }
extern "C" const char *dpvo_corr_v36_parent_source_sha256() {
    return "3cf0c9cf79754fff996efc03c2ca2b162733f10b2b9aa580dee263e5df5b9990";
}

extern "C" int dpvo_corr_v36_prepare_frame(
    const float *gmap,uint64_t ng,const float *fmap,uint64_t nf,
    int64_t m,int64_t c,int64_t ph,int64_t pw,int64_t h,int64_t w,
    int32_t threads,void **frame,dpvo_corr_v27_stats *stats,char *error,size_t cap) {
    return dpvo_corr_v27_prepare(gmap,ng,fmap,nf,1,m,1,c,ph,pw,h,w,
                                threads,frame,stats,error,cap);
}

extern "C" int dpvo_corr_v36_correlate_frames(
    const void *const *frames,uint64_t n,const float *coords,uint64_t nc,
    const int64_t *kk,const int64_t *jj,uint64_t ni,int64_t e,int32_t radius,
    float *output,uint64_t no,dpvo_corr_v27_stats *stats,char *error,size_t cap) {
    const auto begin=Clock::now(); stats_init(stats); message(error,cap,"");
    try {
        if (!frames || n<1 || n>64) throw std::invalid_argument("1..64 frame handles required");
        const Context *first=nullptr;
        uint64_t bytes=0;
        for(uint64_t f=0;f<n;++f) if(frames[f]) {
            const auto *ctx=static_cast<const Context*>(frames[f]);
            if(!first) first=ctx;
            if(ctx->b!=1 || ctx->v!=1 || ctx->p!=first->p || ctx->c!=first->c ||
               ctx->ph!=first->ph || ctx->pw!=first->pw || ctx->h!=first->h ||
               ctx->w!=first->w || ctx->threads!=first->threads)
                throw std::invalid_argument("immutable frame geometry/thread mismatch");
            bytes+=ctx->bytes();
            if(bytes>kMemoryLimit) throw std::invalid_argument("frame set exceeds 512MiB safety budget");
        }
        if(!first) throw std::invalid_argument("no prepared frames");
        const auto &ctx=*first;
        if(stats) { stats->threads=ctx.threads; stats->context_bytes=bytes; }
        if(radius!=3) throw std::invalid_argument("only original radius3 supported");
        const uint64_t expected_nc=product({1,e,2,ctx.ph,ctx.pw});
        const uint64_t expected_no=product({1,e,7,7,ctx.ph,ctx.pw,2});
        if(nc!=expected_nc || no!=expected_no || ni!=static_cast<uint64_t>(e))
            throw std::invalid_argument("coordinate/index/output buffer length mismatch");
        if((e && (!kk || !jj)) || (no && !output)) throw std::invalid_argument("NULL buffer");
        if(overlaps(output,no*4,coords,nc*4) || overlaps(output,no*4,kk,ni*8) ||
           overlaps(output,no*4,jj,ni*8) || overlaps(output,no*4,frames,n*sizeof(void*)))
            throw std::invalid_argument("output may not overlap request buffers");
        const auto scan=Clock::now();
        finite_input(coords,nc,"coords contains nonfinite");
        for(uint64_t i=0;i<nc;++i) if(std::fabs(coords[i])>=1073741824.0f)
            throw std::invalid_argument("coords exceed safe int32 sampling range");
        for(int64_t i=0;i<e;++i) {
            if(kk[i]<0 || static_cast<uint64_t>(kk[i]/ctx.p)>=n || jj[i]<0 ||
               static_cast<uint64_t>(jj[i])>=n)
                throw std::invalid_argument("patch/view index out of range");
            if(!frames[kk[i]/ctx.p] || !frames[jj[i]])
                throw std::invalid_argument("edge refers to an unprepared frame");
        }
        if(stats) { stats->scan_seconds=elapsed(scan); stats->output_elements=no; }
        const auto kernel=Clock::now();
        const int64_t pp=ctx.ph*ctx.pw,tasks=e*pp;
        int invalid=0;
        #pragma omp parallel for schedule(static) num_threads(ctx.threads) reduction(|:invalid)
        for(int64_t task=0;task<tasks;++task) {
            const int64_t pixel=task%pp,edge=task/pp;
            const auto &source=*static_cast<const Context*>(frames[kk[edge]/ctx.p]);
            const auto &destination=*static_cast<const Context*>(frames[jj[edge]]);
            const float *patch=source.patches.data()+((kk[edge]%ctx.p)*pp+pixel)*ctx.c;
            const float cx=coords[edge*2*pp+pixel],cy=coords[edge*2*pp+pp+pixel];
            for(int64_t level=0;level<2;++level) {
                const float factor=level==0 ? 1.0f : 4.0f;
                const float x=cx/factor,y=cy/factor,fx=std::floor(x),fy=std::floor(y);
                const int64_t ix=static_cast<int64_t>(fx),iy=static_cast<int64_t>(fy);
                const float dx=x-fx,dy=y-fy;
                const int64_t h=level==0 ? ctx.h : ctx.h4,w=level==0 ? ctx.w : ctx.w4;
                const float *target=(level==0 ? destination.level1.data() : destination.level4.data());
                float raw[8][8];
                for(int64_t oy=0;oy<8;++oy) for(int64_t ox=0;ox<8;ox+=4) {
                    const int64_t tx=ix+ox-3,ty=iy+oy-3;
                    if(tx>=0 && tx+3<w && ty>=0 && ty<h) {
                        horizontal_dot4_same_order(patch,target+(ty*w+tx)*ctx.c,
                                                   ctx.c,&raw[oy][ox]);
                        for(int q=0;q<4;++q)
                            if(!std::isfinite(raw[oy][ox+q])) invalid=1;
                    } else {
                        for(int q=0;q<4;++q) {
                            const int64_t xq=tx+q;
                            float value=0.0f;
                            if(xq>=0 && xq<w && ty>=0 && ty<h)
                                value=dot(patch,target+(ty*w+xq)*ctx.c,ctx.c);
                            raw[oy][ox+q]=value;
                            if(!std::isfinite(value)) invalid=1;
                        }
                    }
                }
                for(int64_t ox=0;ox<7;++ox) for(int64_t oy=0;oy<7;++oy) {
                    float value=((1.0f-dx)*(1.0f-dy))*raw[oy][ox];
                    value += (dx*(1.0f-dy))*raw[oy][ox+1];
                    value += ((1.0f-dx)*dy)*raw[oy+1][ox];
                    value += (dx*dy)*raw[oy+1][ox+1];
                    const int64_t index=(((edge*7+ox)*7+oy)*pp+pixel)*2+level;
                    output[index]=value;
                    if(!std::isfinite(value)) invalid=1;
                }
            }
        }
        if(stats) { stats->kernel_seconds=elapsed(kernel); stats->total_seconds=elapsed(begin); }
        if(invalid) throw std::invalid_argument("FP32 correlation overflow/nonfinite result");
        return 0;
    } catch(const std::exception &failure) { message(error,cap,failure.what()); }
      catch(...) { message(error,cap,"unknown correlation failure"); }
    if(stats) stats->total_seconds=elapsed(begin);
    return 1;
}

extern "C" int dpvo_corr_v36_destroy_frame(void **frame,char *error,size_t cap) {
    return dpvo_corr_v27_destroy(frame,error,cap);
}
