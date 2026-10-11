/* DPVO MIT research: immutable per-frame reuse, no geometry/model changes.
 * Build with -fno-fast-math -ffp-contract=off. The included frozen parent
 * retains the exact preparation, pooling, scalar/NEON dot and FP32 order.
 * Parent SHA: 3cf0c9cf79754fff996efc03c2ca2b162733f10b2b9aa580dee263e5df5b9990.
 * The only kernel change is choosing immutable source/target frame pointers
 * instead of indexing one newly concatenated historical feature buffer. */
#include "dpvo_frame_corr.h"
#include "dpvo_corr.cpp"

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
                for(int64_t oy=0;oy<8;++oy) for(int64_t ox=0;ox<8;++ox) {
                    const int64_t tx=ix+ox-3,ty=iy+oy-3;
                    float value=0.0f;
                    if(tx>=0 && tx<w && ty>=0 && ty<h)
                        value=dot(patch,target+(ty*w+tx)*ctx.c,ctx.c);
                    raw[oy][ox]=value;
                    if(!std::isfinite(value)) invalid=1;
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
