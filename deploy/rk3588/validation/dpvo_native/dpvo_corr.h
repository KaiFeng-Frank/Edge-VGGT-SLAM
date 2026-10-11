/* DPVO CPU correlation research, derived from DPVO (MIT), commit
 * 0ac95b656d1fda91c271d2a106460d19ad966fc7.
 * Copyright (c) 2022 Princeton Vision & Learning Lab. See LICENSE.
 * No Torch, model, CUDA, BA or pose implementation is linked here. */
#ifndef DPVO_CORR_V27_H
#define DPVO_CORR_V27_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Versioned, fixed-width ABI. Dimensions/element counts include batch.
 * Caller owns valid buffers of the declared lengths. Inputs and output must
 * not overlap; context cannot be destroyed while a call uses it. Errors return
 * nonzero and a bounded NUL-terminated message. Never use output after error. */
typedef struct dpvo_corr_v27_stats {
    uint32_t abi_version;
    uint32_t reserved;
    uint64_t context_bytes;
    uint64_t output_elements;
    double scan_seconds;
    double layout_seconds;
    double pool_seconds;
    double kernel_seconds;
    double total_seconds;
    int32_t threads;
    int32_t neon;
} dpvo_corr_v27_stats;

uint32_t dpvo_corr_v27_abi_version(void);
uint64_t dpvo_corr_v27_stats_size(void);
const char *dpvo_corr_v27_build_info(void);

/* gmap [B,Npatch,C,PH,PW], fmap [B,Nview,C,H,W], C-order FP32.
 * Creates independent owned channel-contiguous copies and a level4 pyramid.
 * Input mutations after prepare have no effect: explicitly prepare a NEW
 * context when feature values/generation change. No pointer-identity cache. */
int dpvo_corr_v27_prepare(
    const float *gmap, uint64_t gmap_elements,
    const float *fmap, uint64_t fmap_elements,
    int64_t batch, int64_t patches, int64_t views, int64_t channels,
    int64_t patch_h, int64_t patch_w, int64_t height, int64_t width,
    int32_t threads, void **context, dpvo_corr_v27_stats *stats,
    char *error, size_t error_capacity);

/* coords [B,E,2,PH,PW] FP32 xy; kk/jj [E] int64 (source patch/target view).
 * Fixed radius3 and independent levels1,4. Output CPU FP32 C-order
 * [B,E,7,7,PH,PW,2] or flattened [B,E,98*PH*PW]. */
int dpvo_corr_v27_correlate(
    const void *context, const float *coords, uint64_t coordinate_elements,
    const int64_t *kk, const int64_t *jj, uint64_t index_elements,
    int64_t edges, int32_t radius, float *output, uint64_t output_elements,
    dpvo_corr_v27_stats *stats, char *error, size_t error_capacity);

/* Idempotent for *context=NULL. Sets the caller's pointer to NULL. */
int dpvo_corr_v27_destroy(void **context, char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
