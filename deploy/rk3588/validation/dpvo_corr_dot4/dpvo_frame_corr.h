/* DPVO immutable per-frame correlation research (MIT).
 * Derived from DPVO 0ac95b656d1fda91c271d2a106460d19ad966fc7.
 * Copyright (c) 2022 Princeton Vision & Learning Lab. See LICENSE.
 * Uses the byte-frozen v27 header/source; no model/BA/device code. */
#ifndef DPVO_FRAME_CORR_V36_H
#define DPVO_FRAME_CORR_V36_H
#include "dpvo_corr.h"
#ifdef __cplusplus
extern "C" {
#endif

uint32_t dpvo_corr_v36_abi_version(void);
uint64_t dpvo_corr_v36_stats_size(void);
const char *dpvo_corr_v36_build_info(void);
const char *dpvo_corr_v36_parent_source_sha256(void);

/* One frame: gmap [M,C,PH,PW], fmap [C,H,W], C-order FP32.
 * The existing v27 preparation creates an independent owned snapshot:
 * patches, channel-last level1, and independent 4x4 level4 pooling.
 * No feature values, geometry or thread count can change after preparation. */
int dpvo_corr_v36_prepare_frame(
    const float *gmap, uint64_t gmap_elements,
    const float *fmap, uint64_t fmap_elements,
    int64_t patches, int64_t channels, int64_t patch_h, int64_t patch_w,
    int64_t height, int64_t width, int32_t threads,
    void **frame, dpvo_corr_v27_stats *stats, char *error, size_t error_capacity);

/* handles follow CURRENT retained-frame indices, not historical slots.
 * NULL entries are permitted only when no edge refers to them. kk refers to
 * frame=kk/M, patch=kk%M, jj to the target frame. Original edge order remains.
 * coords [1,E,2,PH,PW], output [1,E,7,7,PH,PW,2] (level innermost).
 * Caller holds every prepared frame alive until this synchronous call ends.
 * Frames are immutable; independent queries can read them concurrently, but
 * destruction must be serialized by the owner. Output must not alias inputs. */
int dpvo_corr_v36_correlate_frames(
    const void *const *frames, uint64_t frame_count,
    const float *coords, uint64_t coordinate_elements,
    const int64_t *kk, const int64_t *jj, uint64_t index_elements,
    int64_t edges, int32_t radius, float *output, uint64_t output_elements,
    dpvo_corr_v27_stats *stats, char *error, size_t error_capacity);

int dpvo_corr_v36_destroy_frame(void **frame, char *error, size_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
