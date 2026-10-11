/* Research-only AArch64 NEON conversion. No SDK, allocation or FPCR writes.
 * Build without fast-math: -O3 -std=c11 -march=armv8.2-a+fp16 -fPIC -shared
 * -fno-fast-math -ffp-contract=off. NumPy is not part of the timed kernel.
 */
#include "v43_native_fp16_pack.h"
#include <arm_neon.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error "AArch64 with FP16 NEON conversion is required"
#endif
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "The frozen native attribute contract is little endian"
#endif
#if defined(__FAST_MATH__)
#error "fast-math is unsupported: finite and half overflow checks are required"
#endif
_Static_assert(sizeof(float) == 4 && sizeof(uint16_t) == 2, "FP32/FP16 ABI sizes");

/* Require RN, gradual underflow, standard half format, no alternate handling
 * and no enabled FP traps. DN is also rejected to keep the supported state
 * explicit. Read only: caller's thread FP environment is never repaired.
 */
#define UNSUPPORTED_FPCR ( \
    (UINT64_C(3) << 22) | (UINT64_C(1) << 24) | \
    (UINT64_C(1) << 19) | (UINT64_C(1) << 26) | \
    (UINT64_C(1) << 25) | (UINT64_C(31) << 8) | \
    (UINT64_C(1) << 15) | UINT64_C(3))

uint32_t v43_pack_abi_version(void) { return 1; }
uint64_t v43_pack_read_fpcr(void) {
    uint64_t result;
    __asm__ volatile("mrs %0, fpcr" : "=r" (result));
    return result;
}

static int checked_mul(size_t a, size_t b, size_t *result) {
    if (b && a > SIZE_MAX / b) return EOVERFLOW;
    *result = a * b;
    return 0;
}

/* Four CF channel rows become four spatial columns, with channels in lanes. */
static inline void transpose4(const float32x4_t *x, float32x4_t *columns) {
    const float32x4x2_t ab = vtrnq_f32(x[0], x[1]);
    const float32x4x2_t cd = vtrnq_f32(x[2], x[3]);
    columns[0] = vcombine_f32(vget_low_f32(ab.val[0]), vget_low_f32(cd.val[0]));
    columns[1] = vcombine_f32(vget_low_f32(ab.val[1]), vget_low_f32(cd.val[1]));
    columns[2] = vcombine_f32(vget_high_f32(ab.val[0]), vget_high_f32(cd.val[0]));
    columns[3] = vcombine_f32(vget_high_f32(ab.val[1]), vget_high_f32(cd.val[1]));
}

int v43_pack_cf_f32_to_nc1hwc2_f16(
    const float *src, size_t src_elements,
    uint16_t *dst, size_t dst_capacity_bytes,
    uint32_t B, uint32_t C, uint32_t D, uint32_t M) {
    if (!src || !dst || (B != 1 && B != 2) || (C != 384 && C != 882)
        || D < 3 || D > 7 || M != 16) return EINVAL;
    if ((uintptr_t)src % _Alignof(float) || (uintptr_t)dst % _Alignof(uint16_t))
        return EINVAL;

    const size_t blocks = ((size_t)C + 7) / 8;
    size_t spatial, needed_src, src_bytes, needed_dst, dst_bytes;
    if (checked_mul(D, M, &spatial) || checked_mul(B, C, &needed_src)
        || checked_mul(needed_src, spatial, &needed_src)
        || checked_mul(needed_src, sizeof(float), &src_bytes)
        || checked_mul(B, blocks, &needed_dst)
        || checked_mul(needed_dst, spatial, &needed_dst)
        || checked_mul(needed_dst, 8, &needed_dst)
        || checked_mul(needed_dst, sizeof(uint16_t), &dst_bytes)) return EOVERFLOW;
    if (src_elements != needed_src) return EINVAL;
    if (dst_capacity_bytes < dst_bytes) return ENOSPC;
    const uintptr_t s = (uintptr_t)src, t = (uintptr_t)dst;
    if (s > UINTPTR_MAX - src_bytes || t > UINTPTR_MAX - dst_bytes) return EOVERFLOW;
    if (s < t + dst_bytes && t < s + src_bytes) return EINVAL;
    if (v43_pack_read_fpcr() & UNSUPPORTED_FPCR) return ENOTSUP;

    const uint32x4_t exponent32 = vdupq_n_u32(UINT32_C(0x7f800000));
    const uint16x4_t exponent16 = vdup_n_u16(UINT16_C(0x7c00));
    for (size_t b = 0; b < B; ++b) {
        for (size_t block = 0; block < blocks; ++block) {
            for (size_t p = 0; p < spatial; p += 4) {
                float32x4_t rows[8], low[4], high[4];
                uint32x4_t bad = vdupq_n_u32(0);
                for (size_t lane = 0; lane < 8; ++lane) {
                    const size_t channel = block * 8 + lane;
                    rows[lane] = channel < C
                        ? vld1q_f32(src + (b * C + channel) * spatial + p)
                        : vdupq_n_f32(0.0f);
                    bad = vorrq_u32(bad, vceqq_u32(
                        vandq_u32(vreinterpretq_u32_f32(rows[lane]), exponent32), exponent32));
                }
                /* Bit checks reject Inf and all NaNs before any FP conversion. */
                if (vmaxvq_u32(bad)) return EDOM;
                transpose4(rows, low);
                transpose4(rows + 4, high);
                float16x4_t hlow[4], hhigh[4];
                uint16x4_t overflow = vdup_n_u16(0);
                for (size_t position = 0; position < 4; ++position) {
                    hlow[position] = vcvt_f16_f32(low[position]);
                    hhigh[position] = vcvt_f16_f32(high[position]);
                    overflow = vorr_u16(overflow, vceq_u16(vand_u16(
                        vreinterpret_u16_f16(hlow[position]), exponent16), exponent16));
                    overflow = vorr_u16(overflow, vceq_u16(vand_u16(
                        vreinterpret_u16_f16(hhigh[position]), exponent16), exponent16));
                }
                /* Test the actual rounded result: do not clip or reject finite
                 * values just above 65504 that correctly round back to 65504. */
                if (vmaxv_u16(overflow)) return ERANGE;
                uint16_t *out = dst + ((b * blocks + block) * spatial + p) * 8;
                for (size_t position = 0; position < 4; ++position)
                    vst1q_u16(out + position * 8, vreinterpretq_u16_f16(
                        vcombine_f16(hlow[position], hhigh[position])));
            }
        }
    }
    return 0;
}
