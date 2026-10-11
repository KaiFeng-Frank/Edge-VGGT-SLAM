/* Original VGGT IO conversion only; no model/SDK execution in this source.
 * Build on RK3588: -O3 -std=c11 -shared -fPIC -march=armv8.2-a+fp16
 * Use -fno-fast-math and the normal FPCR (round to nearest, no flush-to-zero).
 * The Python caller supplies disjoint dense buffers of at least n elements.
 * No restrict, vector alignment or incompatible typed-pointer aliasing is used.
 * memcpy loads/stores also permit storage not aligned to a vector boundary.
 */
#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error "ARMv8.2 AArch64 FP16 NEON is required"
#endif

void f32_to_f16(const float *source, uint16_t *destination, size_t n)
{
    const unsigned char *src = (const unsigned char *)(const void *)source;
    unsigned char *dst = (unsigned char *)(void *)destination;
    while (n >= 8) {
        float32x4_t low, high;
        memcpy(&low, src, sizeof(low));
        memcpy(&high, src + sizeof(low), sizeof(high));
        uint16x8_t bits = vcombine_u16(vreinterpret_u16_f16(vcvt_f16_f32(low)),
                                      vreinterpret_u16_f16(vcvt_f16_f32(high)));
        memcpy(dst, &bits, sizeof(bits));
        src += 8 * sizeof(float);
        dst += 8 * sizeof(uint16_t);
        n -= 8;
    }
    if (n) {
        float values[8] = {0};
        float32x4_t low, high;
        memcpy(values, src, n * sizeof(float));
        memcpy(&low, values, sizeof(low));
        memcpy(&high, values + 4, sizeof(high));
        uint16x8_t bits = vcombine_u16(vreinterpret_u16_f16(vcvt_f16_f32(low)),
                                      vreinterpret_u16_f16(vcvt_f16_f32(high)));
        memcpy(dst, &bits, n * sizeof(uint16_t));
    }
}

void f16_to_f32(const uint16_t *source, float *destination, size_t n)
{
    const unsigned char *src = (const unsigned char *)(const void *)source;
    unsigned char *dst = (unsigned char *)(void *)destination;
    while (n >= 8) {
        uint16x8_t bits;
        memcpy(&bits, src, sizeof(bits));
        float32x4_t low = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(bits)));
        float32x4_t high = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(bits)));
        memcpy(dst, &low, sizeof(low));
        memcpy(dst + sizeof(low), &high, sizeof(high));
        src += 8 * sizeof(uint16_t);
        dst += 8 * sizeof(float);
        n -= 8;
    }
    if (n) {
        uint16x8_t bits = vdupq_n_u16(0);
        float values[8];
        memcpy(&bits, src, n * sizeof(uint16_t));
        float32x4_t low = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(bits)));
        float32x4_t high = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(bits)));
        memcpy(values, &low, sizeof(low));
        memcpy(values + 4, &high, sizeof(high));
        memcpy(dst, values, n * sizeof(float));
    }
}
