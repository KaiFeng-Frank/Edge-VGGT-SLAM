/* Research-only native input conversion; no SDK or model functions. */
#ifndef V43_NATIVE_FP16_PACK_H
#define V43_NATIVE_FP16_PACK_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

uint32_t v43_pack_abi_version(void);
uint64_t v43_pack_read_fpcr(void);

/* CF [B,C,D,M] -> NC1HWC2 [B,ceil(C/8),D,M,8], IEEE binary16 bits.
 * B=1/2, C=384/882, D=3..7, M=16. Source and destination must not alias.
 * src_elements must equal the CF product; capacity is BYTES, not elements.
 * Return 0 or a positive errno value. After failure dst is unspecified and
 * MUST NOT be submitted. No input is modified, no FPCR state is modified.
 * Numeric validation and conversion occur within this one synchronous call.
 */
int v43_pack_cf_f32_to_nc1hwc2_f16(
    const float *src, size_t src_elements,
    uint16_t *dst, size_t dst_capacity_bytes,
    uint32_t B, uint32_t C, uint32_t D, uint32_t M);

#ifdef __cplusplus
}
#endif
#endif
