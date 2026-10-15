#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF

#define ROWS 1041u
#define KEYS 1041u
#define PAD_KEYS 1056u
#define WG_SIZE 256u
#define ITEMS_PER_LANE 5u

// One workgroup owns one original query row. C4/A8 are the actual native
// MatMul layouts, not NCHW or an SDK custom-op surface. All original keys
// participate. The fifteen storage-padding keys never enter either reduction.
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void original_softmax_c4_to_half_a8(
    __global const float *src, const ulong src_offset_elements,
    __global half *dst, const ulong dst_offset_elements,
    volatile __global unsigned int *error_bits)
{
    const uint row = (uint)get_group_id(0);
    const uint lane = (uint)get_local_id(0);
    __local float reduction[WG_SIZE];
    float score[ITEMS_PER_LANE];
    float weight[ITEMS_PER_LANE];
    float lane_max = -INFINITY;

    for (uint item = 0; item < ITEMS_PER_LANE; ++item) {
        const uint key = lane + item * WG_SIZE;
        float value = -INFINITY;
        if (key < KEYS) {
            const size_t c4 = ((size_t)(key / 4u) * ROWS + row) * 4u + key % 4u;
            value = src[src_offset_elements + c4] * 0.125f;
            if (!isfinite(value)) {
                atomic_or(error_bits, 1u);
                value = 0.0f; // The call fails; retain safe, unconditional barriers.
            }
        }
        score[item] = value;
        lane_max = fmax(lane_max, value);
    }
    reduction[lane] = lane_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint step = WG_SIZE / 2u; step > 0u; step /= 2u) {
        if (lane < step)
            reduction[lane] = fmax(reduction[lane], reduction[lane + step]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float row_max = reduction[0];
    // Every lane must consume row_max before this local array is reused.
    barrier(CLK_LOCAL_MEM_FENCE);

    float lane_sum = 0.0f;
    for (uint item = 0; item < ITEMS_PER_LANE; ++item) {
        const uint key = lane + item * WG_SIZE;
        float value = key < KEYS ? exp(score[item] - row_max) : 0.0f;
        if (!isfinite(value)) {
            atomic_or(error_bits, 2u);
            value = 0.0f;
        }
        weight[item] = value;
        lane_sum += value;
    }
    reduction[lane] = lane_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint step = WG_SIZE / 2u; step > 0u; step /= 2u) {
        if (lane < step)
            reduction[lane] += reduction[lane + step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float denominator = reduction[0];
    const int valid_sum = isfinite(denominator) && denominator > 0.0f;
    if (!valid_sum && lane == 0u)
        atomic_or(error_bits, 4u);

    for (uint item = 0; item < ITEMS_PER_LANE; ++item) {
        const uint key = lane + item * WG_SIZE;
        if (key < PAD_KEYS) {
            float probability = 0.0f;
            if (key < KEYS && valid_sum) {
                probability = weight[item] / denominator;
                if (!isfinite(probability) || probability < 0.0f || probability > 1.0f) {
                    atomic_or(error_bits, 8u);
                    probability = 0.0f;
                }
            }
            const size_t a8 = ((size_t)(key / 8u) * ROWS + row) * 8u + key % 8u;
            dst[dst_offset_elements + a8] = convert_half_rte(probability);
        }
    }
}
