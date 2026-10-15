#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF

// Original VGGT DINO0 only: B=1, all 1041 tokens, 16 heads of 64 channels.
// MatMul is bias-free FP16 x FP16 -> FP32. Parameters, residuals and all
// scalar arithmetic here remain FP32; each native A/B boundary uses Half RTE.
// Build with -cl-std=CL1.2 only. No fast math or native_* approximations.
// Concatenate the frozen R119 attention_softmax.cl unchanged at build time.

#define DINO0_ROWS 1041u
#define DINO0_DIM 1024u
#define DINO0_HIDDEN 4096u
#define DINO0_HEADS 16u
#define DINO0_HEAD_DIM 64u
#define DINO0_PAD_KEYS 1056u
#define DINO0_WG 256u
#define DINO0_NORM_ITEMS 4u

// Offsets on imported pointers are in elements of the pointer's scalar type.
// NPU owners must validate actual native descriptors and imported capacities.
inline size_t dino0_a8(const uint row, const uint column)
{
    return ((size_t)(column / 8u) * DINO0_ROWS + row) * 8u + column % 8u;
}

inline size_t dino0_c4(const uint row, const uint column)
{
    return ((size_t)(column / 4u) * DINO0_ROWS + row) * 4u + column % 4u;
}

inline size_t dino0_b16x32(const uint k, const uint n, const uint k_size)
{
    return (((size_t)(n / 16u) * (k_size / 32u) + k / 32u) * 16u
            + n % 16u) * 32u + k % 32u;
}

// Original full channels; only integer shard selection/local addressing.
inline float dino0_sharded_c4_352(
    const uint row, const uint column,
    __global const float *c0, const ulong offset0,
    __global const float *c1, const ulong offset1,
    __global const float *c2, const ulong offset2)
{
    if (column < 352u)
        return c0[offset0 + dino0_c4(row, column)];
    if (column < 704u)
        return c1[offset1 + dino0_c4(row, column - 352u)];
    return c2[offset2 + dino0_c4(row, column - 704u)];
}

// Original full channels; only integer shard selection/local addressing.
inline float dino0_sharded_c4_1376(
    const uint row, const uint column,
    __global const float *c0, const ulong offset0,
    __global const float *c1, const ulong offset1,
    __global const float *c2, const ulong offset2)
{
    if (column < 1376u)
        return c0[offset0 + dino0_c4(row, column)];
    if (column < 2752u)
        return c1[offset1 + dino0_c4(row, column - 1376u)];
    return c2[offset2 + dino0_c4(row, column - 2752u)];
}

// Two-pass population variance. Every lane contributes four original values;
// the second pass squares centered FP32 values, not E[x*x] - E[x]*E[x].
// Reduction order is explicit and may round differently from Torch's kernel.
inline float dino0_norm_denominator(
    const float values[DINO0_NORM_ITEMS], const uint lane,
    __local float *reduction, __private float *row_mean)
{
    float lane_sum = 0.0f;
    for (uint item = 0u; item < DINO0_NORM_ITEMS; ++item)
        lane_sum = lane_sum + values[item];
    reduction[lane] = lane_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint step = DINO0_WG / 2u; step > 0u; step /= 2u) {
        if (lane < step)
            reduction[lane] = reduction[lane] + reduction[lane + step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float mean = reduction[0] / 1024.0f;
    // All lanes must consume the mean before reusing the local array.
    barrier(CLK_LOCAL_MEM_FENCE);
    float lane_square_sum = 0.0f;
    for (uint item = 0u; item < DINO0_NORM_ITEMS; ++item) {
        const float centered = values[item] - mean;
        const float square = centered * centered;
        lane_square_sum = lane_square_sum + square;
    }
    reduction[lane] = lane_square_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint step = DINO0_WG / 2u; step > 0u; step /= 2u) {
        if (lane < step)
            reduction[lane] = reduction[lane] + reduction[lane + step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float variance = reduction[0] / 1024.0f;
    const float denominator = sqrt(variance + 1.0e-6f);
    *row_mean = mean;
    return denominator;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void norm1_pack(
    __global const float *x,
    __global const float *gamma,
    __global const float *beta,
    __global half *qkvA, const ulong qkvA_offset)
{
    const uint row = (uint)get_group_id(0);
    const uint lane = (uint)get_local_id(0);
    if (row >= DINO0_ROWS) return; // Uniform for the entire workgroup.
    __local float reduction[DINO0_WG];
    float values[DINO0_NORM_ITEMS];
    for (uint item = 0u; item < DINO0_NORM_ITEMS; ++item) {
        const uint column = lane + item * DINO0_WG;
        values[item] = x[(size_t)row * DINO0_DIM + column];
    }
    float mean;
    const float denominator = dino0_norm_denominator(values, lane, reduction, &mean);
    for (uint item = 0u; item < DINO0_NORM_ITEMS; ++item) {
        const uint column = lane + item * DINO0_WG;
        const float centered = values[item] - mean;
        const float normalized = centered / denominator;
        const float scaled = normalized * gamma[column];
        const float shifted = scaled + beta[column];
        qkvA[qkvA_offset + dino0_a8(row, column)] = convert_half_rte(shifted);
    }
}

__kernel void qkv_head_pack(
    __global const float *qkv0C, const ulong qkv0C_offset,
    __global const float *qkv1C, const ulong qkv1C_offset,
    __global const float *qkv2C, const ulong qkv2C_offset,
    __global const float *qkvbias,
    __global half *qkA, const ulong qkA_offset,
    __global half *qkB, const ulong qkB_offset,
    __global half *pvB, const ulong pvB_offset,
    const int head)
{
    const size_t i = get_global_id(0);
    if (i >= (size_t)DINO0_PAD_KEYS * DINO0_HEAD_DIM
            || head < 0 || head >= (int)DINO0_HEADS) return;
    // Traverse a C4-shaped 1056-row tile to coalesce reads of real Q/K/V.
    const uint row = (uint)((i / 4u) % DINO0_PAD_KEYS);
    const uint channel = (uint)((i / 4u / DINO0_PAD_KEYS) * 4u + i % 4u);
    const uint q_column = (uint)head * DINO0_HEAD_DIM + channel;
    const uint k_column = DINO0_DIM + q_column;
    const uint v_column = 2u * DINO0_DIM + q_column;
    half k_value = (half)0.0f;
    half v_value = (half)0.0f;
    if (row < DINO0_ROWS) {
        const float query = qkv0C[qkv0C_offset + dino0_c4(row, q_column)]
                            + qkvbias[q_column];
        const float key = qkv1C[qkv1C_offset + dino0_c4(row, k_column - DINO0_DIM)]
                          + qkvbias[k_column];
        const float value = qkv2C[qkv2C_offset + dino0_c4(row, v_column - 2u * DINO0_DIM)]
                            + qkvbias[v_column];
        qkA[qkA_offset + dino0_a8(row, channel)] = convert_half_rte(query);
        k_value = convert_half_rte(key);
        v_value = convert_half_rte(value);
    }
    // QK B is original K^T[64,1041] with 15 zero storage columns.
    qkB[qkB_offset + dino0_b16x32(channel, row, DINO0_HEAD_DIM)] = k_value;
    // PV B is original V[1041,64] with 15 zero storage rows.
    pvB[pvB_offset + dino0_b16x32(row, channel, DINO0_PAD_KEYS)] = v_value;
}

__kernel void pv_head_merge(
    __global const float *pvC, const ulong pvC_offset,
    __global half *projA, const ulong projA_offset,
    const int head)
{
    const size_t i = get_global_id(0);
    if (i >= (size_t)DINO0_ROWS * DINO0_HEAD_DIM
            || head < 0 || head >= (int)DINO0_HEADS) return;
    // Traverse native PV C4; place this head in original [row,head,channel].
    const uint row = (uint)((i / 4u) % DINO0_ROWS);
    const uint channel = (uint)((i / 4u / DINO0_ROWS) * 4u + i % 4u);
    const uint column = (uint)head * DINO0_HEAD_DIM + channel;
    projA[projA_offset + dino0_a8(row, column)] = convert_half_rte(pvC[pvC_offset + i]);
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void proj_residual_norm(
    __global const float *proj0C, const ulong proj0C_offset,
    __global const float *proj1C, const ulong proj1C_offset,
    __global const float *proj2C, const ulong proj2C_offset,
    __global const float *bias,
    __global const float *ls1,
    __global const float *x,
    __global const float *gamma2,
    __global const float *beta2,
    __global float *residual1,
    __global half *fc1A, const ulong fc1A_offset)
{
    const uint row = (uint)get_group_id(0);
    const uint lane = (uint)get_local_id(0);
    if (row >= DINO0_ROWS) return;
    __local float reduction[DINO0_WG];
    float values[DINO0_NORM_ITEMS];
    for (uint item = 0u; item < DINO0_NORM_ITEMS; ++item) {
        const uint column = lane + item * DINO0_WG;
        const size_t index = (size_t)row * DINO0_DIM + column;
        const float projected = dino0_sharded_c4_352(row, column, proj0C, proj0C_offset, proj1C, proj1C_offset, proj2C, proj2C_offset) + bias[column];
        const float layer_scaled = projected * ls1[column];
        const float residual = x[index] + layer_scaled;
        values[item] = residual;
        residual1[index] = residual; // Retain original FP32 r1 for final residual.
    }
    float mean;
    const float denominator = dino0_norm_denominator(values, lane, reduction, &mean);
    for (uint item = 0u; item < DINO0_NORM_ITEMS; ++item) {
        const uint column = lane + item * DINO0_WG;
        const float centered = values[item] - mean;
        const float normalized = centered / denominator;
        const float scaled = normalized * gamma2[column];
        const float shifted = scaled + beta2[column];
        fc1A[fc1A_offset + dino0_a8(row, column)] = convert_half_rte(shifted);
    }
}

__kernel void gelu_pack(
    __global const float *fc10C, const ulong fc10C_offset,
    __global const float *fc11C, const ulong fc11C_offset,
    __global const float *fc12C, const ulong fc12C_offset,
    __global const float *bias,
    __global half *fc2A, const ulong fc2A_offset)
{
    const size_t i = get_global_id(0);
    if (i >= (size_t)DINO0_ROWS * DINO0_HIDDEN) return;
    // Traverse destination A8, as in the original R81 Erf bridge.
    const uint row = (uint)((i / 8u) % DINO0_ROWS);
    const uint column = (uint)((i / 8u / DINO0_ROWS) * 8u + i % 8u);
    const float hidden = dino0_sharded_c4_1376(row, column, fc10C, fc10C_offset, fc11C, fc11C_offset, fc12C, fc12C_offset) + bias[column];
    const float scaled = hidden / 0x1.6a09e6p+0f; // FP32 sqrt(2), not inverse multiply.
    const float plus_one = 1.0f + erf(scaled);
    const float product = hidden * plus_one;
    const float gelu = product * 0.5f;
    fc2A[fc2A_offset + i] = convert_half_rte(gelu);
}

__kernel void final_residual(
    __global const float *fc20C, const ulong fc20C_offset,
    __global const float *fc21C, const ulong fc21C_offset,
    __global const float *fc22C, const ulong fc22C_offset,
    __global const float *bias,
    __global const float *ls2,
    __global const float *residual1,
    __global float *out)
{
    const size_t i = get_global_id(0);
    if (i >= (size_t)DINO0_ROWS * DINO0_DIM) return;
    const uint row = (uint)(i / DINO0_DIM);
    const uint column = (uint)(i % DINO0_DIM);
    const float mlp = dino0_sharded_c4_352(row, column, fc20C, fc20C_offset, fc21C, fc21C_offset, fc22C, fc22C_offset) + bias[column];
    const float layer_scaled = mlp * ls2[column];
    out[i] = residual1[i] + layer_scaled;
}
