#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF

// Append after the frozen original DINO0 and R119 softmax sources.
// Original arithmetic bodies are copied verbatim; only per-head
// indexing and uniform fixed-pointer selection are substituted.
// Fixed 16 heads, all1041 keys,15 storage-padding keys. No slab or
// host/GPU pointer table. Build only with -cl-std=CL1.2.

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void pack_all_heads(
    __global const float *qkvC,
    const ulong qkvC_offset,
    __global const float *qkvbias,
    __global half * qkA_h0,
    ulong qkA_offset_h0,
    __global half * qkB_h0,
    ulong qkB_offset_h0,
    __global half * pvB_h0,
    ulong pvB_offset_h0,
    __global half * qkA_h1,
    ulong qkA_offset_h1,
    __global half * qkB_h1,
    ulong qkB_offset_h1,
    __global half * pvB_h1,
    ulong pvB_offset_h1,
    __global half * qkA_h2,
    ulong qkA_offset_h2,
    __global half * qkB_h2,
    ulong qkB_offset_h2,
    __global half * pvB_h2,
    ulong pvB_offset_h2,
    __global half * qkA_h3,
    ulong qkA_offset_h3,
    __global half * qkB_h3,
    ulong qkB_offset_h3,
    __global half * pvB_h3,
    ulong pvB_offset_h3,
    __global half * qkA_h4,
    ulong qkA_offset_h4,
    __global half * qkB_h4,
    ulong qkB_offset_h4,
    __global half * pvB_h4,
    ulong pvB_offset_h4,
    __global half * qkA_h5,
    ulong qkA_offset_h5,
    __global half * qkB_h5,
    ulong qkB_offset_h5,
    __global half * pvB_h5,
    ulong pvB_offset_h5,
    __global half * qkA_h6,
    ulong qkA_offset_h6,
    __global half * qkB_h6,
    ulong qkB_offset_h6,
    __global half * pvB_h6,
    ulong pvB_offset_h6,
    __global half * qkA_h7,
    ulong qkA_offset_h7,
    __global half * qkB_h7,
    ulong qkB_offset_h7,
    __global half * pvB_h7,
    ulong pvB_offset_h7,
    __global half * qkA_h8,
    ulong qkA_offset_h8,
    __global half * qkB_h8,
    ulong qkB_offset_h8,
    __global half * pvB_h8,
    ulong pvB_offset_h8,
    __global half * qkA_h9,
    ulong qkA_offset_h9,
    __global half * qkB_h9,
    ulong qkB_offset_h9,
    __global half * pvB_h9,
    ulong pvB_offset_h9,
    __global half * qkA_h10,
    ulong qkA_offset_h10,
    __global half * qkB_h10,
    ulong qkB_offset_h10,
    __global half * pvB_h10,
    ulong pvB_offset_h10,
    __global half * qkA_h11,
    ulong qkA_offset_h11,
    __global half * qkB_h11,
    ulong qkB_offset_h11,
    __global half * pvB_h11,
    ulong pvB_offset_h11,
    __global half * qkA_h12,
    ulong qkA_offset_h12,
    __global half * qkB_h12,
    ulong qkB_offset_h12,
    __global half * pvB_h12,
    ulong pvB_offset_h12,
    __global half * qkA_h13,
    ulong qkA_offset_h13,
    __global half * qkB_h13,
    ulong qkB_offset_h13,
    __global half * pvB_h13,
    ulong pvB_offset_h13,
    __global half * qkA_h14,
    ulong qkA_offset_h14,
    __global half * qkB_h14,
    ulong qkB_offset_h14,
    __global half * pvB_h14,
    ulong pvB_offset_h14,
    __global half * qkA_h15,
    ulong qkA_offset_h15,
    __global half * qkB_h15,
    ulong qkB_offset_h15,
    __global half * pvB_h15,
    ulong pvB_offset_h15)
{
    const size_t batched_index = get_global_id(0);
    const int head = (int)(batched_index / 67584u);
    const size_t i = batched_index % 67584u;
    __global half * qkA;
    ulong qkA_offset;
    __global half * qkB;
    ulong qkB_offset;
    __global half * pvB;
    ulong pvB_offset;
    switch (head) {
        case 0: qkA = qkA_h0; qkA_offset = qkA_offset_h0; qkB = qkB_h0; qkB_offset = qkB_offset_h0; pvB = pvB_h0; pvB_offset = pvB_offset_h0; break;
        case 1: qkA = qkA_h1; qkA_offset = qkA_offset_h1; qkB = qkB_h1; qkB_offset = qkB_offset_h1; pvB = pvB_h1; pvB_offset = pvB_offset_h1; break;
        case 2: qkA = qkA_h2; qkA_offset = qkA_offset_h2; qkB = qkB_h2; qkB_offset = qkB_offset_h2; pvB = pvB_h2; pvB_offset = pvB_offset_h2; break;
        case 3: qkA = qkA_h3; qkA_offset = qkA_offset_h3; qkB = qkB_h3; qkB_offset = qkB_offset_h3; pvB = pvB_h3; pvB_offset = pvB_offset_h3; break;
        case 4: qkA = qkA_h4; qkA_offset = qkA_offset_h4; qkB = qkB_h4; qkB_offset = qkB_offset_h4; pvB = pvB_h4; pvB_offset = pvB_offset_h4; break;
        case 5: qkA = qkA_h5; qkA_offset = qkA_offset_h5; qkB = qkB_h5; qkB_offset = qkB_offset_h5; pvB = pvB_h5; pvB_offset = pvB_offset_h5; break;
        case 6: qkA = qkA_h6; qkA_offset = qkA_offset_h6; qkB = qkB_h6; qkB_offset = qkB_offset_h6; pvB = pvB_h6; pvB_offset = pvB_offset_h6; break;
        case 7: qkA = qkA_h7; qkA_offset = qkA_offset_h7; qkB = qkB_h7; qkB_offset = qkB_offset_h7; pvB = pvB_h7; pvB_offset = pvB_offset_h7; break;
        case 8: qkA = qkA_h8; qkA_offset = qkA_offset_h8; qkB = qkB_h8; qkB_offset = qkB_offset_h8; pvB = pvB_h8; pvB_offset = pvB_offset_h8; break;
        case 9: qkA = qkA_h9; qkA_offset = qkA_offset_h9; qkB = qkB_h9; qkB_offset = qkB_offset_h9; pvB = pvB_h9; pvB_offset = pvB_offset_h9; break;
        case 10: qkA = qkA_h10; qkA_offset = qkA_offset_h10; qkB = qkB_h10; qkB_offset = qkB_offset_h10; pvB = pvB_h10; pvB_offset = pvB_offset_h10; break;
        case 11: qkA = qkA_h11; qkA_offset = qkA_offset_h11; qkB = qkB_h11; qkB_offset = qkB_offset_h11; pvB = pvB_h11; pvB_offset = pvB_offset_h11; break;
        case 12: qkA = qkA_h12; qkA_offset = qkA_offset_h12; qkB = qkB_h12; qkB_offset = qkB_offset_h12; pvB = pvB_h12; pvB_offset = pvB_offset_h12; break;
        case 13: qkA = qkA_h13; qkA_offset = qkA_offset_h13; qkB = qkB_h13; qkB_offset = qkB_offset_h13; pvB = pvB_h13; pvB_offset = pvB_offset_h13; break;
        case 14: qkA = qkA_h14; qkA_offset = qkA_offset_h14; qkB = qkB_h14; qkB_offset = qkB_offset_h14; pvB = pvB_h14; pvB_offset = pvB_offset_h14; break;
        case 15: qkA = qkA_h15; qkA_offset = qkA_offset_h15; qkB = qkB_h15; qkB_offset = qkB_offset_h15; pvB = pvB_h15; pvB_offset = pvB_offset_h15; break;
        default: return;
    }
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
        const float query = qkvC[qkvC_offset + dino0_c4(row, q_column)]
                            + qkvbias[q_column];
        const float key = qkvC[qkvC_offset + dino0_c4(row, k_column)]
                          + qkvbias[k_column];
        const float value = qkvC[qkvC_offset + dino0_c4(row, v_column)]
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

__attribute__((reqd_work_group_size(1024, 1, 1)))
__kernel void softmax_all_heads(
    __global const float * src_h0,
    ulong src_offset_elements_h0,
    __global half * dst_h0,
    ulong dst_offset_elements_h0,
    __global const float * src_h1,
    ulong src_offset_elements_h1,
    __global half * dst_h1,
    ulong dst_offset_elements_h1,
    __global const float * src_h2,
    ulong src_offset_elements_h2,
    __global half * dst_h2,
    ulong dst_offset_elements_h2,
    __global const float * src_h3,
    ulong src_offset_elements_h3,
    __global half * dst_h3,
    ulong dst_offset_elements_h3,
    __global const float * src_h4,
    ulong src_offset_elements_h4,
    __global half * dst_h4,
    ulong dst_offset_elements_h4,
    __global const float * src_h5,
    ulong src_offset_elements_h5,
    __global half * dst_h5,
    ulong dst_offset_elements_h5,
    __global const float * src_h6,
    ulong src_offset_elements_h6,
    __global half * dst_h6,
    ulong dst_offset_elements_h6,
    __global const float * src_h7,
    ulong src_offset_elements_h7,
    __global half * dst_h7,
    ulong dst_offset_elements_h7,
    __global const float * src_h8,
    ulong src_offset_elements_h8,
    __global half * dst_h8,
    ulong dst_offset_elements_h8,
    __global const float * src_h9,
    ulong src_offset_elements_h9,
    __global half * dst_h9,
    ulong dst_offset_elements_h9,
    __global const float * src_h10,
    ulong src_offset_elements_h10,
    __global half * dst_h10,
    ulong dst_offset_elements_h10,
    __global const float * src_h11,
    ulong src_offset_elements_h11,
    __global half * dst_h11,
    ulong dst_offset_elements_h11,
    __global const float * src_h12,
    ulong src_offset_elements_h12,
    __global half * dst_h12,
    ulong dst_offset_elements_h12,
    __global const float * src_h13,
    ulong src_offset_elements_h13,
    __global half * dst_h13,
    ulong dst_offset_elements_h13,
    __global const float * src_h14,
    ulong src_offset_elements_h14,
    __global half * dst_h14,
    ulong dst_offset_elements_h14,
    __global const float * src_h15,
    ulong src_offset_elements_h15,
    __global half * dst_h15,
    ulong dst_offset_elements_h15,
    volatile __global unsigned int *error_bits)
{
    const uint batched_group = (uint)get_group_id(0);
    const uint query_slot = (uint)get_local_id(0) % 4u;
    const uint head = batched_group / 261u;
    const uint row_candidate = (batched_group % 261u) * 4u + query_slot;
    const uint row = min(row_candidate, ROWS - 1u);
    __global const float * src;
    ulong src_offset_elements;
    __global half * dst;
    ulong dst_offset_elements;
    switch (head) {
        case 0: src = src_h0; src_offset_elements = src_offset_elements_h0; dst = dst_h0; dst_offset_elements = dst_offset_elements_h0; break;
        case 1: src = src_h1; src_offset_elements = src_offset_elements_h1; dst = dst_h1; dst_offset_elements = dst_offset_elements_h1; break;
        case 2: src = src_h2; src_offset_elements = src_offset_elements_h2; dst = dst_h2; dst_offset_elements = dst_offset_elements_h2; break;
        case 3: src = src_h3; src_offset_elements = src_offset_elements_h3; dst = dst_h3; dst_offset_elements = dst_offset_elements_h3; break;
        case 4: src = src_h4; src_offset_elements = src_offset_elements_h4; dst = dst_h4; dst_offset_elements = dst_offset_elements_h4; break;
        case 5: src = src_h5; src_offset_elements = src_offset_elements_h5; dst = dst_h5; dst_offset_elements = dst_offset_elements_h5; break;
        case 6: src = src_h6; src_offset_elements = src_offset_elements_h6; dst = dst_h6; dst_offset_elements = dst_offset_elements_h6; break;
        case 7: src = src_h7; src_offset_elements = src_offset_elements_h7; dst = dst_h7; dst_offset_elements = dst_offset_elements_h7; break;
        case 8: src = src_h8; src_offset_elements = src_offset_elements_h8; dst = dst_h8; dst_offset_elements = dst_offset_elements_h8; break;
        case 9: src = src_h9; src_offset_elements = src_offset_elements_h9; dst = dst_h9; dst_offset_elements = dst_offset_elements_h9; break;
        case 10: src = src_h10; src_offset_elements = src_offset_elements_h10; dst = dst_h10; dst_offset_elements = dst_offset_elements_h10; break;
        case 11: src = src_h11; src_offset_elements = src_offset_elements_h11; dst = dst_h11; dst_offset_elements = dst_offset_elements_h11; break;
        case 12: src = src_h12; src_offset_elements = src_offset_elements_h12; dst = dst_h12; dst_offset_elements = dst_offset_elements_h12; break;
        case 13: src = src_h13; src_offset_elements = src_offset_elements_h13; dst = dst_h13; dst_offset_elements = dst_offset_elements_h13; break;
        case 14: src = src_h14; src_offset_elements = src_offset_elements_h14; dst = dst_h14; dst_offset_elements = dst_offset_elements_h14; break;
        case 15: src = src_h15; src_offset_elements = src_offset_elements_h15; dst = dst_h15; dst_offset_elements = dst_offset_elements_h15; break;
        default: return;
    }
    const uint lane = (uint)get_local_id(0) / 4u;
    __local float reduction[WG_SIZE * 4u];
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
    reduction[lane * 4u + query_slot] = lane_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint step = WG_SIZE / 2u; step > 0u; step /= 2u) {
        if (lane < step)
            reduction[lane * 4u + query_slot] = fmax(reduction[lane * 4u + query_slot], reduction[(lane + step) * 4u + query_slot]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float row_max = reduction[query_slot];
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
    reduction[lane * 4u + query_slot] = lane_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint step = WG_SIZE / 2u; step > 0u; step /= 2u) {
        if (lane < step)
            reduction[lane * 4u + query_slot] += reduction[(lane + step) * 4u + query_slot];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float denominator = reduction[query_slot];
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
            if (row_candidate < ROWS)
                dst[dst_offset_elements + a8] = convert_half_rte(probability);
        }
    }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void merge_all_heads(
    __global const float * pvC_h0,
    ulong pvC_offset_h0,
    __global const float * pvC_h1,
    ulong pvC_offset_h1,
    __global const float * pvC_h2,
    ulong pvC_offset_h2,
    __global const float * pvC_h3,
    ulong pvC_offset_h3,
    __global const float * pvC_h4,
    ulong pvC_offset_h4,
    __global const float * pvC_h5,
    ulong pvC_offset_h5,
    __global const float * pvC_h6,
    ulong pvC_offset_h6,
    __global const float * pvC_h7,
    ulong pvC_offset_h7,
    __global const float * pvC_h8,
    ulong pvC_offset_h8,
    __global const float * pvC_h9,
    ulong pvC_offset_h9,
    __global const float * pvC_h10,
    ulong pvC_offset_h10,
    __global const float * pvC_h11,
    ulong pvC_offset_h11,
    __global const float * pvC_h12,
    ulong pvC_offset_h12,
    __global const float * pvC_h13,
    ulong pvC_offset_h13,
    __global const float * pvC_h14,
    ulong pvC_offset_h14,
    __global const float * pvC_h15,
    ulong pvC_offset_h15,
    __global half *projA,
    const ulong projA_offset)
{
    const size_t batched_index = get_global_id(0);
    const int head = (int)(batched_index / 66816u);
    const size_t i = batched_index % 66816u;
    __global const float * pvC;
    ulong pvC_offset;
    switch (head) {
        case 0: pvC = pvC_h0; pvC_offset = pvC_offset_h0; break;
        case 1: pvC = pvC_h1; pvC_offset = pvC_offset_h1; break;
        case 2: pvC = pvC_h2; pvC_offset = pvC_offset_h2; break;
        case 3: pvC = pvC_h3; pvC_offset = pvC_offset_h3; break;
        case 4: pvC = pvC_h4; pvC_offset = pvC_offset_h4; break;
        case 5: pvC = pvC_h5; pvC_offset = pvC_offset_h5; break;
        case 6: pvC = pvC_h6; pvC_offset = pvC_offset_h6; break;
        case 7: pvC = pvC_h7; pvC_offset = pvC_offset_h7; break;
        case 8: pvC = pvC_h8; pvC_offset = pvC_offset_h8; break;
        case 9: pvC = pvC_h9; pvC_offset = pvC_offset_h9; break;
        case 10: pvC = pvC_h10; pvC_offset = pvC_offset_h10; break;
        case 11: pvC = pvC_h11; pvC_offset = pvC_offset_h11; break;
        case 12: pvC = pvC_h12; pvC_offset = pvC_offset_h12; break;
        case 13: pvC = pvC_h13; pvC_offset = pvC_offset_h13; break;
        case 14: pvC = pvC_h14; pvC_offset = pvC_offset_h14; break;
        case 15: pvC = pvC_h15; pvC_offset = pvC_offset_h15; break;
        default: return;
    }
    if (i >= (size_t)DINO0_ROWS * DINO0_HEAD_DIM
            || head < 0 || head >= (int)DINO0_HEADS) return;
    // Traverse native PV C4; place this head in original [row,head,channel].
    const uint row = (uint)((i / 4u) % DINO0_ROWS);
    const uint channel = (uint)((i / 4u / DINO0_ROWS) * 4u + i % 4u);
    const uint column = (uint)head * DINO0_HEAD_DIM + channel;
    projA[projA_offset + dino0_a8(row, column)] = convert_half_rte(pvC[pvC_offset + i]);
}
