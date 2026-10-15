#!/usr/bin/env python3
"""Static four-query softmax layout rewrite; no model/OpenCL/RKNN calls."""
import argparse
import hashlib
import json
from pathlib import Path

PARENT_SHA = '34a216960e61f7a895e92341cf93f4332cc3db0c5db16740fe8a3a586aebd822'
PARENT_PROOF_SHA = '925f1d4011af5707a487a3830deb979714531d023c3b7eddfae55f5c0d109f0d'
DINO_SHA = 'da7e8d3c15bd4f67732b6d4a727f065eef0349635ad7b81770b08ecf9719dc47'
SOFTMAX_SHA = '5cbb36e03e09c38db9e6769f067e3168e1cf7f67d719009601c4456e9f2b150a'
EXPECTED_OUTPUT_RAW = '70de59362e3bb03eb875bc3ee9be56a7af538b4098e662d9fd6b817d5c3c3e0d'
ROWS, LANES, SLOTS, GROUPS_PER_HEAD, HEADS = 1041, 256, 4, 261, 16


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pin(path, expected):
    path = path.resolve(strict=True)
    data = path.read_bytes()
    require(sha(data) == expected, 'Pinned parent differs: ' + str(path))
    return path, data


def softmax_region(source):
    marker = '__attribute__((reqd_work_group_size(256, 1, 1)))\n__kernel void softmax_all_heads('
    require(source.count(marker) == 1, 'Exactly one parent softmax kernel required')
    start = source.index(marker)
    opening = source.index('{', start)
    depth = 0
    for end in range(opening, len(source)):
        depth += (source[end] == '{') - (source[end] == '}')
        if depth == 0:
            return start, end + 1
    raise ValueError('Incomplete original softmax body')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parent = Path(__file__).resolve().parent.parent
    parser.add_argument('--parent-batched-cl', type=Path, default=parent/'dino0_batched_bridge.cl')
    parser.add_argument('--parent-proof', type=Path, default=parent/'batched-kernel-static-proof.json')
    parser.add_argument('--original-dino-cl', type=Path, default=Path(__file__).with_name('dino0_native_bridge.cl'))
    parser.add_argument('--original-softmax-cl', type=Path, default=Path(__file__).with_name('attention_softmax.cl'))
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    parent_path, raw = pin(args.parent_batched_cl, PARENT_SHA)
    proof_path, proof_bytes = pin(args.parent_proof, PARENT_PROOF_SHA)
    original_proof = json.loads(proof_bytes)
    require(original_proof['original_math_bodies_inverse_byte_equal'] is True and
            original_proof['generated_kernel']['sha256'] == PARENT_SHA, 'Original scalar-body inverse proof missing')
    dino_path, dino = pin(args.original_dino_cl, DINO_SHA)
    soft_path, soft = pin(args.original_softmax_cl, SOFTMAX_SHA)
    source = raw.decode('utf-8')
    start, end = softmax_region(source)
    original = source[start:end]
    changed = original
    replacements = [
        ('__attribute__((reqd_work_group_size(256, 1, 1)))',
         '__attribute__((reqd_work_group_size(1024, 1, 1)))', 1),
        ('    const uint batched_group = (uint)get_group_id(0);\n'
         '    const uint head = batched_group / ROWS;\n'
         '    const uint row = batched_group % ROWS;',
         '    const uint batched_group = (uint)get_group_id(0);\n'
         '    const uint query_slot = (uint)get_local_id(0) % 4u;\n'
         '    const uint head = batched_group / 261u;\n'
         '    const uint row_candidate = (batched_group % 261u) * 4u + query_slot;\n'
         '    const uint row = min(row_candidate, ROWS - 1u);', 1),
        ('    const uint lane = (uint)get_local_id(0);',
         '    const uint lane = (uint)get_local_id(0) / 4u;', 1),
        ('    __local float reduction[WG_SIZE];',
         '    __local float reduction[WG_SIZE * 4u];', 1),
        ('reduction[lane + step]', 'reduction[(lane + step) * 4u + query_slot]', 2),
        ('reduction[lane]', 'reduction[lane * 4u + query_slot]', 5),
        ('reduction[0]', 'reduction[query_slot]', 2),
        ('            dst[dst_offset_elements + a8] = convert_half_rte(probability);',
         '            if (row_candidate < ROWS)\n'
         '                dst[dst_offset_elements + a8] = convert_half_rte(probability);', 1)]
    for old, new, count in replacements:
        require(changed.count(old) == count, 'Original source anchor count differs: ' + old)
        changed = changed.replace(old, new)
    inverse = changed
    for old, new, count in reversed(replacements):
        require(inverse.count(new) == count, 'Inverse source anchor count differs: ' + new)
        inverse = inverse.replace(new, old)
    require(inverse == original, 'Four-row softmax inverse changed original scalar arithmetic')
    data = (source[:start] + changed + source[end:]).encode('utf-8')
    require(data[:start] == raw[:start] and data[start+len(changed.encode()):] == raw[end:],
            'Any source outside the softmax kernel changed')
    require((source[:start] + inverse + source[end:]).encode() == raw, 'Whole-parent inverse differs')

    seen, inactive = [], 0
    for group in range(HEADS * GROUPS_PER_HEAD):
        head = group // GROUPS_PER_HEAD
        for slot in range(SLOTS):
            candidate = group % GROUPS_PER_HEAD * SLOTS + slot
            row = min(candidate, ROWS-1)
            require(0 <= head < HEADS and 0 <= row < ROWS, 'Invalid clamped source query')
            if candidate < ROWS:
                seen.append((head, candidate))
            else:
                inactive += 1
                require(row == ROWS-1, 'Inactive query must still execute valid source/barriers')
    require(seen == [(head,row) for head in range(HEADS) for row in range(ROWS)] and
            inactive == HEADS*3, 'Original active query/store coverage changed')
    cells = [lane*SLOTS+slot for lane in range(LANES) for slot in range(SLOTS)]
    require(cells == list(range(LANES*SLOTS)), 'Reduction cells collide or omit a lane')
    for step in (128,64,32,16,8,4,2,1):
        for lane in range(step):
            for slot in range(SLOTS):
                read = (lane+step)*SLOTS+slot
                require(read % SLOTS == slot and read // SLOTS == lane+step,
                        'Reduction crosses queries or changes original lane tree')
    keys = sorted(lane+item*LANES for lane in range(LANES) for item in range(5) if lane+item*LANES < ROWS)
    require(keys == list(range(ROWS)), 'Original all1041 keys changed')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir/'dino0_batched_bridge.cl'
    require(output.resolve() != parent_path, 'Cannot overwrite original parent kernel')
    if output.exists():
        require(output.read_bytes() == raw, 'R2 destination must initially be the unmodified parent copy')
        output.write_bytes(data)
    else:
        with output.open('xb') as stream:
            stream.write(data)
    report = dict(schema='v122_r2_original_DINO0_four_query_softmax_static_proof',
                  status='generated_original_math_layout_variant_not_OpenCL_or_model_validated',
                  generator_sha256=sha(Path(__file__).read_bytes()),
                  parent_kernel=dict(file=str(parent_path),bytes=len(raw),sha256=PARENT_SHA),
                  parent_static_proof=dict(file=str(proof_path),sha256=PARENT_PROOF_SHA),
                  generated_kernel=dict(file=output.name,bytes=len(data),sha256=sha(data)),
                  original_source_pins=[dict(file=str(dino_path),sha256=DINO_SHA),dict(file=str(soft_path),sha256=SOFTMAX_SHA)],
                  required_combined_source_sha256=sha(dino+b'\n'+soft+b'\n'+data),
                  pack_and_merge_kernels_and_all_outside_softmax_source_byte_unchanged=True,
                  whole_parent_inverse_byte_equal=True, original_softmax_scalar_arithmetic_inverse_byte_equal=True,
                  source_replacements=[dict(old=old,new=new,count=count) for old,new,count in replacements],
                  parameter_count_unchanged=65,parameter_bytes_64_bit_unchanged=520,
                  global_size=[HEADS*GROUPS_PER_HEAD*LANES*SLOTS],local_size=[LANES*SLOTS],
                  query_slots_per_workgroup=SLOTS,original_lanes_per_query=LANES,
                  original_items_per_lane=5,actual_keys=ROWS,padded_keys=1056,
                  reduction_layout='Interleaved local cells[lane*4+slot], unchanged per-query128..1 reduction tree',
                  original_active_queries_once_per_head=True,inactive_clamped_query_slots=inactive,
                  inactive_queries_execute_all_barriers_and_never_store=True,
                  R121_native_B_output_raw_sha256_required=EXPECTED_OUTPUT_RAW,
                  unchanged_original_FP32_gate=dict(relative_l2_max=1e-5,max_abs_max=1e-3,rtol=1e-4,atol=1e-5),
                  FP_CONTRACT_OFF=True,build_options='-cl-std=CL1.2',
                  OpenCL_or_RKNN_or_model_execution=False,actual_precision_or_performance_accepted=False)
    proof=args.output_dir/'four-row-kernel-static-proof.json'
    with proof.open('x') as stream:
        json.dump(report,stream,indent=2)
        stream.write('\n')
    print(json.dumps(dict(kernel=report['generated_kernel'],whole_parent_inverse_byte_equal=True,
                          original_per_query_math_and_reduction_tree_retained=True,
                          global_size=report['global_size'],local_size=report['local_size'])))


if __name__ == '__main__':
    main()
