#!/usr/bin/env python3
"""Static batch-head source generation; no model/OpenCL/RKNN execution."""
import argparse
import hashlib
import json
from pathlib import Path
import re

DINO_SHA = 'da7e8d3c15bd4f67732b6d4a727f065eef0349635ad7b81770b08ecf9719dc47'
SOFTMAX_SHA = '5cbb36e03e09c38db9e6769f067e3168e1cf7f67d719009601c4456e9f2b150a'
FROZEN = Path('/path/to/research-artifacts/v121-native518-native-block-r1')
HEADS, ROWS, PAD, HEAD_DIM, WG = 16, 1041, 1056, 64, 256


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def pinned(path, expected):
    path = path.resolve(strict=True)
    data = path.read_bytes()
    require(digest(data) == expected, 'Frozen original source differs: ' + str(path))
    return path, data, data.decode('utf-8')


def body(source, name):
    marker = '__kernel void ' + name + '('
    require(source.count(marker) == 1, 'Exactly one original kernel required: ' + name)
    start = source.index('{', source.index(marker))
    depth = 0
    for end in range(start, len(source)):
        depth += (source[end] == '{') - (source[end] == '}')
        if depth == 0:
            return source[start:end + 1]
    raise ValueError('Original kernel braces incomplete')


def choose(fields):
    declarations = ''.join('    ' + kind + ' ' + name + ';\n' for kind, name in fields)
    cases = []
    for head in range(HEADS):
        assignments = ' '.join(name + ' = ' + name + '_h' + str(head) + ';' for _, name in fields)
        cases.append('        case ' + str(head) + ': ' + assignments + ' break;')
    return declarations + '    switch (head) {\n' + '\n'.join(cases) + '\n        default: return;\n    }'


def generate(dino, softmax):
    kernels, proofs = [], []
    pack_fields = [('__global half *', 'qkA'), ('ulong', 'qkA_offset'),
                   ('__global half *', 'qkB'), ('ulong', 'qkB_offset'),
                   ('__global half *', 'pvB'), ('ulong', 'pvB_offset')]
    soft_fields = [('__global const float *', 'src'), ('ulong', 'src_offset_elements'),
                   ('__global half *', 'dst'), ('ulong', 'dst_offset_elements')]
    merge_fields = [('__global const float *', 'pvC'), ('ulong', 'pvC_offset')]
    specs = [
        ('qkv_head_pack', 'pack_all_heads', dino,
         ['__global const float *qkvC', 'const ulong qkvC_offset', '__global const float *qkvbias'],
         pack_fields, [],
         '    const size_t i = get_global_id(0);',
         '    const size_t batched_index = get_global_id(0);\n'
         '    const int head = (int)(batched_index / 67584u);\n'
         '    const size_t i = batched_index % 67584u;\n',
         67584, 67584, 99, 792),
        ('original_softmax_c4_to_half_a8', 'softmax_all_heads', softmax,
         [], soft_fields, ['volatile __global unsigned int *error_bits'],
         '    const uint row = (uint)get_group_id(0);',
         '    const uint batched_group = (uint)get_group_id(0);\n'
         '    const uint head = batched_group / ROWS;\n'
         '    const uint row = batched_group % ROWS;\n',
         ROWS * WG, ROWS * WG, 65, 520),
        ('pv_head_merge', 'merge_all_heads', dino,
         [], merge_fields, ['__global half *projA', 'const ulong projA_offset'],
         '    const size_t i = get_global_id(0);',
         '    const size_t batched_index = get_global_id(0);\n'
         '    const int head = (int)(batched_index / 66816u);\n'
         '    const size_t i = batched_index % 66816u;\n',
         66816, ROWS * HEAD_DIM, 34, 272)]
    for old, name, source, first, fields, last, anchor, indexing, span, live, argc, argument_bytes in specs:
        original_body = body(source, old)
        require(original_body.count(anchor) == 1, 'Exact original indexing anchor required')
        selection = indexing + choose(fields)
        changed_body = original_body.replace(anchor, selection, 1)
        recovered = changed_body.replace(selection, anchor, 1)
        require(recovered == original_body, 'Original complete arithmetic body inverse differs')
        parameters = first + [kind + ' ' + field + '_h' + str(head)
                              for head in range(HEADS) for kind, field in fields] + last
        require(len(parameters) == argc and 8 * argc == argument_bytes, '64-bit ABI argument budget differs')
        kernels.append('__attribute__((reqd_work_group_size(256, 1, 1)))\n'
                       '__kernel void ' + name + '(\n    ' + ',\n    '.join(parameters) + ')\n' + changed_body)
        for base in range(0, HEADS * span, WG):
            require(base // span == (base + WG - 1) // span, 'Head selection diverges within workgroup')
        require(live <= span and span % WG == 0, 'Invalid original per-head coverage')
        for head in range(HEADS):
            require(head * span // span == head and (head * span + live - 1) % span == live - 1,
                    'Per-head original live range changed')
        proofs.append(dict(kernel=name, original_kernel=old, arguments=parameters,
                           argument_count=argc, argument_bytes_64_bit=argument_bytes,
                           global_size=[HEADS * span], local_size=[WG],
                           head_span=span, live_work_items_per_head=live,
                           guarded_tail_per_head=span-live,
                           original_body_sha256=digest(original_body.encode()),
                           batched_body_sha256=digest(changed_body.encode()),
                           inverse_original_body_sha256=digest(recovered.encode()),
                           inverse_body_byte_equal=True, unchanged_original_arithmetic_body=True,
                           every_workgroup_has_one_uniform_head=True,
                           every_original_live_index_per_head_retained=True))
    prefix = ('#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n'
              '#pragma OPENCL FP_CONTRACT OFF\n\n'
              '// Append after the frozen original DINO0 and R119 softmax sources.\n'
              '// Original arithmetic bodies are copied verbatim; only per-head\n'
              '// indexing and uniform fixed-pointer selection are substituted.\n'
              '// Fixed 16 heads, all1041 keys,15 storage-padding keys. No slab or\n'
              '// host/GPU pointer table. Build only with -cl-std=CL1.2.\n\n')
    return (prefix + '\n\n'.join(kernels) + '\n').encode(), proofs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-dino-cl', type=Path, default=FROZEN/'dino0_native_bridge.cl')
    parser.add_argument('--original-softmax-cl', type=Path, default=FROZEN/'attention_softmax.cl')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    dino_path, dino_bytes, dino = pinned(args.original_dino_cl, DINO_SHA)
    soft_path, soft_bytes, softmax = pinned(args.original_softmax_cl, SOFTMAX_SHA)
    data, kernels = generate(dino, softmax)
    require(re.findall(rb'__kernel\s+void\s+(\w+)\s*\(', data) ==
            [b'pack_all_heads', b'softmax_all_heads', b'merge_all_heads'], 'Exactly3 added kernels required')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/'dino0_batched_bridge.cl'
    with path.open('xb') as stream:
        stream.write(data)
    report = dict(schema='v122_original_DINO0_batched_kernel_static_generation_proof',
                  status='generated_original_math_batch_kernels_not_OpenCL_or_model_validated',
                  generator_sha256=digest(Path(__file__).read_bytes()),
                  original_sources=[dict(file=str(dino_path),bytes=len(dino_bytes),sha256=DINO_SHA),
                                    dict(file=str(soft_path),bytes=len(soft_bytes),sha256=SOFTMAX_SHA)],
                  generated_kernel=dict(file=path.name,bytes=len(data),sha256=digest(data)),
                  required_combined_source_order=['original_DINO0','original_R119_softmax','batched_append'],
                  combined_source_separator='one newline byte between the three unchanged source blobs',
                  required_combined_source_sha256=digest(dino_bytes+b'\n'+soft_bytes+b'\n'+data),
                  original_source_bytes_unchanged=True, original_math_bodies_inverse_byte_equal=True,
                  source_change_scope='Only added ABI, uniform fixed-pointer selection, head-local index mapping and explicit256-workgroup attribute. Original arithmetic statements, barriers, reductions, scale, tail checks and stores byte-identical after inverse.',
                  head_count=HEADS, actual_keys=ROWS, padded_keys=PAD,
                  FP_CONTRACT_OFF=True, build_options='-cl-std=CL1.2', kernels=kernels,
                  R121_native_B_output_raw_sha256_required='70de59362e3bb03eb875bc3ee9be56a7af538b4098e662d9fd6b817d5c3c3e0d',
                  unchanged_original_FP32_gate=dict(relative_l2_max=1e-5,max_abs_max=1e-3,rtol=1e-4,atol=1e-5),
                  OpenCL_or_RKNN_or_model_execution=False, actual_numerical_or_performance_acceptance=False)
    with (args.output_dir/'batched-kernel-static-proof.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(kernel=report['generated_kernel'],inverse_proof_passed=True,
                          OpenCL_or_RKNN_or_model_execution=False)))


if __name__ == '__main__':
    main()
