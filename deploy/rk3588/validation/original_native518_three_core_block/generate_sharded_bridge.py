#!/usr/bin/env python3
"""Generate the original DINO0 three-column-shard GPU bridge; static work only.

Only native C4 buffer addressing and pointer arguments change. All original
floating-point statements, barriers, launch geometry and safety paths remain.
The generated integer layout check is not an FP32/model equivalence test.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

PARENT_PINS = {
    'native_dino_bridge_opencl.py': 'd3eb7dca3f9370ae54f089ca63c00d0e1c7bbf69442bf99ad0c142a45e6607de',
    'dino0_native_bridge.cl': 'da7e8d3c15bd4f67732b6d4a727f065eef0349635ad7b81770b08ecf9719dc47',
    'dino0_batched_bridge.cl': '34a216960e61f7a895e92341cf93f4332cc3db0c5db16740fe8a3a586aebd822',
    'attention_softmax.cl': '5cbb36e03e09c38db9e6769f067e3168e1cf7f67d719009601c4456e9f2b150a',
}
SPLITS = {'qkv': (1024, 1024, 1024), 'proj': (352, 352, 320),
          'fc1': (1376, 1376, 1344), 'fc2': (352, 352, 320)}
ROWS = 1041


def sha(value):
    return hashlib.sha256(value).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def edit(text, before, after, changes):
    require(text.count(before) == 1, 'Expected exactly one source match: '+before[:90])
    changes.append((before, after))
    return text.replace(before, after)


def inverse(text, changes):
    for before, after in reversed(changes):
        require(text.count(after) == 1, 'Inverse substitution ambiguous: '+after[:90])
        text = text.replace(after, before)
    return text


def args(name, multiline=False):
    if multiline:
        return '\n'.join(f'    __global const float *{name}{i}C,\n    const ulong {name}{i}C_offset,' for i in range(3))
    return '\n'.join(f'    __global const float *{name}{i}C, const ulong {name}{i}C_offset,' for i in range(3))


def selector(first):
    second = first*2
    return f"""// Original full channels; only integer shard selection/local addressing.
inline float dino0_sharded_c4_{first}(
    const uint row, const uint column,
    __global const float *c0, const ulong offset0,
    __global const float *c1, const ulong offset1,
    __global const float *c2, const ulong offset2)
{{
    if (column < {first}u)
        return c0[offset0 + dino0_c4(row, column)];
    if (column < {second}u)
        return c1[offset1 + dino0_c4(row, column - {first}u)];
    return c2[offset2 + dino0_c4(row, column - {second}u)];
}}

"""


def read_shards(name, first):
    pairs = ', '.join(f'{name}{i}C, {name}{i}C_offset' for i in range(3))
    return f'dino0_sharded_c4_{first}(row, column, {pairs})'


def rewrite_core(parent):
    changes = []
    result = parent
    marker = '// Two-pass population variance.'
    helpers = selector(352)+selector(1376)
    result = edit(result, marker, helpers+marker, changes)
    result = edit(result, '    __global const float *qkvC, const ulong qkvC_offset,',
                  args('qkv'), changes)
    for lane, column, local in ((0, 'q_column', 'q_column'),
                                (1, 'k_column', 'k_column - DINO0_DIM'),
                                (2, 'v_column', 'v_column - 2u * DINO0_DIM')):
        result = edit(result,
            f'qkvC[qkvC_offset + dino0_c4(row, {column})]',
            f'qkv{lane}C[qkv{lane}C_offset + dino0_c4(row, {local})]', changes)
    for name, first in (('proj', 352), ('fc1', 1376), ('fc2', 352)):
        result = edit(result, f'    __global const float *{name}C, const ulong {name}C_offset,',
                      args(name), changes)
        result = edit(result, f'{name}C[{name}C_offset + dino0_c4(row, column)]',
                      read_shards(name, first), changes)
    require(inverse(result, changes) == parent, 'Whole core-source inverse differs')
    return result, changes


def rewrite_batch(parent):
    changes = []
    result = edit(parent, '    __global const float *qkvC,\n    const ulong qkvC_offset,',
                  args('qkv', multiline=True), changes)
    for lane, column, local in ((0, 'q_column', 'q_column'),
                                (1, 'k_column', 'k_column - DINO0_DIM'),
                                (2, 'v_column', 'v_column - 2u * DINO0_DIM')):
        result = edit(result,
            f'qkvC[qkvC_offset + dino0_c4(row, {column})]',
            f'qkv{lane}C[qkv{lane}C_offset + dino0_c4(row, {local})]', changes)
    require(inverse(result, changes) == parent, 'Whole batch-source inverse differs')
    # The whole original softmax and merge text remains unchanged.
    start = '__attribute__((reqd_work_group_size(256, 1, 1)))\n__kernel void softmax_all_heads('
    require(result[result.index(start):] == parent[parent.index(start):],
            'Softmax or merge source changed')
    return result, changes


def rewrite_bridge(parent, core_sha, batch_sha):
    changes = []
    result = edit(parent,
        '"""Board-only batched-head glue between original DINO0 native RKNN MatMuls.',
        '"""Board-only original DINO0 glue with three native output-column shards.', changes)
    result = edit(result,
        "DINO_KERNEL_SHA = '"+PARENT_PINS['dino0_native_bridge.cl']+"'",
        "DINO_KERNEL_SHA = '"+core_sha+"'\nBATCH_KERNEL_SHA = '"+batch_sha+"'", changes)
    old = """IMPORT_SPECS = {'qkvA': (2,ROWS*DIM,2), 'qkvC': (4,ROWS*3*DIM,4),
    'projA': (2,ROWS*DIM,2), 'projC': (4,ROWS*DIM,4),
    'fc1A': (2,ROWS*DIM,2), 'fc1C': (4,ROWS*4*DIM,4),
    'fc2A': (2,ROWS*4*DIM,2), 'fc2C': (4,ROWS*DIM,4)}"""
    new = """WEIGHTED_OUTPUT_SPLITS = {'qkv': (1024,1024,1024), 'proj': (352,352,320),
    'fc1': (1376,1376,1344), 'fc2': (352,352,320)}
IMPORT_SPECS = {'qkvA': (2,ROWS*DIM,2), 'projA': (2,ROWS*DIM,2),
    'fc1A': (2,ROWS*DIM,2), 'fc2A': (2,ROWS*4*DIM,2)}
for _weighted,_sizes in WEIGHTED_OUTPUT_SPLITS.items():
    for _part,_columns in enumerate(_sizes):
        IMPORT_SPECS[_weighted+str(_part)+'C'] = (4,ROWS*_columns,4)"""
    result = edit(result, old, new, changes)
    result = edit(result, '_v122_frozen_v81_opencl_abi', '_v124_frozen_v81_opencl_abi', changes)
    result = edit(result, "'Exactly104 native DINO buffer boundaries required'",
                  "'Exactly112 native DINO buffer boundaries required'", changes)
    result = edit(result,
        "            batched_all16_heads=True, GPU_kernels_per_complete_block=7,",
        "            batched_all16_heads=True, GPU_kernels_per_complete_block=7,\n"
        "            weighted_output_channel_splits=copy.deepcopy(WEIGHTED_OUTPUT_SPLITS),\n"
        "            shared_weighted_A_buffers=4, independent_weighted_C_buffers=12,\n"
        "            DMA_import_count=112, weighted_C_gather_buffers_or_kernels=False,", changes)
    result = edit(result,
        "            # The root bundle pins the new batched source separately. Compile\n"
        "            # its exact bytes after the two unchanged frozen source files.\n"
        "            batch = Path(batch_source_path).resolve(strict=True)",
        "            # Pin the generated shard-addressing source and unchanged softmax.\n"
        "            batch = _pin(batch_source_path, BATCH_KERNEL_SHA)", changes)
    result = edit(result,
        "            _require(parameter_limit.value >= 792, 'Device cannot accept the792-byte complete head-pack argument list')",
        "            _require(parameter_limit.value >= 824, 'Device cannot accept the824-byte complete head-pack argument list')", changes)
    result = edit(result, "            packed_args = imported('qkvC')+internal('attn.qkv.bias')",
        "            def weighted_c(name):\n"
        "                return [arg for part in range(3) for arg in imported(name+str(part)+'C')]\n"
        "            packed_args = weighted_c('qkv')+internal('attn.qkv.bias')", changes)
    for name in ('proj', 'fc1', 'fc2'):
        result = edit(result, f"imported('{name}C')", f"weighted_c('{name}')", changes)
    result = edit(result, "{'pack_all':792,'softmax_all':520,'merge_all':272}",
                  "{'pack_all':824,'softmax_all':520,'merge_all':272}", changes)
    # Strict arity and byte guards also cover the changed weighted consumers.
    result = edit(result,
        "                if method in ('pack_all','softmax_all','merge_all'):\n"
        "                    _require(argument_bytes == {'pack_all':824,'softmax_all':520,'merge_all':272}[method],\n"
        "                        'Exact batched parameter ABI byte count differs: ' + name)",
        "                expected_args = {'norm1':5,'pack_all':103,'softmax_all':65,'merge_all':34,\n"
        "                    'proj_norm':14,'gelu':9,'final':10}\n"
        "                expected_bytes = {'norm1':40,'pack_all':824,'softmax_all':520,'merge_all':272,\n"
        "                    'proj_norm':112,'gelu':72,'final':80}\n"
        "                _require(argc.value == expected_args[method] and argument_bytes == expected_bytes[method],\n"
        "                    'Exact sharded parameter ABI differs: ' + name)", changes)
    require(inverse(result, changes) == parent, 'Whole bridge-source inverse differs')
    ast.parse(result)
    return result, changes


def integer_layout_check():
    import numpy as np
    checks = {}
    # Unique generated uint32 indices stand for native C4 addresses, not weights.
    for name, sizes in SPLITS.items():
        width = sum(sizes)
        source = np.arange(width*ROWS, dtype=np.uint32).reshape(width//4,ROWS,4)
        offsets = (3,11,19)
        outputs = []
        start = 0
        for size, offset in zip(sizes, offsets):
            require(size % 32 == 0 and start % 4 == 0, 'Native channel alignment differs')
            local = np.empty(offset+size*ROWS, dtype=np.uint32)
            local[:offset] = np.uint32(0xffffffff)
            local[offset:] = source[start//4:(start+size)//4].reshape(-1)
            outputs.append(local); start += size
        start = 0
        for part, (size, offset) in enumerate(zip(sizes, offsets)):
            # Check every channel and every real row, including shard boundaries,
            # using the exact selector's local-channel C4 addressing.
            rows = np.arange(ROWS, dtype=np.int64)
            for local_column in range(size):
                column = start+local_column
                selected = 0 if column < sizes[0] else (1 if column < sizes[0]+sizes[1] else 2)
                local_channel = column-sum(sizes[:selected])
                addresses = offsets[selected]+((local_channel//4)*ROWS+rows)*4+local_channel%4
                actual = outputs[selected][addresses]
                expected = source[column//4,:,column%4]
                require(selected == part and np.array_equal(actual,expected),
                        'Generated sharded C4 read differs')
            start += size
        require(start == width and np.array_equal(np.concatenate(
            [v[o:] for v,o in zip(outputs,offsets)]),source.reshape(-1)),
            'Native C4 partition/inverse differs')
        checks[name] = dict(channel_splits=list(sizes),all_rows=ROWS,all_channels=width,
            elements_checked=ROWS*width,nonzero_import_offsets=list(offsets),
            complete_native_C4_inverse_passed=True,every_selected_read_passed=True)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args_ = parser.parse_args()
    original = {}
    for name, digest in PARENT_PINS.items():
        value = (args_.parent_dir/name).read_bytes()
        require(sha(value) == digest, 'Frozen R122 R1 source changed: '+name)
        original[name] = value.decode()
    core, core_changes = rewrite_core(original['dino0_native_bridge.cl'])
    batch, batch_changes = rewrite_batch(original['dino0_batched_bridge.cl'])
    bridge, bridge_changes = rewrite_bridge(original['native_dino_bridge_opencl.py'],
                                            sha(core.encode()),sha(batch.encode()))
    outputs = {'dino0_native_bridge.cl':core,'dino0_batched_bridge.cl':batch,
               'native_dino_bridge_opencl.py':bridge,'attention_softmax.cl':original['attention_softmax.cl']}
    proof = dict(schema='v124_original_DINO0_sharded_C4_static_source_layout_check',
        status='passed_static_indices_and_inverse_source_only',
        generator_sha256=sha(Path(__file__).read_bytes()),parent_source_sha256=PARENT_PINS,
        only_buffer_selection_integer_addressing_and_argument_lists_changed=True,
        original_floating_point_statements_barriers_and_launches_preserved=True,
        core_whole_source_inverse_byte_equal=True,batch_whole_source_inverse_byte_equal=True,
        bridge_whole_source_inverse_byte_equal=True,softmax_and_merge_whole_source_unchanged=True,
        substitutions={'core':len(core_changes),'batch':len(batch_changes),'bridge':len(bridge_changes)},
        generated_integer_layout=integer_layout_check(),DMA_imports=112,
        weighted_C_gather_buffers_or_kernels=False,
        expected_active_kernel_arguments={'norm1':5,'pack_all':103,'softmax_all':65,'merge_all':34,
                                         'proj_norm':14,'gelu':9,'final':10},
        expected_active_kernel_parameter_bytes={'norm1':40,'pack_all':824,'softmax_all':520,'merge_all':272,
                                               'proj_norm':112,'gelu':72,'final':80},
        FP32_equivalence_or_hardware_precision_proven=False,
        model_Torch_ONNX_RKNN_OpenCL_or_board_executed=False,
        output_sources={name:dict(bytes=len(value.encode()),sha256=sha(value.encode()))
                        for name,value in outputs.items()})
    args_.output_dir.mkdir(parents=True,exist_ok=True)
    for name,value in outputs.items():
        (args_.output_dir/name).write_text(value)
    (args_.output_dir/'static-layout-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:proof[k] for k in ('status','generator_sha256','output_sources')}))


if __name__ == '__main__':
    raise SystemExit(main())
