#!/usr/bin/env python3
"""Saved-only original CPU/NPU retained-token discrepancy on the same pair.

No model/operator inference. These cumulative discrepancies do not isolate
the error introduced by a block, nor attribute full trajectory degradation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/dev/shm/edge-vggt-v113-camera-precision')
PRIVATE = Path('/path/to/research-artifacts')
CPU = PRIVATE / 'v99-native518-components-review-r1/head-reference.json'
NPU = PRIVATE / 'v100-native518-integration-r1/full-model-sequential-core1-r1/result.json'
PINS = {
    CPU: '969ab6f5611efa62d81b5d62625a15815248668e1e079db17ac4d9ed9c96932b',
    NPU: '4af7b9c4cbd38dbfd54c14b1308dd6035da25941f98f858a5ad13b781c3c8af6',
    ROOT / 'board-r1/result.json': '22869179086da3473d2b945b61a452791f53c50a6e83545dfc5a3ecaba31c914',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(4 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def stats(actual, reference):
    a, r = actual.astype(np.float64), reference.astype(np.float64)
    delta = a - r
    denominator = float(np.linalg.norm(r.ravel()))
    require(denominator > 0, 'Zero reference norm')
    return {
        'elements': int(a.size),
        'reference_l2_norm': denominator,
        'difference_l2_norm': float(np.linalg.norm(delta.ravel())),
        'relative_l2': float(np.linalg.norm(delta.ravel()) / denominator),
        'max_abs': float(np.max(np.abs(delta))),
        'strict_outlier_count': int(np.count_nonzero(np.abs(delta) > 1e-5 + 1e-4 * np.abs(r))),
    }


def main():
    watched, ledger = {}, []
    for path, digest in PINS.items():
        require(sha(path) == digest, 'Pinned receipt differs: ' + str(path))
        watched[path] = digest
    cpu, npu = json.loads(CPU.read_text()), json.loads(NPU.read_text())
    diagnostic = json.loads((ROOT / 'board-r1/result.json').read_text())
    require(cpu['source_images'] == npu['source_images'] == diagnostic['source_images'], 'Source images differ')
    require(cpu['retained_layer_indices'] == [4, 11, 17, 23], 'Retained layer order differs')
    require(cpu['completed'] and cpu['all_requested_final_outputs_bitwise_equal_v92'] and
            npu['complete_expected_execution'] and npu['resources_balanced'], 'Incomplete source execution')

    def load(directory, record):
        path = directory / Path(record['file']).name
        require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], 'Array file identity differs: ' + str(path))
        value = np.load(path, allow_pickle=False)
        require(value.dtype == np.float32 and list(value.shape) == record['shape'] and np.isfinite(value).all(), 'Array contract differs')
        require(hashlib.sha256(value.tobytes()).hexdigest() == record['raw_sha256'], 'Array payload differs')
        watched[path] = record['sha256']
        ledger.append({'file': str(path), 'sha256': record['sha256'], 'raw_sha256': record['raw_sha256'], 'shape': list(value.shape)})
        return value

    rows = []
    for group, layer in enumerate((4, 11, 17, 23)):
        reference = load(ROOT / 'upstream-reference', cpu['artifacts'][f'tokens_{layer}'])
        frame = load(NPU.parent, npu['calls'][0]['actual_raw_group_outputs'][group][0])
        global_tokens = load(NPU.parent, npu['calls'][0]['actual_raw_group_outputs'][group][1])
        require(frame.shape == (2, 1041, 1024) and global_tokens.shape == (1, 2082, 1024), 'Raw group layout differs')
        actual = np.concatenate((frame.reshape(1, 2, 1041, 1024), global_tokens.reshape(1, 2, 1041, 1024)), axis=-1)
        require(actual.shape == reference.shape == (1, 2, 1041, 2048), 'Retained-token layout differs')
        row = {'retained_layer': layer, 'group': group, 'comparisons': {}}
        for name, tokens in (('all_tokens', slice(None)), ('camera_token', slice(0, 1)), ('register_tokens', slice(1, 5)), ('patch_tokens', slice(5, None))):
            row['comparisons'][name] = {}
            for branch, channels in (('concatenated', slice(None)), ('frame', slice(0, 1024)), ('global', slice(1024, None))):
                row['comparisons'][name][branch] = stats(actual[:, :, tokens, channels], reference[:, :, tokens, channels])
        if layer == 23:
            for name, value in (('cpu_control_input', reference[:, :, 0]), ('npu_tokens_cpu_head_input', actual[:, :, 0])):
                expected = diagnostic['artifacts'][name]['raw_sha256']
                require(hashlib.sha256(value.tobytes()).hexdigest() == expected, 'R113 final camera input identity differs')
            row['final_camera_inputs_exactly_match_R113'] = True
        rows.append(row)
    require(all(sha(path) == digest for path, digest in watched.items()), 'Saved evidence changed during analysis')
    result = {
        'schema': 'v113_saved_original_retained_token_discrepancy_trend',
        'status': 'completed_saved_only_diagnostic',
        'script_sha256': sha(Path(__file__)),
        'NN_SDK_ORT_GPU_board_execution': False,
        'source_pair_indices': [32, 40],
        'native_prepared_HW': [392, 518],
        'source_receipts': {str(path): digest for path, digest in PINS.items()},
        'array_ledger': ledger,
        'rows': rows,
        'unchanged_saved_evidence': True,
        'limits': [
            'Cumulative same-input CPU/NPU path discrepancies, not local block error or an additive error budget.',
            'Each layer has a different reference norm and representation; changes in relative L2 alone do not identify a faulty layer.',
            'No neural execution, per-block intervention, trajectory metric, precision acceptance or speed measurement.',
        ],
    }
    (ROOT / 'retained-token-trend.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps([{'layer': r['retained_layer'], 'camera_relative_l2': r['comparisons']['camera_token']['concatenated']['relative_l2'], 'patch_relative_l2': r['comparisons']['patch_tokens']['concatenated']['relative_l2']} for r in rows]))


if __name__ == '__main__':
    main()
