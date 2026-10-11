#!/usr/bin/env python3
"""Extract the frozen actual observation34 B2D6 CF feed; saved arrays only."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

RUN_SHA = '78f15e36e5444dfe17a70610eb4410f0605c262eb7102bc2fee39ba5cfdf427b'
ADAPTER_SHA = '0f427739037e2178dddc4631a32fb602ac0a2ddc259e48e147028604ae6733f1'
PLAN_SHA = '04e240c2c35bd1359dbe5649fb038490a0a7ff99654d8d112ad7c83acc87ac2e'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                finite=bool(np.isfinite(value).all()), c_contiguous=bool(value.flags.c_contiguous),
                owned=bool(value.flags.owndata))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory')
    args = parser.parse_args()
    require(sha(args.run) == RUN_SHA and sha(args.adapter) == ADAPTER_SHA
            and sha(args.adapter.with_name('source_frame_plan.py')) == PLAN_SHA, 'Frozen input/source SHA differs')
    run = json.loads(args.run.read_text())
    capture = next(row for row in run['update_captures'] if row['frame_index'] == 34)
    path = args.run.parent / capture['input']['file']
    require(sha(path) == capture['input']['sha256'] and path.stat().st_size == capture['input']['bytes'], 'Capture NPZ differs')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], order='C', copy=True) for name in archive.files}
    require(set(arrays) == set(capture['input']['arrays']) == {'net', 'inp', 'corr', 'ii', 'jj', 'kk'}, 'Exactly six source arrays required')
    for name, value in arrays.items():
        actual, expected = metadata(value), capture['input']['arrays'][name]
        require(all(actual[key] == expected[key] for key in ('shape', 'dtype', 'bytes', 'tensor_sha256'))
                and actual['finite'], 'Source array metadata differs')
    spec = importlib.util.spec_from_file_location('_v43_saved_group_adapter', args.adapter)
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    plan = adapter.build_degree_plan(arrays['ii'], arrays['jj'], arrays['kk'], patches_per_frame=16,
                                     frame_capacity=64, max_sources_per_batch=2)
    saved_call = next(row for row in run['online_update_records'] if row['index'] == capture['update_index'])
    require(saved_call['route'] == 'full' and saved_call['degree_plan'] == plan.metadata(), 'Actual source plan differs')
    index, group = next((i, row) for i, row in enumerate(plan.batches)
                        if len(row.source_slots) == 2 and row.degree == 6 and row.patches == 16)
    require(group.source_slots == (2, 5), 'Frozen selected group differs')
    for offset, slot in enumerate(group.source_slots):
        require(np.array_equal(np.sort(group.canonical_edge_indices[offset].ravel()), np.flatnonzero(arrays['ii'] == slot)),
                'Selected source is incomplete')
    feed = {name: adapter.pack_batch(arrays[name], group) for name in ('net', 'inp', 'corr')}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    target = args.output_dir / 'actual-obs034-B2D6-feed.npz'
    np.savez_compressed(target, **feed)
    record = dict(schema='v43_saved_actual_complete_source_CF_group_extraction',
        status='saved_array_integrity_verified_no_neural_execution', extract_script_sha256=sha(__file__),
        candidate_accepted=False, model_execution=False, SDK_loaded=False, SDK_run_calls=0,
        GT_or_teacher_used=False, same_call_original_Torch=False,
        source_run=dict(file='vggt-v41-online-bothnpu-r2/run.json', sha256=RUN_SHA,
                        original_status=run['status'], original_error=run['error'], status_modified=False),
        source_capture=dict(frame_index=34, update_index=capture['update_index'], edges=capture['edges'],
                            input_file=path.name, input_sha256=sha(path), input_bytes=path.stat().st_size,
                            original_six_input_array_metadata={name: metadata(value) for name, value in arrays.items()},
                            all_descriptor_fields_matched=True),
        source_pins=dict(source_frame_batches_py=ADAPTER_SHA, source_frame_plan_py=PLAN_SHA),
        extraction=dict(method='Frozen build_degree_plan and pack_batch only', patches_per_frame=16,
                        frame_capacity=64, max_sources_per_batch=2, full_plan=plan.metadata(),
                        selected_group_index=index, selected_group=group.metadata(), complete_source_edges_verified=True,
                        canonical_edge_indices=group.canonical_edge_indices.tolist(),
                        canonical_edge_indices_sha256=hashlib.sha256(group.canonical_edge_indices.tobytes()).hexdigest(),
                        original_full_degree_plan_metadata_matched=True, source_order_unchanged=True,
                        no_padding_or_dropped_edges=True),
        output=dict(file=target.name, bytes=target.stat().st_size, sha256=sha(target),
                    arrays={name: metadata(value) for name, value in feed.items()}))
    (args.output_dir / 'actual-obs034-B2D6-feed.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps(record['output']))


if __name__ == '__main__':
    main()
