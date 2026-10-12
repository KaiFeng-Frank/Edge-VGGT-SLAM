#!/usr/bin/env python3
"""Saved-array context diagnostic; no model, SDK or loop implementation."""
import hashlib
import json
from pathlib import Path

import numpy as np

RUN = Path('/dev/shm/edge-vggt-v75/complete-1ghz-r1')
OUTPUT = Path(__file__).with_name('target-token-context-saved.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(count=int(values.size), min=float(values.min()),
                median=float(np.median(values)), mean=float(values.mean()),
                p95=float(np.percentile(values, 95)), max=float(values.max()))


def main():
    receipt = RUN / 'result.json'
    result = json.loads(receipt.read_text())
    assert result['status'] == 'completed_not_quality_acceptance' and result['save_aggregator_outputs']
    assert result['original_model_only'] and len(result['branches']) == 1
    branch_name, branch = next(iter(result['branches'].items()))
    assert branch['status'] == 'completed'
    indices, timestamps = result['input_indices'], result['input_timestamps']
    assert len(indices) == len(timestamps) == 17
    assert len(set(indices)) == len(set(timestamps)) == 17
    assert all(a < b for a, b in zip(timestamps, timestamps[1:]))
    updates = branch['updates']
    assert [u['pair'] for u in updates] == list(range(1, 17))
    descriptors, artifacts = [], []
    for pair, update in enumerate(updates, 1):
        assert update['timestamp'] == timestamps[pair]
        meta = update['aggregator_artifact']
        path = RUN / branch_name / meta['file']
        assert path.parent == RUN / branch_name
        assert path.stat().st_size == meta['bytes'] and sha(path) == meta['sha256']
        with np.load(path, allow_pickle=False) as archive:
            tokens = archive['target_tokens']
        assert tokens.shape == (2, 53, 1024) and tokens.dtype == np.float32
        assert np.isfinite(tokens).all()
        means = tokens[:, 5:, :].mean(axis=1, dtype=np.float64)
        norms = np.linalg.norm(means, axis=1, keepdims=True)
        assert np.isfinite(norms).all() and (norms > 0).all()
        descriptors.append(means / norms)
        artifacts.append(dict(pair=pair, file=meta['file'], sha256=meta['sha256'],
            target_raw_sha256=hashlib.sha256(tokens.tobytes()).hexdigest(),
            frame_indices=indices[pair - 1:pair + 1],
            frame_timestamps=timestamps[pair - 1:pair + 1]))
    descriptors = np.stack(descriptors)
    # Query: tail of pair i. Gallery: head of every pair; same-frame head is i+1.
    distances = np.linalg.norm(descriptors[:-1, 1, None, :] -
                               descriptors[None, :, 0, :], axis=-1)
    truth = np.arange(1, 16)
    same = distances[np.arange(15), truth]
    different_mask = np.arange(16)[None, :] != truth[:, None]
    orders = np.argsort(distances, axis=1, kind='stable')
    ranks = np.argmax(orders == truth[:, None], axis=1) + 1
    nearest = orders[:, 0]
    rows = [dict(frame_index=indices[i + 1], timestamp=timestamps[i + 1],
        tail_pair=i + 1, matching_head_pair=i + 2, same_frame_l2=float(same[i]),
        nearest_other_frame_l2=float(distances[i, different_mask[i]].min()),
        same_frame_rank=int(ranks[i]), nearest_head_frame_index=indices[int(nearest[i])],
        nearest_head_timestamp=timestamps[int(nearest[i])],
        nearest_head_l2=float(distances[i, nearest[i]])) for i in range(15)]
    report = dict(schema='v77_original_target_tokens_saved_context_diagnostic',
        status='saved_data_analyzed_not_retrieval_or_loop_acceptance',
        source_sha256=sha(Path(__file__)), result_sha256=sha(receipt),
        branch=branch_name, original_model_only=True, new_model=False,
        model_OR_ORT_OR_SDK_OR_board_execution=False,
        descriptor='mean(target_tokens[:,5:,:], patch_axis, float64), then L2 normalize',
        target_source='original global20, already per-token normalized; saved production FP16 backbone outputs',
        token_shape=[2, 53, 1024], patch_count=48, descriptor_dimension=1024,
        input_indices=indices, input_timestamps=timestamps, artifacts=artifacts,
        same_frame_different_pair_context_l2=stats(same),
        different_frame_query_gallery_l2=stats(distances[different_mask]),
        two_different_frames_within_same_pair_l2=stats(np.linalg.norm(
            descriptors[:, 0] - descriptors[:, 1], axis=-1)),
        nearest_neighbor=dict(queries=15, gallery=16, same_frame_top1=int((ranks == 1).sum()),
            same_frame_top3=int((ranks <= 3).sum()), same_frame_rank=stats(ranks),
            nearest_is_other_frame_in_query_pair=int((nearest == np.arange(15)).sum())),
        overlapping_frames=rows, retrieval_quality_accepted=False, loop_implemented=False,
        limitations=['One short forward sequence and exact overlapping images, not natural loop closures.',
            'Adjacent gallery frames and shared pair context can confound image identity.',
            'Descriptor computation adds no model and does not estimate pose or create an SL4 edge.',
            'These are saved FP16-backbone outputs, not a fresh original FP32 descriptor experiment.'])
    with OUTPUT.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('same_frame_different_pair_context_l2',
        'different_frame_query_gallery_l2', 'two_different_frames_within_same_pair_l2',
        'nearest_neighbor')}, allow_nan=False))
    print(str(OUTPUT), sha(OUTPUT))


if __name__ == '__main__':
    main()
