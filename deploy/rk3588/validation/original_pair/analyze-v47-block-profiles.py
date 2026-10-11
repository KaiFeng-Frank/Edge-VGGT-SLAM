#!/usr/bin/env python3
"""Read historical original-VGGT SDK receipts only; never call any executor."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics

ROW = re.compile(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(CPU|NPU|GPU)\s+(.+?)\s+'
                 r'(\d+/\d+/\d+)\s+(\d+)\s+(.*?)\s+'
                 r'(\d+(?:\.\d+)?%/\d+(?:\.\d+)?%/\d+(?:\.\d+)?%)\s+(\d+)\s+(.+?)\s*$')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(normal_dir, profile_dir):
    normal_dir, profile_dir = Path(normal_dir), Path(profile_dir)
    runs = []
    for directory, normal_mode in ((normal_dir, True), (profile_dir, False)):
        wrapper = json.loads((directory / 'run.json').read_text())
        driver = json.loads((directory / 'driver.json').read_text())
        if (wrapper['kind'] != 'original_VGGT_aggregator_block_RKNN_probe'
                or wrapper['normal_benchmark'] is not normal_mode
                or wrapper['driver_receipt']['sha256'] != sha(directory / 'driver.json')
                or wrapper['status'] != 'completed' or driver['status'] != 'completed'
                or driver['npu_run_api_call_count'] != 4 or len(driver['calls']) != 4
                or driver['context_destroyed'] is not True or driver['cleanup_errors']
                or driver['source_and_inputs_unchanged'] is not True):
            raise ValueError('Expected four completed original-VGGT synchronous calls and clean lifecycle')
        runs.append(driver)
    normal, profile = runs
    tables = []
    for call in profile['calls']:
        detail = call['perf_detail']
        raw = detail['raw_text']
        if len(raw.encode()) != detail['bytes'] or hashlib.sha256(raw.encode()).hexdigest() != detail['raw_sha256']:
            raise ValueError('SDK raw profile byte/hash mismatch')
        rows = []
        for line in raw.splitlines():
            m = ROW.match(line)
            if m:
                rows.append({'id': int(m[1]), 'op': m[2], 'target': m[4], 'us': int(m[7]), 'name': m[11]})
        total = int(re.search(r'Total Operator Elapsed Per Frame Time\(us\):\s*(\d+)', raw)[1])
        if ([row['id'] for row in rows] != list(range(1, len(rows) + 1))
                or sum(row['us'] for row in rows) != total):
            raise ValueError('SDK operator table is incomplete or total is inconsistent')
        if any(row['target'] != 'NPU' for row in rows if row['op'] not in ('InputOperator', 'OutputOperator')):
            raise ValueError('Historical complete block has an internal non-NPU operator')
        tables.append(rows)
    counts = Counter((r['op'], r['target']) for r in tables[0])
    if any(Counter((r['op'], r['target']) for r in table) != counts for table in tables):
        raise ValueError('Operator placement changed between profile calls')
    families = [{'op': key[0], 'target': key[1], 'count_per_call': count,
                 'mean_microseconds': statistics.mean(sum(r['us'] for r in table
                                              if (r['op'], r['target']) == key) for table in tables)}
                for key, count in counts.items()]
    named = [{'name': row['name'], 'op': row['op'], 'target': row['target'],
              'mean_microseconds': statistics.mean(table[index]['us'] for table in tables)}
             for index, row in enumerate(tables[0])]
    return {'normal_run_sha256': sha(normal_dir / 'run.json'),
            'normal_driver_sha256': sha(normal_dir / 'driver.json'),
            'profile_run_sha256': sha(profile_dir / 'run.json'),
            'profile_driver_sha256': sha(profile_dir / 'driver.json'),
            'normal_warm_seconds': {key: statistics.mean(row[key] for row in normal['calls'][1:])
              for key in ('input_set_seconds', 'run_api_seconds', 'outputs_get_seconds',
                          'output_copy_seconds', 'outputs_release_seconds', 'io_total_seconds')},
            'profile_all_four_operator_mean_microseconds': statistics.mean(sum(r['us'] for r in t) for t in tables),
            'profile_operator_families': families, 'profile_named_operators': named,
            'profile_internal_targets': {'NPU': len(tables[0]) - 2, 'CPU': 0},
            'profile_CPU_IO_operators': 2, 'memory_reported_bytes': normal['model_memory_bytes'],
            'standalone_relative_l2': normal['calls'][0]['versus_cpu_reference']['relative_l2'],
            'standalone_reference_input_raw_sha256': normal['capture_metadata'].get('original_input_raw_sha256'),
            'scope': 'Standalone historical block on its original CPU input; four profile calls separate from four normal calls.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('frame-normal', 'frame-profile', 'global-normal', 'global-profile', 'output'):
        p.add_argument('--' + key, required=True)
    a = p.parse_args()
    result = {'schema': 'v47_original_VGGT_saved_block_profile_review',
              'status': 'saved_receipts_checked', 'script_sha256': sha(__file__),
              'host_model_or_ORT_execution': False, 'board_operations': False,
              'historical_RKNN_FP32_rejections_retained': True,
              'frame0_R2': summarize(a.frame_normal, a.frame_profile),
              'global0_R3': summarize(a.global_normal, a.global_profile),
              'limits': ['Do not add profile operator clocks to normal SDK IO clocks: they are nested and from different calls.',
                        'Old global0 input is CPU frame0 output, not an actual preceding RKNN frame0 output.',
                        'No measured sequential two-block speed or accumulated error is available.',
                        'Pair removes one logical host bridge; internal layout, SDPA/MLP cost and actual DMA must be measured anew.',
                        'Weights of frame0 and global0 are distinct; pairing does not deduplicate original learned parameters.',
                        'Historical FP16 block outputs fail the original FP32 gate; this report does not change acceptance.',
                        'Original DINO/head/depth and 23 other frame/global pairs remain outside the proposed pair probe.']}
    with Path(a.output).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'output_sha256': sha(a.output)}))


if __name__ == '__main__':
    main()
