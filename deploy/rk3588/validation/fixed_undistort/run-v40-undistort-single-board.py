"""RK3588-only single-remap pixel/CPU preprocessing; no neural inference or VO.

All first192 v39 inputs are compared. Fixed24 uniformly spaced indices receive
ABBA timings on the same decoded pixels. imread is measured once separately;
neither paired pipeline clock includes imread, hashing, comparison or saving.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import time
import traceback

MANIFEST_SHA = 'ad2c7eadef84cb4ecc550e2b4dc0f27611cb116bcefaf7eb697620646bf4bb10'
POLICY_SHA = 'cc6a31f89bdcff0b43c0d51266ad62c48d0e297346845a98d42bad5c8874ef4a'
CACHE_SHA = '6ae53bd252de2333bf105e0f5f98c842dd2cc79244f8f1acf9572809bd627c63'
NORMALIZER_SHA = 'fc6fc662613a720cfb779a360a63906a593a96ff08addecb9d01cfd8767ca7a8'
BASE = Path('/mnt/vggt-data/vggt-slam2/experiments')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def require_board(system=None, machine=None, compatible=None):
    system = platform.system() if system is None else system
    machine = platform.machine() if machine is None else machine
    if compatible is None:
        compatible = Path('/proc/device-tree/compatible').read_bytes()
    require(system == 'Linux' and machine == 'aarch64' and b'rockchip,rk3588' in compatible,
            'Real image processing is RK3588/S board-only')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def performance_indices():
    return [i * 191 // 23 for i in range(24)]


def validate_manifest(manifest, index_text):
    require(manifest['schema'] == 'v39_first_contiguous_TUM_RGB_frames'
            and manifest['GT_read'] is False and len(manifest['frames']) == 192,
            'Frozen first192 RGB manifest required')
    entries = [line.split() for line in index_text.splitlines()
               if line.strip() and not line.lstrip().startswith('#')][:192]
    require(len(entries) == 192, 'RGB index contains fewer than192 observations')
    for item, (stamp, name) in zip(manifest['frames'], entries):
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts
                and name.startswith('rgb/') and item['relative_path'] == name
                and item['frame_id'] == stamp and item['timestamp'] == float(stamp)
                and math.isfinite(item['timestamp']), 'Ordered source identity/path differs')
    require(all(b['timestamp'] > a['timestamp'] for a,b in
                zip(manifest['frames'], manifest['frames'][1:])), 'RGB times are not increasing')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-dir', 'input-manifest', 'policy', 'cache-helper',
                 'normalization-helper', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require_board()  # Before NumPy/OpenCV imports, decode, maps or output creation.
    out = args.output_dir.resolve()
    require(out.is_relative_to(BASE) and not out.exists(), 'New board experiment output required')
    out.mkdir()
    report = dict(schema='v40_fixed_undistort_cache_pixel_and_CPU_benchmark', status='failed',
        neural_forward_calls=0, ORT_calls=0, SDK_calls=0, BA_calls=0, GT_read=False,
        physical_camera=False, realtime_validated=False, default_pipeline_changed=False,
        all192_pixel_gate_passed=False, frames=[], performance_calls=[], cleanup_errors=[])
    begin = time.monotonic()
    pins, handle, cv2 = {}, None, None
    affinity = os.sched_getaffinity(0)
    old_cv_threads = None
    def pin(path, expected):
        path = path.resolve()
        require(sha(path) == expected, 'Pinned SHA differs: ' + str(path))
        pins[str(path)] = expected
        return path
    try:
        pin(Path(__file__), sha(__file__))
        pin(args.input_manifest, MANIFEST_SHA); pin(args.policy, POLICY_SHA)
        pin(args.cache_helper, CACHE_SHA); pin(args.normalization_helper, NORMALIZER_SHA)
        pin(args.cache_helper.with_name('fixed_undistort_cache.py'), '14c7331fcb9bc84e13fa8d9b38f0a0575b529729b08dd539915f3714a632972c')
        manifest = json.loads(args.input_manifest.read_text())
        rgb_index = pin(args.dataset_dir / 'rgb.txt', manifest['rgb_index_sha256'])
        validate_manifest(manifest, rgb_index.read_text())
        for item in manifest['frames']:
            path = pin(args.dataset_dir / item['relative_path'], item['sha256'])
            require(path.is_relative_to(args.dataset_dir.resolve())
                    and path.stat().st_size == item['bytes'], 'RGB path/size differs')
        policy = json.loads(args.policy.read_text())
        pre = policy['preprocessing']
        require(pre['source_shape_HWC'] == [480,640,3]
                and pre['crop_y_start_stop_x_start_stop'] == [8,472,16,624]
                and pre['resized_shape_HWC'] == [232,304,3], 'Original pixel geometry differs')
        require({4,5,6,7} <= affinity, 'Four A76 cores required')
        os.sched_setaffinity(0, {4,5,6,7})
        import cv2
        import numpy as np
        require(cv2.__version__ == '4.10.0', 'Original board OpenCV4.10.0 required')
        old_cv_threads = cv2.getNumThreads()
        report['OpenCV_threads_before'] = old_cv_threads
        require(old_cv_threads == 4, 'Original post-affinity OpenCV default4 threads required')
        # Match v39: bind CPUs before importing OpenCV; keep its actual default.
        # The unbound process default (8) is not this replay's thread setting.
        cache = load(args.cache_helper, '_v40_fixed_cache')
        normalizer = load(args.normalization_helper, '_v40_frozen_normalization')
        fx,fy,cx,cy = pre['original_intrinsics_fx_fy_cx_cy']
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float64)
        D = np.array(pre['original_distortion_k1_k2_p1_p2_k3'], np.float64)
        report.update(cpu_affinity=sorted(os.sched_getaffinity(0)), OpenCV_version=cv2.__version__,
            NumPy_version=np.__version__, OpenCV_threads_before=old_cv_threads,
            OpenCV_threads_both_paths=cv2.getNumThreads(), input_manifest_sha256=MANIFEST_SHA,
            policy_sha256=POLICY_SHA, performance_source_indices=performance_indices(),
            performance_order=['original','cached','cached','original'],
            scope='No model/pose/quality test. Same decoded image per pair; source192 pixel gate. CPU prep ABBA only.',
            timing_scope='imread and cold map construction separate; warm clocks include owned output allocation, remap, crop/resize, FP32 normalization; hashes/comparisons/save excluded.')
        start = time.monotonic()
        handle = cache.SingleRemapUndistorter(K,D,(480,640),expected_opencv_version='4.10.0')
        report['cold_map_construction_seconds'] = time.monotonic() - start
        def array_info(a):
            return dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,
                        tensor_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())
        def process(image, method):
            start = time.monotonic()
            full = cv2.undistort(image,K,D) if method == 'original' else handle.undistort_bgr(image)
            remapped = time.monotonic()
            crop = np.ascontiguousarray(full[8:-8,16:-16])
            half = np.ascontiguousarray(cv2.resize(crop,(304,232),interpolation=cv2.INTER_AREA))
            resized = time.monotonic()
            normalized = normalizer.normalize_bgr(half)
            finished = time.monotonic()
            return (full,half,normalized), dict(undistort_or_cached_remap_seconds=remapped-start,
                crop_resize_seconds=resized-remapped, normalization_seconds=finished-resized,
                CPU_prep_without_decode_seconds=finished-start)
        def equivalent(a,b):
            require(all(x.shape==y.shape and x.dtype==y.dtype and x.tobytes(order='C')==y.tobytes(order='C')
                        for x,y in zip(a,b)), 'Full/cropped-half/FP32 original bitwise pixel gate failed')
        samples = {}
        selected_perf = set(performance_indices())
        for index,item in enumerate(manifest['frames']):
            start = time.monotonic()
            image = cv2.imread(str(args.dataset_dir/item['relative_path']),cv2.IMREAD_COLOR)
            read_end = time.monotonic()
            require(image is not None and image.shape==(480,640,3) and image.dtype==np.uint8,
                    'Original uint8 BGR geometry differs')
            before = hashlib.sha256(image.tobytes()).hexdigest()
            original,ot = process(image,'original'); cached,ct = process(image,'cached')
            equivalent(original,cached)
            row = dict(index=index,frame_id=item['frame_id'],source_sha256=item['sha256'],
                imread_seconds=read_end-start, original=ot,cached=ct,
                original_arrays=[array_info(x) for x in original],
                cached_arrays=[array_info(x) for x in cached],bitwise_all_three=True)
            if index in selected_perf:
                for order,method in enumerate(('original','cached','cached','original')):
                    actual,timing = process(image,method)
                    equivalent(original,actual)
                    report['performance_calls'].append(dict(frame_index=index,order=order,
                        method=method,bitwise_all_three=True,**timing))
            if index in (0,95,191):
                for method,arrays in (('original',original),('cached',cached)):
                    samples[f'{method}_half_{index:03d}'] = arrays[1]
                    samples[f'{method}_normalized_{index:03d}'] = arrays[2]
            require(hashlib.sha256(image.tobytes()).hexdigest()==before,'Decoded input mutated')
            report['frames'].append(row)
        sample_file = out/'equivalence-samples.npz'
        np.savez(sample_file,**samples)
        report['samples'] = dict(file=sample_file.name,bytes=sample_file.stat().st_size,
            sha256=sha(sample_file),arrays={k:array_info(v) for k,v in samples.items()})
        report.update(status='completed_preprocessing_pixel_gate_passed_not_realtime',
                      all192_pixel_gate_passed=True,frames_completed=len(report['frames']))
    except Exception as error:
        report.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        if handle is not None:
            handle.close(); report['cache'] = handle.metadata()
        if cv2 is not None and old_cv_threads is not None:
            try:
                cv2.setNumThreads(old_cv_threads)
                require(cv2.getNumThreads() == old_cv_threads, 'OpenCV thread restore failed')
                report['OpenCV_threads_restored'] = True
            except Exception as error: report['cleanup_errors'].append(repr(error))
        try:
            os.sched_setaffinity(0,affinity)
            require(os.sched_getaffinity(0)==affinity,'Affinity restore failed')
            report['affinity_restored'] = True
        except Exception as error: report['cleanup_errors'].append(repr(error))
        try:
            after = {path:sha(path) for path in pins}
            report.update(source_pins=pins,source_hashes_after=after,sources_and_inputs_unchanged=after==pins)
            require(after==pins,'Pinned source/input changed')
        except Exception as error: report['cleanup_errors'].append(repr(error))
        if report['cleanup_errors']: report['status']='failed'
        report.update(actual_driver_seconds=time.monotonic()-begin,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        (out/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','error','frames_completed','cleanup_errors')}),flush=True)
    return 0 if report['status'].startswith('completed_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
