"""Profile the first five original cache fingerprints within one board comparator."""
import cProfile
import hashlib
import json
from pathlib import Path
import platform
import pstats
import runpy
import sys
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
if platform.machine() != 'aarch64' or b'rockchip,rk3588' not in Path('/proc/device-tree/compatible').read_bytes():
    raise RuntimeError('Actual model execution is restricted to RK3588S')
from vggt_slam import patch_cache
output = Path(sys.argv.pop(1))
if output.exists():
    raise FileExistsError(output)
source = Path(patch_cache.__file__)
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
if source_sha != '3a8cc8c5a68c3bb934969e86053538f0d2026da1846274f4e6085297474a1e4b':
    raise ValueError('Original v61 cache source required')
original = patch_cache.PatchCacheHandle._fingerprint
profiler = cProfile.Profile()
calls, observed_seconds = 0, []
def fingerprint(self):
    global calls
    calls += 1
    if calls > 5:
        return original(self)
    start = time.perf_counter()
    profiler.enable()
    try:
        return original(self)
    finally:
        profiler.disable()
        observed_seconds.append(time.perf_counter() - start)
patch_cache.PatchCacheHandle._fingerprint = fingerprint
report = dict(status='failed', candidate_accepted=False, real_time_validated=False,
              profiler_overhead_included=True, use_timings_for_performance_claim=False,
              original_cache_source_sha256=source_sha,
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
try:
    script = BASE/'src/deploy/rk3588/validation/original_full_aggregator/compare_complete_model.py'
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name='__main__')
    report['status'] = 'original_comparator_completed_with_guard_profiling'
finally:
    patch_cache.PatchCacheHandle._fingerprint = original
    report.update(fingerprint_calls=calls, profiled_calls=len(observed_seconds),
                  profiled_wall_seconds=observed_seconds,
                  source_unchanged=hashlib.sha256(source.read_bytes()).hexdigest() == source_sha,
                  method_restored=patch_cache.PatchCacheHandle._fingerprint is original)
    report['functions'] = [dict(file=key[0], line=key[1], function=key[2],
        primitive_calls=value[0], calls=value[1], own_seconds=value[2], cumulative_seconds=value[3])
        for key, value in sorted(pstats.Stats(profiler).stats.items(), key=lambda pair: pair[1][3], reverse=True)]
    output.write_text(json.dumps(report, indent=2) + '\n')
