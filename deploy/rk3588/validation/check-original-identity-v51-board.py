#!/usr/bin/env python3
"""Read and hash the board's original checkpoint without deserializing it."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--source-sha256', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
assert b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes()
source_hash = hashlib.sha256(a.source.read_bytes()).hexdigest()
assert source_hash == a.source_sha256
spec = importlib.util.spec_from_file_location('original_checkpoint_identity', a.source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
base = Path('/mnt/vggt-data/vggt-slam2')
models = sorted(x.name for x in (base/'src/third_party').iterdir() if x.is_dir())
weights = sorted(x.name for x in (base/'weights').iterdir() if x.is_file())
assert models == ['vggt'] and weights == ['model.pt']
started = time.monotonic_ns()
identity = module.verify_original_checkpoint(base/'weights/model.pt')
elapsed_ns = time.monotonic_ns() - started
assert hashlib.sha256(a.source.read_bytes()).hexdigest() == source_hash
result = {'schema': 'v51_original_checkpoint_identity_board_check',
          'status': 'original_checkpoint_identity_verified',
          'verifier_sha256': source_hash,
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'checkpoint': {'file': 'model.pt', **identity},
          'startup_file_verification_ns': elapsed_ns,
          'active_model_directories': models, 'active_weight_files': weights,
          'checkpoint_deserialized': False, 'model_constructed_or_executed': False,
          'performance_or_trajectory_acceptance': False}
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
