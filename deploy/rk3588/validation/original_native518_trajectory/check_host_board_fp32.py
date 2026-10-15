#!/usr/bin/env python3
"""One same-pixel original FP32 host/board control; not a performance test."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

R = Path(__file__).resolve().parent
P = Path('/dev/shm/edge-vggt-v50/public-snapshot')
REFERENCE = Path('/dev/shm/edge-vggt-v92/complete-capture')
VENDOR = R / 'original-vggt-source'
CHECKPOINT = Path('/path/to/research-artifacts/checkpoints/vggt-original-d15bf50a/model.pt')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    out = R / 'host-board-fp32-control-r1'
    out.mkdir(exist_ok=False)
    before_affinity = os.sched_getaffinity(0)
    pins = {}
    result = dict(status='running',scope='One original FP32 forward on host CPU versus saved original board CPU FP32 outputs; not board latency or a trajectory acceptance',script_sha256=sha(__file__),host_gpu_executed=False)
    def pin(path, digest):
        path = path.resolve(strict=True)
        assert sha(path) == digest, str(path)
        pins[path] = digest
        return path
    try:
        pin(REFERENCE/'capture.json','3cdfa55556896b0f5356f710c65069b8b5e53f37a0f8cb6f550439f6dfb06730')
        capture = json.loads((REFERENCE/'capture.json').read_text())
        assert capture['full_model_forward_completed'] and not capture['cleanup_errors']
        for name,digest in capture['source_hashes'].items():
            path = VENDOR/Path(name).relative_to('third_party/vggt') if name.startswith('third_party/vggt/') else P/name
            pin(path,digest)
        pin(P/'vggt_slam/model_execution.py','dc5365fd3e823385261b45be2b4b00085c9fb30afbc40825c6c55af0b0253e87')
        images = [pin(Path('/dev/shm/edge-vggt-v110-execution-review/source-images')/Path(row['file']).name,row['sha256']) for row in capture['source_images']]
        sys.path[:0] = [str(VENDOR),str(P)]
        import numpy as np
        import torch
        from vggt_slam.runtime import load_model
        from vggt_slam.model_execution import TorchVGGTExecutor,prepare_rgb_images,execute
        os.sched_setaffinity(0,set(sorted(before_affinity)[:4]))
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        prepared = prepare_rgb_images([str(p) for p in images],518)
        assert hashlib.sha256(prepared.tobytes()).hexdigest()==capture['preprocessed_images_raw_sha256']
        model = load_model(CHECKPOINT,device='cpu',image_size=518,prune_unused_heads=False,offline=True)
        assert all(p.device.type=='cpu' and p.dtype==torch.float32 for p in model.parameters())
        start = time.monotonic()
        with torch.inference_mode():
            prediction = execute(TorchVGGTExecutor(model),prepared,compute_similarity=False)
        result.update(host_forward_seconds_not_board_latency=time.monotonic()-start,host_torch=torch.__version__,board_torch=capture['versions']['torch'],checkpoint=capture['checkpoint'],prepared_pair_raw_sha256=capture['preprocessed_images_raw_sha256'],metrics={})
        for name,metadata in capture['references'].items():
            reference = np.load(pin(REFERENCE/metadata['file'],metadata['sha256']),allow_pickle=False)[0]
            actual = np.asarray(prediction[name])
            assert actual.shape==reference.shape and actual.dtype==reference.dtype==np.float32 and np.isfinite(actual).all()
            error = actual.astype(np.float64)-reference.astype(np.float64)
            path = out/f'host-{name}.npy'
            np.save(path,actual,allow_pickle=False)
            result['metrics'][name] = dict(relative_l2=float(np.linalg.norm(error)/np.linalg.norm(reference.astype(np.float64))),max_absolute_error=float(np.max(np.abs(error))),outside_original_element_gate=int(np.count_nonzero(np.abs(error)>2e-4+2e-3*np.abs(reference))),output_sha256=sha(path),reference_sha256=metadata['sha256'])
        assert hashlib.sha256(prepared.tobytes()).hexdigest()==capture['preprocessed_images_raw_sha256']
        result['status']='completed_same_pixel_original_FP32_host_board_control'
    except BaseException as exc:
        result.update(status='failed',error=repr(exc))
        raise
    finally:
        os.sched_setaffinity(0,before_affinity)
        result.update(source_inputs_unchanged=all(sha(p)==h for p,h in pins.items()),affinity_restored=os.sched_getaffinity(0)==before_affinity,source_input_sha256={str(p):h for p,h in pins.items()})
        (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result.get('metrics',result)),flush=True)

if __name__=='__main__':
    main()
