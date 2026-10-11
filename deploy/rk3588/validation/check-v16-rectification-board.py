#!/usr/bin/env python3
"""Check shared live rectification on RK3588S; no model forward is executed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party/vggt')]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--images', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    compatible = Path('/proc/device-tree/compatible').read_bytes().replace(b'\0', b',').decode()
    if 'rockchip,rk3588' not in compatible:
        raise RuntimeError('This real-image preprocessing check must run on RK3588/S')
    if a.output.exists():
        raise FileExistsError(a.output)
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from vggt_slam.image_rectification import CalibrationRectifier
    from vggt_slam.edge_mapping import prepare_rectified_images
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    cfg = json.loads(a.calibration.read_text())
    capture, mapper = CalibrationRectifier(cfg), CalibrationRectifier(cfg)
    rows = []
    for path in a.images:
        digest = sha(path)
        begin = time.monotonic()
        bgr = capture.decode_bgr(str(path))
        elapsed = time.monotonic()-begin
        assert np.array_equal(bgr, mapper.decode_bgr(str(path)))
        # An independent direct PIL resize from the capture pixel array must
        # match the mapper's PNG bridge plus unchanged upstream preprocessing.
        width, height = 112, max(14, round(bgr.shape[0]*112/bgr.shape[1]/14)*14)
        rgb = Image.fromarray(np.ascontiguousarray(bgr[..., ::-1]))
        expected = np.array(rgb.resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32)
        if height > width:
            start = (height-width)//2
            expected = expected[start:start+width]
        expected = np.transpose(expected/255., (2, 0, 1))
        tensor = prepare_rectified_images([str(path)], mapper, 112)
        actual = tensor[0].numpy()
        assert np.array_equal(expected, actual)
        assert sha(path) == digest
        rows.append(dict(image=path.name, original_sha256=digest, image_hw=list(bgr.shape[:2]),
                         capture_and_mapper_pixels_equal=True, original_preprocessing_equal=True,
                         rectified_bgr_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(),
                         model_tensor_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                         decode_plus_remap_seconds=elapsed, model_input_shape=list(tensor.shape)))
    report = dict(status='passed', kind='board_real_image_shared_rectification_check',
                  device_tree_compatible=compatible, metadata=capture.metadata(), images=rows,
                  host_image_processing=False, model_forward_calls=0, real_time_validated=False,
                  script_sha256=sha(__file__), calibration_sha256=sha(a.calibration),
                  source_sha256={name:sha(ROOT/name) for name in [
                      'vggt_slam/image_rectification.py', 'vggt_slam/edge_mapping.py',
                      'vggt_slam/solver.py',
                      'third_party/vggt/vggt/utils/load_fn.py']})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status':report['status'], 'images':len(rows), 'output':str(a.output)}))


if __name__ == '__main__':
    main()
