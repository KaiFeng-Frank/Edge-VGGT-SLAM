"""Original-weight, board-only FP32 encoder for a declared half-resolution run.

Reuses original DPVO neural modules and frozen NumPy patch sampling. No new
tracking algorithm, learned parameters, ORT or NPU implementation is introduced.
DPVO attribution: Princeton Vision & Learning Lab, MIT.
"""
import hashlib
from pathlib import Path
import platform
import time

import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


class BoardTorchEncoder:
    def __init__(self, neural, common, sampling, torch, *, source_files):
        require(platform.system() == 'Linux' and platform.machine() == 'aarch64'
                and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
                'Real neural execution requires RK3588/S')
        self.neural = common.make_encoder(neural)
        self.sampling, self.torch = sampling, torch
        self.pins = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                     for p in source_files}
        self.closed = False
        self.data = dict(backend='original_DPVO_Torch_CPU_FP32', completed_frames=0,
                         normalization_seconds=0., forward_seconds=0., sampling_seconds=0.,
                         total_encode_seconds=0., closed=False, input_shape=[1,1,3,232,304],
                         no_weights_changed=True, patches_per_frame=16)

    def encode(self, image, *, generator, patch_count):
        require(not self.closed and image.dtype == np.uint8 and image.shape == (232,304,3)
                and patch_count == 16, 'Declared half-resolution M16 BGR input required')
        begin = time.monotonic()
        normalized = self.sampling.normalize_bgr(image)
        normalized_end = time.monotonic()
        with self.torch.inference_mode():
            fmap, imap = self.neural(self.torch.from_numpy(normalized)[:, None])
            fmap, imap = fmap.detach().numpy().copy(), imap.detach().numpy().copy()
        forward_end = time.monotonic()
        require(fmap.shape == (1,1,128,58,76) and imap.shape == (1,1,384,58,76),
                'Half-resolution original feature shape differs')
        centers = self.sampling.random_centers(76,58,patch_count,generator)
        result = self.sampling.features_to_patches(fmap,imap,centers)
        end = time.monotonic()
        timing = dict(normalization_seconds=normalized_end-begin,
                      forward_seconds=forward_end-normalized_end,
                      sampling_seconds=end-forward_end,total_encode_seconds=end-begin)
        self.data['completed_frames'] += 1
        for k,v in timing.items(): self.data[k] += v
        result['timing'] = timing
        return result

    def check_sources(self):
        require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
                    for p,digest in self.pins.items()), 'Encoder source changed')
        return dict(self.pins)

    def metrics(self):
        return dict(self.data)

    def close(self):
        self.neural = None
        self.closed = True
        self.data['closed'] = True
