#!/usr/bin/env python3
"""Validate offline SALAD descriptors on images; does not validate SLAM loop constraints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/vggt")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image-folder", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.frames < 2 or args.threads < 1:
        p.error("frames must be >= 2 and threads positive")
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    from PIL import Image
    import torch
    from torchvision.transforms.functional import to_tensor
    from vggt_slam.loop_closure import ImageRetrieval
    from vggt_slam.slam_utils import sort_images_by_number
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    files = sort_images_by_number([str(path) for path in args.image_folder.iterdir()
                                  if path.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if len(files) < args.frames:
        raise ValueError("Insufficient input images")
    selected = [files[index] for index in np.linspace(0, len(files) - 1, args.frames, dtype=int)]
    report = dict(status="failed", backend="cpu", input_images=selected,
                  slam_loop_constraints_validated=False, model="SALAD", threads=args.threads)
    started = time.monotonic()
    try:
        with args.checkpoint.open("rb") as f:
            report["checkpoint_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
        retrieval = ImageRetrieval(device="cpu", checkpoint=args.checkpoint, offline=True)
        report["load_seconds"] = time.monotonic() - started
        descriptors, durations = [], []
        for path in selected:
            with Image.open(path) as image:
                value = to_tensor(image.convert("RGB").resize((224, 224)))
            tick = time.monotonic()
            descriptors.append(retrieval.get_batch_descriptors(value[None]).cpu().numpy()[0])
            durations.append(time.monotonic() - tick)
            print(json.dumps(dict(image=path, seconds=durations[-1])), flush=True)
        vectors = np.stack(descriptors)
        norms = np.linalg.norm(vectors, axis=1)
        if vectors.shape != (args.frames, 8448) or not np.isfinite(vectors).all() or not np.allclose(norms, 1, atol=1e-4):
            raise RuntimeError("SALAD descriptors have invalid dimensions, values or normalization")
        # Avoid materializing an N x N x 8448 tensor for a full sequence.
        precise = vectors.astype(np.float64)
        squared_norms = np.einsum("ij,ij->i", precise, precise)
        squared_distances = squared_norms[:, None] + squared_norms[None, :] - 2 * (precise @ precise.T)
        distances = np.sqrt(np.maximum(squared_distances, 0))
        np.fill_diagonal(distances, 0)
        report.update(status="passed", shape=list(vectors.shape), norms=norms.tolist(),
                      inference_seconds=durations,
                      pairwise_l2=distances.tolist())
        np.savez_compressed(args.output / "descriptors.npz", descriptors=vectors, frame_names=selected)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
