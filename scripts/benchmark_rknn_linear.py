#!/usr/bin/env python3
"""Exercise real NPU Linear operations; this does not validate SLAM or its weights."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--library", default="/usr/lib/librknnrt.so")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--rows", type=int, nargs="+", help="Test these row counts for all four Transformer projection shapes")
    p.add_argument("--core-mask", type=int, default=0, choices=[0, 1, 2, 4])
    p.add_argument("--weight-layout", default="normal", choices=["normal", "native"])
    args = p.parse_args()
    if args.repeats < 1 or args.threads < 1:
        p.error("threads and repeats must be positive")
    if args.rows and any(rows < 1 for rows in args.rows):
        p.error("row counts must be positive")
    import numpy as np
    from PIL import Image
    import torch
    from torch import nn
    from vggt_slam.rknn_linear import install_rknn_linear

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        p.error("output already exists")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    rng = torch.Generator().manual_seed(42)
    with Image.open(args.image) as image:
        pixels = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32)
    pixels = (pixels.ravel() / 255.0 - 0.5) * 2.0
    report = dict(status="failed", full_model_validated=False, pretrained_weights_used=False,
                  input_image=str(args.image), input_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
                  npu_inference_executed=False, threads=torch.get_num_threads(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), core_mask=args.core_mask,
                  weight_layout=args.weight_layout, cases=[])
    started = time.monotonic()
    try:
        shapes = [(64, 1024, 3072), (128, 1024, 1024), (128, 1024, 4096), (128, 4096, 1024)]
        if args.rows:
            shapes = [(rows, width, outputs) for rows in args.rows
                      for width, outputs in [(1024, 3072), (1024, 1024), (1024, 4096), (4096, 1024)]]
        for rows, width, outputs in shapes:
            model = nn.ModuleList([nn.Linear(width, outputs), nn.Linear(width, outputs)]).eval()
            with torch.no_grad():
                for layer in model:
                    layer.weight.copy_(torch.randn(outputs, width, generator=rng) / math.sqrt(width))
                    layer.bias.copy_(torch.randn(outputs, generator=rng) * 0.01)
            inputs = torch.from_numpy(np.resize(pixels, (rows, width)).copy())
            alternate_inputs = -inputs.roll(7, dims=-1)
            with torch.inference_mode():
                examples = [inputs, alternate_inputs]
                expected = [layer(value) for layer, value in zip(model, examples)]
                rounded = [torch.nn.functional.linear(value.half().float(), layer.weight.half().float(), layer.bias)
                           for layer, value in zip(model, examples)]
                cpu_times = []
                for i in range(args.repeats):
                    tick = time.perf_counter()
                    model[i % 2](examples[i % 2])
                    cpu_times.append(time.perf_counter() - tick)
                with install_rknn_linear(model, library_path=args.library, core_mask=args.core_mask,
                                         weight_layout=args.weight_layout) as handle:
                    tick = time.perf_counter()
                    result = model[0](examples[0])
                    cold_seconds = time.perf_counter() - tick
                    results = [(0, result)]
                    times = []
                    for i in range(1, args.repeats + 1):
                        tick = time.perf_counter()
                        result = model[i % 2](examples[i % 2])
                        times.append(time.perf_counter() - tick)
                        results.append((i % 2, result))
                    metrics = handle.metrics()
                error = np.stack([(result - expected[index]).numpy() for index, result in results])
                rounded_error = np.stack([(result - rounded[index]).numpy() for index, result in results])
                case = dict(shape=[rows, width, outputs], cpu_seconds=cpu_times,
                            npu_linear_seconds=times, npu_cold_seconds=cold_seconds,
                            alternates_weights_and_inputs=True,
                            max_abs_error=float(np.max(np.abs(error))),
                            rmse=float(np.sqrt(np.mean(error ** 2))),
                            max_abs_error_vs_fp16_inputs_weights=float(np.max(np.abs(rounded_error))),
                            finite=all(bool(torch.isfinite(result).all()) for _, result in results), metrics=metrics)
                case["calls"] = []
                for call, (index, result) in enumerate(results):
                    difference = error[call]
                    worst = np.unravel_index(np.argmax(np.abs(difference)), difference.shape)
                    case["calls"].append(dict(layer=index,
                        max_abs_error=float(np.max(np.abs(difference))),
                        rmse=float(np.sqrt(np.mean(difference ** 2))),
                        elements_error_above_0_01=int(np.count_nonzero(np.abs(difference) > 0.01)),
                        worst_index=list(map(int, worst)), actual=float(result[worst]),
                        expected=float(expected[index][worst])))
                case["passed"] = case["finite"] and case["max_abs_error"] < 0.01 and case["rmse"] < 0.001
                report["cases"].append(case)
                report["npu_inference_executed"] |= metrics["npu_calls"] > 0
                print(json.dumps(case), flush=True)
                if not case["passed"]:
                    np.savez_compressed(args.output.with_suffix(".failure.npz"),
                                        actual=np.stack([value.numpy() for _, value in results]),
                                        expected=np.stack([expected[index].numpy() for index, _ in results]))
                    raise RuntimeError("NPU Linear numerical check failed")
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
