#!/usr/bin/env python3
"""Bounded CPU kernel probe; synthetic weights do not validate SLAM accuracy.

Each Torch thread count runs in a fresh process. Low precision errors are
reported without retrying the operation in float32. Both native dtype timings
and float32 input/output conversion timings are measured where supported.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


class BudgetExpired(Exception):
    pass


def cases(args):
    result = []
    for rows in args.rows:
        if "linear" in args.ops:
            for width, outputs in ((1024, 3072), (1024, 1024), (1024, 4096), (4096, 1024)):
                result.append(dict(op="linear", shape=[rows, width, outputs]))
        if "layernorm" in args.ops:
            for width in (1024, 64):
                result.append(dict(op="layernorm", shape=[rows, width]))
        if "gelu" in args.ops:
            result.append(dict(op="gelu", shape=[rows, 4096]))
    if "sdpa" in args.ops:
        for tokens in args.sdpa_tokens:
            result.append(dict(op="sdpa", shape=[1, 16, tokens, 64]))
    if "conv2d" in args.ops:
        # DPT token projection and a residual convolution on the patch grid.
        result.extend([dict(op="conv2d", shape=[1, 2048, 21, 37], outputs=256, kernel=1),
                       dict(op="conv2d", shape=[1, 256, 21, 37], outputs=256, kernel=3)])
    return result


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def platform_sample():
    result = dict(cpu_frequency_khz={}, temperature_c={})
    for policy in Path("/sys/devices/system/cpu/cpufreq").glob("policy*"):
        try:
            result["cpu_frequency_khz"][policy.name] = int((policy / "scaling_cur_freq").read_text())
        except (OSError, ValueError):
            pass
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            result["temperature_c"][(zone / "type").read_text().strip()] = int((zone / "temp").read_text()) / 1000
        except (OSError, ValueError):
            pass
    return result


def cpu_features():
    try:
        lines = Path("/proc/cpuinfo").read_text().splitlines()
    except OSError:
        return []
    return sorted({value for line in lines if line.startswith(("Features", "flags"))
                   for value in line.split(":", 1)[-1].split()})


def build_case(torch, spec):
    nn = torch.nn
    op, shape = spec["op"], spec["shape"]
    if op == "linear":
        rows, width, outputs = shape
        # A container is required: quantize_dynamic does not replace a root Linear.
        return nn.Sequential(nn.Linear(width, outputs)).eval(), (torch.randn(rows, width),)
    if op == "layernorm":
        return nn.LayerNorm(shape[-1]).eval(), (torch.randn(*shape),)
    if op == "gelu":
        return nn.GELU().eval(), (torch.randn(*shape),)
    if op == "conv2d":
        return nn.Conv2d(shape[1], spec["outputs"], spec["kernel"],
                         padding=spec["kernel"] // 2).eval(), (torch.randn(*shape),)
    if op == "sdpa":
        def attention(query, key, value):
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False)
        return attention, tuple(torch.randn(*shape) for _ in range(3))
    raise ValueError(f"Unknown operation: {op}")


def timed(fn, warmup, repeats, deadline):
    durations = []
    cpu_durations = []
    output = None
    for index in range(warmup + repeats):
        if time.monotonic() >= deadline:
            raise BudgetExpired()
        start = time.perf_counter()
        cpu_start = time.process_time()
        output = fn()
        cpu_elapsed = time.process_time() - cpu_start
        elapsed = time.perf_counter() - start
        if index >= warmup:
            durations.append(elapsed)
            cpu_durations.append(cpu_elapsed)
    return output, dict(seconds=durations, median_seconds=statistics.median(durations),
                        minimum_seconds=min(durations), cpu_seconds=cpu_durations,
                        median_cpu_seconds=statistics.median(cpu_durations),
                        aggregate_cpu_to_wall_ratio=sum(cpu_durations) / sum(durations))


def variant(torch, args, spec, module, inputs, reference, dtype_name, deadline):
    row = dict(**spec, dtype=dtype_name, status="error", platform_at_start=platform_sample())
    started = time.perf_counter()
    try:
        if time.monotonic() >= deadline:
            raise BudgetExpired()
        if dtype_name == "qint8":
            if args.quantized_engine not in torch.backends.quantized.supported_engines:
                raise NotImplementedError(f"Quantized engine unavailable: {args.quantized_engine}")
            torch.backends.quantized.engine = args.quantized_engine
            selected = torch.ao.quantization.quantize_dynamic(
                module, {torch.nn.Linear}, dtype=torch.qint8, inplace=False)
            if not isinstance(selected[0], torch.ao.nn.quantized.dynamic.Linear):
                raise RuntimeError("quantize_dynamic did not replace the Linear")
            row["quantized_engine"] = torch.backends.quantized.engine
            row["weight_dtype"] = str(selected[0].weight().dtype)
            arguments = inputs
        else:
            dtype = getattr(torch, dtype_name)
            selected = copy.deepcopy(module).to(dtype=dtype) if isinstance(module, torch.nn.Module) else module
            arguments = tuple(value.to(dtype=dtype) for value in inputs)
        row["prepare_seconds"] = time.perf_counter() - started
        result, row["native_dtype"] = timed(lambda: selected(*arguments),
            args.warmup, args.repeats, deadline)
        row["output_dtype"] = str(result.dtype)
        finite = bool(torch.isfinite(result).all())
        row["finite_output"] = finite
        if not finite:
            raise RuntimeError("Kernel output contains non-finite values")
        difference = result.float() - reference
        row["max_absolute_error_vs_float32"] = float(difference.abs().max())
        row["rmse_vs_float32"] = float(difference.square().mean().sqrt())
        row["relative_l2_error_vs_float32"] = float(
            torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference).clamp_min(1e-12))
        if dtype_name in ("float16", "bfloat16"):
            # Includes activation/output casts, excludes the once-only weight cast.
            def float32_io():
                return selected(*(value.to(dtype=getattr(torch, dtype_name)) for value in inputs)).float()
            _, row["float32_io"] = timed(float32_io, args.warmup, args.repeats, deadline)
        else:
            # Dynamic INT8 already quantizes FP32 inputs and returns FP32 outputs.
            row["float32_io"] = dict(row["native_dtype"])
        row["status"] = "supported"
    except BudgetExpired:
        row["status"] = "budget_exhausted"
    except (RuntimeError, NotImplementedError, TypeError) as error:
        row["status"] = "unsupported" if isinstance(error, NotImplementedError) or any(
            token in str(error).lower() for token in ("not implemented", "not supported", "unsupported")) else "error"
        row["error"] = f"{type(error).__name__}: {error}"
    row["platform_at_end"] = platform_sample()
    row["elapsed_seconds"] = time.perf_counter() - started
    return row


def worker(args):
    start = time.monotonic()
    deadline = start + args.max_seconds
    import torch
    from threadpoolctl import threadpool_info
    before = threadpool_info()
    torch.set_num_threads(args.threads[0])
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    report = dict(status="running", machine=platform.machine(), python=platform.python_version(),
        torch=str(torch.__version__), torch_build=torch.__config__.show(), cpu_features=cpu_features(),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        threadpools_before_torch_set_num_threads=before, threadpools_at_start=threadpool_info(),
        environment={name: os.environ.get(name) for name in
            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS")},
        quantized_engines=list(torch.backends.quantized.supported_engines),
        pretrained_weights_used=False, full_model_validated=False, accuracy_validated=False,
        warmup=args.warmup, repeats=args.repeats, seed=args.seed, cases=[],
        limitations="Synthetic tensors and fixed cached weights; conversion/packing reported separately; default CPU SDPA dispatch; no CPU dtype fallback; no end-to-end speed or accuracy inference")
    write_json(args.output, report)
    try:
        with torch.inference_mode():
            for spec in cases(args):
                if time.monotonic() >= deadline:
                    raise BudgetExpired()
                module, inputs = build_case(torch, spec)
                reference = module(*inputs).float()
                names = list(args.dtypes)
                if args.int8 and spec["op"] == "linear":
                    names.insert(1, "qint8")
                baseline = None
                for name in names:
                    row = variant(torch, args, spec, module, inputs, reference, name, deadline)
                    if name == "float32" and row["status"] == "supported":
                        baseline = row["native_dtype"]["median_seconds"]
                    if baseline is not None and row["status"] == "supported":
                        row["native_speedup_vs_float32"] = baseline / row["native_dtype"]["median_seconds"]
                        row["float32_io_speedup_vs_float32"] = baseline / row["float32_io"]["median_seconds"]
                    report["cases"].append(row)
                    write_json(args.output, report)
                    print(json.dumps(dict(threads=report["torch_threads"], **row)), flush=True)
                    if row["status"] == "budget_exhausted":
                        raise BudgetExpired()
                del module, inputs, reference
        report["status"] = "completed"
    except BudgetExpired:
        report["status"] = "budget_exhausted"
    finally:
        report["seconds"] = time.monotonic() - start
        report["threadpools_at_end"] = threadpool_info()
        write_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ops", nargs="+", choices=["linear", "sdpa", "layernorm", "gelu", "conv2d"], default=["linear"])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--rows", nargs="+", type=int, default=[128, 1024])
    parser.add_argument("--sdpa-tokens", nargs="+", type=int, default=[37, 782])
    parser.add_argument("--dtypes", nargs="+", choices=["float32", "float16", "bfloat16"], default=["float32", "float16", "bfloat16"])
    parser.add_argument("--int8", action="store_true", help="Also try dynamic quantized Linear; no fallback")
    parser.add_argument("--quantized-engine", default="qnnpack")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seconds", type=float, default=55, help="Total soft wall-clock budget including child imports")
    parser.add_argument("--describe", action="store_true", help="List cases without importing Torch or running kernels")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if any(number < 1 for number in args.threads + args.rows + args.sdpa_tokens) or args.warmup < 0 or args.repeats < 1:
        parser.error("threads, rows, tokens and repeats must be positive; warmup nonnegative")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("max-seconds must be positive and finite")
    args.dtypes = list(dict.fromkeys(["float32", *args.dtypes]))
    if args.describe:
        print(json.dumps(dict(threads=args.threads, dtypes=args.dtypes, int8=args.int8,
                             max_seconds=args.max_seconds, cases=cases(args)), indent=2))
        return
    if args.output is None or args.output.exists():
        parser.error("--output must name a new JSON file")
    if args.worker:
        if len(args.threads) != 1:
            parser.error("worker requires one thread count")
        worker(args)
        return
    start = time.monotonic()
    report = dict(status="running", full_model_validated=False, accuracy_validated=False, workers=[])
    parts = args.output.parent / (args.output.stem + ".workers")
    parts.mkdir(parents=True, exist_ok=True)
    for index, threads in enumerate(args.threads):
        remaining = args.max_seconds - (time.monotonic() - start)
        if remaining <= 0:
            report["status"] = "budget_exhausted"
            break
        target = parts / f"threads-{threads}.json"
        budget = remaining / (len(args.threads) - index)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--threads", str(threads),
            "--output", str(target), "--ops", *args.ops, "--rows", *map(str, args.rows),
            "--sdpa-tokens", *map(str, args.sdpa_tokens), "--dtypes", *args.dtypes,
            "--warmup", str(args.warmup), "--repeats", str(args.repeats), "--seed", str(args.seed),
            "--quantized-engine", args.quantized_engine, "--max-seconds", str(budget)]
        if args.int8:
            command.append("--int8")
        environment = dict(os.environ, OMP_NUM_THREADS=str(threads))
        try:
            process = subprocess.run(command, env=environment, timeout=budget + 2)
            if target.is_file():
                entry = json.loads(target.read_text())
            else:
                entry = dict(status="failed", error="Worker produced no report")
            entry["worker_exit_code"] = process.returncode
        except subprocess.TimeoutExpired:
            entry = json.loads(target.read_text()) if target.is_file() else {}
            entry.update(status="timeout", requested_threads=threads)
        report["workers"].append(entry)
        write_json(args.output, report)
    if report["status"] == "running":
        report["status"] = "completed" if all(worker["status"] == "completed" for worker in report["workers"]) else "partial"
    report["seconds"] = time.monotonic() - start
    write_json(args.output, report)


if __name__ == "__main__":
    main()
