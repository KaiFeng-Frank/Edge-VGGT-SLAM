#!/usr/bin/env python3
"""Bounded, independent probe of Torch 2.3.1's pinned XNNPACK FP16 ABI.

No model or fallback is used. Synthetic error does not establish SLAM accuracy.
Weights/bias are FP16 [N,K]/[N]; both XNNPACK cache arguments are NULL.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


XNNPACK_COMMIT = "fcbf55af6cf28a4627bcd1f703ab7ad843f0f3a2"
PTHREADPOOL_COMMIT = "4fe0e1e183925bf8cfa6aae24237e724a96479b8"
HEADER_SHA256 = {
    "xnnpack.h": "62a48ebae3f2d77c41989eadf4a6a3c97a68af9126777323da73e9059e728b02",
    "pthreadpool.h": "afae65ccebb810dde2a3370875df42f89ef2e43502a3430208d7b6e980f4b74c",
}
EXTRA_BYTES = 16
YIELD_WORKERS = 0x10
STATUS_NAMES = {0: "success", 1: "uninitialized", 2: "invalid_parameter",
                3: "invalid_state", 4: "unsupported_parameter",
                5: "unsupported_hardware", 6: "out_of_memory",
                7: "reallocation_required"}
SHAPES = ((1024, 3072), (1024, 1024), (1024, 4096), (4096, 1024))


class BudgetExpired(Exception):
    pass


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_budget(deadline):
    if time.monotonic() >= deadline:
        raise BudgetExpired("soft wall budget exhausted")


def measured(fn):
    start, cpu_start = time.perf_counter(), time.process_time()
    value = fn()
    wall, cpu = time.perf_counter() - start, time.process_time() - cpu_start
    return value, dict(wall_seconds=wall, cpu_seconds=cpu,
                       cpu_to_wall_ratio=cpu / wall if wall else 0.0)


def summarize(samples):
    wall = [s["wall_seconds"] for s in samples]
    cpu = [s["cpu_seconds"] for s in samples]
    return dict(wall_seconds=wall, cpu_seconds=cpu,
                median_wall_seconds=statistics.median(wall),
                minimum_wall_seconds=min(wall),
                aggregate_cpu_to_wall_ratio=sum(cpu) / sum(wall))


def repeated(fn, args, deadline):
    samples, value = [], None
    for index in range(args.warmup + args.repeats):
        check_budget(deadline)
        value, sample = measured(fn)
        if index >= args.warmup:
            samples.append(sample)
    return value, summarize(samples)


def error_metrics(torch, value, reference):
    actual = value.float()
    finite = bool(torch.isfinite(actual).all())
    if not finite:
        return {"finite": False}
    difference = actual - reference
    return dict(finite=True, max_abs=float(difference.abs().max()),
                rmse=float(difference.square().mean().sqrt()),
                relative_l2=float(difference.norm() / reference.norm().clamp_min(1e-12)))


class Native:
    """Signatures copied from exact installed header; no guessed parameters."""
    def __init__(self, torch):
        torch_root = Path(torch.__file__).resolve().parent
        self.headers = {}
        for name, expected in HEADER_SHA256.items():
            path = torch_root / "include" / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.headers[name] = dict(path=str(path), sha256=actual)
            if actual != expected:
                raise RuntimeError(f"Unverified ABI: {name} SHA256 {actual} != {expected}")
        self.library = torch_root / "lib" / "libtorch_cpu.so"
        self.lib = ctypes.CDLL(str(self.library))
        void, size = ctypes.c_void_p, ctypes.c_size_t
        float_pointer = ctypes.POINTER(ctypes.c_float)
        signatures = {
            "xnn_initialize": ([void], ctypes.c_int),
            "xnn_deinitialize": ([], ctypes.c_int),
            "xnn_create_fully_connected_nc_f16": (
                [size, size, size, size, void, void, ctypes.c_float, ctypes.c_float,
                 ctypes.c_uint32, void, void, ctypes.POINTER(void)], ctypes.c_int),
            "xnn_reshape_fully_connected_nc_f16": ([void, size, void], ctypes.c_int),
            "xnn_setup_fully_connected_nc_f16": ([void, void, void], ctypes.c_int),
            "xnn_create_fully_connected_nc_f32": (
                [size, size, size, size, float_pointer, float_pointer,
                 ctypes.c_float, ctypes.c_float, ctypes.c_uint32,
                 void, void, ctypes.POINTER(void)], ctypes.c_int),
            "xnn_reshape_fully_connected_nc_f32": ([void, size, void], ctypes.c_int),
            "xnn_setup_fully_connected_nc_f32": (
                [void, float_pointer, float_pointer], ctypes.c_int),
            "xnn_run_operator": ([void, void], ctypes.c_int),
            "xnn_delete_operator": ([void], ctypes.c_int),
            "pthreadpool_create": ([size], void),
            "pthreadpool_get_threads_count": ([void], size),
            "pthreadpool_destroy": ([void], None),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, result
        self.check("xnn_initialize", None)

    def check(self, name, *arguments):
        status = getattr(self.lib, name)(*arguments)
        if status:
            raise RuntimeError(f"{name}: {STATUS_NAMES.get(status, 'unknown')} ({status})")

    def create(self, width, outputs, weight, bias):
        operator = ctypes.c_void_p()
        self.check("xnn_create_fully_connected_nc_f16", width, outputs, width, outputs,
                   weight.data_ptr(), bias.data_ptr(), -math.inf, math.inf,
                   YIELD_WORKERS, None, None, ctypes.byref(operator))
        if not operator.value:
            raise RuntimeError("XNNPACK returned a null operator on success")
        return operator


def padded(torch, shape, fill=0.0, dtype=None):
    count = math.prod(shape)
    dtype = dtype or torch.float16
    extra_elements = EXTRA_BYTES // torch.empty((), dtype=dtype).element_size()
    storage = torch.full((count + extra_elements,), fill, dtype=dtype)
    return storage, storage[:count].view(shape)


def trial_f32(torch, native, args, pool, rows, width, outputs,
              inputs, weight, bias, reference, deadline):
    """FP32 prepacked control on the very same pthreadpool as the FP16 trial."""
    result = dict(status="error", dtype="float32", create_flags=YIELD_WORKERS,
                  code_cache=None, weights_cache=None)
    operator = None
    pointer = lambda tensor: ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))
    try:
        check_budget(deadline)
        input_storage, input_view = padded(torch, (rows, width), dtype=torch.float32)
        output_storage, output_view = padded(torch, (rows, outputs), fill=123.5, dtype=torch.float32)
        weight_storage, weight_view = padded(torch, (outputs, width), dtype=torch.float32)
        bias_storage, bias_view = padded(torch, (outputs,), dtype=torch.float32)

        def prepare():
            input_view.copy_(inputs)
            weight_view.copy_(weight)
            bias_view.copy_(bias)

        _, result["buffer_copy_once"] = measured(prepare)

        def create():
            op = ctypes.c_void_p()
            native.check("xnn_create_fully_connected_nc_f32", width, outputs, width, outputs,
                         pointer(weight_storage), pointer(bias_storage), -math.inf, math.inf,
                         YIELD_WORKERS, None, None, ctypes.byref(op))
            if not op.value:
                raise RuntimeError("XNNPACK returned a null FP32 operator on success")
            return op

        operator, result["create_pack_once"] = measured(create)

        def bind():
            native.check("xnn_reshape_fully_connected_nc_f32", operator, rows, pool)
            native.check("xnn_setup_fully_connected_nc_f32", operator,
                         pointer(input_storage), pointer(output_storage))

        _, result["reshape_setup"] = measured(bind)
        _, result["native_fp32"] = repeated(
            lambda: native.check("xnn_run_operator", operator, pool), args, deadline)
        result["error_vs_torch_fp32"] = error_metrics(torch, output_view, reference)
        result["output_guard_preserved"] = bool((output_storage[rows * outputs:] == 123.5).all())
        result["status"] = "ok" if (result["error_vs_torch_fp32"]["finite"] and
            result["error_vs_torch_fp32"]["max_abs"] <= args.max_abs and
            result["error_vs_torch_fp32"]["relative_l2"] <= args.max_relative_l2 and
            result["output_guard_preserved"]) else "numerical_or_guard_error"
    finally:
        if operator is not None:
            native.check("xnn_delete_operator", operator)
    return result


def trial(torch, native, args, rows, width, outputs, inputs, weight, bias,
          reference, rounded_reference, weight_b, bias_b, reference_b,
          threads, deadline):
    result = dict(threads_requested=threads, status="error", create_flags=YIELD_WORKERS,
                  weights_format="FP16 contiguous [N,K]", bias_format="FP16 [N]",
                  code_cache=None, weights_cache=None)
    pool, operator = None, None
    try:
        check_budget(deadline)
        pool = native.lib.pthreadpool_create(threads)
        if not pool:
            raise RuntimeError("pthreadpool_create failed")
        result["threads_actual"] = native.lib.pthreadpool_get_threads_count(pool)
        input_storage, input_half = padded(torch, (rows, width))
        output_storage, output_half = padded(torch, (rows, outputs), fill=123.5)
        weight_storage, weight_half = padded(torch, (outputs, width))
        bias_storage, bias_half = padded(torch, (outputs,))

        def convert_weights_a():
            weight_half.copy_(weight)
            bias_half.copy_(bias)

        _, result["weights_bias_convert_a"] = measured(convert_weights_a)
        _, result["initial_input_convert"] = measured(lambda: input_half.copy_(inputs))
        operator, result["create_pack_a"] = measured(
            lambda: native.create(width, outputs, weight_storage, bias_storage))

        def reshape_setup(batch_rows=rows, output=output_storage):
            native.check("xnn_reshape_fully_connected_nc_f16", operator, batch_rows, pool)
            native.check("xnn_setup_fully_connected_nc_f16", operator,
                         input_storage.data_ptr(), output.data_ptr())

        _, result["initial_reshape_setup"] = measured(reshape_setup)
        _, result["native_fp16"] = repeated(
            lambda: native.check("xnn_run_operator", operator, pool), args, deadline)
        result["error_vs_fp32"] = error_metrics(torch, output_half, reference)
        result["error_vs_rounded_fp16_fp32_math"] = error_metrics(
            torch, output_half, rounded_reference)
        result["output_guard_preserved"] = bool((output_storage[rows * outputs:] == 123.5).all())

        phases = {name: [] for name in ("input_cast", "reshape_setup", "run", "output_cast")}

        def float32_io():
            _, a = measured(lambda: input_half.copy_(inputs))
            _, b = measured(reshape_setup)
            _, c = measured(lambda: native.check("xnn_run_operator", operator, pool))
            output, d = measured(lambda: output_half.float())
            for name, sample in zip(phases, (a, b, c, d)):
                phases[name].append(sample)
            return output

        actual, result["float32_io"] = repeated(float32_io, args, deadline)
        result["float32_io_phases"] = {
            name: summarize(samples[args.warmup:]) for name, samples in phases.items()}
        result["float32_io_error_vs_fp32"] = error_metrics(torch, actual, reference)

        # Reuse the packed operator for a smaller, non-tile-aligned batch.
        check_budget(deadline)
        tail_rows = min(args.tail_rows, rows)
        tail_storage, tail_half = padded(torch, (tail_rows, outputs), fill=123.5)
        _, result["tail_reshape_setup"] = measured(lambda: reshape_setup(tail_rows, tail_storage))
        _, result["tail_run"] = measured(lambda: native.check("xnn_run_operator", operator, pool))
        result["tail_rows"] = tail_rows
        result["tail_error_vs_fp32"] = error_metrics(torch, tail_half, reference[:tail_rows])
        result["tail_output_guard_preserved"] = bool(
            (tail_storage[tail_rows * outputs:] == 123.5).all())

        # Static weights require a new operator, not merely another setup call.
        native.check("xnn_delete_operator", operator)
        operator = None
        check_budget(deadline)

        def convert_weights_b():
            weight_half.copy_(weight_b)
            bias_half.copy_(bias_b)

        _, result["weights_bias_convert_b"] = measured(convert_weights_b)
        operator, result["create_pack_b"] = measured(
            lambda: native.create(width, outputs, weight_storage, bias_storage))
        _, result["rebuild_b_reshape_setup"] = measured(reshape_setup)
        _, result["rebuild_b_run"] = measured(lambda: native.check("xnn_run_operator", operator, pool))
        result["rebuild_b_error_vs_fp32"] = error_metrics(torch, output_half, reference_b)
        result["rebuild_b_difference_from_a_max_abs"] = float((output_half.float() - reference).abs().max())
        result["rebuild_b_output_guard_preserved"] = bool(
            (output_storage[rows * outputs:] == 123.5).all())
        result["status"] = "ok"
        result["numeric_smoke_limits"] = dict(max_abs=args.max_abs,
                                              relative_l2=args.max_relative_l2)
        result["numeric_checks_passed"] = True
        for key in ("error_vs_fp32", "error_vs_rounded_fp16_fp32_math",
                    "tail_error_vs_fp32", "rebuild_b_error_vs_fp32"):
            if not result[key]["finite"]:
                result["status"] = "nonfinite_output"
                result["numeric_checks_passed"] = False
            elif (result[key]["max_abs"] > args.max_abs or
                  result[key]["relative_l2"] > args.max_relative_l2):
                result["status"] = "numerical_limit_exceeded"
                result["numeric_checks_passed"] = False
        if not all(result[key] for key in ("output_guard_preserved",
                   "tail_output_guard_preserved", "rebuild_b_output_guard_preserved")):
            result["status"] = "output_guard_corrupted"
        if width == 1024 and outputs == 3072:
            result["f32_same_pool_control"] = trial_f32(torch, native, args, pool,
                rows, width, outputs, inputs, weight, bias, reference, deadline)
            if result["f32_same_pool_control"]["status"] != "ok":
                result["status"] = "f32_control_error"
    except BudgetExpired as error:
        result["status"], result["error"] = "budget_exhausted", str(error)
    except Exception as error:
        result["error_type"], result["error"] = type(error).__name__, str(error)
    finally:
        if operator is not None:
            try:
                native.check("xnn_delete_operator", operator)
            except Exception as error:
                result["cleanup_error"] = str(error)
        if pool:
            native.lib.pthreadpool_destroy(pool)
    return result


def worker(args):
    start, deadline = time.monotonic(), time.monotonic() + args.max_seconds
    report = dict(status="running", backend="XNNPACK FP16 fully_connected_nc_f16",
                  synthetic_only=True, no_fallback=True, cases=[],
                  xnnpack_commit=XNNPACK_COMMIT, pthreadpool_commit=PTHREADPOOL_COMMIT,
                  arguments={k: str(v) if isinstance(v, Path) else v
                             for k, v in vars(args).items() if k != "worker"},
                  environment={k: os.environ.get(k) for k in
                               ("OMP_NUM_THREADS", "OMP_WAIT_POLICY", "OPENBLAS_NUM_THREADS")})
    native = None
    try:
        import torch
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(args.seed)
        report["torch_version"] = torch.__version__
        report["torch_threads"] = torch.get_num_threads()
        report["affinity"] = sorted(os.sched_getaffinity(0))
        report["machine"] = platform.machine()
        native = Native(torch)
        report["verified_installed_headers"] = native.headers
        report["library"] = str(native.library)
        write_json(args.output, report)
        with torch.inference_mode():
            for rows in args.rows:
                for width, outputs in SHAPES:
                    check_budget(deadline)
                    inputs = torch.randn(rows, width)
                    weight = torch.randn(outputs, width) / math.sqrt(width)
                    bias = torch.randn(outputs) * 0.1
                    weight_b, bias_b = weight * -0.75, bias + 0.125
                    linear = lambda: torch.nn.functional.linear(inputs, weight, bias)
                    reference, baseline = repeated(linear, args, deadline)
                    reference_b = torch.nn.functional.linear(inputs, weight_b, bias_b)
                    rounded = torch.nn.functional.linear(inputs.half().float(),
                        weight.half().float(), bias.half().float())
                    item = dict(shape=[rows, width, outputs], fp32_baseline=baseline,
                                baseline_backend="torch.nn.functional.linear", trials=[])
                    report["cases"].append(item)
                    write_json(args.output, report)
                    for threads in args.threads:
                        check_budget(deadline)
                        result = trial(torch, native, args, rows, width, outputs,
                            inputs, weight, bias, reference, rounded,
                            weight_b, bias_b, reference_b, threads, deadline)
                        item["trials"].append(result)
                        write_json(args.output, report)
                        print(json.dumps(dict(shape=item["shape"], threads=threads,
                            status=result["status"], native=result.get("native_fp16"),
                            float32_io=result.get("float32_io"), error=result.get("error"))), flush=True)
                        if result["status"] == "budget_exhausted":
                            raise BudgetExpired(result["error"])
        report["status"] = "ok" if all(t["status"] == "ok" for c in report["cases"]
                                                for t in c["trials"]) else "errors"
    except BudgetExpired as error:
        report["status"], report["error"] = "partial", str(error)
    except Exception as error:
        report["status"] = "error"
        report["error_type"], report["error"] = type(error).__name__, str(error)
    finally:
        if native:
            try:
                native.check("xnn_deinitialize")
            except Exception as error:
                report["deinitialize_error"] = str(error)
        report["elapsed_wall_seconds"] = time.monotonic() - start
        write_json(args.output, report)
    return 0 if report["status"] in ("ok", "partial") else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path)
    result.add_argument("--rows", type=int, nargs="+", default=[128])
    result.add_argument("--threads", type=int, nargs="+", default=[1, 3, 4])
    result.add_argument("--torch-threads", type=int, default=4)
    result.add_argument("--tail-rows", type=int, default=17)
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--max-abs", type=float, default=0.25,
                        help="Synthetic numeric smoke limit; not a model accuracy guarantee")
    result.add_argument("--max-relative-l2", type=float, default=0.03)
    result.add_argument("--max-seconds", type=float, default=27)
    result.add_argument("--describe", action="store_true")
    result.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return result


def main():
    arg_parser = parser()
    args = arg_parser.parse_args()
    if args.describe:
        print(json.dumps(dict(shapes=[[m, k, n] for m in args.rows for k, n in SHAPES],
            pthreadpool_threads=args.threads, tail_rows=args.tail_rows,
            verified_header_sha256=HEADER_SHA256, caches="NULL/NULL",
            create_flags=YIELD_WORKERS, extra_bytes=EXTRA_BYTES), indent=2))
        return 0
    if not args.output:
        arg_parser.error("--output is required")
    if any(n < 1 for n in args.rows + args.threads + [args.torch_threads, args.tail_rows, args.repeats]):
        arg_parser.error("rows, threads, tail rows and repeats must be positive")
    if args.warmup < 0 or args.max_seconds <= 0:
        arg_parser.error("invalid warmup or wall budget")
    if args.worker:
        return worker(args)
    if args.output.exists():
        arg_parser.error("output already exists; use a new filename")
    # Initialize OpenMP policy before Torch is imported in the worker.
    environment = os.environ.copy()
    environment["OMP_WAIT_POLICY"] = "PASSIVE"
    environment["OMP_NUM_THREADS"] = str(args.torch_threads)
    try:
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
        return subprocess.run(command, env=environment, timeout=args.max_seconds + 2).returncode
    except subprocess.TimeoutExpired:
        report = json.loads(args.output.read_text()) if args.output.exists() else {}
        report.update(status="partial", coordinator_timeout=True,
                      error="worker exceeded wall budget; only recorded completed trials are valid")
        write_json(args.output, report)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
