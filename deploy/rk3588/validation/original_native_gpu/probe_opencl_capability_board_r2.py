#!/usr/bin/env python3
"""Bounded RK3588/S OpenCL discovery; no context, kernel, buffer or model run."""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

CL_SUCCESS = 0
CL_DEVICE_NOT_FOUND = -1
CL_PLATFORM_NOT_FOUND_KHR = -1001
CL_DEVICE_TYPE_GPU = 1 << 2


def source_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def board_identity():
    compatible = next((p.read_bytes() for p in (
        Path('/proc/device-tree/compatible'),
        Path('/sys/firmware/devicetree/base/compatible')) if p.is_file()), b'')
    if platform.machine().lower() not in ('aarch64', 'arm64') or b'rk3588' not in compatible:
        raise RuntimeError('This discovery probe is restricted to the connected RK3588/S board')
    return {'machine': platform.machine(),
            'compatible': compatible.replace(b'\0', b',').decode(errors='replace')}


def discover(library_name):
    lib = ct.CDLL(library_name)
    signatures = {
        'clGetPlatformIDs': ([ct.c_uint32, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_uint32)], ct.c_int32),
        'clGetPlatformInfo': ([ct.c_void_p, ct.c_uint32, ct.c_size_t, ct.c_void_p, ct.POINTER(ct.c_size_t)], ct.c_int32),
        'clGetDeviceIDs': ([ct.c_void_p, ct.c_uint64, ct.c_uint32, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_uint32)], ct.c_int32),
        'clGetDeviceInfo': ([ct.c_void_p, ct.c_uint32, ct.c_size_t, ct.c_void_p, ct.POINTER(ct.c_size_t)], ct.c_int32),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result

    def check(code, operation):
        if code != CL_SUCCESS:
            raise RuntimeError(f'{operation} returned {code}')

    def info(fn, handle, key, kind=None):
        if kind is not None:
            value = kind()
            check(fn(handle, key, ct.sizeof(value), ct.byref(value), None), 'device scalar query')
            return int(value.value)
        count = ct.c_size_t()
        check(fn(handle, key, 0, None, ct.byref(count)), 'discovery string size query')
        if not 0 < count.value <= 32768:
            raise RuntimeError(f'Unexpected discovery string size {count.value}')
        buf = ct.create_string_buffer(count.value)
        check(fn(handle, key, count.value, buf, None), 'discovery string query')
        return buf.value.decode('utf-8', errors='replace')

    count = ct.c_uint32()
    code = lib.clGetPlatformIDs(0, None, ct.byref(count))
    if code == CL_PLATFORM_NOT_FOUND_KHR or (code == CL_SUCCESS and count.value == 0):
        return {'status': 'no_opencl_platform', 'library': library_name, 'platforms': [],
                'gpu_device_count': 0}
    check(code, 'clGetPlatformIDs(count)')
    if not 0 < count.value <= 16:
        raise RuntimeError(f'Unexpected platform count {count.value}')
    platforms = (ct.c_void_p * count.value)()
    check(lib.clGetPlatformIDs(count.value, platforms, None), 'clGetPlatformIDs(list)')
    report = {'status': 'discovered', 'library': library_name, 'platforms': [], 'gpu_device_count': 0}
    strings = {'name': 0x102B, 'vendor': 0x102C, 'driver_version': 0x102D,
               'device_version': 0x102F, 'extensions': 0x1030, 'opencl_c_version': 0x103D}
    scalars = {'compute_units': (0x1002, ct.c_uint32),
               'reported_max_clock_MHz': (0x100C, ct.c_uint32),
               'global_mem_bytes': (0x101F, ct.c_uint64),
               'max_allocation_bytes': (0x1010, ct.c_uint64),
               'memory_base_alignment_bits': (0x1019, ct.c_uint32),
               'max_work_group_size': (0x1004, ct.c_size_t),
               'available': (0x1027, ct.c_uint32),
               'compiler_available': (0x1028, ct.c_uint32),
               'host_unified_memory': (0x1035, ct.c_uint32)}
    for handle in platforms:
        row = {k: info(lib.clGetPlatformInfo, handle, key) for k, key in
               {'name': 0x0902, 'vendor': 0x0903, 'version': 0x0901}.items()}
        row['gpu_devices'] = []
        devices_count = ct.c_uint32()
        code = lib.clGetDeviceIDs(handle, CL_DEVICE_TYPE_GPU, 0, None, ct.byref(devices_count))
        if code == CL_DEVICE_NOT_FOUND:
            report['platforms'].append(row)
            continue
        check(code, 'clGetDeviceIDs(count)')
        if not 0 < devices_count.value <= 32:
            raise RuntimeError(f'Unexpected GPU device count {devices_count.value}')
        devices = (ct.c_void_p * devices_count.value)()
        check(lib.clGetDeviceIDs(handle, CL_DEVICE_TYPE_GPU, devices_count.value, devices, None),
              'clGetDeviceIDs(list)')
        for device in devices:
            item = {k: info(lib.clGetDeviceInfo, device, key) for k, key in strings.items()}
            item.update({k: info(lib.clGetDeviceInfo, device, key, kind)
                         for k, (key, kind) in scalars.items()})
            extensions = set(item['extensions'].split())
            item['extensions'] = sorted(extensions)
            item['has_arm_import_memory_extension'] = 'cl_arm_import_memory' in extensions
            item['has_arm_dma_buf_import_extension'] = 'cl_arm_import_memory_dma_buf' in extensions
            item['has_fp16_extension'] = 'cl_khr_fp16' in extensions
            import_address = None
            get_extension = getattr(lib, 'clGetExtensionFunctionAddressForPlatform', None)
            if get_extension is not None:
                get_extension.argtypes, get_extension.restype = [ct.c_void_p, ct.c_char_p], ct.c_void_p
                import_address = get_extension(handle, b'clImportMemoryARM')
            item['arm_import_function_address_available'] = bool(import_address)
            item['arm_import_library_symbol_available'] = hasattr(lib, 'clImportMemoryARM')
            item['DMA_buf_import_or_kernel_compilation_actually_tested'] = False
            row['gpu_devices'].append(item)
            report['gpu_device_count'] += 1
        report['platforms'].append(row)
    if not report['gpu_device_count']:
        report['status'] = 'no_gpu_device'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--library', default='libOpenCL.so.1')
    parser.add_argument('--timeout', type=float, default=8.0)
    parser.add_argument('--_child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.timeout <= 15:
        parser.error('--timeout must be in (0, 15] seconds')
    if args._child:
        try:
            board_identity()
            print(json.dumps(discover(args.library), sort_keys=True), flush=True)
            return 0
        except BaseException as error:
            print(json.dumps({'status': 'failed', 'error': f'{type(error).__name__}: {error}'}), flush=True)
            return 1
    if args.output is None:
        parser.error('--output is required')
    identity = board_identity()
    path = args.output.resolve()
    if path.exists():
        raise FileExistsError('Keep previous probe results: ' + str(path))
    report = {'schema': 'original_VGGT_OpenCL_capability_discovery_only',
              'source_sha256': source_sha(), 'board': identity,
              'model_or_MatMul_or_kernel_execution': False,
              'context_or_command_queue_or_buffer_created': False,
              'frequency_or_governor_changed': False,
              'timeout_seconds': args.timeout,
              'extension_presence_does_not_prove_import_or_performance': True}
    started = time.monotonic()
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--_child',
                                '--library', args.library], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=args.timeout)
        report['child_exit_code'] = process.returncode
        try:
            report['discovery'] = json.loads(stdout.decode())
        except (UnicodeError, json.JSONDecodeError):
            report['child_stdout_tail'] = stdout[-32768:].decode(errors='replace')
            rows = []
            for line in stdout.decode(errors='replace').splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and 'status' in item:
                    rows.append(item)
            report['discovery'] = rows[0] if len(rows) == 1 else {'status': 'invalid_child_output'}
            report['native_library_extra_stdout_preserved'] = True
        if stderr:
            report['child_stderr_tail'] = stderr[-32768:].decode(errors='replace')
        report['status'] = ('completed_discovery_only' if process.returncode == 0 and report['discovery'].get('status') in ('discovered', 'no_opencl_platform', 'no_gpu_device') else 'failed_discovery')
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        report.update(status='timeout_child_killed', child_exit_code=process.returncode,
                      child_stdout_tail=stdout[-32768:].decode(errors='replace'),
                      child_stderr_tail=stderr[-32768:].decode(errors='replace'))
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        report['child_terminal_confirmed'] = process.poll() is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as output:
            json.dump(report, output, indent=2, sort_keys=True)
            output.write('\n')
        print(json.dumps({'status': report['status'], 'output': str(path),
                          'seconds': report['elapsed_seconds']}), flush=True)
    return 0 if report['status'] == 'completed_discovery_only' else 1


if __name__ == '__main__':
    raise SystemExit(main())
