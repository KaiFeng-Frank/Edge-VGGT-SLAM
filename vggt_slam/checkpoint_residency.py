"""Explicit self-process checkpoint residency experiment for original VGGT.

The caller records stat before the pinned loader and calls this only after that
loader completed its original SHA verification and a backend installation.
No tensor data is read here, and no alternative advice or retry is permitted.
"""
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import time

ORIGINAL_BYTES = 5026874952
ORIGINAL_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
SOURCE_PINS = {
    'runtime.py': 'f36bb5b87fbdbf03984ea3e1466e6ee27865a540ae3e5abd4ee07a45538e23f5',
    'checkpoint_identity.py': '9fc0b927f6a68ac6ca55194616104febb1c488370593d4be60e8dba5bc336461',
}
MADV_PAGEOUT = 21
HEADER = re.compile(r'^([0-9a-f]+)-([0-9a-f]+)\s+(\S+)\s+([0-9a-f]+)\s+([0-9a-f]+):([0-9a-f]+)\s+(\d+)(?:\s+(.*))?$')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, row):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def checkpoint_stat(path):
    path = Path(path).resolve(strict=True)
    value = path.stat()
    require(stat.S_ISREG(value.st_mode) and value.st_size == ORIGINAL_BYTES,
            'Pinned original regular checkpoint size required')
    return dict(path=str(path), dev=value.st_dev, inode=value.st_ino, bytes=value.st_size,
                mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)


def parse_smaps(text):
    rows, row = [], None
    for line in text.splitlines():
        match = HEADER.fullmatch(line)
        if match:
            a, b, permissions, offset, major, minor, inode, name = match.groups()
            row = dict(start=int(a, 16), end=int(b, 16), permissions=permissions,
                       offset=int(offset, 16), dev_major=int(major, 16), dev_minor=int(minor, 16),
                       inode=int(inode), path=name or '', metrics={})
            rows.append(row)
        elif row is not None and ':' in line:
            key, value = line.split(':', 1)
            fields = value.split()
            if key == 'VmFlags':
                row[key] = fields
            elif fields and fields[0].isdigit():
                row['metrics'][key] = int(fields[0])
    return rows


def selected(rows, identity):
    return [r for r in rows if (r['dev_major'], r['dev_minor'], r['inode']) ==
            (os.major(identity['dev']), os.minor(identity['dev']), identity['inode'])]


def validate_vmas(rows, identity, page_size):
    require(rows, 'Original checkpoint has no matching self file VMA; mmap is required')
    forbidden = {'lo', 'io', 'pf', 'ht', 'mm'}
    for row in rows:
        require(row in selected([row], identity) and row['path'] == identity['path']
                and row['permissions'] in ('r--p', 'rw-p'), 'Original private checkpoint VMA identity differs')
        start, end, offset = row['start'], row['end'], row['offset']
        require(0 < start < end and start % page_size == end % page_size == offset % page_size == 0
                and offset >= 0 and offset + end - start <= ((identity['bytes'] + page_size - 1) // page_size) * page_size,
                'Checkpoint VMA alignment or file offset extent differs')
        metrics = row['metrics']
        require(all(k in metrics for k in ('Size', 'Rss', 'Pss', 'Private_Clean', 'Shared_Clean',
                                           'Private_Dirty', 'Shared_Dirty', 'Anonymous', 'Swap', 'Locked'))
                and metrics['Size'] * 1024 == end - start,
                'Complete checkpoint smaps size/clean evidence required')
        require(all(metrics[k] == 0 for k in ('Private_Dirty', 'Shared_Dirty', 'Anonymous', 'Swap', 'Locked'))
                and all(metrics.get(k, 0) == 0 for k in ('AnonHugePages', 'ShmemPmdMapped', 'FilePmdMapped'))
                and 'VmFlags' in row and not forbidden.intersection(row['VmFlags']),
                'Dirty, anonymous, swapped, locked, huge or device checkpoint VMA rejected')
    return rows


def mapping_identity(row):
    return {k: row[k] for k in ('start', 'end', 'offset', 'permissions', 'dev_major', 'dev_minor', 'inode', 'path')}


def fault_fields(text):
    # /proc/self/stat comm may contain spaces and parentheses.
    fields = text[text.rfind(')') + 2:].split()
    return dict(minor_faults=int(fields[7]), major_faults=int(fields[9]), starttime_ticks=int(fields[19]))


def memory_snapshot(checkpoint, identity):
    proc = Path('/proc/self')
    smaps = selected(parse_smaps((proc / 'smaps').read_text()), identity)
    maps = selected(parse_smaps((proc / 'maps').read_text()), identity)
    require([mapping_identity(r) for r in maps] == [mapping_identity(r) for r in smaps],
            'Self maps/smaps checkpoint VMAs changed between reads')
    process_status, rollup = {}, {}
    for name, result in (('status', process_status), ('smaps_rollup', rollup)):
        for line in (proc / name).read_text().splitlines():
            key, separator, value = line.partition(':')
            if separator and value.strip().endswith('kB'):
                result[key] = int(value.split()[0])
    raw_stat = (proc / 'stat').read_text().strip()
    return dict(monotonic=time.monotonic(), checkpoint_stat=checkpoint_stat(checkpoint),
                checkpoint_vmas=smaps, status_kB=process_status, smaps_rollup_kB=rollup,
                stat=raw_stat, statm=(proc / 'statm').read_text().strip(), faults=fault_fields(raw_stat),
                measurement_scope='self procfs reads outside timed model service; reads are sequential')


def tensor_metadata(model):
    rows = []
    for registry, named in (('parameter', model.named_parameters()), ('buffer', model.named_buffers())):
        for name, value in named:
            require(value.device.type == 'cpu' and not value.is_meta, 'CPU materialized original tensor required')
            storage = value.untyped_storage()
            rows.append(dict(registry=registry, name=name, tensor_id=id(value),
                             storage_cdata=storage._cdata, storage_ptr=storage.data_ptr(), storage_bytes=storage.nbytes(),
                             data_ptr=value.data_ptr(), shape=list(value.shape), stride=list(value.stride()),
                             dtype=str(value.dtype), version=value._version, requires_grad=value.requires_grad))
    require(rows, 'Original model tensors required')
    return rows


def pageout_checkpoint(model, checkpoint, loaded_stat, receipt_path, *,
                       source_root=Path('/mnt/vggt-data/vggt-slam2/src'), stage='after_backend_install'):
    """One advice per verified clean VMA, after original load_model SHA validation.

    loaded_stat is captured before that loader and rechecked afterward by the
    caller. Return the persisted receipt on success; any failure raises after
    writing its receipt. No hash or tensor-value read occurs after pageout.
    """
    receipt_path, checkpoint = Path(receipt_path), Path(checkpoint)
    require(not receipt_path.exists(), 'New pageout receipt required')
    report = dict(schema='v74_original_checkpoint_MADV_PAGEOUT_experiment', status='running',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  advice=MADV_PAGEOUT, stage=stage, pid=os.getpid(), advice_calls=[], errors=[],
                  checkpoint_sha256=ORIGINAL_SHA, checkpoint_bytes=ORIGINAL_BYTES,
                  checkpoint_SHA_verification='pinned original load_model before deserialization; stable pre/post stat',
                  checkpoint_rehashed_by_pageout_helper=False, tensor_values_read_by_helper=False,
                  source_sha256=SOURCE_PINS, affects_prior_initialization_HWM=False,
                  startup_peak_or_8GB_hardware_acceptance_established=False)
    before_tensors, descriptor, vmas, page_size, started = None, None, None, None, time.monotonic()
    try:
        require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
                'RK3588 board required')
        for name, expected in SOURCE_PINS.items():
            require(hashlib.sha256((Path(source_root) / 'vggt_slam' / name).read_bytes()).hexdigest() == expected,
                    'Pinned original checkpoint loader source differs')
        identity = checkpoint_stat(checkpoint)
        require(identity == loaded_stat, 'Original checkpoint changed after verified loader')
        descriptor = os.open(identity['path'], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        value = os.fstat(descriptor)
        require((value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns) ==
                tuple(identity[k] for k in ('dev', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')),
                'Opened checkpoint identity changed')
        before_tensors = tensor_metadata(model)
        report['tensor_metadata_before_sha256'] = hashlib.sha256(json.dumps(before_tensors, sort_keys=True).encode()).hexdigest()
        report['tensor_count'] = len(before_tensors)
        report['before'] = memory_snapshot(checkpoint, identity)
        page_size = os.sysconf('SC_PAGE_SIZE')
        report['page_size'] = page_size
        vmas = validate_vmas(report['before']['checkpoint_vmas'], identity, page_size)
        mapped = [t for t in before_tensors if t['storage_bytes'] and any(
            r['start'] <= t['storage_ptr'] and t['storage_ptr'] + t['storage_bytes'] <= r['end'] for r in vmas)]
        require(mapped, 'Original model has no storage inside verified checkpoint VMAs')
        report['checkpoint_backed_storage_count'] = len({t['storage_cdata'] for t in mapped})
        write(receipt_path, report)
        libc = ct.CDLL(None, use_errno=True)
        libc.madvise.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_int]
        libc.madvise.restype = ct.c_int
        for vma in vmas:
            current = memory_snapshot(checkpoint, identity)
            require(current['checkpoint_stat'] == identity, 'Checkpoint file changed before advice')
            validate_vmas(current['checkpoint_vmas'], identity, page_size)
            require([mapping_identity(r) for r in current['checkpoint_vmas']] == [mapping_identity(r) for r in vmas],
                    'Checkpoint VMA layout changed before advice')
            call = dict(mapping=mapping_identity(vma), bytes=vma['end'] - vma['start'], status='calling')
            report['advice_calls'].append(call)
            write(receipt_path, report)
            ct.set_errno(0)
            call_start = time.monotonic()
            code = libc.madvise(ct.c_void_p(vma['start']), ct.c_size_t(call['bytes']), MADV_PAGEOUT)
            call.update(returncode=int(code), errno=ct.get_errno(), seconds=time.monotonic() - call_start,
                        status='returned')
            write(receipt_path, report)
            require(code == 0, 'MADV_PAGEOUT failed with errno ' + str(call['errno']))
        report['status'] = 'advice_completed_not_memory_or_quality_acceptance'
    except BaseException as exc:
        report.update(status='failed')
        report['errors'].append(f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            report['after_immediate'] = memory_snapshot(checkpoint, loaded_stat)
            if vmas is not None:
                validate_vmas(report['after_immediate']['checkpoint_vmas'], loaded_stat, page_size)
                report['checkpoint_VMA_layout_unchanged'] = (
                    [mapping_identity(r) for r in report['after_immediate']['checkpoint_vmas']] ==
                    [mapping_identity(r) for r in vmas])
                require(report['checkpoint_VMA_layout_unchanged'], 'Checkpoint VMA layout changed during advice')
            if before_tensors is not None:
                after = tensor_metadata(model)
                report['tensor_metadata_after_sha256'] = hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest()
                report['tensor_storage_VA_version_unchanged'] = before_tensors == after
                require(before_tensors == after, 'Original tensor/storage/VA/version changed')
            require(report['after_immediate']['checkpoint_stat'] == loaded_stat, 'Checkpoint changed during pageout')
        except BaseException as exc:
            report['status'] = 'failed'
            report['errors'].append(f'postcheck: {type(exc).__name__}: {exc}')
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                report['status'] = 'failed'
                report['errors'].append(f'close checkpoint: {exc}')
        report['seconds'] = time.monotonic() - started
        write(receipt_path, report)
    require(report['status'] != 'failed', 'Pageout verification or cleanup failed: ' + '; '.join(report['errors']))
    return report
