"""Process-local registry for explicit RKNN same-file shared-weight init.

No SDK import, global API monkeypatch, inference or physical-memory claim.
The caller pins each original RKNN file/API before calling. One RLock
serializes initialization/destruction; SDK calls stay on the calling worker.
Deferred adapter closes are logical releases: snapshot reports actual resources.
"""
import ctypes as ct
import os
from pathlib import Path
import threading


class InitExtend(ct.Structure):
    _fields_ = [('ctx', ct.c_uint64), ('real_model_offset', ct.c_int32),
                ('real_model_size', ct.c_uint32), ('model_buffer_fd', ct.c_int32),
                ('model_buffer_flags', ct.c_uint32), ('reserved', ct.c_uint8 * 112)]


if ct.sizeof(InitExtend) != 136 or InitExtend.ctx.offset != 0 or InitExtend.reserved.offset != 24:
    raise RuntimeError('Pinned aarch64 rknn_init_extend ABI differs')

_PID = os.getpid()
_LOCK = threading.RLock()
_GROUPS, _RECORDS, _EVENTS, _ERRORS = {}, {}, [], []


def _check_process():
    if os.getpid() != _PID:
        raise RuntimeError('Shared context registry cannot be inherited across processes')


def _number(value):
    return int(getattr(value, 'value', value))


def _identity(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _api_identity(api):
    return getattr(api, '_handle', id(api))


def _register(api, handle, filename, identity, master, code, model_buffer=None):
    previous = _RECORDS.get(handle)
    if previous is not None and previous['context'].value:
        _ERRORS.append('SDK initialization reused a still-live handle')
        raise RuntimeError(_ERRORS[-1])
    record = dict(api=api, context=ct.c_uint64(handle), filename=filename, identity=identity, model_buffer=model_buffer,
                  master=master, children=set(), pending=False, init_code=code,
                  owner_thread=threading.get_ident(), destroy_requests=0,
                  actual_destroy_attempts=0, actual_destroy_code=None)
    _RECORDS[handle] = record
    if master is None:
        _GROUPS[filename] = handle
    else:
        _RECORDS[master]['children'].add(handle)
    return record


def initialize(api, context_pointer, model_pointer, size, flags, extend):
    """Forward ordinary same-file init, adding flag0x20/ctx only for children.

    Only filename/size0/flags0/no-extend requests are supported. File identity
    must stay fixed between the caller's validated initializations. SDK error
    codes are returned unchanged; any nonzero partial handle remains tracked.
    """
    with _LOCK:
        _check_process()
        if _number(size) != 0 or _number(flags) != 0 or bool(extend):
            raise ValueError('Shared registry requires filename, size0, flags0 and no caller extend')
        slot = ct.cast(context_pointer, ct.POINTER(ct.c_uint64)).contents
        if slot.value:
            raise ValueError('Initialization output context must be empty')
        path = Path(os.fsdecode(ct.string_at(model_pointer))).resolve(strict=True)
        if not path.is_file():
            raise ValueError('Validated RKNN model must be a regular file')
        filename, identity = str(path), _identity(path)
        master = _GROUPS.get(filename)
        actual_flags, argument = 0, None
        model_buffer = None
        actual_model, actual_size = model_pointer, 0
        if master is not None:
            parent = _RECORDS[master]
            if (not parent['context'].value or parent['pending'] or parent['init_code'] != 0
                    or parent['identity'] != identity or _api_identity(parent['api']) != _api_identity(api)):
                raise RuntimeError('Original model file/master/API changed or master is closing/failed')
            argument = InitExtend(ctx=parent['context'].value)
            actual_flags = 0x20
            actual_size = identity[2]
            if not 0 < actual_size < (1 << 32):
                raise ValueError('Model byte size exceeds RKNN buffer API')
            model_buffer = ct.create_string_buffer(actual_size)
            with path.open('rb') as stream:
                if stream.readinto(model_buffer) != actual_size or stream.read(1):
                    raise ValueError('Frozen model buffer size changed')
            if _identity(path) != identity:
                raise ValueError('Frozen model file changed during buffer load')
            actual_model = ct.cast(model_buffer, ct.c_void_p)
        event = dict(operation='initialize', filename=filename, requested_flags=0,
                     actual_flags=actual_flags, sharing_requested=master is not None,
                     thread=threading.get_ident(), code=None, context_created=False,
                     model_input=('owned_CPU_buffer' if master is not None else 'filename'),
                     retained_CPU_model_bytes=actual_size)
        _EVENTS.append(event)
        try:
            code = _number(api.rknn_init(context_pointer, actual_model, actual_size, actual_flags,
                                         ct.byref(argument) if argument is not None else None))
        except BaseException as exc:
            event['exception'] = f'{type(exc).__name__}: {exc}'
            if slot.value:
                _register(api, slot.value, filename, identity, master, None, model_buffer)
                event['context_created'] = True
            _ERRORS.append('SDK initialization raised: ' + event['exception'])
            raise
        event['code'], event['context_created'] = code, bool(slot.value)
        if slot.value:
            _register(api, slot.value, filename, identity, master, code, model_buffer)
        if code != 0:
            _ERRORS.append(f'SDK initialization returned {code}; partial handle retained if present')
        elif not slot.value:
            _ERRORS.append('SDK initialization returned success without a context')
        return code


def _physical_destroy(record, *, trigger):
    """Return SDK rc; leave the original copy live on error, never retry here."""
    record['actual_destroy_attempts'] += 1
    event = dict(operation='physical_destroy', filename=record['filename'], trigger=trigger,
                 master=record['master'] is None, thread=threading.get_ident(), code=None)
    _EVENTS.append(event)
    try:
        code = _number(record['api'].rknn_destroy(record['context']))
    except BaseException as exc:
        event['exception'] = f'{type(exc).__name__}: {exc}'
        _ERRORS.append('SDK destroy raised: ' + event['exception'])
        raise
    event['code'] = record['actual_destroy_code'] = code
    if code == 0:
        record['context'].value = 0
        record['model_buffer'] = None
        if record['master'] is None:
            _GROUPS.pop(record['filename'], None)
    else:
        _ERRORS.append(f'SDK destroy returned {code}; actual context retained')
    return code


def destroy(api, context):
    """Logical master release may defer its actual SDK destroy to last child.

    Children remain live until their own SDK destroy succeeds. If a child
    succeeds but the deferred master fails, return the child's original rc0;
    snapshot retains the master/error so final registry gates must reject it.
    Unknown nonzero handles are physically cleaned with the calling API.
    """
    with _LOCK:
        _check_process()
        handle = _number(context)
        if not handle:
            return 0
        record = _RECORDS.get(handle)
        if record is None:
            record = dict(api=api, context=ct.c_uint64(handle), filename=None, identity=None,
                          master=None, children=set(), pending=False, init_code=None,
                          owner_thread=threading.get_ident(), destroy_requests=0,
                          actual_destroy_attempts=0, actual_destroy_code=None)
            _RECORDS[handle] = record
            _EVENTS.append(dict(operation='unregistered_handle_cleanup', thread=threading.get_ident()))
        if not record['context'].value:
            return 0
        if _api_identity(api) != _api_identity(record['api']):
            raise RuntimeError('Context destroy API differs from its initialization API')
        record['destroy_requests'] += 1
        if record['master'] is None and record['children']:
            record['pending'] = True
            _EVENTS.append(dict(operation='defer_master_destroy', filename=record['filename'],
                                dependent_children=len(record['children']), thread=threading.get_ident()))
            return 0
        code = _physical_destroy(record, trigger='owner_close')
        if code == 0 and record['master'] is not None:
            parent = _RECORDS[record['master']]
            parent['children'].discard(handle)
            if parent['pending'] and not parent['children'] and parent['context'].value:
                try:
                    _physical_destroy(parent, trigger='last_child_closed_pending_master')
                except BaseException:
                    pass  # Failure stays in the registry; the child is already destroyed.
        return code


def snapshot():
    """Pure statistics; SDK DMA fields are not physical weight-sharing proof."""
    with _LOCK:
        _check_process()
        records = list(_RECORDS.values())
        created = sum(event['context_created'] for event in _EVENTS if event['operation'] == 'initialize')
        destroyed = sum(not record['context'].value for record in records)
        live = sum(bool(record['context'].value) for record in records)
        return dict(schema='v79_explicit_shared_RKNN_context_registry', process_local=True,
                    initialization_serialized=True, same_validated_filename_required=True,
                    rknn_dup_context_used=False, async_flag_used=False,
                    init_extend_bytes=ct.sizeof(InitExtend), physical_weight_sharing_verified=False,
                    actual_contexts_created=created, actual_contexts_destroyed=destroyed,
                    registered_contexts=len(records), live_contexts=live,
                    retained_CPU_model_bytes=sum(ct.sizeof(record['model_buffer']) for record in records if record.get('model_buffer') is not None),
                    pending_masters=sum(record['pending'] and bool(record['context'].value) for record in records),
                    share_init_requests=sum(event.get('sharing_requested', False) for event in _EVENTS),
                    deferred_master_destroy_requests=sum(event['operation'] == 'defer_master_destroy' for event in _EVENTS),
                    all_registered_contexts_final_zero=bool(records) and live == 0,
                    cleanup_errors=list(_ERRORS), events=[dict(event) for event in _EVENTS])
