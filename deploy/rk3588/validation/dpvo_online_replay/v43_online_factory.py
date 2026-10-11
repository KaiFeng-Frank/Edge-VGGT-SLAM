"""Metadata-only wrapper around the frozen nine-shape v42 board factory.

DPVO, Princeton University (2022), MIT; see adjacent LICENSE. No inference on
import. The frozen factory still owns all model/SDK validation and lifecycle;
only the callback's NPU_used reporting query is changed. Historical numerical
and workload rejection remain, with no native-input or model adoption option.
"""
import importlib.util
from pathlib import Path
import sys
import uuid

PARENT_FACTORY_SHA256 = 'c829269de17d775ef2df7663c561272025d3a080c671c01de92bd99ecf3939cf'
PARENT_CALLBACK_SHA256 = '3bbba0a4629fae4577e61efabcfeb005e11a7f5c9170aadf15bd5d386bc8beaa'
CALLBACK_SHA256 = '4f4a5457dbdec5aebe1ddd6204a37c9073896ac99ed90abd3fbfe3d1f78db566'
CONTRACTS_SHA256 = '30e68837815bc6f135f6d244b9ae4d166cf3ed43991249ca97bc07aa40a6ebb0'
METADATA_HELPER_SHA256 = '58acede7cd759ed6954d5b95e72c0f432408ebdf59e24de56afd6fab314872e4'


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_parent(path):
    name = '_v43_metadata_factory_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return value


def _recheck(pins):
    for path, expected in pins.items():
        if sha256(path) != expected:
            raise ValueError('Frozen metadata-only source changed: ' + path)


def open_board_online_from_index(*, v42_factory, v42_factory_sha256,
        v42_callback, v42_callback_sha256, metadata_helper, metadata_helper_sha256,
        **kwargs):
    """Same v42 keyword API plus six explicit frozen source-provenance fields.

    All remaining arguments are delegated unchanged to the frozen v42 factory.
    Its nine-entry catalogue, snapshot, attrs, owned output, context budget and
    explicit startup-probe rules remain active. This wrapper does not import
    the helper at runtime: its byte-identical function is in the pinned callback.
    """
    if kwargs.get('diagnostic_allow_fp32_rejected') is not True:
        raise ValueError('Explicit FP32-rejected online research diagnostic required')
    for actual, expected, name in (
            (v42_factory_sha256, PARENT_FACTORY_SHA256, 'v42 factory'),
            (v42_callback_sha256, PARENT_CALLBACK_SHA256, 'v42 callback'),
            (metadata_helper_sha256, METADATA_HELPER_SHA256, 'metadata helper'),
            (kwargs.get('callback_sha256'), CALLBACK_SHA256, 'v43 callback'),
            (kwargs.get('contracts_sha256'), CONTRACTS_SHA256, 'v42 contracts')):
        if actual != expected:
            raise ValueError('Exact frozen ' + name + ' SHA differs')
    pins = {str(Path(path).resolve(strict=True)): digest for path, digest in (
        (v42_factory, PARENT_FACTORY_SHA256),
        (v42_callback, PARENT_CALLBACK_SHA256),
        (metadata_helper, METADATA_HELPER_SHA256),
        (kwargs['callback_path'], CALLBACK_SHA256),
        (kwargs['contracts'], CONTRACTS_SHA256),
        (__file__, sha256(__file__)))}
    _recheck(pins)
    parent = _load_parent(Path(v42_factory).resolve(strict=True))
    callback = parent.open_board_online_from_index(**kwargs)
    try:
        _recheck(pins)
        callback.source_pins = dict(callback.source_pins)
        callback.source_pins.update(pins)
        callback.catalogue_provenance = dict(callback.catalogue_provenance,
            metadata_only_reporting=True, numerical_execution_changed=False,
            native_input_transport_added=False,
            original_v42_factory_sha256=PARENT_FACTORY_SHA256,
            original_v42_callback_sha256=PARENT_CALLBACK_SHA256,
            metadata_helper_sha256=METADATA_HELPER_SHA256,
            metadata_only_callback_sha256=CALLBACK_SHA256)
    except BaseException:
        callback.close()
        raise
    return callback
