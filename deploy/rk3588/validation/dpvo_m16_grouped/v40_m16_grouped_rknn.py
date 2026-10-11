"""M16 research binding of the byte-pinned v37 SDK implementation.

No Torch, ORT, SDK library or model is loaded on import. A private ancestor
module receives the new M16 contract; its transport, execute, bounded cache and
complete-edge scatter implementations remain unchanged. Real initialization is
board-only. DPVO MIT: Copyright (c) 2022 Princeton University.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
import uuid

sys.dont_write_bytecode = True
PARENT_RUNTIME_SHA256 = '55d7b04e782a687fedb6475b8388a12d52361cc56b1a7f260537bd3e1d679d1d'
PARENT_CONTRACTS_SHA256 = 'aa9e6af938f64ae4a4a82ffc84b487321fff208547703b0d65646c72b96d85c9'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _load(path, expected, label):
    path = Path(path).resolve(strict=True)
    if sha256(path) != expected:
        raise ValueError('Pinned ' + label + ' SHA differs')
    # Unique name prevents any shared/previous runtime module being rebound.
    name = '_v40_private_' + label + '_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return value


def load_runtime(parent_runtime, contracts_path, contracts_sha256, parent_contracts):
    """Return an isolated API; old files and other loaded modules are untouched.

    The frozen runtime imports its own adjacent v37 contract on load. That exact
    original dependency must remain adjacent when staged on board; the new M16
    file is a separate contract, never masquerading as the M48 ancestor.
    """
    parent_runtime = Path(parent_runtime).resolve(strict=True)
    parent_contracts = Path(parent_contracts).resolve(strict=True)
    adjacent = parent_runtime.with_name('v37_grouped_contracts.py')
    if (sha256(parent_contracts) != PARENT_CONTRACTS_SHA256
            or sha256(adjacent) != PARENT_CONTRACTS_SHA256):
        raise ValueError('Frozen adjacent/explicit parent contract SHA differs')
    c = _load(contracts_path, contracts_sha256, 'M16_contracts')
    c.load_parent(parent_contracts)
    parent = _load(parent_runtime, PARENT_RUNTIME_SHA256, 'v37_runtime')
    if c.INPUT_NAMES != parent.INPUT_NAMES or c.OUTPUT_NAMES != parent.OUTPUT_NAMES:
        raise ValueError('M16 transport operand names changed')
    parent.C = c  # This module instance is private to this returned API.

    class GroupedSession(parent.GroupedSession):
        @classmethod
        def open_board(cls, *args, pins=None, **kwargs):
            sources = dict(pins or {})
            c.pin(parent_runtime, PARENT_RUNTIME_SHA256, sources)
            c.pin(adjacent, PARENT_CONTRACTS_SHA256, sources)
            c.pin(parent_contracts, PARENT_CONTRACTS_SHA256, sources)
            c.pin(contracts_path, contracts_sha256, sources)
            c.pin(__file__, sha256(__file__), sources)
            c.recheck_pins(sources)
            # The frozen method enforces ARM/RK3588 before CDLL/init, records
            # actual attrs/memory, and cleans partial startup failures.
            return super().open_board(*args, pins=sources, **kwargs)

    return types.SimpleNamespace(
        C=c, GroupedSession=GroupedSession,
        BoundedContextCache=parent.BoundedContextCache,
        execute_full_graph=parent.execute_full_graph,
        pack_inputs=parent.pack_inputs, compare=parent.compare,
        StartupError=parent.StartupError, GroupExecutionError=parent.GroupExecutionError,
        parent=parent, source_pins={str(parent_runtime): PARENT_RUNTIME_SHA256,
            str(adjacent): PARENT_CONTRACTS_SHA256,
            str(parent_contracts): PARENT_CONTRACTS_SHA256,
            str(Path(contracts_path).resolve()): contracts_sha256,
            str(Path(__file__).resolve()): sha256(__file__)})
