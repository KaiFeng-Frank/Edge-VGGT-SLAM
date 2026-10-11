"""Strict CPU neural-only loader for the pinned MIT-licensed DPVO source.

The official extractor is imported unchanged. Only the original class ASTs
needed by Update are executed from blocks.py/net.py; their forward methods
are not rewritten. Real CPU operators replace the called scatter/neighbor
bindings. CUDA training, bundle adjustment and a complete VO system are not
imported or claimed. Real checkpoint loading is restricted to aarch64 Linux.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
from types import ModuleType


COMMIT = '0ac95b656d1fda91c271d2a106460d19ad966fc7'
MANIFEST_SHA256 = 'ce2ecaf96b086f69ef4d90a2e07e7b81afe8ae094656eb41260182000a69e375'
CHECKPOINT_SHA256 = '30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12'
CHECKPOINT_BYTES = 14167743
SOURCE_PINS = {
    'LICENSE': 'c5b0105dcef65f5423bec0aafe796364a086375e23746cf75402aba0f897489a',
    'dpvo/extractor.py': 'f5cbcbe14c42e44bcec6e827a3429ebe93979a439ba9b9ef70d4f4baa3454a2b',
    'dpvo/blocks.py': '9c9e2848da524829374e56bb1b14a891e9e2c19bc683af8e958e8982ba33d661',
    'dpvo/net.py': '1aee33d54f0b1cc1f3f842b60790ff1299ef9c1e20bca6467ce7114dca802acc',
    'dpvo/dpvo.py': '7efcb45689c2da8d18e2c69774e221799c9f8472e99531955f55dcb145626fd7',
}
BLOCK_CLASSES = ('GatedResidual', 'SoftAgg', 'GradClip', 'GradientClip')
LEGACY_SHAPES = {
    'update.lmbda.0.weight': (384, 384),
    'update.lmbda.0.bias': (384,),
    'update.lmbda.2.weight': (1, 384),
    'update.lmbda.2.bias': (1,),
}
INFERENCE_KEYS = 94
INFERENCE_PARAMETERS = 3384324
RELEASED_KEYS = 98
RELEASED_PARAMETERS = 3532549


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_source(source_dir, manifest_path):
    source_dir = Path(source_dir).resolve(strict=True)
    manifest_path = Path(manifest_path).resolve(strict=True)
    require(sha256(manifest_path) == MANIFEST_SHA256, 'Pinned v25 DPVO source manifest SHA mismatch')
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get('repo') == 'princeton-vl/DPVO' and manifest.get('commit') == COMMIT,
            'Official DPVO repository/commit mismatch')
    listed = manifest['retrieved_files']
    rows = {r['path']: r for r in listed}
    require(len(rows) == len(listed), 'Duplicate source manifest path')
    hashes = {}
    for name, expected in SOURCE_PINS.items():
        row = rows[name]
        path = (source_dir / name).resolve(strict=True)
        require(path.is_relative_to(source_dir), 'Source path escapes the official source directory')
        raw = path.read_bytes()
        require(row.get('sha256') == expected and hashlib.sha256(raw).hexdigest() == expected
                and len(raw) == row['bytes'], 'Pinned official source bytes mismatch: ' + name)
        git_blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        require(git_blob == row['git_blob_sha1'], 'Official Git blob mismatch: ' + name)
        hashes[name] = dict(sha256=expected, git_blob_sha1=git_blob, bytes=len(raw))
    return dict(repo=manifest['repo'], commit=COMMIT, manifest_sha256=MANIFEST_SHA256,
                source_hashes=hashes, source_files_modified=False, upstream_license='MIT')


def extract_class_asts(text, names):
    """Select original top-level class nodes without transforming their ASTs."""
    tree = ast.parse(text)
    selected = []
    records = {}
    for name in names:
        matches = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
        require(len(matches) == 1, 'Expected one original class definition: ' + name)
        node = matches[0]
        selected.append(node)
        records[name] = dict(class_ast_sha256=hashlib.sha256(
            ast.dump(node, include_attributes=False).encode()).hexdigest(),
            source_segment_sha256=hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest(),
            original_line=node.lineno, original_end_line=node.end_lineno,
            original_class_AST_preserved=True)
    result = ast.Module(body=selected, type_ignores=[])
    require([ast.dump(n, include_attributes=False) for n in result.body]
            == [ast.dump(n, include_attributes=False) for n in selected], 'Class AST changed')
    return result, records


def _execute_classes(path, names, namespace):
    text = Path(path).read_text()
    tree, records = extract_class_asts(text, names)
    exec(compile(tree, str(path), 'exec'), namespace)
    return records


def _load_cpu_ops(path, expected_sha256=None):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        require(_valid_sha(expected_sha256) and digest == expected_sha256, 'CPU operator source SHA mismatch')
    module = ModuleType('_dpvo_probe_cpu_ops_' + digest[:16])
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    try:
        exec(compile(raw, str(path), 'exec'), module.__dict__)
        for name in ('neighbors', 'scatter_sum', 'scatter_softmax'):
            require(callable(getattr(module, name, None)), 'Missing real CPU operator: ' + name)
        require(sha256(path) == digest, 'CPU operator source changed during import')
    except BaseException:
        sys.modules.pop(module.__name__, None)
        raise
    return module, dict(file=path.name, sha256=digest, expected_sha256=expected_sha256)


def construct_neural(source_dir, manifest_path, cpu_ops_path, *, cpu_ops_sha256=None):
    """Construct original modules without loading checkpoint tensors.

    Host use is for synthetic tests only; this does not establish inference.
    """
    import torch
    import torch.nn as nn

    source_dir = Path(source_dir).resolve(strict=True)
    provenance = validate_source(source_dir, manifest_path)
    cpu_ops, cpu_info = _load_cpu_ops(cpu_ops_path, cpu_ops_sha256)
    path = source_dir / 'dpvo/extractor.py'
    name = '_dpvo_original_extractor_' + SOURCE_PINS['dpvo/extractor.py'][:16]
    spec = importlib.util.spec_from_file_location(name, path)
    extractor = importlib.util.module_from_spec(spec)
    sys.modules[name] = extractor
    spec.loader.exec_module(extractor)
    require(sha256(path) == SOURCE_PINS['dpvo/extractor.py'], 'Extractor changed during import')

    bindings = dict(__name__='_dpvo_original_update_classes', torch=torch, nn=nn,
                    torch_scatter=cpu_ops, fastba=cpu_ops, DIM=384)
    class_records = _execute_classes(source_dir / 'dpvo/blocks.py', BLOCK_CLASSES, bindings)
    net_tree = ast.parse((source_dir / 'dpvo/net.py').read_text())
    dim_assignments = [n for n in net_tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'DIM' for t in n.targets)]
    require(len(dim_assignments) == 1 and ast.literal_eval(dim_assignments[0].value) == 384,
            'Original Update DIM must be 384')
    class_records.update(_execute_classes(source_dir / 'dpvo/net.py', ('Update',), bindings))

    class Neural(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patchify = torch.nn.Module()
            self.patchify.fnet = extractor.BasicEncoder4(output_dim=128, norm_fn='instance')
            self.patchify.inet = extractor.BasicEncoder4(output_dim=384, norm_fn='none')
            self.update = bindings['Update'](3)
            self.DIM, self.RES, self.P = 384, 4, 3

        @property
        def fnet(self):
            return self.patchify.fnet

        @property
        def inet(self):
            return self.patchify.inet

    neural = Neural().cpu().float().eval()
    state = neural.state_dict()
    require(len(state) == INFERENCE_KEYS and sum(v.numel() for v in state.values()) == INFERENCE_PARAMETERS,
            'Official inference state coverage differs from the released 94-key architecture')
    provenance.update(original_classes=class_records, cpu_ops=cpu_info,
        uncalled_imports_isolated=['CUDA bundle adjustment', 'lietorch', 'altcorr CUDA',
                                  'training forward', 'matplotlib', 'torch_scatter extension'],
        inference_state_keys=INFERENCE_KEYS, inference_parameters=INFERENCE_PARAMETERS,
        original_Update_class_AST_preserved=True, original_extractor_file_imported=True,
        full_DPVO_or_BA_implemented=False, VO_accuracy_validated=False, production_changed=False,
        original_VGGT_weights_used=False, different_model_from_VGGT=True, ORB_used=False)
    return neural, provenance


def tensor_info(value):
    raw = value.detach().contiguous().numpy().tobytes(order='C')
    return dict(shape=list(value.shape), dtype=str(value.dtype), numel=value.numel(),
                bytes=len(raw), tensor_sha256=hashlib.sha256(raw).hexdigest())


def strict_load_state(neural, released_state):
    """Load a synthetic or released flat 98-key mapping with no dtype coercion."""
    import torch

    require(isinstance(released_state, Mapping) and len(released_state) == RELEASED_KEYS,
            'Exactly 98 released state keys required')
    state = OrderedDict()
    for name, value in released_state.items():
        require(isinstance(name, str) and name.startswith('module.'), 'Only the exact module. prefix may be removed')
        target = name[len('module.'):]
        require(target not in state and isinstance(value, torch.Tensor)
                and value.device.type == 'cpu' and value.dtype == torch.float32
                and value.layout == torch.strided and torch.isfinite(value).all().item(),
                'Duplicate, nonfinite, non-CPU or non-FP32 released state: ' + target)
        state[target] = value
    expected = neural.state_dict()
    require(len(expected) == INFERENCE_KEYS and sum(v.numel() for v in expected.values()) == INFERENCE_PARAMETERS,
            'Expected exact original 94-key neural state')
    require(set(state) == set(expected) | set(LEGACY_SHAPES),
            'Only the four exact legacy update.lmbda keys may be dropped')
    for key, target in expected.items():
        require(state[key].shape == target.shape and state[key].dtype == target.dtype,
                'Inference state shape/dtype mismatch: ' + key)
    for key, shape in LEGACY_SHAPES.items():
        require(tuple(state[key].shape) == shape, 'Legacy state shape mismatch: ' + key)
    require(sum(v.numel() for v in state.values()) == RELEASED_PARAMETERS, 'Released parameter coverage mismatch')
    kept = OrderedDict((k, state[k]) for k in expected)
    before = {k: tensor_info(v) for k, v in kept.items()}
    discarded = {k: tensor_info(state[k]) for k in LEGACY_SHAPES}
    result = neural.load_state_dict(kept, strict=True)
    require(not result.missing_keys and not result.unexpected_keys, 'strict=True loading did not consume every inference key')
    after = {k: tensor_info(v) for k, v in neural.state_dict().items()}
    require(before == after, 'Loaded inference parameter bytes differ from released tensors')
    neural.eval().requires_grad_(False)
    return dict(released_state_keys=RELEASED_KEYS, released_parameters=RELEASED_PARAMETERS,
                inference_state_keys=INFERENCE_KEYS, inference_parameters=INFERENCE_PARAMETERS,
                removed_prefix='module.', exact_dropped_legacy_keys=list(LEGACY_SHAPES),
                strict=True, inference_parameter_coverage=1.0,
                loaded_tensors=after, dropped_legacy_tensors=discarded,
                all_loaded_parameter_bytes_equal=True)


def load_neural(source_dir, manifest_path, checkpoint_path, checkpoint_sha256, cpu_ops_path,
                *, cpu_ops_sha256=None):
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64',
            'Real DPVO checkpoint loading is restricted to aarch64 Linux; host tests use synthetic state only')
    require(checkpoint_sha256 == CHECKPOINT_SHA256
            and Path(checkpoint_path).stat().st_size == CHECKPOINT_BYTES
            and sha256(checkpoint_path) == checkpoint_sha256,
            'Official DPVO checkpoint SHA mismatch')
    import torch

    neural, provenance = construct_neural(source_dir, manifest_path, cpu_ops_path,
                                          cpu_ops_sha256=cpu_ops_sha256)
    released = torch.load(str(checkpoint_path), map_location='cpu', weights_only=True)
    coverage = strict_load_state(neural, released)
    require(sha256(checkpoint_path) == checkpoint_sha256, 'Checkpoint changed during loading')
    validate_source(source_dir, manifest_path)
    require(sha256(cpu_ops_path) == provenance['cpu_ops']['sha256'], 'CPU operators changed during loading')
    provenance.update(checkpoint=dict(sha256=checkpoint_sha256, bytes=Path(checkpoint_path).stat().st_size),
                      load_state=coverage, weights_only=True, map_location='cpu',
                      helper_sha256=sha256(__file__), sources_and_checkpoint_unchanged=True,
                      model_device='cpu', model_precision='float32', weights_loaded_on_board=True)
    return neural, provenance


def encode(neural, normalized_bgr):
    """Original fnet/inet followed by /4, without re-normalizing BGR pixels."""
    import torch

    require(isinstance(normalized_bgr, torch.Tensor) and normalized_bgr.device.type == 'cpu'
            and normalized_bgr.dtype == torch.float32 and normalized_bgr.ndim == 5
            and normalized_bgr.shape[0] > 0 and normalized_bgr.shape[1] > 0
            and normalized_bgr.shape[2] == 3 and min(normalized_bgr.shape[-2:]) >= 8
            and all(n % 4 == 0 for n in normalized_bgr.shape[-2:])
            and torch.isfinite(normalized_bgr).all().item()
            and normalized_bgr.min().item() >= -0.5 and normalized_bgr.max().item() <= 1.5,
            'Already normalized CPU FP32 BGR [B,S,3,H,W] with H/W multiples of four required')
    with torch.no_grad():
        return neural.patchify.fnet(normalized_bgr) / 4.0, neural.patchify.inet(normalized_bgr) / 4.0


def make_encoder(neural):
    """Independent export wrapper; it does not add registrations to neural."""
    import torch

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fnet = neural.patchify.fnet
            self.inet = neural.patchify.inet

        def forward(self, normalized_bgr):
            return self.fnet(normalized_bgr) / 4.0, self.inet(normalized_bgr) / 4.0

    return Encoder().eval()
