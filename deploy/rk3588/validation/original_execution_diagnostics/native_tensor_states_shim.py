"""Private instance-only injection; original module/guards remain unchanged."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import types

CACHE_SHA = '3a8cc8c5a68c3bb934969e86053538f0d2026da1846274f4e6085297474a1e4b'


def fingerprint_ast(source):
    tree = ast.parse(source)
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == 'PatchCacheHandle')
    return next(node for node in owner.body if isinstance(node, ast.FunctionDef)
                and node.name == '_fingerprint')


def replace_tensor_loops(function, bridge_name):
    """Change exactly two RHS expressions; no module-walk or guard rewrite."""
    original = ast.dump(function, include_attributes=False)
    expected = {
        'parameters': 'tuple(_tensor_state(name, tensor, parameter=True) for name, tensor in self.model.named_parameters())',
        'buffers': 'tuple(_tensor_state(name, tensor, parameter=False) for name, tensor in self.model.named_buffers())',
    }
    changes = []
    for node in function.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in expected:
            continue
        name = target.id
        oracle = ast.parse(expected[name], mode='eval').body
        if ast.dump(node.value, include_attributes=False) != ast.dump(oracle, include_attributes=False):
            raise RuntimeError('Pinned tensor expression changed: ' + name)
        changes.append((node, node.value))
        iterator = 'named_parameters' if name == 'parameters' else 'named_buffers'
        node.value = ast.parse(f'{bridge_name}(self.model.{iterator}(), {name == "parameters"})', mode='eval').body
    if len(changes) != 2:
        raise RuntimeError('Exactly two original tensor generators required')
    # Inverse reconstruction proves all other AST remains identical.
    changed = [node.value for node, _ in changes]
    for node, old in changes:
        node.value = old
    if ast.dump(function, include_attributes=False) != original:
        raise RuntimeError('Unexpected fingerprint AST mutation')
    for (node, _), new in zip(changes, changed):
        node.value = new
    return ast.fix_missing_locations(function)


class NativeTensorStateHandle:
    def __init__(self, handle, library_path, library_sha256):
        original = type(handle)._fingerprint
        self.cache_handle, self.closed = handle, False
        self.globals = original.__globals__
        source_path = Path(original.__code__.co_filename).resolve(strict=True)
        source = source_path.read_bytes()
        if hashlib.sha256(source).hexdigest() != CACHE_SHA or original.__code__.co_freevars:
            raise RuntimeError('Frozen original cache source and no closure required')
        if '_fingerprint' in handle.__dict__:
            raise RuntimeError('Existing instance fingerprint override cannot be replaced')
        path = Path(library_path).resolve(strict=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != library_sha256:
            raise RuntimeError('Private native library SHA differs')
        spec = importlib.util.spec_from_file_location('_v76_tensor_states', path)
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
        self.bridge_name = f'_v76_native_tensor_state_loop_{id(handle):x}'
        if self.bridge_name in self.globals:
            raise RuntimeError('Private bridge name already exists')
        function = replace_tensor_loops(fingerprint_ast(source.decode()), self.bridge_name)
        compiled = compile(ast.Module(body=[function], type_ignores=[]),
                           '<v76-private-native-tensor-state-loop>', 'exec')
        code = next(value for value in compiled.co_consts if isinstance(value, types.CodeType))
        # Exact original live namespace, never a copied globals dictionary.
        replacement = types.FunctionType(code, self.globals, original.__name__, original.__defaults__)
        replacement.__kwdefaults__ = original.__kwdefaults__
        self.replacement = types.MethodType(replacement, handle)
        self.bridge = lambda iterator, flag: extension.tensor_states(iterator, flag, self.globals)
        self.globals[self.bridge_name] = self.bridge
        try:
            handle._fingerprint = self.replacement
        except BaseException:
            del self.globals[self.bridge_name]
            raise
        self.library_sha256 = library_sha256

    def close(self):
        if self.closed:
            return
        errors = []
        if self.cache_handle.__dict__.get('_fingerprint') is self.replacement:
            del self.cache_handle._fingerprint
        else:
            errors.append('Instance fingerprint changed; caller override not overwritten')
        if self.globals.get(self.bridge_name) is self.bridge:
            del self.globals[self.bridge_name]
        else:
            errors.append('Private bridge changed; caller override not overwritten')
        self.closed = True
        if errors:
            raise RuntimeError('; '.join(errors))

    def metrics(self):
        return dict(closed=self.closed, instance_only=True, replaced_tensor_generators=2,
            library_sha256=self.library_sha256, original_cache_sha256=CACHE_SHA,
            original_module_walk_and_two_guards_unchanged=True, original_live_globals=True)


def install_native_tensor_state_loop(handle, library_path, library_sha256):
    return NativeTensorStateHandle(handle, library_path, library_sha256)
