"""Research-only cache-hit closed query; no neural/native/SDK work on import.

Only the frozen v37 BoundedContextCache._get live-hit predicate is replaced.
The cache lock remains held by its inherited execute(), then the original
session lock protects the closed read. Cold allocation, usage/budget queries,
eviction/history, metrics(), output ownership and close() remain inherited.
DPVO: Copyright (c) 2022 Princeton University; see adjacent MIT LICENSE.
"""
import ast
import copy
import hashlib
import inspect
from pathlib import Path
import textwrap
import types

PARENT_RUNTIME_SHA256 = '55d7b04e782a687fedb6475b8388a12d52361cc56b1a7f260537bd3e1d679d1d'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def session_closed(session):
    """Return precisely the original closed operand, without a metrics copy."""
    with session._lock:
        return session._metrics['closed']


def _get_closed_flag(self, key):
    C.require(not self._closed and not self._metrics['cleanup_errors'], 'Context cache is closed or failed')
    C.require(key in self.specs, 'No compiled context for this actual shape')
    if key in self._live:
        C.require(not session_closed(self._live[key]), 'Failed context cannot be reused')
        self._live.move_to_end(key); self._metrics['hits'] += 1
        return self._live[key]
    spec = self.specs[key]
    estimated = spec['model_bytes']+1
    C.require(estimated <= self.max_model_bytes, 'Individual model host buffer exceeds cache budget')
    while self._live and (len(self._live) >= self.max_contexts or self._usage()[0]+estimated > self.max_model_bytes):
        self._evict(next(iter(self._live)))
    session = self.factory(copy.deepcopy(spec)); self._metrics['created'] += 1
    try:
        record = session.metrics()
        C.require(not record['closed'], 'Factory returned closed context')
        host, dma = record['current_model_host_bytes'], record['current_DMA_bytes']
        self._metrics['peak_transient_init_DMA_bytes'] = max(self._metrics['peak_transient_init_DMA_bytes'], self._usage()[1]+dma)
        C.require(host == estimated and dma <= self.max_dma_bytes, 'Actual context/model allocation exceeds declared budget')
        while self._live and (self._usage()[0]+host > self.max_model_bytes or self._usage()[1]+dma > self.max_dma_bytes):
            self._evict(next(iter(self._live)))
        self._live[key] = session
    except BaseException:
        session.close(); self._history.append(dict(model_key=key, metrics=session.metrics()))
        raise
    host_total, dma_total = self._usage()
    for name, value in (('peak_resident_contexts', len(self._live)), ('peak_resident_model_bytes', host_total), ('peak_resident_DMA_bytes', dma_total)):
        self._metrics[name] = max(self._metrics[name], value)
    return session


def _function(tree, class_name, method_name):
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == method_name)


def source_change_proof(parent_runtime_path):
    """Prove the new method body has exactly one expression substitution."""
    parent_runtime_path = Path(parent_runtime_path).resolve(strict=True)
    if sha256(parent_runtime_path) != PARENT_RUNTIME_SHA256:
        raise ValueError('Frozen v37 runtime SHA differs')
    parent = ast.parse(parent_runtime_path.read_text())
    original = _function(parent, 'BoundedContextCache', '_get')
    target = ast.parse("self._live[key].metrics()['closed']", mode='eval').body
    replacement = ast.parse('session_closed(self._live[key])', mode='eval').body
    count = 0
    class Substitute(ast.NodeTransformer):
        def visit_Subscript(self, node):
            nonlocal count
            if ast.dump(node, include_attributes=False) == ast.dump(target, include_attributes=False):
                count += 1
                return copy.deepcopy(replacement)
            return self.generic_visit(node)
    expected = Substitute().visit(copy.deepcopy(original)); expected.name = '_get_closed_flag'
    template = ast.parse(textwrap.dedent(inspect.getsource(_get_closed_flag))).body[0]
    if count != 1 or ast.dump(expected, include_attributes=False) != ast.dump(template, include_attributes=False):
        raise ValueError('Only the single frozen closed-query expression may change')
    return dict(parent_runtime_sha256=PARENT_RUNTIME_SHA256,
        changed_method='BoundedContextCache._get', changed_expression_count=1,
        original_expression="self._live[key].metrics()['closed']",
        candidate_expression='session_closed(self._live[key])',
        all_other_get_AST_preserved=True,
        inherited_methods=[n.name for n in next(n for n in parent.body if isinstance(n, ast.ClassDef)
            and n.name == 'BoundedContextCache').body if isinstance(n, ast.FunctionDef) and n.name != '_get'],
        lock_order='cache._lock (inherited execute) -> session._lock (closed query)',
        cold_and_budget_full_metrics_preserved=True, lifecycle_full_metrics_preserved=True,
        numerical_execution_changed=False, native_pack_or_SDK_changed=False,
        candidate_accepted=False)


def make_cache_class(parent_cache_class, *, parent_runtime_path):
    """Return a new subclass; never edit the old class or its module globals."""
    path = Path(parent_runtime_path).resolve(strict=True)
    proof = source_change_proof(path)
    if (not isinstance(parent_cache_class, type) or parent_cache_class.__name__ != 'BoundedContextCache'
            or Path(inspect.getsourcefile(parent_cache_class._get)).resolve() != path):
        raise ValueError('Actual frozen parent cache class required')
    actual = ast.parse(textwrap.dedent(inspect.getsource(parent_cache_class._get))).body[0]
    expected = _function(ast.parse(path.read_text()), 'BoundedContextCache', '_get')
    if ast.dump(actual, include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise ValueError('Loaded parent cache method differs from pinned source')
    # Retain the private runtime's exact C/copy references (including M16 binding).
    # This new dictionary does not mutate any original runtime globals.
    namespace = dict(parent_cache_class._get.__globals__)
    namespace['session_closed'] = session_closed
    get = types.FunctionType(_get_closed_flag.__code__, namespace, '_get')
    get.__qualname__ = 'ClosedFlagContextCache._get'
    result = type('ClosedFlagContextCache', (parent_cache_class,),
        dict(_get=get, __module__=__name__, closed_flag_source_proof=proof))
    return result


def bind_runtime(runtime, *, parent_runtime_path, diagnostic_allow_closed_flag=False):
    """Copy the runtime API, replacing only its cache class before construction.

    Use this after the frozen native runtime is loaded and before creating the
    candidate's empty cache. Do not migrate or replace live SDK contexts.
    Calling this function performs source checks only; it opens no context.
    """
    if diagnostic_allow_closed_flag is not True:
        raise ValueError('Explicit diagnostic_allow_closed_flag required')
    path = Path(parent_runtime_path).resolve(strict=True)
    # Raw closed is equivalent only for the original session metrics contract.
    method = runtime.GroupedSession.metrics
    if Path(inspect.getsourcefile(method)).resolve() != path:
        raise ValueError('Original GroupedSession.metrics contract required')
    actual = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]
    expected = _function(ast.parse(path.read_text()), 'GroupedSession', 'metrics')
    if ast.dump(actual, include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise ValueError('Custom session metrics override cannot use a raw closed query')
    new_class = make_cache_class(runtime.BoundedContextCache, parent_runtime_path=path)
    api = dict(vars(runtime)); api['BoundedContextCache'] = new_class
    pins = dict(runtime.source_pins)
    pins[str(path)] = PARENT_RUNTIME_SHA256
    pins[str(Path(__file__).resolve())] = sha256(__file__)
    api.update(source_pins=pins, closed_flag_only=True,
        closed_flag_source_proof=copy.deepcopy(new_class.closed_flag_source_proof),
        candidate_accepted=False)
    if sha256(path) != PARENT_RUNTIME_SHA256:
        raise ValueError('Parent source changed during cache binding')
    return types.SimpleNamespace(**api)
