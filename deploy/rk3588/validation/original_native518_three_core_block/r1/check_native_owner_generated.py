"""Tiny fake-only ownership checks. No arrays, model, SDK, OpenCL or board."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'native_dino_runtime.py'
EXPECTED_SHA = 'bfee2265cd0c8685c45aa04cfbc21f905d54c5f029a63250259606ce59bb0454'


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED_SHA
    spec = importlib.util.spec_from_file_location('_r124_fake_native',SOURCE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.WAIT_SECONDS = .03  # Synthetic wait; deployed source remains unchanged.
    checks = []

    class Product:
        def __init__(self,owner,fail_once=False):
            self.owner,self.fail_once = owner,fail_once
            self.k = self.n = 1
            self.ctx = SimpleNamespace(value=1)
            self.memories = dict(A=1,B=2,C=3)
            self.descriptors,self.cleanup_errors = {},[]

        def close(self):
            self.owner.check_thread()
            if self.fail_once:
                self.fail_once = False
                self.cleanup_errors.append('forced borrower release failure')
            else:
                if self.owner.lane == 0:
                    assert all(not any(p.ctx.value or p.memories for p in w.products.values())
                        for w in self.owner.runtime.workers[1:])
                self.memories.clear()
                self.ctx.value = 0
            return dict(K=1,output_N=1,live_context=int(bool(self.ctx.value)),
                live_memories=len(self.memories),cleanup_errors=list(self.cleanup_errors))

    class GPU:
        def __init__(self): self.closed,self.close_calls = False,0
        def close(self): self.close_calls += 1;self.closed = True
        def metrics(self): return dict(fully_released=self.closed,queue_idle=True,live_events=0)

    def fake(fail_borrower=False):
        x = m.NativeDino.__new__(m.NativeDino)
        x.main_ident = threading.get_ident()
        x.api,x.counter_lock = object(),threading.Lock()
        x.workers,x.inflight,x.products = [],[],{}
        x.closed = x.failed = False
        x.calls_attempted = x.calls_completed = 0
        x.cleanup,x.original_weight_partitions,x._half_weights = {},{},{}
        x.shared_A = dict(fake_leader_A=1)
        x.bridge = GPU()
        for lane in range(3):
            owner = m._Worker(x,lane)
            x.workers.append(owner)
            owner.start()
            def attach(w=owner):
                w.check_thread()
                w.products['fake'] = Product(w,fail_borrower and w.lane == 1)
            x._join([owner.submit(attach)])
        return x

    first = fake(True)
    try:
        try:
            first.close()
            raise AssertionError('Borrower failure was hidden')
        except RuntimeError:
            pass
        assert first.bridge.closed
        assert first.workers[0].products['fake'].ctx.value == 1
        assert first.metrics()['live_owner_workers'] == 3
        assert not first.cleanup['borrowed_A_bindings_released']
        terminal = first.close()
        assert terminal['live_contexts'] == terminal['live_memories'] == terminal['live_owner_workers'] == 0
        assert terminal['borrowed_A_bindings_released'] and len(terminal['resources']) == 1
        # Distinct owner snapshots retain all three fake resources and history.
        assert len(terminal['owner_metrics']['fixed_owner_workers']) == 3
        assert terminal == first.close()
        copied = first.worker_metrics()
        copied[0]['resources']['fake']['live_context'] = 99
        assert first.worker_metrics()[0]['resources']['fake']['live_context'] == 0
        checks.append('borrower failure retains leader and workers; same-owner retry and idempotent ledger')
    finally:
        if any(w.thread.is_alive() for w in first.workers): first.close()

    second = fake()
    release,entered = threading.Event(),threading.Event()
    try:
        def block(): entered.set();release.wait(2.)
        future = second.workers[0].submit(block)
        second.inflight.append(future)
        assert entered.wait(1.)
        try:
            second.close()
            raise AssertionError('Busy owner close unexpectedly succeeded')
        except RuntimeError:
            pass
        assert second.bridge.close_calls == 0
        assert all(w.products['fake'].ctx.value == 1 for w in second.workers)
        try:
            second.metrics()
            raise AssertionError('Busy owner falsely reported terminal metrics')
        except RuntimeError:
            pass
        release.set()
        future.result(timeout=1.)
        terminal = second.close()
        assert terminal['live_contexts'] == terminal['live_memories'] == terminal['live_owner_workers'] == 0
        checks.append('busy timeout preserves GPU/native owners; terminal metrics refuses busy state; retry joins')
    finally:
        release.set()
        if any(w.thread.is_alive() for w in second.workers): second.close()

    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED_SHA
    report = dict(schema='v124_generated_native_owner_failure_smoke',status='passed_fake_only',
        runtime_sha256=EXPECTED_SHA,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checks=checks,check_count=len(checks),all_fake_owner_threads_joined=True,
        model_SDK_OpenCL_board_or_array_execution=False,deployed_source_unchanged=True,
        numerical_or_performance_acceptance=False)
    path = HERE / 'generated-owner-smoke.json'
    with path.open('x') as f: json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
