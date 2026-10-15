#!/usr/bin/env python3
"""Generated SDK-owner fixtures only; no SDK, model, tensor or GPU calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
import types

ROOT=Path(__file__).resolve().parent
RUNTIME_SHA='1785466d9d60b49589fdac27ed6c3ec5ef6fea02a3a3976b65a90d7d8b6bc38d'


class FakeSDKOwner:
    def __init__(self,path,digest,**kwargs):
        self.owner=threading.get_ident();self.closed=False;self.last_timings={}
        self.attempts=[];self.close_ids=[];self.failure=kwargs.get('fixture_failure')
        self.barrier=kwargs.get('fixture_barrier');self.heads=kwargs['input_shapes'][0][1]
        self.close_attempts=0
        if self.failure=='init':
            exc=RuntimeError('generated partial constructor failure');exc.owned_context=self;raise exc
    def run(self,inputs):
        assert threading.get_ident()==self.owner and not self.closed
        self.attempts.append(threading.get_ident())
        if self.barrier is not None:self.barrier.wait(timeout=3.)
        if self.failure=='run':raise RuntimeError('generated partial run failure')
        self.last_timings={'fixture_only':True};return [self.heads]
    def close(self):
        assert threading.get_ident()==self.owner
        self.close_ids.append(threading.get_ident());self.close_attempts+=1
        if self.failure=='close_once' and self.close_attempts==1:raise RuntimeError('generated release failure; owner retained')
        self.closed=True
    def metrics(self):return dict(closed=self.closed,live_contexts=int(not self.closed),live_output_buffers=0,
        resource_counts_balanced=self.closed,cleanup_errors=[],contexts_created=1,contexts_destroyed=int(self.closed),
        runs_completed=len(self.attempts),output_buffers_acquired=0,output_buffers_released=0)


def main():
    path=ROOT/'run_sdpa_head_parallel_abba.py';source_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert source_sha==RUNTIME_SHA
    spec=importlib.util.spec_from_file_location('_v112_generated_owner_check',path)
    M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
    O=types.SimpleNamespace(OwnedContext=FakeSDKOwner)
    results=[];row=dict(model_path='generated_fixture_no_model',rknn={'sha256':'fixture_no_file'})

    def workers(failures,barrier=None):
        return [M.DedicatedWorker(O,row,part,dict(fixture_failure=failure,fixture_barrier=barrier))
            for part,failure in zip(M.PARTS,failures)]
    def clean(items):
        for item in items:
            if not item.shutdown:item.close(5.)
    def expect_failure(job):
        try:job.result(timeout=5.)
        except RuntimeError:return
        raise AssertionError('Generated failing job unexpectedly succeeded')

    items=workers([None]*3,threading.Barrier(3))
    try:
        for item in items:item.init_future.result(timeout=5.)
        jobs=[item.submit([]) for item in items]
        assert [job.result(timeout=5.)[0] for job in jobs]==[6,5,5]
        assert len({item.thread_id for item in items})==3
        clean(items)
        assert all(item.metrics()['SDK_thread_ownership_preserved'] and item.shutdown and item.owner.closed for item in items)
        results.append(dict(case='three_distinct_workers_reach_barrier_concurrently_and_close_on_owner_threads',passed=True))
    finally:clean(items)

    items=workers(['init'])
    try:
        expect_failure(items[0].init_future)
        assert items[0].owner is not None and not items[0].owner.closed
        clean(items);assert items[0].owner.closed and items[0].owner.close_ids==[items[0].thread_id]
        results.append(dict(case='partial_constructor_owner_remains_reachable_for_worker_cleanup',passed=True))
    finally:clean(items)

    items=workers(['run',None,None],threading.Barrier(3))
    try:
        for item in items:item.init_future.result(timeout=5.)
        jobs=[item.submit([]) for item in items];expect_failure(jobs[0])
        clean(items)
        assert all(item.owner.closed and item.shutdown and item.owner.attempts==[item.thread_id] for item in items)
        results.append(dict(case='one_failed_future_keeps_all_three_owners_reachable_until_worker_queue_cleanup',passed=True))
    finally:clean(items)

    items=workers(['close_once'])
    try:
        item=items[0];item.init_future.result(timeout=5.)
        try:item.close(5.)
        except RuntimeError:pass
        else:raise AssertionError('Generated first release failure missing')
        assert not item.shutdown and not item.owner.closed
        item.close(5.);assert item.shutdown and item.owner.closed and item.owner.close_ids==[item.thread_id]*2
        results.append(dict(case='failed_close_keeps_same_worker_owner_available_for_retry',passed=True))
    finally:clean(items)

    assert hashlib.sha256(path.read_bytes()).hexdigest()==RUNTIME_SHA
    report=dict(schema='v112_generated_dedicated_worker_owner_checks',status='passed',runtime_source_sha256=RUNTIME_SHA,
        test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=results,
        actual_SDK_NN_GPU_board_execution=False,learned_model_or_tensor_math_executed=False,
        all_fixture_worker_pools_shutdown=True,runtime_source_unchanged=True,
        limitation='Thread ownership/reachable cleanup fixtures only; not real SDK cleanup, numerical or performance acceptance.')
    (ROOT/'runtime-synthetic-owner-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));return 0


if __name__=='__main__':raise SystemExit(main())
