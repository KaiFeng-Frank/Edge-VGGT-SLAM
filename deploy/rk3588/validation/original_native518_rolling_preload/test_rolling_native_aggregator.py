#!/usr/bin/env python3
"""Portable fake-only concurrency tests; no private artifacts or NN/SDK calls.

The public r2 runtime is imported without constructing its board-only model
executor. Temporary text fixtures simulate close-time source pins; these are
not model sources, original artifact verification or a board precision test.
"""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np

SOURCE=Path(__file__).resolve().parent/'r2/rolling_native_aggregator.py'
EXPECTED_SHA='99a08a3cb7823e5212a12184e487413318f206a10bedea59fe87110df6065f21'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==EXPECTED_SHA
spec=importlib.util.spec_from_file_location('_v123_public_portable_fake_runtime',SOURCE)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
M.FRAME=(2,3,4);M.GLOBAL=(1,6,4)
FAKE_PINS={}
_pin_fixture=None
_original_frozen,_original_sequential_sha=M.FROZEN,M.SEQUENTIAL_SHA


def setUpModule():
    global _pin_fixture
    _pin_fixture=tempfile.TemporaryDirectory(prefix='v123-public-fake-pins-')
    M.FROZEN=Path(_pin_fixture.name)
    for name in ('native_sequential_runtime.py','run_native_retained_groups.py','native_heads_runtime.py'):
        # Read/hash only: these deliberately non-model fixtures are never imported.
        content=('# Generated fake-only close-pin fixture: '+name+'\n').encode()
        (M.FROZEN/name).write_bytes(content)
        FAKE_PINS[name]=hashlib.sha256(content).hexdigest()
    M.SEQUENTIAL_SHA=FAKE_PINS['native_sequential_runtime.py']


def tearDownModule():
    global _pin_fixture
    M.FROZEN,M.SEQUENTIAL_SHA=_original_frozen,_original_sequential_sha
    if _pin_fixture is not None:
        _pin_fixture.cleanup();_pin_fixture=None


class Harness:
    def __init__(self,*,init_fail=None,run_fail=None,close_fail=False,blocked_init=False):
        self.init_fail,self.run_fail,self.close_fail=init_fail,run_fail,close_fail
        self.blocked_init=blocked_init;self.allow_init=threading.Event()
        self.lock=threading.RLock();self.live=0;self.peak=0;self.running=0;self.peak_running=0
        self.records=[];self.inputs={};self.contexts=[]

    def factory(self,row):
        if self.blocked_init and row['index']==0:self.allow_init.wait()
        context=FakeContext(self,row);self.contexts.append(context)
        if row['index']==self.init_fail:
            error=RuntimeError('generated partial init failure');error.owned_context=context;raise error
        return context


class FakeContext:
    def __init__(self,h,row):
        self.h,self.index,self.owner=h,row['index'],threading.get_ident()
        self.closed=False;self.last_timings={}
        self.stats={key:0 for key in M.COUNTERS}
        self.stats.update(init_attempts=1,contexts_created=1,cleanup_errors=[],SDK_memory={'dma_bytes':100+self.index})
        with h.lock:
            h.live+=1;h.peak=max(h.peak,h.live);h.records.append((self.index,'init',self.owner))

    def run(self,inputs):
        assert threading.get_ident()==self.owner and not self.closed
        h=self.h
        with h.lock:
            h.running+=1;h.peak_running=max(h.peak_running,h.running)
            h.inputs[self.index]=inputs[0].copy();h.records.append((self.index,'run',threading.get_ident()))
        try:
            time.sleep(.02)
            if self.index==h.run_fail:raise RuntimeError('generated run failure')
            y=np.add(inputs[0],np.float32(10*(self.index+1)),dtype=np.float32)
            values=[np.add(inputs[0],np.float32(self.index+1),dtype=np.float32),y.reshape(M.GLOBAL).copy()]
            if self.index==3:values.append(np.add(y,np.float32(7),dtype=np.float32).reshape(M.GLOBAL).copy())
            self.stats['run_attempts']+=1;self.stats['runs_completed']+=1
            self.stats['output_buffers_acquired']+=len(values);self.stats['output_buffers_released']+=len(values)
            self.last_timings={'SDK_run_seconds':.02,'complete_boundary_seconds':.02}
            return values
        finally:
            with h.lock:h.running-=1

    def close(self):
        assert threading.get_ident()==self.owner
        if self.closed:return
        if self.index==0 and self.h.close_fail:raise RuntimeError('generated retained close failure')
        with self.h.lock:
            self.h.live-=1;self.h.records.append((self.index,'close',threading.get_ident()))
        self.stats['destroy_attempts']+=1;self.stats['contexts_destroyed']+=1;self.closed=True

    def metrics(self):
        assert threading.get_ident()==self.owner
        return copy.deepcopy(dict(self.stats,closed=self.closed,failed=False,
            live_contexts=int(not self.closed),live_output_buffers=0))


def make_runtime(h,directory):
    obj=M.NativeRollingGroups.__new__(M.NativeRollingGroups);obj._rolling_setup()
    obj.np=np;obj.torch=SimpleNamespace(inference_mode=contextlib.nullcontext,from_numpy=lambda x:x)
    obj.core_mask=7;obj._wait_seconds=.5
    obj._frozen=SimpleNamespace(ORIGINAL_RUNTIME_SHA=FAKE_PINS['run_native_retained_groups.py'],
        OWNED_RUNTIME_SHA=FAKE_PINS['native_heads_runtime.py'])
    obj.rows=[]
    for index,last in enumerate(M.RETAINED):
        path=Path(directory)/('group'+str(index));path.write_bytes(b'fake')
        obj.rows.append(dict(index=index,indices=[last],file=str(path),bytes=4))
    obj._factory=h.factory;obj.observe=lambda: {'generated_only':True}
    def consume(row,ys,retained):
        retained[row['indices'][-1]]=np.concatenate([ys[0].reshape(1,*M.FRAME),ys[1].reshape(1,*M.FRAME)],axis=-1)
        target=None
        if row['index']==3:
            value=ys[2].reshape(M.FRAME).copy();target=value/(np.linalg.norm(value,axis=-1,keepdims=True)+1e-8)
        return ys[1].reshape(M.FRAME),target
    obj._consume=consume
    return obj


class Tests(unittest.TestCase):
    def test_dependency_owner_limit_capture_and_snapshots(self):
        with tempfile.TemporaryDirectory(prefix='v123-generated-') as directory:
            h=Harness();r=make_runtime(h,directory);self.assertEqual(r._workers,[])
            x=np.arange(24,dtype=np.float32).reshape(M.FRAME)
            result=r.run(x,capture=True);r.close();ledger=r.metrics()
            for i,delta in enumerate((0,10,30,60)):self.assertTrue(np.array_equal(h.inputs[i],x+delta))
            self.assertLessEqual(h.peak,2);self.assertEqual(h.peak_running,1)
            self.assertEqual(h.live,0);self.assertEqual(ledger['contexts_created'],4)
            self.assertEqual(ledger['contexts_destroyed'],4);self.assertEqual(ledger['output_buffers_acquired'],9)
            self.assertEqual(len(result['diagnostic_outputs']),9);self.assertEqual(len(result['aggregated_tokens_list']),24)
            self.assertEqual([i for i,v in enumerate(result['aggregated_tokens_list']) if v is not None],list(M.RETAINED))
            for context in h.contexts:
                self.assertTrue(all(t==context.owner for i,_,t in h.records if i==context.index))
                self.assertNotEqual(context.owner,threading.get_ident())
            timeline=ledger['timeline']
            for following,current in ((2,0),(3,1)):
                opened=next(t['monotonic_seconds'] for t in timeline if t['group']==following and t['phase']=='init_begin')
                closed=next(t['monotonic_seconds'] for t in timeline if t['group']==current and t['phase']=='close_end')
                self.assertGreaterEqual(opened,closed)
            ledger['per_group_closed_contexts'][0]['contexts_created']=99
            self.assertEqual(r.metrics()['contexts_created'],4)

    def test_partial_init_and_run_failure_cleanup(self):
        for mode in ('init','run'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(prefix='v123-generated-') as directory:
                h=Harness(init_fail=1 if mode=='init' else None,run_fail=1 if mode=='run' else None)
                r=make_runtime(h,directory)
                with self.assertRaises(RuntimeError):r.run(np.zeros(M.FRAME,np.float32))
                r.close();ledger=r.metrics()
                self.assertTrue(r.failed);self.assertEqual(h.live,0);self.assertLessEqual(h.peak,2)
                self.assertNotIn(3,[i for i,phase,_ in h.records if phase=='init'])
                self.assertTrue(ledger['resources_balanced']);self.assertEqual(ledger['live_owner_workers'],0)

    def test_close_failure_retains_owner_until_same_thread_retry(self):
        with tempfile.TemporaryDirectory(prefix='v123-generated-') as directory:
            h=Harness(close_fail=True);r=make_runtime(h,directory)
            try:
                with self.assertRaises(RuntimeError):r.run(np.zeros(M.FRAME,np.float32))
                self.assertTrue(any(w.thread.is_alive() for w in r._workers))
                self.assertTrue(any(w.context is not None and not w.context.closed for w in r._workers))
                with self.assertRaises(RuntimeError):r.metrics()
                self.assertNotIn(2,[i for i,phase,_ in h.records if phase=='init'])
            finally:
                h.close_fail=False;r.close()
            self.assertEqual(h.live,0);self.assertTrue(r.metrics()['resources_balanced'])

    def test_unpublished_initializer_timeout_does_not_claim_zero_live(self):
        with tempfile.TemporaryDirectory(prefix='v123-generated-') as directory:
            h=Harness(blocked_init=True);r=make_runtime(h,directory);r._wait_seconds=.02
            try:
                with self.assertRaises(TimeoutError):r.run(np.zeros(M.FRAME,np.float32))
                with self.assertRaises(RuntimeError):r.metrics()
                self.assertTrue(r._workers[0].thread.is_alive())
            finally:
                h.allow_init.set();r._wait_seconds=.5;r.close()
            self.assertEqual(h.live,0);self.assertFalse(any(w.thread.is_alive() for w in r._workers))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,help='Optional fresh small JSON receipt; no arrays are saved')
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt=dict(schema='v123_public_portable_rolling_preload_fake_concurrency_check',
        status='passed_portable_fake_checks_only' if result.wasSuccessful() else 'failed',runtime_sha256=EXPECTED_SHA,
        test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),tests_run=result.testsRun,
        fake_source_pin_sha256=FAKE_PINS,temporary_pin_fixtures_cleaned=_pin_fixture is None,
        public_runtime_source_unchanged=hashlib.sha256(SOURCE.read_bytes()).hexdigest()==EXPECTED_SHA,
        private_artifacts_required=False,model_Torch_OpenCL_RKNN_or_board_executed=False,
        actual_precision_performance_or_8GB_accepted=False)
    if args.report is not None:
        with args.report.open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt))
    raise SystemExit(0 if result.wasSuccessful() else 1)
