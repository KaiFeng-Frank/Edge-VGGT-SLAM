"""Generated pair lifecycle/bridge checks; no real Torch, SDK or model load."""
import contextlib
import ctypes as ct
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from vggt_slam import rknn_aggregator_pairs as runtime
import test_rknn_aggregator_blocks as original
from test_rknn_aggregator_blocks import FakeTensor, FakeSDK, fake_abi, original_forwards


class PairSDK(FakeSDK):
    def rknn_init(self, pointer, filename, *unused):
        context = len(self.initialized) + 1
        ct.cast(pointer, ct.POINTER(ct.c_uint64))[0] = context
        self.initialized.append(context)
        self.shapes[context] = runtime.SHAPES['frame']
        return -9 if context == self.fail_init_at else 0

    def query(self, api, context, command, attr):
        if command == 0:
            return type('Counts', (), dict(n_input=1, n_output=2))()
        if command in (1, 2):
            binding = (dict(index=0, name='input', shape=list(runtime.SHAPES['frame']))
                       if command == 1 else runtime.OUTPUT_BINDINGS[attr.index])
            return dict(index=binding['index'], name=binding['name'], shape=binding['shape'],
                n_elems=int(np.prod(binding['shape'])), type=1, quantization_type=0,
                zero_point=0, scale=1., format=3, pass_through=0,
                size=int(np.prod(binding['shape']))*2)
        return super().query(api, context, command, attr)

    def rknn_inputs_set(self, context, count, inputs):
        assert count == 1 and inputs[0].type == 0 and inputs[0].pass_through == 0 and inputs[0].fmt == 3
        assert inputs[0].size == int(np.prod(runtime.SHAPES['frame']))*4
        self.input = np.ctypeslib.as_array(ct.cast(inputs[0].buf, ct.POINTER(ct.c_float)),
            shape=(int(np.prod(runtime.SHAPES['frame'])),)).copy()
        self.events.append(('input', context.value, float(self.input[0])))
        return 0

    def rknn_run(self, context, pointer):
        frame = self.input + np.float32(context.value)
        self.buffers = [frame, frame + np.float32(100)]
        ct.cast(pointer, ct.POINTER(original.RunExtend)).contents.frame_id = 42
        self.events.append(('run', context.value))
        return 0

    def rknn_outputs_get(self, context, count, outputs, pointer):
        assert count == 2 and [o.index for o in outputs] == [0, 1]
        assert all(o.want_float == 1 and o.is_prealloc == 0 for o in outputs)
        for index in range(1 if self.partial_get else 2):
            outputs[index].buf, outputs[index].size = self.buffers[index].ctypes.data, self.buffers[index].nbytes
        ct.cast(pointer, ct.POINTER(original.OutputExtend)).contents.frame_id = 42
        return -17 if self.partial_get else 0

    def rknn_outputs_release(self, context, count, outputs):
        assert count == 2
        for index, output in enumerate(outputs):
            if output.buf:
                assert output.buf == self.buffers[index].ctypes.data
                self.buffers[index].fill(-999.)
                output.buf = None
        self.releases += 1
        return 0


@contextlib.contextmanager
def fixture(api):
    # Reuse the existing tiny original-parameter/source/ABI installation fixture.
    with original.installation_fixture(api) as (model, args, options):
        manifest = args[0]
        before = json.loads(manifest.read_text())
        parents = {(b['role'], b['index']): b for b in before['blocks']}
        rows = []
        for index in range(24):
            paths = [manifest.parent / f'pair-{index:02d}.{extension}' for extension in ('onnx', 'rknn')]
            for path in paths:
                path.write_bytes(b'generated complete original-pair placeholder')
            def item(path):
                return dict(file=path.name, bytes=path.stat().st_size, sha256=original.runtime.sha(path))
            rows.append(dict(item(paths[0]), role='pair', index=index, rknn=item(paths[1]),
                input_name='input', input_shape=list(runtime.SHAPES['frame']),
                output_bindings=runtime.OUTPUT_BINDINGS, heads=16,
                parents={r: parents[(r, index)] for r in runtime.SHAPES},
                positions={r: parents[(r, index)]['positions'] for r in runtime.SHAPES},
                parameter_raw_sha256={r: parents[(r, index)]['parameter_raw_sha256'] for r in runtime.SHAPES},
                all_36_original_parameter_values_preserved_by_parent_composition=True,
                parent_nodes_plus_one_reshape=True, original_aggregator_concat_must_be_retained=True))
        before.update(schema='v55_original_VGGT_24_frame_global_pair_RKNN_manifest', blocks=rows,
            pair_composer_source_sha256=runtime.COMPOSER_SHA, all_original_48_blocks_retained=True,
            original_aggregator_concat_must_be_retained=True)
        manifest.write_text(json.dumps(before))
        with contextlib.ExitStack() as stack:
            for name in ('ABI_SHA', 'HEADER_SHA', 'LIBRARY_SHA'):
                stack.enter_context(patch.object(runtime, name, getattr(original.runtime, name)))
            yield model, (manifest, original.runtime.sha(manifest)), options


def positions(handle, index, role):
    # Position fixtures are all zero; the adapter still verifies the real shapes.
    return FakeTensor(np.zeros((*runtime.SHAPES[role][:2], 2), np.int64))


class OriginalPairTests(unittest.TestCase):
    def test_two_full_loops_use_24_sdk_calls_each_and_both_owned_outputs(self):
        api = PairSDK()
        with fixture(api) as (model, args, options):
            originals = original_forwards(model)
            handle = runtime.install_rknn_aggregator_pairs(model, *args, **options)
            try:
                agg = model.aggregator
                images = FakeTensor(np.zeros((1,2,3,84,112), np.float32))
                for _ in range(2):
                    handle._guard(agg, (images,), {})
                    value, expected = FakeTensor(np.zeros(runtime.SHAPES['frame'], np.float32)), 0.
                    retained = []
                    for index in range(24):
                        frame = agg.frame_blocks[index].forward(value, positions(handle,index,'frame'))
                        expected += index + 1
                        np.testing.assert_array_equal(frame.numpy(), np.full(runtime.SHAPES['frame'], expected, np.float32))
                        self.assertEqual(handle.metrics()['pending_pairs'], 1)
                        self.assertTrue(all(np.all(b == -999.) for b in api.buffers))
                        # Exactly the two original .view calls; no CPU reference reset.
                        bridge = FakeTensor(frame.numpy().reshape(1,2,53,1024).reshape(runtime.SHAPES['global']))
                        global_ = agg.global_blocks[index].forward(bridge, positions(handle,index,'global'))
                        expected += 100
                        np.testing.assert_array_equal(global_.numpy(), np.full(runtime.SHAPES['global'], expected, np.float32))
                        if index in (4,11,17,23):
                            retained.append(np.concatenate((frame.numpy(), global_.numpy().reshape(frame.shape)), axis=-1))
                        value = FakeTensor(global_.numpy().reshape(runtime.SHAPES['frame']))
                    self.assertEqual(len(retained),4)
                    self.assertTrue(all(a.shape==(2,53,2048) for a in retained))
                    self.assertEqual(handle.metrics()['pending_pairs'],0)
                stats=handle.metrics()
                self.assertEqual((stats['completed_blocks'],stats['completed_pairs'],stats['sdk_runs_succeeded']), (96,48,48))
                self.assertEqual([e[1] for e in api.events if e[0]=='run'], list(range(1,25))*2)
                self.assertEqual(stats['block_calls'], {f'{r}{i}':2 for r in runtime.SHAPES for i in range(24)})
                self.assertEqual(api.releases,48)
            finally:
                handle.close()
            for block,had,value,forward in originals:
                self.assertEqual('forward' in block.__dict__,had)
                self.assertIs(block.__dict__.get('forward'),value)
                self.assertEqual(block.forward,forward)
            self.assertEqual(api.destroyed,list(range(1,25)))
            self.assertEqual(model.hooks[0].removed,1)
            self.assertFalse(hasattr(model.aggregator,'_vggt_rknn_blocks_owner'))
            self.assertEqual(handle.metrics()['live_contexts'],0)
            handle.close()
            self.assertEqual(api.destroyed,list(range(1,25)))

    def test_partial_initialization_failure_releases_without_patching(self):
        api=PairSDK(fail_init_at=3)
        with fixture(api) as (model,args,options):
            before=[row[3] for row in original_forwards(model)]
            with self.assertRaisesRegex(RuntimeError,'rknn_init returned -9'):
                runtime.install_rknn_aggregator_pairs(model,*args,**options)
            self.assertEqual([row[3] for row in original_forwards(model)],before)
            self.assertEqual(api.destroyed,[1,2,3])
            self.assertFalse(model.hooks)
            self.assertFalse(hasattr(model.aggregator,'_vggt_rknn_blocks_owner'))

    def test_second_output_descriptor_mismatch_releases_all_unpatched_contexts(self):
        class WrongIndexSDK(PairSDK):
            def query(self,api,context,command,attr):
                row=super().query(api,context,command,attr)
                if command==2 and attr.index==1 and context.value==4:
                    row['index']=0
                return row
        api=WrongIndexSDK()
        with fixture(api) as (model,args,options):
            before=[row[3] for row in original_forwards(model)]
            with self.assertRaisesRegex(ValueError,'floating pair output'):
                runtime.install_rknn_aggregator_pairs(model,*args,**options)
            self.assertEqual([row[3] for row in original_forwards(model)],before)
            self.assertEqual(api.destroyed,[1,2,3,4])
            self.assertFalse(model.hooks)

    def test_failed_partial_outputs_get_releases_and_blocks_further_calls(self):
        api=PairSDK(partial_get=True)
        with fixture(api) as (model,args,options):
            handle=runtime.install_rknn_aggregator_pairs(model,*args,**options)
            try:
                forward=model.aggregator.frame_blocks[0].forward
                x=FakeTensor(np.zeros(runtime.SHAPES['frame'],np.float32))
                with self.assertRaisesRegex(RuntimeError,'rknn_outputs_get returned -17'):
                    forward(x,positions(handle,0,'frame'))
                self.assertEqual(api.releases,1)
                self.assertEqual(handle.metrics()['pending_pairs'],0)
                with self.assertRaisesRegex(RuntimeError,'no fallback'):
                    forward(x,positions(handle,0,'frame'))
                self.assertEqual(handle.metrics()['sdk_run_attempts'],1)
            finally:
                handle.close()

    def test_copied_or_mutated_bridge_and_duplicate_frame_reject_and_clear_pending(self):
        for kind in ('copy','mutation','duplicate','wrong_index'):
            with self.subTest(kind=kind):
                api=PairSDK()
                with fixture(api) as (model,args,options):
                    handle=runtime.install_rknn_aggregator_pairs(model,*args,**options)
                    try:
                        agg=model.aggregator
                        frame=agg.frame_blocks[0].forward(FakeTensor(np.zeros(runtime.SHAPES['frame'],np.float32)),positions(handle,0,'frame'))
                        array=frame.numpy().reshape(runtime.SHAPES['global'])
                        if kind=='copy':
                            array=array.copy()  # Identical values cannot replace provenance.
                        if kind=='mutation':
                            array[0,0,0]+=1  # Same pointer cannot hide an in-place write.
                        with self.assertRaises(RuntimeError):
                            if kind=='duplicate':
                                agg.frame_blocks[0].forward(frame,positions(handle,0,'frame'))
                            else:
                                index=1 if kind=='wrong_index' else 0
                                agg.global_blocks[index].forward(FakeTensor(array),positions(handle,index,'global'))
                        self.assertTrue(handle.failed)
                        self.assertEqual(handle.metrics()['pending_pairs'],0)
                        self.assertEqual(handle.metrics()['sdk_runs_succeeded'],1)
                    finally:
                        handle.close()

    def test_close_clears_unconsumed_frame_and_fixed_mode_guards_reject(self):
        api=PairSDK()
        with fixture(api) as (model,args,options):
            handle=runtime.install_rknn_aggregator_pairs(model,*args,**options)
            model.aggregator.frame_blocks[0].forward(FakeTensor(np.zeros(runtime.SHAPES['frame'],np.float32)),positions(handle,0,'frame'))
            handle.close()
            self.assertEqual(handle.metrics()['pending_pairs'],0)
            self.assertEqual(handle.metrics()['live_contexts'],0)
        for kwargs in ({'compute_similarity':True}, {}):
            api=PairSDK()
            with fixture(api) as (model,args,options):
                handle=runtime.install_rknn_aggregator_pairs(model,*args,**options)
                try:
                    shape=(1,2,3,84,112) if kwargs else (1,3,3,84,112)
                    with self.assertRaisesRegex(RuntimeError,'no fallback'):
                        handle._guard(model.aggregator,(FakeTensor(np.zeros(shape,np.float32)),),kwargs)
                    self.assertTrue(handle.failed)
                    self.assertEqual(handle.metrics()['sdk_run_attempts'],0)
                finally:
                    handle.close()


if __name__=='__main__':
    unittest.main()
