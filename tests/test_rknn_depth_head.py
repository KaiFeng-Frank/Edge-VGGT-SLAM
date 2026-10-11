"""Small generated DPT IO/chunk checks; no real model, Torch or SDK execution."""
import ctypes as ct
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from vggt_slam import rknn_depth_head as runtime
from test_rknn_aggregator_blocks import FakeTensor, FakeSDK, fake_torch, fake_abi, RunExtend, OutputExtend


class Tensor(FakeTensor):
    def view(self,*shape):
        return Tensor(self.array.reshape(shape))


class Head:
    training=False
    activation,conf_activation='exp','expp1'

    def _forward_impl(self,*args,**kwargs):
        return 'generated original implementation placeholder'

    def forward(self,tokens,images,patch_start_idx,frames_chunk_size=1):
        # A service double for the original CPU chunk loop, never a real head.
        predictions,confidences=[],[]
        for start in range(0,images.shape[1],frames_chunk_size):
            pred,conf=self._forward_impl(tokens,images,patch_start_idx,start,min(start+frames_chunk_size,images.shape[1]))
            predictions.append(pred.numpy());confidences.append(conf.numpy())
        return Tensor(np.concatenate(predictions,axis=1)),Tensor(np.concatenate(confidences,axis=1))


class DepthSDK(FakeSDK):
    def __init__(self,*,partial_get=False):
        super().__init__(partial_get=partial_get)
        self.frame_inputs=[]

    def rknn_inputs_set(self,context,count,inputs):
        assert count==4
        values=[]
        for index in range(4):
            row=inputs[index]
            assert row.index==index and row.type==0 and row.pass_through==0 and row.fmt==3 and row.size==48*2048*4
            value=np.ctypeslib.as_array(ct.cast(row.buf,ct.POINTER(ct.c_float)),shape=(48*2048,))
            assert np.all(value==value[0])  # No special tokens enter any graph input.
            values.append(float(value[0]))
        self.frame_inputs.append(values)
        return 0

    def rknn_run(self,context,pointer):
        ct.cast(pointer,ct.POINTER(RunExtend)).contents.frame_id=17
        self.buffer=np.full((1,2,84,112),self.frame_inputs[-1][0],np.float32)
        return 0

    def rknn_outputs_get(self,context,count,outputs,pointer):
        assert count==1 and outputs[0].want_float==1
        outputs[0].buf,outputs[0].size=self.buffer.ctypes.data,self.buffer.nbytes
        ct.cast(pointer,ct.POINTER(OutputExtend)).contents.frame_id=17
        return -17 if self.partial_get else 0

    def rknn_outputs_release(self,context,count,outputs):
        assert count==1 and outputs[0].buf==self.buffer.ctypes.data
        self.releases+=1
        self.buffer.fill(-999.)
        outputs[0].buf=None
        return 0


def generated_handle(*,partial_get=False):
    api=DepthSDK(partial_get=partial_get)
    handle=runtime.RKNNOriginalDepthHead.__new__(runtime.RKNNOriginalDepthHead)
    handle.base,handle.api,handle.np,handle.torch=fake_abi(api),api,np,fake_torch()
    handle.torch.from_numpy=Tensor
    handle.lock,handle.timeout=threading.RLock(),30000
    handle.closed=handle.failed=False
    handle.contract=SimpleNamespace(LAYERS=(4,11,17,23),OUTPUT_SHAPE=(1,2,84,112))
    handle.contexts=[ct.c_uint64(1)]
    handle.inputs=[{'format':3}]*4
    handle.stats=dict(sdk_run_attempts=0,sdk_runs_succeeded=0,sdk_io_seconds=0.,contexts_destroyed=0,
        cleanup_errors=[],completed_frames=0,completed_chunk_calls=0)
    handle.head=Head()
    handle.activation_calls=[]
    def activation(out,**kwargs):
        assert kwargs=={'activation':'exp','conf_activation':'expp1'}
        handle.activation_calls.append(out.numpy().copy())
        # A function-call double, deliberately not a reimplementation of activation math.
        return Tensor(out.numpy()[:,0,:,:,None]),Tensor(out.numpy()[:,1])
    handle.activate_head=activation
    installed=handle._make_forward_impl()
    handle.patch=(False,None,handle.head._forward_impl,installed)
    handle.head._forward_impl=installed
    handle.head._vggt_rknn_depth_owner=handle
    tokens=[None]*24
    for layer in handle.contract.LAYERS:
        array=np.empty((1,2,53,2048),np.float32)
        array[:,0]=layer;array[:,1]=layer+100
        array[:,:,:5]=-12345.
        tokens[layer]=Tensor(array)
    return handle,api,tokens,Tensor(np.zeros((1,2,3,84,112),np.float32))


class OriginalDepthTests(unittest.TestCase):
    def test_original_chunk_loop_selects_each_frame_and_calls_original_activation_binding(self):
        handle,api,tokens,images=generated_handle()
        original_forward=handle.head.forward
        try:
            pred,conf=handle.head.forward(tokens,images,5)
            self.assertEqual(pred.shape,(1,2,84,112,1));self.assertEqual(conf.shape,(1,2,84,112))
            self.assertEqual(api.frame_inputs,[[4.,11.,17.,23.],[104.,111.,117.,123.]])
            self.assertEqual(handle.head.forward,original_forward)
            self.assertEqual((handle.stats['completed_chunk_calls'],handle.stats['completed_frames'],api.releases),(2,2,2))
            np.testing.assert_array_equal(pred.numpy()[0,0],np.full((84,112,1),4.,np.float32))
            np.testing.assert_array_equal(pred.numpy()[0,1],np.full((84,112,1),104.,np.float32))
            self.assertEqual([float(x.flat[0]) for x in handle.activation_calls],[4.,104.])
            self.assertTrue(np.all(api.buffer==-999.))
        finally:
            handle.close()

    def test_whole_selected_chunk_still_uses_two_ordered_single_frame_graph_calls(self):
        handle,api,tokens,images=generated_handle()
        try:
            pred,conf=handle.head._forward_impl(tokens,images,5)
            self.assertEqual(pred.shape,(1,2,84,112,1));self.assertEqual(conf.shape,(1,2,84,112))
            self.assertEqual(api.frame_inputs,[[4.,11.,17.,23.],[104.,111.,117.,123.]])
            self.assertEqual(handle.stats['completed_chunk_calls'],1)
            self.assertEqual(len(handle.activation_calls),1)
        finally:
            handle.close()

    def test_partial_get_releases_owned_sdk_buffer_and_terminal_failure_prevents_retry(self):
        handle,api,tokens,images=generated_handle(partial_get=True)
        try:
            with self.assertRaisesRegex(RuntimeError,'rknn_outputs_get returned -17'):
                handle.head.forward(tokens,images,5)
            self.assertEqual(api.releases,1);self.assertTrue(handle.failed)
            self.assertFalse(handle.activation_calls)
            with self.assertRaisesRegex(RuntimeError,'no fallback'):
                handle.head.forward(tokens,images,5)
            self.assertEqual(handle.stats['sdk_run_attempts'],1)
        finally:
            handle.close()

    def test_close_restores_implementation_and_destroys_single_context_once(self):
        handle,api,_,_=generated_handle()
        original=handle.patch[2]
        handle.close();handle.close()
        self.assertEqual(handle.head._forward_impl,original)
        self.assertNotIn('_forward_impl',handle.head.__dict__)
        self.assertFalse(hasattr(handle.head,'_vggt_rknn_depth_owner'))
        self.assertEqual(api.destroyed,[1]);self.assertEqual(handle.metrics()['live_contexts'],0)


if __name__=='__main__':
    unittest.main()
