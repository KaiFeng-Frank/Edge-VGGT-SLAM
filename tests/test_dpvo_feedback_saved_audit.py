"""Tiny saved-data audit checks; no real inputs, model or BA."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
if not (ROOT / "dpvo_cpu_geometry.py").is_file():
    ROOT = ROOT.parent / "deploy/rk3588/validation/dpvo_geometry"
spec = importlib.util.spec_from_file_location("saved_audit", ROOT / "audit-v28-dpvo-feedback.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class SavedAuditTests(unittest.TestCase):
    def test_raw_metadata_and_file_bytes_are_enforced(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            folder=Path(temporary)
            x=np.arange(6,dtype=np.float32).reshape(2,3)
            path=folder/"toy.npz"
            np.savez(path,value=x)
            artifact=dict(file=path.name,bytes=path.stat().st_size,sha256=audit.sha(path),arrays={"value":audit.meta(x)})
            np.testing.assert_array_equal(audit.load_artifact(folder,artifact)["value"],x)
            artifact["arrays"]["value"]["shape"]=[3,2]
            with self.assertRaisesRegex(ValueError,"tensor SHA/shape/dtype"):
                audit.load_artifact(folder,artifact)
            artifact["arrays"]["value"]=audit.meta(x)
            path.write_bytes(path.read_bytes()+b"modified")
            with self.assertRaisesRegex(ValueError,"file SHA/bytes"):
                audit.load_artifact(folder,artifact)

    def test_fixed_old_mask_exposes_error_hidden_by_mask_drop(self):
        residual=np.array([[[3.,4.],[6.,8.]]],np.float32)
        weight=np.ones_like(residual)
        old=np.array([True,True])
        new=np.array([True,False])
        current=audit.residual_summary(residual,new,weight)
        fixed=audit.residual_summary(residual,new,weight,old)
        self.assertEqual(current["weighted_squared_residual"],25)
        self.assertEqual(fixed["weighted_squared_residual"],125)
        self.assertEqual((current["selected_edges"],fixed["selected_edges"]),(1,2))

    def test_w2c_translation_and_inverse_depth_gauge(self):
        poses=np.zeros((1,2,7),np.float32)
        poses[...,6]=1
        poses[0,1,0]=-.1
        patches=np.zeros((1,1,3,3,3),np.float32)
        patches[0,0,0]=50
        patches[0,0,1]=40
        patches[0,0,2]=2
        K=np.broadcast_to(np.array([100,100,50,40],np.float32),(1,2,4)).copy()
        ii,jj,kk=(np.array([v],np.int64) for v in (0,1,0))
        xy,_=audit.selected_projection(poses,patches,K,ii,jj,kk,0)
        np.testing.assert_allclose(xy[0],30,atol=1e-6)
        np.testing.assert_allclose(xy[1],40,atol=1e-6)
        poses[...,:3]*=3
        patches[:,:,2]/=3
        scaled,_=audit.selected_projection(poses,patches,K,ii,jj,kk,0)
        np.testing.assert_allclose(xy,scaled,atol=1e-6)

    def test_raw_fastBA_and_normalized_LieTorch_are_distinct(self):
        poses=np.zeros((1,2,7),np.float32)
        poses[...,6]=1
        angle=.2
        poses[0,1,3:]=np.array([0,0,np.sin(angle/2),np.cos(angle/2)],np.float32)*np.float32(1.00002)
        patches=np.zeros((1,1,3,3,3),np.float32)
        patches[0,0,0]=75
        patches[0,0,1]=45
        patches[0,0,2]=1
        K=np.broadcast_to(np.array([100,100,50,40],np.float32),(1,2,4)).copy()
        ii,jj,kk=(np.array([v],np.int64) for v in (0,1,0))
        normalized,_=audit.selected_projection(poses,patches,K,ii,jj,kk,0)
        raw,_=audit.selected_projection(poses,patches,K,ii,jj,kk,0,raw_fastba=True)
        self.assertGreater(float(np.max(np.abs(raw-normalized))),1e-5)


if __name__=="__main__":
    unittest.main()
