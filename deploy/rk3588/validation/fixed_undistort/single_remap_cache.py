"""Independent candidate: concatenate the frozen stripe maps, remap once.

No map recomputation/crop/resize/normalization/learned work. Changing the remap
schedule still needs a separate board pixel gate. See OPENCV_NOTICE.md for the
official OpenCV undistort source and its retained license notices.
Single owner only: do not call close concurrently with undistort_bgr.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import uuid

import cv2
import numpy as np

PARENT_SHA256 = '14c7331fcb9bc84e13fa8d9b38f0a0575b529729b08dd539915f3714a632972c'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_parent(path=None):
    path = Path(path or Path(__file__).with_name('fixed_undistort_cache.py')).resolve(strict=True)
    if _sha(path) != PARENT_SHA256:
        raise ValueError('Frozen stripe map helper SHA differs')
    name = '_v40_single_remap_parent_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    if _sha(path) != PARENT_SHA256:
        raise ValueError('Stripe helper changed during import')
    return module, path


class SingleRemapUndistorter:
    """Reuse every original fixed-point map value; change only remap schedule.

    Retained map-array storage has the same max_map_bytes bound as the parent.
    Concatenation temporarily holds both representations (up to twice that
    bound); this is not a process-RSS or all-temporary-allocation bound.
    """
    def __init__(self, camera_matrix, distortion, image_hw=(480,640), *,
                 expected_opencv_version, max_map_bytes=2*1024*1024, parent_helper=None):
        parent, path = load_parent(parent_helper)
        previous = parent.FixedStripeUndistorter(camera_matrix,distortion,image_hw,
            expected_opencv_version=expected_opencv_version,max_map_bytes=max_map_bytes)
        try:
            metadata = previous.metadata()
            def ordered_hash(index):
                digest = hashlib.sha256()
                for stripe in previous._stripes:
                    digest.update(stripe[index].tobytes(order='C'))
                return digest.hexdigest()
            expected = (ordered_hash(2),ordered_hash(3))
            # Copies from the exact original stripe maps, not a new full-grid
            # initUndistortRectifyMap call with different floating-point order.
            map1 = np.concatenate([stripe[2] for stripe in previous._stripes],axis=0)
            map2 = np.concatenate([stripe[3] for stripe in previous._stripes],axis=0)
            actual = tuple(hashlib.sha256(value.tobytes(order='C')).hexdigest() for value in (map1,map2))
            if expected != actual:
                raise ValueError('Concatenated map bytes differ from ordered original stripes')
            height,width = image_hw
            if (map1.shape != (height,width,2) or map1.dtype != np.int16
                    or map2.shape != (height,width) or map2.dtype != np.uint16
                    or not map1.flags.owndata or not map2.flags.owndata
                    or not map1.flags.c_contiguous or not map2.flags.c_contiguous):
                raise ValueError('Owned full-frame map contract differs')
            byte_count = map1.nbytes+map2.nbytes
            if byte_count != metadata['map_bytes'] or byte_count > max_map_bytes:
                raise ValueError('Concatenated map-array budget differs')
            map1.flags.writeable = map2.flags.writeable = False
            self._map1,self._map2 = map1,map2
            self._height,self._width = height,width
            self._version = expected_opencv_version
            self._original_metadata = metadata
            self._parent_path = str(path)
            self._map_hashes = actual
            self._map_bytes = byte_count
            self._construction_peak_map_array_bytes = 2*byte_count
            self._closed,self._calls = False,0
        finally:
            previous.close()
        if _sha(path) != PARENT_SHA256:
            self.close()
            raise ValueError('Stripe helper changed during map construction')

    def undistort_bgr(self,image):
        if self._closed:
            raise RuntimeError('Single-remap cache is closed')
        if cv2.__version__ != self._version:
            raise ValueError('OpenCV version changed after map creation')
        if (not isinstance(image,np.ndarray) or image.dtype != np.uint8
                or image.shape != (self._height,self._width,3)
                or not image.flags.c_contiguous):
            raise ValueError('Original C-order uint8 BGR input required')
        output = np.empty_like(image)
        result = cv2.remap(image,self._map1,self._map2,cv2.INTER_LINEAR,dst=output,
                           borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        if result.shape != output.shape or not np.shares_memory(result,output):
            raise RuntimeError('OpenCV remap did not fill the owned full-frame destination')
        self._calls += 1
        return output

    def metadata(self):
        result = dict(self._original_metadata)
        result.update(method='concatenated_original_stripe_maps_single_remap_candidate',
            parent_helper_path=self._parent_path,parent_helper_sha256=PARENT_SHA256,
            ordered_original_maps_bitwise_preserved=True,full_map_raw_sha256=list(self._map_hashes),
            map_bytes=self._map_bytes,current_map_bytes=0 if self._closed else self._map_bytes,
            construction_peak_map_array_bytes=self._construction_peak_map_array_bytes,
            remap_calls_per_image=1,original_stripe_maps_released_after_concatenation=True,
            completed_calls=self._calls,closed=self._closed,
            actual_board_pixels_validated=False,performance_validated=False,
            default_pipeline_changed=False)
        return result

    def close(self):
        self._map1 = self._map2 = None
        self._closed = True
