"""Small source/schedule guards, without real images or a board session."""
import importlib.util
from pathlib import Path

import pytest

ENTRY = Path(__file__).with_name('run-v40-undistort-cache-board.py')
spec = importlib.util.spec_from_file_location('v40_board_pixel_driver', ENTRY)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def manifest():
    rows = [dict(frame_id=str(i+.25),timestamp=i+.25,relative_path=f'rgb/{i}.png')
            for i in range(192)]
    return dict(schema='v39_first_contiguous_TUM_RGB_frames',GT_read=False,frames=rows)


def index_text(value):
    return '# synthetic index\n' + '\n'.join(f"{r['frame_id']} {r['relative_path']}" for r in value['frames'])


def test_fixed24_selection_matches_declared_first192_population():
    indices = module.performance_indices()
    assert len(indices)==len(set(indices))==24 and indices[0]==0 and indices[-1]==191
    assert indices==sorted(indices) and set(b-a for a,b in zip(indices,indices[1:])) <= {8,9}


@pytest.mark.parametrize('system,machine,compatible', [
    ('Linux','x86_64',b'rockchip,rk3588'), ('Darwin','aarch64',b'rockchip,rk3588'),
    ('Linux','aarch64',b'other,soc')])
def test_platform_guard_does_not_treat_any_ARM_as_target(system,machine,compatible):
    with pytest.raises(ValueError,match='board-only'):
        module.require_board(system,machine,compatible)


def test_target_and_original_manifest_pass_without_any_decode():
    module.require_board('Linux','aarch64',b'rockchip,rk3588s\x00')
    value=manifest()
    module.validate_manifest(value,index_text(value))


@pytest.mark.parametrize('mutation', ['order','escape','count','GT'])
def test_identity_selection_and_no_GT_guards(mutation):
    value=manifest(); original=index_text(value)
    if mutation=='order': value['frames'][0],value['frames'][1]=value['frames'][1],value['frames'][0]
    elif mutation=='escape':
        value['frames'][0]['relative_path']='rgb/../../bad.png';original=index_text(value)
    elif mutation=='count': value['frames'].pop()
    elif mutation=='GT': value['GT_read']=True
    with pytest.raises(ValueError): module.validate_manifest(value,original)
