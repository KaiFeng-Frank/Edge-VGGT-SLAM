"""Tiny generated procfs/fake-libc checks; no torch, model or advice execution."""
import ast
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import checkpoint_pageout as p


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected rejection')


def run():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / 'model.pt'
        checkpoint.write_bytes(b'x' * 8192)
        with patch.object(p, 'ORIGINAL_BYTES', 8192):
            identity = p.checkpoint_stat(checkpoint)
            major, minor = p.os.major(identity['dev']), p.os.minor(identity['dev'])
            header = f"10000-12000 rw-p 00000000 {major:x}:{minor:x} {identity['inode']} {identity['path']}"
            smaps = header + '\n' + '\n'.join([
                'Size: 8 kB', 'Rss: 8 kB', 'Pss: 8 kB', 'Private_Clean: 8 kB', 'Shared_Clean: 0 kB',
                'Private_Dirty: 0 kB', 'Shared_Dirty: 0 kB', 'Anonymous: 0 kB', 'Swap: 0 kB',
                'Locked: 0 kB', 'VmFlags: rd wr mr mw me ac sd'])
            row = p.parse_smaps(smaps)[0]
            assert p.validate_vmas([row], identity, 4096) == [row]
            for key in ('Private_Dirty', 'Shared_Dirty', 'Anonymous', 'Swap', 'Locked'):
                bad = copy.deepcopy(row); bad['metrics'][key] = 1
                rejected(lambda: p.validate_vmas([bad], identity, 4096))
            for field in ('Private_Dirty', 'Swap', 'Locked'):
                bad = copy.deepcopy(row); del bad['metrics'][field]
                rejected(lambda: p.validate_vmas([bad], identity, 4096))
            for field in ('inode', 'offset'):
                bad = copy.deepcopy(row); bad[field] += 1
                rejected(lambda: p.validate_vmas([bad], identity, 4096))
            bad = copy.deepcopy(row); bad['VmFlags'].append('io')
            rejected(lambda: p.validate_vmas([bad], identity, 4096))
            short = dict(identity, bytes=8191)
            assert p.validate_vmas([row], short, 4096) == [row]
            fields = ['R'] + [str(i) for i in range(1, 50)]
            faults = p.fault_fields('123 (comm with (nested) parentheses) ' + ' '.join(fields))
            assert faults == {'minor_faults': 7, 'major_faults': 9, 'starttime_ticks': 19}

            storage = SimpleNamespace(_cdata=91, data_ptr=lambda: 0x10100, nbytes=lambda: 1024)
            tensor = SimpleNamespace(device=SimpleNamespace(type='cpu'), is_meta=False,
                untyped_storage=lambda: storage, data_ptr=lambda: 0x10100,
                shape=(256,), stride=lambda: (1,), dtype='float32', _version=2, requires_grad=False)
            model = SimpleNamespace(named_parameters=lambda: iter([('generated', tensor)]),
                                    named_buffers=lambda: iter([]))
            calls = []
            class FakeAdvice:
                def __init__(self, code=0): self.code = code
                def __call__(self, pointer, size, advice):
                    calls.append((pointer.value, size.value, advice))
                    if self.code: p.ct.set_errno(22)
                    return self.code
            def snapshot(checkpoint, loaded):
                return dict(checkpoint_stat=p.checkpoint_stat(checkpoint), checkpoint_vmas=copy.deepcopy([row]),
                            status_kB={'VmRSS': 8, 'VmHWM': 20}, faults={'minor_faults': 3, 'major_faults': 0})
            read_bytes = Path.read_bytes
            def board_bytes(path):
                return b'rockchip,rk3588' if str(path) == '/proc/device-tree/compatible' else read_bytes(path)
            source_root = Path(__file__).resolve().parents[4]
            with patch.object(p.platform, 'machine', return_value='aarch64'), \
                    patch.object(Path, 'read_bytes', board_bytes), patch.object(p, 'memory_snapshot', snapshot):
                with patch.object(p.ct, 'CDLL', return_value=SimpleNamespace(madvise=FakeAdvice())):
                    receipt = p.pageout_checkpoint(model, checkpoint, identity, root/'success.json', source_root=source_root)
                assert calls == [(0x10000, 8192, 21)]
                assert receipt['tensor_storage_VA_version_unchanged'] and receipt['checkpoint_VMA_layout_unchanged']
                assert receipt['checkpoint_rehashed_by_pageout_helper'] is False
                with patch.object(p.ct, 'CDLL', return_value=SimpleNamespace(madvise=FakeAdvice(-1))):
                    rejected(lambda: p.pageout_checkpoint(model, checkpoint, identity, root/'failed.json', source_root=source_root))
                failed = json.loads((root/'failed.json').read_text())
                assert failed['status'] == 'failed' and failed['advice_calls'][0]['errno'] == 22
                assert len(calls) == 2 and all(value[2] == 21 for value in calls)
                count = [0]
                def drifting(checkpoint, loaded):
                    count[0] += 1
                    result = snapshot(checkpoint, loaded)
                    if count[0] >= 3: result['checkpoint_vmas'] = []
                    return result
                with patch.object(p, 'memory_snapshot', drifting), \
                        patch.object(p.ct, 'CDLL', return_value=SimpleNamespace(madvise=FakeAdvice())):
                    rejected(lambda: p.pageout_checkpoint(model, checkpoint, identity, root/'drift.json', source_root=source_root))
                assert json.loads((root/'drift.json').read_text())['status'] == 'failed'
        for name in ('checkpoint_pageout.py', 'compare_complete_pageout.py'):
            ast.parse((Path(__file__).parent / name).read_text())
        wrapper = (Path(__file__).parent / 'compare_complete_pageout.py').read_text()
        assert "p.add_argument('--checkpoint-pageout', action='store_true'" in wrapper
        assert wrapper.index('cache = install_patch_cache(model)') < wrapper.index('result[\'checkpoint_pageout\'] = pageout_checkpoint') < wrapper.index('for i in range(1,len(selected))')
        assert wrapper.index('complete_service_seconds=time.monotonic()-start') < wrapper.index("row['checkpoint_pageout_post_inference']")
        assert 'post_cleanup_RSS_MiB' in wrapper and 'steady_RSS_MiB' not in wrapper
    print(json.dumps(dict(status='passed', generated_procfs_and_fake_libc_only=True,
        clean_mapping=True, dirty_and_missing_and_identity_rejections=11, final_page_rounding=True,
        stat_comm_faults=True, only_advice_21=True, errno_failure_no_retry=True,
        post_advice_VMA_drift_rejected=True, tensor_metadata_unchanged=True,
        wrapper_install_and_timing_order=True, actual_model_SDK_ORΤ_advice_execution=False)))


if __name__ == '__main__':
    run()
