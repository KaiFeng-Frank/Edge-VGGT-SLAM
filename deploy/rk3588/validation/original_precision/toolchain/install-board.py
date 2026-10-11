#!/usr/bin/env python3
"""Install pinned compiler dependencies offline in a new isolated board venv."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def main():
    base = Path('/mnt/vggt-data/vggt-slam2')
    work = base/'experiments/vggt-v49-toolchain-install-r1'
    envdir = base/'experiments/vggt-v49-toolchain-venv-r1'
    planpath = Path(sys.argv[1])
    assert platform.machine() == 'aarch64'
    assert b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes()
    assert not work.exists() and not envdir.exists()
    plan = json.loads(planpath.read_text())
    prod_cfg = base/'venv/pyvenv.cfg'
    before = sha(prod_cfg)
    work.mkdir()
    report = dict(schema='v49_isolated_original_VGGT_toolchain_install', status='failed',
                  script_sha256=sha(__file__), wheel_plan_sha256=sha(planpath),
                  venv=str(envdir), model_forward_executed=False, calibration_executed=False,
                  RKNN_build_executed=False, production_venv_config_before=before, commands=[])
    environment = dict(os.environ)
    for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'): environment.pop(key,None)
    environment.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
                       PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
    try:
        for row in plan['packages']:
            p = Path(row['path'])
            assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], row['filename']
            if 'git_blob_sha1' in row:
                h=hashlib.sha1(b'blob '+str(p.stat().st_size).encode()+b'\0')
                h.update(p.read_bytes())
                assert h.hexdigest() == row['git_blob_sha1']
        report['all_wheels_verified'] = True
        commands = [[str(base/'python/bin/python3.11'),'-m','venv',str(envdir)],
                    [str(envdir/'bin/python'),'-m','pip','install','--no-index','--no-deps',
                     *[row['path'] for row in plan['packages']]],
                    [str(envdir/'bin/python'),'-m','pip','check'],
                    [str(envdir/'bin/python'),'-m','pip','list','--format=json']]
        for i, cmd in enumerate(commands):
            start = time.monotonic()
            with (work/f'command-{i}.log').open('w') as f:
                result = subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=environment)
            report['commands'].append(dict(argv=cmd,returncode=result.returncode,seconds=time.monotonic()-start))
            if result.returncode: raise RuntimeError(f'command {i} failed')
        report['installed_packages'] = json.loads((work/'command-3.log').read_text())
        assert 'include-system-site-packages = false' in (envdir/'pyvenv.cfg').read_text()
        for row in plan['packages']:
            actual = {p['name'].lower().replace('_','-'):p['version'] for p in report['installed_packages']}
            assert actual[row['name'].lower().replace('_','-')] == row['version']
        report['status']='installed_metadata_checked'
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
    finally:
        report['production_venv_config_unchanged']=sha(prod_cfg)==before
        (work/'install.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','error','all_wheels_verified','production_venv_config_unchanged')}),flush=True)
    return 0 if report['status']=='installed_metadata_checked' else 2

if __name__ == '__main__': sys.exit(main())
