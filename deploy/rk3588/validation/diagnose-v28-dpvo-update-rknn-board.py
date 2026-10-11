#!/usr/bin/env python3
"""One SDK call only: preserve invalid outputs/profile without accepting them.

The frozen normal runner remains unchanged. Only its two post-copy output
domain checks are intercepted for this diagnostic, so their raw arrays can be
saved. Every source/input/SDK/ABI/shape/ownership check still raises normally.
No comparison tolerance is widened and no output is corrected or clipped.
"""
import argparse,hashlib,importlib.util,json,platform,sys
from pathlib import Path
FROZEN_SHA='7d44c4ef260126ba0551d7596e9861a3007ec92d56cb33aa99e2223c2b110928'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frozen-runner',type=Path,required=True)
    a,forward=p.parse_known_args(argv)
    if platform.machine()!='aarch64' or b'rockchip,rk3588' not in Path('/proc/device-tree/compatible').read_bytes():raise ValueError('RK3588 board only')
    if sha(a.frozen_runner)!=FROZEN_SHA:raise ValueError('Frozen SDK runner differs')
    own=sha(__file__)
    if '--collect-perf-detail' not in forward:forward.append('--collect-perf-detail')
    out=Path(forward[forward.index('--output-dir')+1])
    spec=importlib.util.spec_from_file_location('_v28_single_invalid_diagnostic',a.frozen_runner);runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    original_require,original_compare,original_execute=runner.require,runner.compare,runner.execute
    ignored=[];executed=0
    def capture_output_check(ok,message):
        if not ok and message in ('Nonfinite SDK output','Invalid sigmoid weight range'):
            ignored.append(message);return
        original_require(ok,message)
    def diagnostic_compare(actual,reference):
        import numpy as np
        original_require(actual.shape==reference.shape and actual.dtype==reference.dtype==np.float32,'Diagnostic output shape/dtype differs')
        if np.isfinite(actual).all():return original_compare(actual,reference)
        mask=np.isfinite(actual)
        return dict(rtol=1e-4,atol=1e-5,gate_passed=False,failed_elements=None,
            comparison_undefined_due_to_nonfinite=True,nan_elements=int(np.isnan(actual).sum()),
            positive_inf_elements=int(np.isposinf(actual).sum()),negative_inf_elements=int(np.isneginf(actual).sum()),
            finite_elements=int(mask.sum()),element_count=actual.size,
            finite_min=float(actual[mask].min()) if mask.any() else None,finite_max=float(actual[mask].max()) if mask.any() else None)
    def one_call(*args,**kwargs):
        nonlocal executed
        if executed:raise RuntimeError('Diagnostic intentionally stops before a second SDK call')
        executed+=1
        return original_execute(*args,**kwargs)
    runner.require=capture_output_check;runner.compare=diagnostic_compare;runner.execute=one_call
    code=runner.main(forward)
    report=json.loads((out/'run.json').read_text())
    report.update(diagnostic_entry_sha256=own,frozen_runner_sha256=FROZEN_SHA,
        diagnostic_post_copy_checks_intercepted=ignored,diagnostic_SDK_calls=executed,
        original_runner_return_code=code,original_runner_terminal_status=report['status'],
        candidate_runtime_accepted=False,outputs_clipped_or_modified=False,FP32_thresholds_relaxed=False)
    good=(executed==1 and len(report['calls'])==1 and not report['cleanup_errors']
          and report['destroy_code']==0 and report['sources_and_inputs_unchanged']
          and sha(a.frozen_runner)==FROZEN_SHA and sha(__file__)==own)
    report['status']='single_call_invalid_output_diagnostic_saved_not_accepted' if good and ignored else 'failed'
    report['diagnostic_sources_unchanged']=sha(a.frozen_runner)==FROZEN_SHA and sha(__file__)==own
    (out/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','diagnostic_SDK_calls','diagnostic_post_copy_checks_intercepted','cleanup_errors')}),flush=True)
    return 0 if report['status']=='single_call_invalid_output_diagnostic_saved_not_accepted' else 1
if __name__=='__main__':raise SystemExit(main())
