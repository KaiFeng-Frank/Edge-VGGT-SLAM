#!/usr/bin/env python3
"""Check the installed MatMul ABI/driver without executing an NPU inference."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--library", default="/usr/lib/librknnrt.so")
p.add_argument("--core-mask", type=int, default=0, choices=[0, 1, 2, 4])
p.add_argument("--weight-layout", default="normal", choices=["normal", "native"])
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
report = dict(npu_inference_executed=False, full_model_validated=False,
              library=a.library, core_mask=a.core_mask, weight_layout=a.weight_layout, status="failed")
start = time.monotonic()
context = None
try:
    from vggt_slam.rknn_linear import _API, _MatmulContext
    api = _API(a.library)
    context = _MatmulContext(api, (16, 32, 16), a.core_mask, a.weight_layout)
    report["context_created"] = True
    report["attributes"] = {
        name: dict(shape=list(getattr(context.attrs, name).dims[:getattr(context.attrs, name).n_dims]),
                   size=getattr(context.attrs, name).size,
                   type=getattr(context.attrs, name).type)
        for name in "ABC"
    }
    report["status"] = "passed"
except Exception as exc:
    report["error"] = f"{type(exc).__name__}: {exc}"
finally:
    if context is not None:
        try:
            context.close()
        except Exception as exc:
            report["status"] = "failed"
            report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
    report["seconds"] = time.monotonic()-start
    encoded = json.dumps(report, indent=2)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(encoded+"\n")
    print(encoded)
raise SystemExit(0 if report["status"] == "passed" else 1)
