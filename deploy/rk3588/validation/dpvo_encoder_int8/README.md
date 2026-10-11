# Optional DPVO INT8 encoder diagnostics

The original sparse INT8 graph and spatial rewrite run only as isolated diagnostics on RK3588S. Neither is adopted. Original VGGT remains the default; no ORB or added LK/PnP is introduced. Source licensing follows the adjacent MIT notice; weights and compiled models remain external.

Actual results: original graph warm SDK mean 429.429 ms, spatial graph 905.223 ms. Each has six normal calls and three separate profile calls on the same three saved input/indices pairs. Both reject the original `rtol=1e-4, atol=1e-5` tensor gate. This is not fresh-pose throughput or a trajectory test.

## Reproduction order

1. Reproduce the fixed original captures, sparse CPU reference and [board PTQ preparation](../board_ptq/README.md). Image preprocessing and calibration execute on the board. Preserve FP32 BGR input and INT64 index order/repetitions.
2. Original deployment uses `run-v36-dpvo-int8-board.py` with the v36 compile/query receipts. New spatial compilation uses `../board_ptq/compile-v37-spatial-int8-board.py`; its raw dataset-list SHA is distinct from the resolved absolute-path list SHA.
3. Query the new spatial model with `query-v37-spatial-int8-io-board.py` before inference. Supply the frozen v36 query helper from `../board_ptq/`. This query binds only init/query/destroy and performs zero forwards.
4. Use `run-v37-spatial-int8-board.py` for spatial inference. Required `--frozen-runtime-helper` is the original runtime in this directory; `--frozen-query-helper` is the original query in `../board_ptq/`. Supply the new spatial compile/query receipts and their exact SHA256. The explicit `--diagnostic-allow-fp32-rejected` retains the parent's rejection. `--help` lists external capture, reference, ABI, header and transport paths.
5. Normal and `--collect-perf-detail` runs require separate new output directories. The runtime uses the existing frozen typed transport from `../dpvo_sparse_context_sdk/v34_sparse_rknn.py`, validates queried formats, copies logical FP32 output before release, and preserves raw arrays before numerical validation. Do not infer buffer lengths from native stride storage.
6. Audit saved outputs using the supplied NumPy-only auditors; no model replay is needed on the host. Actual SDK, original CPU and rewritten CPU comparisons remain separate.

The SDK header/library and model/data artifacts are intentionally absent. Source/receipt hashes and actual command inputs identify the tested external files in [original INT8 evidence](../../../../results/dpvo_encoder_int8_v37.json) and [spatial evidence](../../../../results/spatial_encoder_ptq_v37.json).

## Synthetic checks

The `test_*.py` modules use tiny generated arrays or fake ABI calls. For a portable checkout, set `RKNN_ABI_HELPER` to `../run-v12-rknn-block-board.py`, `V34_TYPED_TRANSPORT` to `../dpvo_sparse_context_sdk/v34_sparse_rknn.py`, `V36_IO_QUERY_HELPER` and `V36_QUERY_IMPLEMENTATION` to `../board_ptq/query-v36-int8-io-board.py`, `V37_IO_QUERY_HELPER` to this directory's new query, and `V36_RUNTIME_IMPLEMENTATION` to this directory's original runtime. Use absolute paths before running pytest. These tests establish transport/guard behavior, not neural accuracy or speed.
