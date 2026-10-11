# 原 VGGT 配对图板端 PTQ

这里的源码只量化原 `frame0→global0` 完整配对图。实际校准、build 和验证均由主代理在 RK3588S 执行；本目录作者没有运行实际 SDK、模型、ORT 或编译。生产运行入口未改动，候选不会自动采用。API 和数据约定的官方出处见 [SDK_API_RESEARCH](SDK_API_RESEARCH.md)。

`compile-v49-pair-ptq-board.py` 受两处板端身份门槛保护：主入口在载入实际数组和 SDK 之前确认 ARM64/device-tree/CPython 3.11/显式独立环境路径，实际 SDK 操作函数再次确认板端。采用主代理新建工具链中的 Torch 2.2.0、NumPy 1.26.4、ONNX 1.16.1、ORT 1.16.0、RKNN 2.3.2，排除生产 Torch 2.3.1。

捕获依赖 sibling `../capture/v49_capture_contracts.py` 的冻结 `load_capture(directory,sha)`。该合同绑定原 checkpoint、源闭包、原图 36 参数、预处理、先于推理固定的真实图像计划以及实际原模型捕获。量化合同另行核验八份真实 frame0 FP32 输入及其固定位置，排除两份 evaluation、两张旧配对评估图片和重复 tensor。校准数据文件严格八行，每行只包含一个完整 `[2,437,1024]` 原输入 NPY 的绝对路径；heldout 输出及位置数组都不进入 SDK dataset。

配置是 `rk3588 / float16 / optimization_level=3 / w8a8 / normal / channel`；build 明确 `do_quantization=True,auto_hybrid=False`。输入 mean/std 为 None，RGB2BGR False，不加维、不转置和不再归一化。SDK 的 rank3 输入解释及具体算子精度须依靠实际后续结果确认。只执行 `config,load_onnx,build,export_rknn`，没有 runtime/inference API。成功记录 `compiled_not_runtime_validated`，即使完成校准也不等于通过精度或实时门槛。

每次实验必须用新的输出目录及板端 `/dev/shm` 工作目录；后者也是 SDK cwd 和进程 TMPDIR，保存全部失败记录。finally 释放 SDK、核验源/输入/SDK Python 源 SHA 并恢复进程 cwd/环境。收据保存全部实际版本、prefix、模块路径、四操作返回值/计时、dataset SHA/8 行身份、2 个 heldout 身份和峰值 RSS。RSS 是编译进程内存，不能替代 8 GB SLAM 部署验证。

## 主代理执行

先按 `source-freeze-r1.json` 的 SHA 将量化源码传到板端新目录，例如 `experiments/vggt-v49-quant-source`。原 pair contracts 可位于生产源码 `src/deploy/rk3588/validation/original_pair/v47_pair_contracts.py`，也可用已经冻结部署的 `experiments/vggt-v48-pair-source/v47_pair_contracts.py`；文件必须保持冻结 SHA。`run-ptq-board.sh` 已绑定主代理完成的板端捕获 `experiments/vggt-v49-capture-r1`，实际收据 SHA 为 `bf39dfa65526a4135cbd2d7ee8b927ce2c4a40da75aebf64652ed2cd87a7e7fc`；这不表示本目录作者执行了捕获或 SDK。可通过 `VGGT_QUANT_SOURCE`、`VGGT_CAPTURE_SOURCE` 指定源码目录，用 `VGGT_PTQ_OUTPUT`、`VGGT_PTQ_WORK` 指定新的实验目录。

```bash
cd /mnt/vggt-data/vggt-slam2
QUANT=experiments/vggt-v49-quant-source
TOOLCHAIN_PREFIX=/mnt/vggt-data/vggt-slam2/experiments/vggt-v49-toolchain-venv-r1
env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH=/mnt/vggt-data/vggt-slam2/sysroot/usr/lib/aarch64-linux-gnu \
  "$TOOLCHAIN_PREFIX/bin/python" -B "$QUANT/compile-v49-pair-ptq-board.py" \
  --contracts "$QUANT/v49_quant_contracts.py" --contracts-sha256 "$QUANT_CONTRACTS_SHA" \
  --pair-contracts experiments/vggt-v48-pair-source/v47_pair_contracts.py \
  --pair-dir experiments/vggt-v48-pair \
  --board-ort-receipt experiments/vggt-v48-pair-ort-r2/check.json \
  --capture-contracts "$CAPTURE_CONTRACTS" --capture-contracts-sha256 "$CAPTURE_CONTRACTS_SHA" \
  --capture-dir "$CAPTURE_DIR" --capture-sha256 "$CAPTURE_SHA" \
  --toolchain-prefix "$TOOLCHAIN_PREFIX" \
  --output-dir experiments/vggt-v49-pair-ptq-r1 \
  --work-dir /dev/shm/vggt-v49-pair-ptq-r1
```

读取最终 `compile.json`，不能仅凭 shell 或 ADB 返回值判断成功。任何失败留存原目录；源码修复须新 freeze，新实验使用新目录。实际 SDK 推理另用 runtime sibling 的冻结合同、同两份 evaluation 原 Torch 最终参考和固定 `rtol=1e-4,atol=1e-5`，保留拒绝事实。

## 小型验证

`test_v49_pair_ptq.py` 的 38 项检查只生成每份 24 个浮点元素的小数组并使用 fake 编译器对象。测试实际 dataset 输入集合、heldout/旧评估泄漏拒绝、原数组 shape/dtype/raw SHA/位置、路径约束、明确量化开关/不改变输入、失败停止与 SDK 释放、严格整数成功返回值、独立环境版本门槛以及旧 FP32 gate 不可放宽。它没有导入实际 SDK，没有实际校准或原模型推理。
