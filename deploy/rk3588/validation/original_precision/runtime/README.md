# 原 VGGT 配对图 FP16 / PTQ 板端对照

本目录是未采用的原模型数值诊断工具。仅保留原 VGGT 完整 frame0 → global0，同一 36 个原参数张量和同一静态图；`nonquant_fp16` 与 `ptq_w8a8` 是该图的两种 RKNN 数值实现。当前只完成源码和生成数组 / fake SDK 检查；本目录冻结时尚未运行新的实际 SDK 对照。原 v48 非量化配对结果没有通过固定 FP32 门槛，失败记录保留。

`run-v49-original-pair-precision-board.py` 只接受 Linux ARM64、little-endian LP64、RK3588 device-tree 和固定 SDK 2.3.2 / driver 0.7.2。它复用 SHA 固定的 `run-v12-rknn-block-board.py` 的 typed ABI / query 定义；不继承旧脚本的“拒绝整数 IO”结果声明。实际编译、校准和 SDK 推理由主任务在板端串行调度。主机不得执行真实推理。

捕获接口来自兄弟 `../capture/v49_capture_contracts.py`。先校验固定真实图像计划、原 checkpoint/source/preprocessing、原 camera/register first/other 角色及十份输入，再取两个 `evaluation` 的完整原 Torch FP32 最终参考。八个 `calibration` 输入、两份保留参考、两张旧评估图像之间保持排除关系。新量化 receipt 必须绑定当前捕获 SHA、冻结编译器和合同、板端校准事实、四项 completed 操作、实际模型 SHA、八行 dataset 内容、全部来源前后 SHA、SDK release 和进程状态恢复。旧非量化 receipt / 模型 SHA 也固定。

实际输入必须为 `[2,437,1024]`、原 FP32、C contiguous；最终参考/输出为 `[1,874,1024]`。真实 SDK rank3 IO 描述符可为 FP32/FP16，或具有有效正 scale / 可表示 zero point 的 INT8 / UINT8 affine。拒绝未知类型、packed format、非有限 scale、错误 shape / size 和不受支持的浮点 affine。无论逻辑描述符是什么，提交始终是 FP32、`pass_through=0`、format3，取回始终是 `want_float=1`；转换由 SDK 完成，不手工归一化、量化、转置或增加 batch 维。返回的 FP32 字节数重新校验，frame_id 必须同步一致，先复制到自有数组再 release；部分 get 失败返回的 buffer 也释放。

两个上下文同进程共存，以固定 core_mask7 和进程 CPU6/7 运行。每个保留输入两种实现各四次：第一组 AB/BA/AB/BA，第二组 BA/AB/BA/AB，共 16 SDK calls、8 对结果，没有隐含 warmup。各组首调用加三次复测单列；上下文实际首次调用独立标记，第二组首调用不称为冷启动。每次输出均对该输入的原 Torch 参考计算固定 `rtol=1e-4, atol=1e-5`，两种 RKNN 输出之间的比较仅是差异诊断，不能充当 FP32 参考。候选始终 `candidate_accepted=False`。

SDK IO 时钟只含 inputs_set、同步 run、outputs_get、检查与自有 FP32 copy、release，保留完整起止纳秒和分项时长。输入 hashes / finite checks、数值误差、重复检查、NPY/JSON 保存、queries 和温度 / 内存采样在此区间外。每种实现平均分母是两个输入各三次复测，共六次；变化百分比以非量化实现为分母。只观察温度、内存及 CPU 当前频率，不改系统时钟 / governor / 散热。两上下文进程 RSS 不证明完整模型的 8 GB 部署。

可另起新的进程和输出目录加 `--collect-perf-detail`；这会创建全新的 flag8 profiling 上下文，普通运行是 flag0。原始 operator 文本和 SHA 单列，profile IO 不产生普通速度改善百分比，不能与普通时钟混加。w8a8 配置和整数边界 IO 都不证明所有内部算子是 INT8 或在 NPU 上；须独立读取实际 operator 表。

板端调用示例如下。路径由主任务实际 staging 决定，所有新 SHA 必须取自已完成并保存的实际收据，不允许用示例值替代：

```bash
cd /mnt/vggt-data/vggt-slam2
python -B "$RUNTIME_SRC/run-v49-original-pair-precision-board.py" \
  --contracts "$RUNTIME_SRC/v49_quantized_pair_runtime_contracts.py" \
  --contracts-sha256 "$RUNTIME_CONTRACT_SHA" \
  --capture-contracts "$CAPTURE_SRC/v49_capture_contracts.py" \
  --capture-contracts-sha256 "$CAPTURE_CONTRACT_SHA" \
  --capture-dir "$CAPTURE_DIR" --capture-sha256 "$CAPTURE_RECEIPT_SHA" \
  --ptq-compiler "$QUANT_SRC/compile-v49-pair-ptq-board.py" \
  --ptq-compiler-sha256 "$PTQ_COMPILER_SHA" \
  --ptq-contracts "$QUANT_SRC/v49_quant_contracts.py" \
  --ptq-contracts-sha256 "$PTQ_CONTRACT_SHA" \
  --ptq-compile-receipt "$PTQ_DIR/compile.json" \
  --ptq-compile-sha256 "$PTQ_RECEIPT_SHA" \
  --ptq-model "$PTQ_DIR/pair-int8.rknn" --ptq-model-sha256 "$PTQ_MODEL_SHA" \
  --nonquant-compile-receipt "$NONQUANT_DIR/compile.json" \
  --nonquant-model "$NONQUANT_DIR/compiler/block.rknn" \
  --runtime-helper src/deploy/rk3588/validation/run-v12-rknn-block-board.py \
  --sdk-header src/deploy/rk3588/reference/rknn_api_v2.3.2.h \
  --diagnostic-allow-FP32-rejected \
  --output-dir experiments/vggt-v49-precision-normal-r1
```

`test_v49_original_pair_precision.py` 的 81 项检查只使用每份 24 个浮点元素的数组、合成描述符 / receipt 以及 fake API，不加载实际 SDK 库、模型、ORT session 或编译器。覆盖 SDK-owned buffer 在 release 时被破坏后 owned copy 仍有效、错误整数返回字节数拒绝、同步 frame 错误、partial get / init 清理、destroy 失败继续清理、真实描述符字段验证、PTQ/捕获来源绑定、原 FP32 拒绝保留、计时分母及 profile 分离。测试可原样置于 `deploy/rk3588/validation/original_precision/runtime/`，capture / quant 在兄弟目录，原 typed SDK helper 在 validation 目录。

这些结果只涉及两层原图和两份同序列保留输入，不是完整模型 / SLAM 精度、轨迹或相机实时验收。源码、模型源码、checkpoint 和 SDK 许可分开；发布不包含 checkpoint、模型数组、含权重的导出或 SDK binary。本目录代码沿用项目代码许可与引用来源，不能据此推导原模型权重的商业使用权限。
