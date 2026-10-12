# 原 VGGT 执行诊断

本目录保存 FC2 输出通道分块、Tensor 元数据循环和现有四组聚合器 profile 的复现源码。均保留原 VGGT 的全部层、token 与原 checkpoint；两条候选没有用于生产。源码不包含权重、ONNX／RKNN 图、SDK 二进制或捕获数组，执行需要板端本地已有、满足固定 SHA 的材料。本仓库不提供这些私有材料的下载入口。量化或分块检查通过不等于原 FP32、轨迹质量或实时验收。

从仓库根目录设置 `VGGT_REPO` 为板端源码根、`BOARD_PYTHON` 为生产 Python 3.11、`COMPILER_PYTHON` 为已安装 RKNN Toolkit2 的隔离编译环境 Python，`VGGT_DIAG_DIR=deploy/rk3588/validation/original_execution_diagnostics`。编译与生产环境不同，按已有部署说明分别加载对应的库路径；编译环境不继承生产环境的 `PYTHONPATH`、`PYTHONHOME` 或 `VIRTUAL_ENV`。以下新输出目录与文件名均为示例，不能覆盖既有证据。模型、ORT、SDK 与编译任务只在 RK3588 板端串行执行。

FC2 分 2／4 保持每个输出的完整 K，不拆归约。构建器分别做两真实输入的固定 FP32 检查，成功后按原非量化 FP16 配置编译；探针绑定两份实际编译收据 SHA，用原图 A、分 2 B、分 4 C 的 ABCCBA 顺序。正常与 profile 必须使用独立新进程／输出目录；`--profile` 只有一次 ABC／输入，不能作为正常统计计时。

```sh
"$COMPILER_PYTHON" -B "$VGGT_DIAG_DIR/build_pair_fc2_n_splits_board.py" \
  --chunks 2 --output-dir NEW_SPLIT2
"$COMPILER_PYTHON" -B "$VGGT_DIAG_DIR/build_pair_fc2_n_splits_board.py" \
  --chunks 4 --output-dir NEW_SPLIT4
"$BOARD_PYTHON" -B "$VGGT_DIAG_DIR/probe_pair_fc2_n_splits_board.py" \
  --repo "$VGGT_REPO" --compiled NEW_SPLIT2 SPLIT2_RESULT_SHA \
  --compiled NEW_SPLIT4 SPLIT4_RESULT_SHA --rounds 10 --output-dir NEW_NORMAL
"$BOARD_PYTHON" -B "$VGGT_DIAG_DIR/probe_pair_fc2_n_splits_board.py" \
  --repo "$VGGT_REPO" --compiled NEW_SPLIT2 SPLIT2_RESULT_SHA \
  --compiled NEW_SPLIT4 SPLIT4_RESULT_SHA --profile --output-dir NEW_PROFILE
```

原生元数据循环只试替换同一 cache 实例中两处 Tensor 元数据遍历，不删除原 module 检查或前后 guard。构建需匹配板端生产 CPython 3.11 的头文件；使用实际构建结果的 library 文件和 SHA。基准加载验证后的原 checkpoint，但不执行模型 forward 或 SDK。它保留所有 48 次样本，不能隐藏原路线的一次未解释停顿以制造结论。

```sh
"$BOARD_PYTHON" -B "$VGGT_DIAG_DIR/build_native_tensor_states.py" \
  --compiler-command cc --output-dir NEW_NATIVE_BUILD
PYTHONPATH="$VGGT_REPO" "$BOARD_PYTHON" -B \
  "$VGGT_DIAG_DIR/benchmark_native_guard_board.py" \
  --library BUILT_LIBRARY --library-sha256 BUILT_LIBRARY_SHA --output NEW_NATIVE_BENCHMARK
```

四组 profile 复用生产 runtime 的 `_run_group`、固定 ABI 与 NEON FP16 转换，不加载 Torch／整网模型，不重新导出或编译。原图出口数固定为 2／2／2／3，末组第三出口保留未归一化的原 `global20`；SDK 出口先复制为 owned FP16，释放后扩大为 FP32。每组使用 v67 原检查收据绑定的 v53 与 v56-calib00 两份真实激活，输出私有 profile 文本、NPZ 及元数据。

```sh
"$BOARD_PYTHON" -B "$VGGT_DIAG_DIR/probe_retained_profiles_board.py" \
  --repo "$VGGT_REPO" --output-dir NEW_RETAINED_PROFILE --expected-npu-hz 300000000
```

该探针仅读取调用前后频率，默认要求 300 MHz，不写系统频率。`--expected-npu-hz 1000000000` 只是核对值；实际 1 GHz 必须由已有明确授权的有界控制器执行，并恢复原值，不能直接把该参数视为升频操作。保存八份单次 profile 的算子组成和资源清理，不据此推断 flags0 整网耗时、轨迹或 20 Hz。原主残差流已经保留通道布局，因此本次不启动整网 channel-first 重写。

冻结的四组 probe 原收据字段 `actual_model_or_ORT_or_compiler_execution=False` 表述过宽：实际执行了 **8 次 SDK 组件推理**；没有执行完整 Torch 模型、ORT 或编译器。源码和原始收据保留，公开摘要以分开的字段说明这一点。

原生测试的首版 fake-device 对象缺少与 `torch.device` 一致的值相等语义；修正版 `test_native_tensor_states_fake_r2.py` 在板端通过 7 项测试。首轮编译缺少隔离工具链的 `libopcodes` 库路径，修复编译环境后构建成功；未改 C 源码。行为检查通过与性能改善分开验收。

后续只编译已在 v73 固定门槛下通过的原 group0 FC2 K4 图，复用 [原构建器](../original_retained_fc2/build_retained_fc2_k4_board.py) 的 `--compile-checked`；其余三组不变。K 切分改变浮点归约顺序，但保留全部原权重、层和 token。新 probe 核验编译与检查收据，调用原 `_run_group`，以 flags0、两常驻 context、两真实输入各 ABBA×6 比较；每输入/路线排除首次，共 22 个暖样本/路线。已在 1 GHz 执行；设置期望频率参数本身不修改频率。

```sh
"$BOARD_PYTHON" -B "$VGGT_DIAG_DIR/probe_retained_fc2_group_board.py" \
  --repo "$VGGT_REPO" --candidate-result COMPILED_GROUP0_RESULT_JSON \
  --candidate-result-sha256 COMPILED_GROUP0_RESULT_SHA \
  --expected-npu-hz 1000000000 --output-dir NEW_GROUP0_COMPARISON
```

group0 平均耗时 134.641→132.429 ms，中位数 135.811→134.914 ms，P95 138.290→138.867 ms；候选有一次 102.546 ms 的低样本。这组组件结果未建立稳定提速，不追加 groups1/3 编译或生产集成，group2 的此前门槛失败保持原记录。无新的整网性能、轨迹或实时验收。

[公开结果摘要](../../../../results/original_execution_diagnostics_v76.json)。原始数组、SDK 日志与图保留私有；本文只提供方法、源码、指标和来源 SHA。
