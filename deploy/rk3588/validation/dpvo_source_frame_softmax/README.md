# DPVO source-frame Update: direct Softmax experiment

这是独立的 v34 研究工具，不修改默认 VGGT。它保留原 DPVO Update 的 50 个参数、c1→c2 顺序、LayerNorm、GRU 和输出分支，仅将完整源帧分组内的显式 max-shift/exp/sum 写法换为 `torch.softmax`。终端 weight 使用与 v33 相同的稳定 Sigmoid 表达；GRU 内部 Sigmoid 不变。

输入是一个完整源块 `[B,D,M,C]`，本次保存参考为两个 `[1,10,96,C]` 状态。`agg_kk` 沿目标轴 1 归约，`agg_ij` 沿 patch 轴 2 归约。它不是完整 E23520 图，也不包含 CNN、相关性、BA 或新观测位姿。CPU/ONNX 的固定 FP32 门限通过不表示逐位等价，且不构成 RKNN、轨迹或实时验收。实际 SDK 数值与算子位置需由独立运行收据验证。

## 文件与依赖

- `source_frame_update_softmax.py`：包装外部原始 Update，参数共享并检查绑定。
- `export-v34-source-frame-softmax-board.py`：仅 RK3588 板端执行原参数 CPU 对照、导出及 CPU ORT 检查。
- `audit-v34-source-frame-softmax.py`：只读取保存数组和静态 ONNX，使用 NumPy/ONNX，不执行模型。
- `audit-v34-source-frame-softmax-rknn.py`：复用冻结 v33 的数值、传输和 profile 解析函数，检查新图实际 SDK 保存输出；通过显式 manifest 路径/SHA 绑定 v34 实验。
- [相邻 v33 目录](../dpvo_source_frame/README.md)：冻结父包装、导出工具及审计函数。

板端导出需要用户提供已核 SHA 的官方 DPVO 源码/权重包和 v33 导出参考。这里只分发研究源码；不包含 checkpoint、ONNX/RKNN 权重、数据或 Rockchip SDK。Torch/ONNX/ORT 及板端运行环境需单独准备。审计无需 SDK。

```bash
python -B export-v34-source-frame-softmax-board.py \
  --v26-source-dir SOURCE_PACKAGE --checkpoint CHECKPOINT \
  --reference-dir V33_REFERENCE \
  --parent-source ../dpvo_source_frame/source_frame_update.py \
  --utility-source ../dpvo_source_frame/export-v33-source-frame-board.py \
  --wrapper-source source_frame_update_softmax.py \
  --wrapper-sha256 b89b9b84d3afb47fd926decddde0077222dc07930bf75ee08acb410dc4f6f3f4 \
  --output-dir NEW_BOARD_EXPERIMENT_DIR

python -B audit-v34-source-frame-softmax.py \
  --run-dir V34_EXPORT --manifest-sha256 EXPORT_MANIFEST_SHA \
  --reference-dir V33_REFERENCE --source-dir . \
  --v33-source-dir ../dpvo_source_frame --output NEW_AUDIT_JSON

python -B audit-v34-source-frame-softmax-rknn.py \
  --normal-dir V34_NORMAL --profile-dir V34_PROFILE \
  --manifest V34_EXPORT/manifest.json --manifest-sha256 EXPORT_MANIFEST_SHA \
  --compile-dir V34_COMPILE --compile-sha256 COMPILE_RECEIPT_SHA \
  --source-dir . --v33-source-dir ../dpvo_source_frame \
  --sdk-source-dir ../dpvo_source_frame --output NEW_RKNN_AUDIT_JSON
```

`manifest` 沿用 v33 的 I/O 合同 schema，实验身份由 `experiment_version=34`、`expression_variant=v34_direct_torch_softmax` 和新源/模型 SHA 单独绑定。本次对照采用历史 v33 原 CPU 参考，没有重新执行完整图来生成新的独立参考。工具不会自动接受或替换候选。

本次 SDK 普通 6 次调用与独立 profile 2 次调用均完成，输出有限且 weight 值域合法，严格 FP32 门限仍失败。实际 profile 中两个 Softmax 在 NPU，两个 ReduceMax 已消失，但终端 weight 的 2 个 Exp 与 1 个 Min 在 CPU，10 个 NPU Transpose 仍存在，因此这是混合执行。没有同进程父图/候选图受控 A/B 计时，不能将单次短块结果外推为完整 Update、VO 或 20 Hz。

## 许可与归属

本工具依据 [Princeton-VL DPVO](https://github.com/princeton-vl/DPVO/tree/0ac95b656d1fda91c271d2a106460d19ad966fc7) 的 Update/SoftAgg 实现，保留 Copyright (c) 2022 Princeton University 及相邻 [MIT LICENSE](LICENSE)。原始来源见 [dpvo/net.py](https://github.com/princeton-vl/DPVO/blob/0ac95b656d1fda91c271d2a106460d19ad966fc7/dpvo/net.py) 与 [dpvo/blocks.py](https://github.com/princeton-vl/DPVO/blob/0ac95b656d1fda91c271d2a106460d19ad966fc7/dpvo/blocks.py)。代码许可不替代外部模型权重或 SDK 的许可；参见仓库 [MODEL_LICENSES.md](../../../../MODEL_LICENSES.md) 和 [THIRD_PARTY_NOTICES.md](../../../../THIRD_PARTY_NOTICES.md)。
