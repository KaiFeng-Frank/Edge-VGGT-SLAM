# 原 VGGT 的 GPU 注意力诊断

本目录记录原注意力的 GPU 接入实验。独立板端 GPU 算子通过了固定输入检查；SDK GPU 回调接入原 frame0/global0 完整块后输出不稳定，CPU 回调转板端 GPU 的对照输出稳定但更慢。两条路径均未采用，当前整机结果见 [v82](../original_calibrated_camera/README.md)。

只使用原 VGGT checkpoint、参数和全部 token。原 Q/K LayerNorm、二维 RoPE、完整 QK/softmax/V 和投影输入布局交给 RK3588S 的 Mali GPU；QKV、投影、残差、MLP 保留在原 RKNN 图。本机 GPU 未参与。

## 实测边界

| 检查 | 结果 |
|---|---|
| 两组原 CPU 输入，frame/global 独立 GPU tail | 28 次输出通过 `rtol=1e-4, atol=1e-5`；48 份保存数组独立复核 |
| 六次暖态完整 GPU tail，含上传/下载/有限性与释放 | frame 7.640–7.969 ms；global 11.642–14.281 ms |
| 原 FP16 pair A / GPU custom-op pair B，同轮各六次暖态 | 26.986 / 92.455 ms |
| 同输入四次重复 | A 稳定；B 的两个出口均每次变化 |
| B 相对原 CPU 块输出 L2 | frame 15.94–32.98%；global 38.10–52.60% |
| SDK CPU 回调转板端 GPU，单独同轮 A/B 暖态 | 27.153 / 72.639 ms；B 重复稳定，原 CPU L2 为 0.425–0.476%，固定 strict 仍失败 |

独立 tail 的通过没有证明 SDK 内部张量传递正确。不同角色的源/名称分离、自建严格 OpenCL program、入口 queue finish/FROM_DEVICE 对照仍失败。SDK 实际 program options 包含 `-cl-fast-relaxed-math -cl-mad-enable -DCL_DTYPE_half`；相同 kernel 文本不能当作实际编译算术相同的证明。这些观察不足以确定根因。

四个回调快照按相同 GPU 侧输入复测，norm/RoPE/输出通过 strict；GPU 输出与 FROM 后 CPU 读回逐字节相同。在两次 frame 回调中，11 个 GPU 侧保存数组和读回输出均相同，而其后的完整 frame 出口有 13,326 个元素变化。额外 output TO/SDK 示例 import 属性未解决；CPU 回调的稳定对照仍慢 2.675 倍，所有原 tail 算术仍在板端 GPU 执行。这些都是固定输入的诊断，不能建立根因或整网质量结论。

详细指标见 [结果 JSON](../../../../results/original_gpu_attention_v83.json)。[来源清单](source-copy-list.json) 固定每份源码字节与 SHA；四份 saved audit 仅读取已有 NumPy 文件，不重新运行模型。诊断回调的保存成本不当作正常性能。

## 复现入口

需要原 RK3588S 环境、原权重和自行产生的私有 v53/v56 激活、ONNX/RKNN 图及收据。公开库不提供这些数组或权重导出。冻结脚本保留实际板端目录和 SHA，启动前应检查所需工件；它们不是下载即用的生产后端。

| 阶段 | 源码入口 |
|---|---|
| 原 FP32 局部捕获，四次原 block forward | `capture_original_attention_tail_board.py` |
| 独立 GPU tail，首次诊断与六次暖态分开 | `probe_original_attention_tail_gpu_board_r2.py`、`original_attention_tail_opencl.py` |
| 两处原 tail 切分及转换期参考 | `build_pair_attention_custom_board.py`、`build_pair_attention_custom_board_r2.py` |
| RKNN 2.3.2 非量化 FP16 编译 | `compile_pair_attention_custom_board_r2.py` |
| 完整 pair A/B，普通 flags0/core7 | `probe_attention_custom_pair_board.py`；`run_attention_custom_pair_1ghz.sh` |
| 来源/严格 program/同步对照 | `run_attention_custom_pair_{distinct,strict,handoff}_1ghz.sh` |
| 有限回调快照与 GPU 侧输入复测 | `rknn_original_attention_custom_diagnostic.py`、`probe_callback_shadow_board_r2.py` |
| CPU 回调的板端 GPU 对照 | `rknn_original_attention_custom_cpu_staging.py`、`run_attention_custom_pair_cpu_staging_1ghz.sh` |
| 本机 custom-op ABI 的 36 项 size/offset | `build_custom_abi_board.py`、`probe_custom_op_abi.c` |

生产 Python 与 RKNN 隔离编译 Python 分开使用，见各实际 `run_*.sh`。GPU 库、NEON 库、typed RKNN ABI 和 [既有 OpenCL helper](../original_native_gpu/native_erf_gelu_opencl.py) 都有独立 pin；不随本目录分发。`custom-op-abi.json` 是本机核对的 36 个 size/offset 数字；原创 C 探针需板上已有 SDK header，本目录不分发 header。

1 GHz 调用必须通过既有 [有界控制器](../original_clock_test/npu_frequency_test.py)，上限 180 秒/75°C，finally 恢复原状态。普通 probe 不写频率；GPU 使用原 `simple_ondemand`，自然变化按实际频率记录，未改 governor/min/max。实际 launcher 需先按目录说明准备工件和新输出目录。

```sh
BASE=/mnt/vggt-data/vggt-slam2
DOC="$BASE/src/deploy/rk3588/validation/original_gpu_attention"
"$BASE/venv/bin/python" "$BASE/src/deploy/rk3588/validation/original_clock_test/npu_frequency_test.py" \
  --execute-frequency-test --frequency-hz 1000000000 \
  --timeout-seconds 180 --temperature-limit-c 75 \
  --launcher "$DOC/run_attention_custom_pair_1ghz.sh" \
  --output "$BASE/experiments/attention-pair-control-new"
```

上式对应这次已执行 launcher 的控制器接口；launcher 中所需私有图/输入/源 snapshot 不由命令自动创建。capture、component 和 pair 的完整保存文件才可用于以下独立审计：

```sh
python audit_attention_custom_pair_saved.py \
  --directory PRIVATE_PAIR_RESULT_DIR \
  --capture-directory PRIVATE_CAPTURE_DIR --output audit.json
```

仍未完成新的整机回放、轨迹或地图质量验证，20 Hz/100 ms 目标未达成。原权重和图受其独立模型许可约束，见 [许可说明](../../../../MODEL_LICENSES.md)。本目录只发布原创源码、哈希和统计，不含 captured arrays、模型权重、SDK 或固件。
