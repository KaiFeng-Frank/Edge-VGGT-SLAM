# v35 DPVO FAST NPU 编码器诊断

这是独立的 DPVO 研究入口，默认 VGGT 不变。原 DPVO 权重、FAST 的 48 patches / lifetime 11 / removal 16 / optimization 7、capacity 64、无闭环及原 Update / 相关性 / BA 保持；未添加 ORB、LK 或 PnP。Torch CPU 显式种子和 x→y→depth 随机数顺序沿用 v34。M48 的随机数消费与历史 M96 不同。

`v35_rknn_encoder.py` 在整个回放复用一个 dense RKNN context，每帧同步执行一次编码器。原 v32 spatial encoder 已被严格 FP32 元素门槛拒绝，运行必须显式提供 `--diagnostic-allow-fp32-rejected`；研究结果始终 `candidate_accepted=false`。没有 CPU 编码器回退、输出裁剪或阈值放宽。SDK 错误日志、非有限或全零特征均停止并保存失败证据。

## 完整 192 帧实测

| 指标 | v35 NPU 编码器 FAST |
| --- | ---: |
| 源帧 / 实际 fresh 位姿 | 192 / 179 |
| 无节拍完整 replay | 284.058 s |
| fresh / 完整 replay 吞吐 | 0.630 Hz |
| fresh 软件处理均值 / P95 | 1.549 s / 2.380 s |
| 同 179 个 common-fresh 的 CPU / NPU ATE RMSE | 约 1.0714 cm / 1.0714 cm |

详见 [完整结果与独立审计](../../../../results/dpvo_fast_rknn_v35.json)。GT 评价固定全源 1 秒计划，162 对中 149 对可用，最近 GT≤20ms，各轨迹正 Sim3 仅评价使用；EOF 12 次离线 Update/重建不计 fresh。短前缀的近似轨迹误差未撤销编码器严格 FP32 拒绝，不表示原 CUDA/VGGT 精度、20Hz、物理相机 P95 或稳定性验收，也不自动采用该候选。

## 板端入口

```sh
python -B run-v35-dpvo-fast-rknn-board.py --help
```

沿用 [v34 FAST](../dpvo_fast/README.md) 的数据、原源码、权重及编码器 CPU 参考参数，另提供 `--fast-parent-driver`（冻结 v34 driver）、`--rknn-encoder-module`（本目录编码器）和以下已绑定文件：

- `--rknn-model` / `--rknn-compile`：v32 dense spatial RKNN 与编译收据。
- `--rknn-rewrite` / `--rknn-failed-ort`：v27 原图改写与数值拒绝收据。
- `--rknn-compiler` / `--rknn-sdk-helper` / `--rknn-parent-runtime`：v32 compiler、v34 dense Context 和 v26 transport。
- `--rknn-abi-helper` / `--rknn-header` / `--rknn-library` / `--rknn-log-helper`：冻结 ABI、SDK header、native library 与 v33 日志捕获实现。

使用 `--diagnostic-allow-fp32-rejected --frames 192 --seed 20261011`。原 `--encoder-model/export/ort` 仅绑定 CPU 参考，不用于回退。首次三帧完整 fmap/imap 和自身 M48 patch 检查保持 `rtol=1e-4, atol=1e-5`；失败保存，只有显式诊断允许继续。大数组和快照留在板端，先取回 run/input-manifest JSON。

整条 encode 计时包含预处理、原生日志捕获、布局打包、SDK I/O、owned-copy、release、数值检查和采样；SDK I/O 是嵌套子项。`geometry_BA_calls` 计时真实原 BA 调用，不能与历史未分解的 stream remainder 混同。fresh 和 EOF 重优化分别统计。

## 保存轨迹评价

```sh
python -B evaluate-v35-fast-rknn-trajectory.py \
  --cpu-run-dir CPU_V34 --cpu-run-sha256 CPU_SHA \
  --npu-run-dir NPU_V35 --npu-run-sha256 NPU_SHA \
  --groundtruth GT --groundtruth-sha256 GT_SHA \
  --parent-evaluator ../dpvo_fast/evaluate-v34-fast-trajectory.py \
  --output NEW.json
```

只读 JSON/GT，保留 CPU/NPU 原 schema，验证完整相同 source 计划；全源固定 1 秒端点、最近 GT ≤20ms、一一关联、fresh/EOF 分开。各自完整 population 和预声明 common-fresh 交集均保留覆盖率，正 Sim3 仅用于评价。没有用 GT 选择配置。短数据回放不证明物理相机、20Hz、30 分钟稳定性或 VGGT 精度。

`analyze-v35-fast.py` 是冻结的历史 v34 JSON/整数拓扑审计脚本，保留原证据路径与字节；搬迁时应先按脚本核对输入位置，它不是通用路径参数 CLI。计时中的 stream remainder 包含多种未分离工作，不能直接称为 BA。

## 合成检查与许可

从仓库根目录运行：

```sh
python -B -m pytest -q -p no:cacheprovider \
  tests/test_v35_rknn_encoder.py tests/test_v35_fast_trajectory.py
```

测试只使用生成像素/特征、fake SDK、toy 轨迹和已公开的编译 metadata；没有真实模型或硬件推理。公开 metadata 已脱敏，测试验证记录的原始 SHA 身份和编译合同，不将重新序列化值冒充原始收据字节。

原 DPVO MIT 归属见 [LICENSE](LICENSE)。SDK header、native library、模型权重和数据不随源码分发，需使用绑定版本自行准备。

## 首三帧完整特征数组独审

`audit-v35-fast-encoder-arrays.py` 只读取本次保存的完整 fmap/imap 和原 v26 CPU capture，核文件/数组 SHA、shape、dtype、finite 及原 FP32 元素门槛，独立于测试中的 metadata 检查：

```sh
python -B audit-v35-fast-encoder-arrays.py \
  --run-dir NPU_V35 --run-sha256 NPU_SHA \
  --capture-dir CPU_CAPTURE --capture-sha256 CAPTURE_SHA \
  --output NEW.json
```

该六项实测数组检查一致地保留严格 FP32 拒绝。relative L2 仅描述数值，不要求 ARM/x86 归约摘要的浮点字节一致，也不替代固定元素门槛；没有再次推理。
