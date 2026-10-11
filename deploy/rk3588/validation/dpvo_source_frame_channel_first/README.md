# v35 source-frame Update: channel-first experiment

本目录是独立 DPVO 原参数 Update 表达实验，不替换默认 VGGT。将完整源块的物理布局从 `[B,D,M,C]` 改为 `[B,C,D,M]`，用原 Linear 参数的精确 reshape 构造 1×1 Conv；保留 c1→c2、边界零输入后的 bias、四个 C 轴 LayerNorm 和两个完整 D/M 分组。只对原末端权重使用 stable sigmoid；内部 GatedResidual sigmoid 不变。50 个原参数的值与绑定不改，没有删除边或减少 MAC。

两个真实 D10/M96 状态的板端 Torch/CPU ORT 84 项比较通过原 `rtol=1e-4, atol=1e-5`。后续 RKNN 的 6 次普通调用和 2 次独立 profile 有限且权重域合法，但原 FP32 门限失败，候选未采用。每次实际 profile 为 50 个内部 NPU 算子、3 个内部 CPU 算子（2 Exp、1 Min）和 6 个 CPU I/O；仍有 6 个 NPU Transpose。普通 SDK 同步调用的两个状态 warm 均值为 54.287/66.368 ms，包含输入打包、set/run/get、owned FP32 输出复制和 release。它不是同进程父图受控提速、全图 Update、完整 VO 或 20 Hz 验收。完整 E=23,520 gold 继承冻结 v33/v34，本轮未重新执行全图参考。

## 文件与依赖

- `source_frame_update_channel_first.py`：原参数包装器；`export-v35-source-frame-channel-first-board.py`：仅 RK3588 板端原权重、CPU 及 ORT 导出对照。
- `compile-v35-source-frame-channel-first.py`：独立 RKNN 工具链编译；`v35_rknn_source_frame_cf.py` 和 `run-v35-channel-first-rknn-board.py`：显式 CF SDK 合同，普通计时与 flags=8 profile 分开。
- `audit-v35-channel-first-{export,rknn}.py`：只读取保存数组/静态图的独立审核，不执行模型或 SDK。通过 CLI 指定实际证据与父源码目录；不会把数值拒绝重标为接受。
- `v33_rknn_source_frame_padded.py`：原字节复制的历史 transport 祖先，供显式 ancestry pin 与同目录运行校验，未修改或伪装为 v35 新实现。

需要已发布的相邻 `dpvo_source_frame`、`dpvo_source_frame_softmax` 源码；板端还需自行准备固定官方 DPVO 源码/权重、外部 RKNN ABI helper、合法取得的 SDK header/library 与匹配版本。此目录不分发权重、ONNX/RKNN 大产物、SDK header、library 或设备数据。实测脚本保留严格 source/weight/artifact SHA 和板端执行检查，路径移动后需按原字节部署并传入明确路径；旧绝对路径收据不能直接冒充新执行记录。

## 保存结果审核

两个 auditor 的 `--help` 列出显式输入目录、manifest/compile SHA 及父源码参数。RKNN auditor 的 `--source-dir` 与 `--sdk-source-dir` 可同时指向本目录；`--v33-source-dir` 指向相邻 `dpvo_source_frame`，`--v34-source-dir` 指向相邻 `dpvo_source_frame_softmax`。`--sdk-parent-source` 可指定其他位置的同字节祖先，仍必须通过冻结 SHA。生成全新审核 JSON，不修改原始收据。

仓库根目录可运行 `pytest -q tests/test_v35_source_frame_channel_first.py tests/test_v35_channel_first_export.py tests/test_v35_channel_first_export_audit.py tests/test_v35_channel_first_rknn_audit.py tests/test_v35_cf_sdk.py`。这些测试只用小型随机 toy、合成 SDK 记录和静态 ONNX；不使用真实模型输入。

## 许可

原 DPVO 算法和参数绑定出处为官方 commit `0ac95b656d1fda91c271d2a106460d19ad966fc7`，本目录的移植/表达保留其 MIT 许可及 Princeton University 版权，见 [LICENSE](LICENSE)。权重及外部 SDK 的许可仍须分别核查；MIT 不重新授权这些未分发的资产。项目根许可与第三方 notices 仍适用各自范围。
