# 两份 heldout 的原图 ORT 检查

此补充脚本保持 `source-freeze-r1.json` 的编译源码不变，独立冻结在 `heldout-source-freeze-r1.json`。它复用原 pair 合同的 `compare` 和冻结 capture 的 `load_capture/load_array`；只执行冻结原 `pair.onnx`，不运行 Torch 模型、RKNN SDK 或校准。

主代理串行运行 `bash experiments/vggt-v49-quant-source/run-heldout-ort-board.sh`。入口确认 ARM64/RK3588 device-tree、板端 cwd、独立工具链 prefix 和全部冻结依赖版本，实际 NumPy/ORT 模块来自同一环境。ORT 1.16.0 使用 CPUExecutionProvider、进程 CPU6/7、2/1 线程和 sequential mode，禁用 fallback。

两份 `evaluation_00/evaluation_01` 各调用一次 session.run，合计两次；没有 warmup，也不对八份 calibration 或旧 0/8 输入做推理。每次使用该样本原 frame0 FP32 输入，参考该次原 Torch frame0/global0 最终输出；固定 `rtol=1e-4,atol=1e-5`。完整实际输入副本和完整输出以独占新文件保存。收据 `check.json` 绑定源/原图/原 checkpoint/capture SHA，并分别计时原数组载入、输入存盘、session.run、输出验证和复制、输出存盘、比对与清理。第一次/第二次只是同进程调用顺序，不能作为实时吞吐结果。

finally 清除 session 的 Python 引用并 garbage collect（不宣称显式 native close），恢复进程亲和性和环境，复核源、capture-owned NPY、保存的输入/输出文件哈希。校验器复核捕获所有数组和原图旧参考的文件身份，但只有两份 heldout 输入进入 ORT。capture 中原 board PNG/checkpoint/source 的身份沿用其实际捕获证明，loader 不重新执行原模型或声称重新哈希那些物理源文件。

`passed` 仅表示这两份 heldout 的冻结 FP32 导出图通过门槛，量化精度与性能仍须 SDK 对照。`numerical_rejected/failed` 保留全部输出/失败目录；不自动采用候选，不改写历史原 FP32 拒绝，不证明完整 VGGT-SLAM 或实时性能。目录默认 `experiments/vggt-v49-heldout-pair-ort-r1`，重跑用 `VGGT_HELDOUT_ORT_OUTPUT` 选择新目录。

`test_v49_heldout_ort.py` 的六项必要检查只用 24 浮点元素的 generated 数组和 fake session：实际输入/完整输出保存、两份 heldout 各一次、固定 FP32 数值拒绝、输入突变、CPU/shape 描述符拒绝及 host 入口门槛。作者没有执行真实 ORT、原模型或 SDK。
