# 原 VGGT pair 的板端 hybrid step1

此源仅准备原 `frame0→global0` 的 SDK 中间文件，未选择或采用精度候选。原 ONNX、36 个原参数、checkpoint、八份 calibration 与两份 evaluation 都由冻结合同绑定。固定 FP32 门槛 `rtol=1e-4, atol=1e-5` 和先前的数值拒绝保留。API 与精度范围见 [SDK_API_PLAN](SDK_API_PLAN.md)。

实际执行只由主调度在 RK3588S 串行进行，使用 v49 隔离工具链。先核对 source freeze，再原样复制此目录到 `experiments/vggt-v50-hybrid-source`；既有 v48 pair、v49 capture 和冻结合同目录保留板端原路径。调用：

```sh
bash /mnt/vggt-data/vggt-slam2/experiments/vggt-v50-hybrid-source/run-v50-hybrid-step1-board.sh \
  /mnt/vggt-data/vggt-slam2/experiments/vggt-v50-hybrid-source \
  /mnt/vggt-data/vggt-slam2/experiments/vggt-v50-hybrid-step1-r1 \
  /dev/shm/vggt-v50-hybrid-step1-r1
```

输出和临时目录均须全新。入口先确认 Linux ARM64/RK3588 device-tree、隔离 CP311 prefix、依赖版本与 source SHA，再延迟导入 SDK。原 graph 按字节复制到临时目录并再核 SHA，使 SDK 根据 graph 路径派生的中间文件也留在新目录。dataset 只列八个原 calibration 输入，不含 evaluation、位置或参考输出。调用只用 `config/load_onnx/hybrid_quantization_step1/release`，不创建 runtime。

`step1.json` 的成功状态为 `hybrid_intermediates_generated_not_precision_approved`，不是编译候选或数值通过。SDK 生成的 `.model/.data/.quantization.cfg` 按原始字节保存在输出目录并记 SHA；**这些文件包含私有模型相关数据，不公开**。源/输入哈希、dataset、版本、API 返回值与耗时、release、环境/cwd 恢复和生成文件前后身份均记入收据。即使失败也保留新目录。

实际配置 catalog 到齐后单独审查 alias 与精度请求，只有审过的副本可以用于 step2。`v50_hybrid_api.py` 的 step2 函数仅保存经核对的三参数 API 形式；真实调用器仍须独立绑定 step1、生成文件和审批策略，当前没有可执行 step2 主入口。仅有配置请求不能证明融合算子只覆盖原 MLP；最终 SDK operator 范围另查。

源码检查与微型 fake/generated 检查可在主机运行，不导入真实 SDK、模型或校准数据：

```sh
python3 -B test_v50_hybrid_api.py
bash -n run-v50-hybrid-step1-board.sh
```
