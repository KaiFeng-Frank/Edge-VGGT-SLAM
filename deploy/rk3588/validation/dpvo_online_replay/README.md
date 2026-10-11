# M16 完整在线 Update 的诊断 callback
Latest integration: [v44 native FP16 complete replay](V44.md), with independent full-state parity and unchanged quality rejection.

后续九模型目录、已完成的 EOF 回放与来源合同见 [v42](V42.md)；本页保留 v41 八模型接口及其原失败边界。

此目录只接同一 DPVO 原参数的完整 Update，未改变默认 VGGT、边集合、
patch 数、采样、BA、初始化或关键帧策略。SDK 的历史逐元素 FP32 门槛
拒绝继续保留；返回值仅供显式研究回放，不代表候选采用或轨迹精度。
调用 RKNN 不等于内部全 NPU：已测父图有 Exp/Min 等 CPU 算子。

`OnlineM16Update` 保持 `update(net,inp,corr,ii,jj,kk)` 六参数接口。
原图为 FP32 `[1,E,C]`，int64 原边序；每源帧保留完整 D×16 依赖，
只以 `max_sources_per_batch=2` 拆独立源帧，全部源帧成功后才 scatter
回原 E 顺序。没有补零、裁边、未知形状 CPU 回退或部分图返回。

## 初始化

`v41_online_factory.open_board_online_from_index` 负责验证两份 export、
八份编译收据/模型、冻结执行器/ABI/真实 attrs，再创建有界 cache。
显式路径和 SHA 参数见函数签名；运行前必须声明
`diagnostic_allow_fp32_rejected=True`。实际上下文创建仍经原 ARM/RK3588
guard、原 native layout 打包及同步 set/run/get/owned-copy/release。
SDK祖先保留原目录；新 C 旁需放原字节 `v40_m16_grouped_contracts.py`
和 `v37_grouped_contracts.py`，不改祖先源码。

```python
callback = open_board_online_from_index(
    # compile_index/sha、parent/additions manifest/sha、exporter、compiler、
    # contracts、v40 bridge、v37 runtime、adapter、ABI/header/library/log
    **pinned_paths,
    motion_probe=original_torch_motion_probe,
    diagnostic_allow_fp32_rejected=True,
    failure_dir=new_board_experiment_directory,
)
```

回放 owner 最薄子类只从冻结 `_neural(..., probe=...)` 的真实参数设路由：

```python
def _neural(self, *args, probe, **kwargs):
    previous = callback.route
    callback.route = 'motion_probe' if probe else 'full'
    try:
        return super()._neural(*args, probe=probe, **kwargs)
    finally:
        callback.route = previous
```

只有显式 `motion_probe` 的完整单源 D1 请求执行原 Torch；初始化完整图、
新观测图和 EOF 更新均走 RKNN。不能根据 E=16 猜测路由。EOF 仍必须
由原 owner 记录为离线更新，不计新观测或 fresh pose。

`identity_reader` 可提供 adapter `SnapshotIdentity`，逐组及最终返回前
复核；未提供时，使用 callback 串行锁下本地 token，明确没有外部 graph/
geometry revision。返回自有 FP32 数组，SDK release 后仍有效。owner
只能同步提交，研究 callback 自身不提交 VO 状态。

## 八模型 index

`index.json` 结构：

```json
{
  "schema": "v41_online_M16_Update_compile_index",
  "status": "staged_eight_online_shapes_diagnostic",
  "parent_manifest_sha256": "5a9d6c571b672fbb174cf6adf9f8f578689fdd9e5f58ee4ec327320b55b62bfb",
  "additions_manifest_sha256": "<本轮实际 manifest SHA>",
  "candidate_accepted": false,
  "models": {
    "B1_D4_M16": {
      "lineage": "v41",
      "compile_dir": "<板端绝对目录>",
      "compile_sha256": "<compile.json SHA>",
      "model": {"file": "block.rknn", "bytes": 1, "sha256": "<实际 SHA>"}
    }
  }
}
```

示例仅展示一条记录；实际必须恰好 B1/B2×D4–7 八条。B1D4/B1D6
使用本轮新 manifest/新 compiler/new-C；B1D5/B1D7/B2D4–7 保留原 v40
manifest/编译器/合同链。旧 B3D6/B5D6 不纳入在线目录。D1 不复用 SDK
目录。支持的形状不是未来图保证：未知 D/B 拒绝并保存完整输入，不能
以旧 CPU 轨迹的 catalog 证明新数值轨迹一定覆盖。

## 时钟、失败和关闭

`metrics()` 返回 JSON 值深拷贝，关闭后仍可读；`close()` 幂等关闭所有
上下文，核对 context/model/DMA 清零和清理错误。owner 必须 finally
关闭，即使先前执行失败。正常记录只保留形状、group 时钟、实际 SDK
释放、cache 生命周期与拒绝标签；没有保存每帧全特征。预声明正常
数值捕获由 board owner 做。

callback 时钟含等待锁、输入自有快照、全 topology 计划、packing、cache
初始化/驱逐、SDK/owned-copy/release、全图还原及输出拷贝。嵌套 group/
cache 时钟不能重复相加；正常 metadata 和 owner 写日志在外，应另用
owner admission-to-publication 时钟验证真正新观测延迟。

失败先保存全部六输入和可用的原错误输出/记录，再抛异常；不返回部分
图、不继续回退。任何输入/SDK/值域错误均使 callback 不可再使用。
固定 FP32 比较没有自动变为通过。实际 SDK、新鲜在线位姿率、GT 精度、
长时复现均由根板端作业验证，本目录合成测试不提供这些结论。

## 可复现的小型测试

从仓库根目录运行，依赖 Python、NumPy 和 pytest。以下五个文件使用
生成数组、fake SDK/cache 或纯静态收据检查，不执行模型、ORT、RKNN
编译或板端作业。encoder 测试还读取已公开
[v39 结果](../../../../results/dpvo_half_workload_v39.json) 中的两份收据，
重建原 JSON 字节并核 SHA；这一项不是 fake 推理，也不依赖私有归档。

```bash
VALIDATION=deploy/rk3588/validation
ONLINE_REPLAY="$VALIDATION/dpvo_online_replay"
env PYTHONDONTWRITEBYTECODE=1 \
  V41_GROUPED_SOURCE="$VALIDATION/dpvo_grouped_update" \
  V41_PARENT_SDK="$VALIDATION/dpvo_grouped_update" \
  V41_PARENT_COMPILER="$VALIDATION/dpvo_grouped_update/compile-v37-grouped-update.py" \
  V41_ONLINE_CONTRACTS_DIR="$ONLINE_REPLAY" \
  python -B -m pytest -q -p no:cacheprovider \
    "$ONLINE_REPLAY/test_v41_half_rknn_encoder.py" \
    "$ONLINE_REPLAY/test_v41_online_update.py" \
    "$ONLINE_REPLAY/test_v41_missing_source_export.py" \
    "$ONLINE_REPLAY/test_v41_online_compiler.py" \
    "$ONLINE_REPLAY/test_v41_missing_saved_audit.py"
```

测试保留在此目录，以同目录寻找新 helper；原 transport、degree plan
和 compiler 从相邻 `dpvo_grouped_update` 读取冻结字节。公开的测试
路径适配副本有自己的 SHA，不与研发阶段的旧测试收据混用。实际板端
入口仍须显式提供模型、数据和 SDK 路径及 SHA；测试不提供这些二进制。

DPVO 相关实现保留相邻 [MIT LICENSE](LICENSE)。权重、数据、Rockchip
SDK/header/library 不随本目录分发；调用者自行获取并核对固定 SHA。

## v41 实际板端边界

`run-v41-cached-cpu-board.py` 只接入已验证的精确单次 remap。完整 CPU 回放与 v39 的全部非时钟结果、共享 RNG 和 55 对保存数组一致，但总耗时由 92.874 s 变成 107.043 s；未控温控频，不把局部收益当成整机收益。

`run-v41-online-npu-board.py` 是独立研究入口。冻结 CPU driver、factory/callback、源/模型/收据都由 CLI 路径与 SHA 绑定；`--online-config` 使用相邻 `board-config.json` 的目录合同，并由 owner 注入原 Torch motion probe 和本轮故障目录。默认仍是 CPU encoder；另传 `--npu-encoder`、`--encoder-config` 及其 SHA 才接半分辨率 RKNN encoder。必须显式 `--diagnostic-allow-fp32-rejected`，未知完整形状仍停止。

本轮 CPU encoder + RKNN Update 和两个神经阶段均用 RKNN 的回放都完成 192 个在线观测、158 个 fresh 位姿，分别为 1.421 / 2.602 Hz、软件读入至事件 flush 的 P95 725.280 / 411.798 ms。**两次完整运行都失败于首个 EOF Update 的 B1D3/M16 目录缺口**。不得把这些在线前缀写成整个运行完成；EOF 没有成功的终态轨迹，也不计为 fresh。所有实际推理均在 RK3588S，SDK 内部仍有 CPU 算子。

相机/队列/30 分钟/稠密地图/8 GB/20 Hz 未验收。NPU 输出捕获标为 `output`，不是相同调用的原 Torch gold；递归反馈后不可拿旧 CPU 轨迹的不同输入输出充当逐元素参考。原数值门槛与工作量配置的精度拒绝不因回放产生位姿而消失。

温度和 cpufreq 只在启动、每 16 次 publication 后及结尾读取，无设置写入。周期读数计入整段分母，但在该帧 publication 时刻之后，不属于该帧 latency。稀疏读数不能证明未发生过降频或定位耗时变化原因。模型加载在 replay 时钟前，缺失 Update context 的惰性初始化在完整调用内。

原始失败和成功前缀、配置/源码 SHA、轨迹独评及保存数组审核汇总于 [v41 公开结果](../../../../results/dpvo_online_update_v41.json)。代码遵循相邻 MIT notice；模型、数据与 SDK 的许可单独处理，本包不分发这些二进制。

板端路径须按部署 manifest 配置并优先核对冻结源码 SHA；不能假定仓库目录就是实际板端部署目录。

### 保存轨迹评价与数组复核

`evaluate-v41-online.py --help` 列出两份基线、候选回放、GT、policy 和冻结数学 helper 的路径/SHA。默认要求完成的运行；本轮失败记录须显式 `--allow-failed-EOF-only`，只允许已核完整在线前缀和首个 EOF D3 缺口，输出仍为失败运行的 fresh 轨迹诊断。`audit-v41-online-saved-arrays.py` 重读两次回放捕获及失败数组，首个相同输入可对原 Torch gold 和 v40 输出核对，后续不同反馈输入不套旧参考。

新增评价器的 13 个生成数据测试（不执行模型）：

```sh
V41_V39_EVALUATOR="$VALIDATION/dpvo_half_workload/evaluate-v39-workload.py" \
V41_MATH_HELPER="$VALIDATION/dpvo_fast/evaluate-v34-fast-trajectory.py" \
python -B -m pytest -q -p no:cacheprovider \
  "$ONLINE_REPLAY/test_v41_online_evaluation.py" \
  "$ONLINE_REPLAY/test_v41_failed_eof_evaluation.py"
```

`ONLINE_REPLAY`、`VALIDATION` 沿用上面的仓库相对配置。61 项接口/合成/静态收据测试与这 13 项评价测试在公开包中分别通过；测试结果不能替代真实板端精度或速度结论。
