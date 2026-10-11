# 完整 Update 的 metadata 与输入传输研究

本目录提供 metadata 候选、保存 JSON 的形状/时钟分析、generated/fake 检查、独立板端 native 输入查询和固定状态 callback 对照。静态候选检查不执行模型、SDK、编译或校准；原 VGGT 默认路径保持，DPVO 研究候选未采用，历史严格 FP32 拒绝与两份 v41 run 的 failed EOF 状态保留。

`v42_metadata_only_update.py` 与受测候选字节相同：只将每次 full callback 用于 NPU 活动报告的完整 `cache.metrics()` 深复制替换为锁内布尔值查询。相同 helper 单独位于 `v42_cache_activity.py`。数学、owned snapshots、finite/domain 检查、完整图失败、原 SDK执行、输出拷贝/release、公开 metrics 和 close 保持。候选目录仍为 v41 的 B1/B2×D4–7，不提供 D3 EOF 支持；若移植到另一 callback，只取 helper 和对应 reporting block。

七个 generated/fake 检查在固定研发源码上通过。发布的 candidate/helper 未改字节，test 仅使冻结父 callback 与 SDK 源码的位置可配置，analyzer 仅移除机器固定路径并使用证据编号。发布副本的 Python 语法在准备阶段静态解析，未把该步骤写成新的板端验证。默认目录与同一 validation 目录中的 `dpvo_online_replay` 和 `dpvo_grouped_update` 相邻；也可设置 `V42_FROZEN_CALLBACK`、`V42_PARENT_SDK` 指向固定父源码。

```sh
python3 -m pytest -q test_v42_metadata_only.py
python3 audit-v41-update-performance.py --runs-dir SAVED_RUNS_DIRECTORY --output saved-analysis.json
```

只读 analyzer 要求用户本地提供两份固定 run JSON，严格核对 SHA，不加载 captures、GT 或任何执行器。发布的精简结果仅保留证据 SHA、均值、完整形状和组数，不分发原输入、权重、模型、SDK header/binary 或机器路径。[时钟与目录](../../../../results/dpvo_update_performance_v42.json)、[native 查询条件与 FP16 反例](../../../../results/dpvo_native_input_feasibility_v42.json)、[原 fake 检查收据及发布源码 SHA](../../../../results/synthetic_metadata_performance_v42.json)。

所有时钟都是已有反馈 run 的嵌套时钟，不能再次相加；169.551−136.546ms 的差额不能全部称为 metadata 成本。845→620 是静态调用组数，输入量/MAC不减，实际大 batch 或 NEON/native transport 的收益需要同状态板端对照。逻辑输入 command1 不能冒充 native8/10；pass-through1也不自动成为 zero-copy。数组所有权、失败保存、DMA预算、关闭与既定数值门槛都属于后续验证合同。

源码沿用 DPVO（Princeton University，2022）MIT 归属，见相邻 LICENSE。SDK与模型产物未包含。

## 实际 native 输入查询

`query-v42-native-inputs-board.py` 仅绑定 init、core mask、query、destroy。
九个实际模型均成功查询并销毁，零 forward；command8和command10都返回
FP16 NC1HWC2/C2=8，不是普通NHWC。corr882有效通道扩为888存储通道。
[实际完整查询和81份属性的独立审查](../../../../results/dpvo_native_input_query_v42.json)
固定了下一打包候选的布局，但没有验证pass-through1提交、尾部padding、
输出等价性或速度。`--help` 给出固定index/ABI/header/library及新输出目录参数，
查询只能在板端执行。原SDK与权重不随代码分发。

## 实际 metadata callback 对照

`run-v42-callback-ab-board.py` 在同一RK3588S进程中，为原callback和候选
创建独立cache，复用固定v41八模型与三份真实六输入capture。各状态ABBA两轮，
分别排除每种方法首次调用，余各3个暖样本。原版/候选完整callback均值为
93.686/81.816、146.065/124.458、122.133/105.570ms；24份完整输出逐字节相同。
两路各56次SDK均release，各6个context关闭。没有反馈、BA或新鲜位姿发布。
[实际调用、计时、生命周期与保存审核](../../../../results/dpvo_metadata_callback_ab_v42.json)。

`--help` 列出factory/config、A/B callback及各自SHA、原输入目录和新输出目录。
输入目录是保存run.json及update-034/127/191-input.npz的v41运行根目录；
不是只含EOF失败输入的online-sdk子目录。板端脚本核固定来源、原完整输入
和所有权，候选没有D3模型能力。这里测试的是元数据查询优化，未接入v42
完整在线回放，也不证明CPU数值等价或真实相机性能。
