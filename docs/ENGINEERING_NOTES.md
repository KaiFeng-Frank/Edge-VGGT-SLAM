# 可复用的工程问题记录

这里记录移植中能够复现、值得复用的故障定位方法。v24 的 DA3-SMALL 是独立模型研究，原 VGGT 默认模型没有替换；ONNX 的 CPU 数值验证也不代表 NPU 或实时 SLAM 已通过。各条状态以链接中的固定源码、输入和收据为准。

## 1. 严格加载遇到 437 个存储键、443 个模型键

- **现象：** DA3 官方模型的 `state_dict` 有 443 个键，发布的 safetensors 只有 437 个；首次加载在任何前向执行前被拒绝。
- **根因证据：** 官方 DualDPT 只构造一次 `ln_seq`，再将同一个 LayerNorm 放入四个辅助输出层。权重和偏置各有三个额外名称，共六个别名，实际只对应两个唯一参数。checkpoint 的 metadata 明确声明了这些别名；模型中也能核对到相同 Parameter 对象、storage、offset、shape 和 stride。这是共享参数的存储去重，不是缺少六组权重。
- **办法：** 仅在 checkpoint 的声明与官方模型的共享关系逐项一致时，将已存储张量映射到这些别名，再执行 `strict=True`。不使用 `strict=False`，不补随机或零值参数。记录存储键数、展开键数和唯一参数量，避免重复名称造成参数量误报。
- **验证边界：** 该映射覆盖本次固定版本的六个别名；任何未被声明或不共享 storage 的缺失键仍须拒绝。已完成板端加载、前向与权重前后哈希核对，不说明其他 checkpoint 可以套用相同映射。
- **链接：** [CPU 与别名证据](../results/da3_cpu_v24.json)、[严格加载实现](../deploy/rk3588/validation/da3_probe_common.py)、[官方源码清单](../results/da3_sources_v24.json)。

## 2. 原网络能运行，但 `cartesian_prod` 无法导出

- **现象：** 官方 CPU 前向和包装器对照成功，Torch 的 opset 17 导出却在 `aten::cartesian_prod` 处失败。
- **根因证据：** PositionGetter 由整数高、宽生成二维位置网格。这一网格只依赖固定输入几何，不依赖图像内容或学习参数；不受支持的是导出算子，不是已经验证失败的网络数学。
- **办法：** 用未修改的官方 getter 生成一次原网格，与独立构造的 row-major `(y,x)` 网格逐元素核对，然后仅在固定尺寸包装器及 trace 中使用其独立副本。此次固定网格是 `[2,432,2]`、int64，对应两视图的 `18×24` patch grid。未知尺寸或设备明确拒绝；结束时恢复原 getter。
- **验证边界：** 这是固定两视图、`252×336` 输入的导出适配，不是动态输入支持。三对原网络参考先由未修改 getter 产生，随后核对包装器与导出图；不能用常量位置网格冒充动态几何。
- **链接：** [导出与独立网格审计](../results/da3_export_v24.json)、[固定 getter 适配](../deploy/rk3588/validation/export-v24-da3-board-r2.py)。

## 3. eager 为 FP32，导出的 DPT 位置算术却出现类型冲突

- **现象：** 固定整数位置网格后，DPT 的位置编码导出仍遇到 float64/float32 `Einsum` 类型冲突。
- **根因证据：** 官方 `_add_pos_embed` 依次执行 UV 网格、正弦余弦编码、乘 `0.1`、通道置换和加法。eager 调用明确使用特征的 FP32 dtype，但导出中的 `linspace` 路径产生了 DOUBLE 网格，而频率张量是 FLOAT。不能由 eager dtype 推断导出图每个中间值的类型。
- **办法：** 仅在包装器中，调用原官方函数生成各固定尺寸的 FP32 PE 常量；保留原乘法、置换和 `x + PE`，用 singleton 广播代替原 batch `expand`。先运行未修改的三对原网络参考，再建立缓存；trace 遇到未预计算的 key 硬拒绝。`finally` 恢复原 head 方法和 getter，不修改官方文件或学习权重。
- **验证边界：** 已核对六种 PE 的尺寸、dtype、字节数和哈希，以及包装器和板端 ORT 输出。适配只对当前固定 CPU FP32 导出成立；不声称混合精度、其他输入尺寸或编译器融合也等价。
- **链接：** [固定 PE 适配](../deploy/rk3588/validation/export-v24-da3-board-r3.py)、[导出与数值证据](../results/da3_export_v24.json)。

## 4. 导出成功之前，先建立原网 → 包装器 → 板端 ORT 的对照链

- **现象：** 图能导出、checker 能通过，并不能排除视图顺序、归一化、输出维度或相机 codec 改错。
- **根因证据：** 这些错误可能保持合法 shape，却改变像素和几何。DA3 的 raw pose 需要它自己的 quaternion/FOV codec；不能因为字段也叫 `pose_enc` 就套用 VGGT 的 codec。
- **办法：** 固定原 FP32 RGB 内容与 frame ID，只执行一次官方归一化；分别保存原网络、包装器和板端 ORT 的 depth、confidence、raw pose、world-to-camera 与 K。逐文件及逐数组记录 SHA，显式核对去除 singleton batch/depth channel 的规则。用预先固定的 `rtol=1e-4`、`atol=1e-5` 逐元素判定，失败保留原误差，不事后放宽门限。
- **验证边界：** v24 三对新原网络参考的 15 个数组与先前 CPU 参考字节一致；三对板端 ORT 的五种输出均为零失败元素，独立审计共 101 项通过。ORT 使用 **CPUExecutionProvider**。这证明本次固定图的数值一致性，不证明 NPU、轨迹精度、持续吞吐或端到端延迟；未使用的辅助 ray 输出也可能被导出图裁去，不能据单次时长宣称公平加速。
- **链接：** [数值收据](../results/da3_export_v24.json)、[可复现保存数组审计](../deploy/rk3588/validation/audit-v24-da3-export.py)、[CPU 参考](../results/da3_cpu_v24.json)。

## 5. RKNN 报错的规则名不等于已定位、已修复的根因

- **现象：** 相同 ONNX 已在板端 ORT 通过，但 RKNN 2.3.2 默认编译在 `gather_to_conv` 阶段发生 `IndexError`。仅设置 `disable_rules=['gather_to_conv']` 的下一次编译也失败：常量折叠中的 Gather 出现索引 48 超出 `[-32,31]` 的错误。
- **根因证据：** 日志显示该节点来自此前 `fuse_channel_shuffle_split_concat_to_gather` 重写，随后又经过 `swap_mul_gather`。目前能将问题缩小到编译重写与常量折叠路径，但尚未证明唯一根因，也没有证明关闭单条规则就能得到可用模型。
- **办法：** 保持已通过 ORT 的原图 SHA 不变，逐次保存编译配置、日志和产物；用局部规则差分定位前置重写。不要为了越过编译错误，直接改变 RoPE 索引、模型权重或数值门限。
- **验证边界：** **已编译并完成板端混合执行，数值与实时目标未通过。** 前两个尝试失败；第三次仅额外关闭产生坏节点的 `fuse_channel_shuffle_split_concat_to_gather`，同一 ONNX 成功生成 133,478,082 字节 RKNN。关闭的规则固定为这两条，其他配置不变。随后完成 6 次普通调用和 3 次独立 profile；存在内部 CPU 算子，严格 FP32 门限未通过，性能见第 8 条。
- **链接：** [RKNN 调查状态与同一 ONNX 证据](../results/da3_export_v24.json)。

## 6. 更换执行器不能顺便丢掉回环或数组所有权

- **现象：** 将 Torch 前向替换成 SDK 调用后，普通 depth/pose 路径可能能用，但回环请求或下一次推理会暴露接口和缓冲区寿命问题。
- **根因证据：** Solver 的普通预测和回环验证均需模型执行。回环请求包含原 query、retrieved 图像及明确的 similarity 能力；SDK 输出缓冲区释放后，返回数组仍须有效。Submap 还会保留预处理图像，采集环形缓冲区不能覆盖这些共享像素。
- **办法：** 执行器显式声明 `compute_similarity` 能力，不支持就报错；保留原图像顺序及匹配决策。原生 adapter 在 release 前完成必要的输出拷贝，保留输入使用稳定、不可变的 storage；错误向上传递，不静默改用 CPU。
- **验证边界：** v21 已验证原 FP32 adapter 的普通与 similarity 请求兼容性，以及首个 SL4 图。接口存在不意味着整网 NPU、自然闭环或另一厂商后端已经通过。SL4 projective 坐标也不能被接口改名成 metric SE3。
- **链接：** [执行契约与所有权](../ARCHITECTURE.md)、[原模型板端回归](../results/model_execution_v21.json)。

## 7. 仓库代码许可不能覆盖模型权重与编译产物

- **现象：** 看到本仓 BSD-2-Clause 或独立 DA3 的 Apache-2.0，容易误认为原 VGGT checkpoint 和它的 RKNN 导出也可按相同许可使用。
- **根因证据：** 本仓代码、VGGT_SPARK 源码、Meta 原权重、DA3 独立权重和 Rockchip SDK 是不同的授权对象。当前固定 Meta 原 checkpoint 的模型卡声明 CC-BY-NC-4.0；DA3 的代码和独立权重有自己的 Apache-2.0 证据，不能互相覆盖。
- **办法：** 同时记录模型 ID、revision、文件 SHA、代码来源和各自许可；发布源码与元数据时保留归属。编译、FP16 转换和使用 BSD wrapper 不移除权重条款。本仓不分发 checkpoint 或含权重的 ONNX/RKNN 文件。
- **验证边界：** 许可结论绑定文档列出的固定版本，不能外推到同名商业 checkpoint、后续版本或其他供应商 SDK。
- **链接：** [模型与产物许可](../MODEL_LICENSES.md)、[第三方声明](../THIRD_PARTY_NOTICES.md)、[许可来源收据](../results/model_license_v21.json)。

## 8. RKNN 编译成功不等于全 NPU，也不保证比 CPU 快

- **现象：** DA3 的完整 learned depth/pose 图成功调用 SDK，首对三次重复平均约 1.075 秒；同板 CPU ORT 的三次固定输入约 0.91–0.95 秒。此短测未显示净收益。
- **根因证据：** 三次实际 profile 均为 520 个内部 NPU 算子、23 个内部 CPU 算子，另有 4 个 CPU I/O。内部 CPU 包括 Transpose、Reshape、Resize、ScatterND 和 Exp；camera decoder 的单个 Transpose 约 54 ms，两个大尺寸深度头 Resize 各约 35–40 ms。不能由编译目标或算子支持表推断实际执行位置。
- **办法：** 普通计时与 profile 分开，逐算子检查实际 target；完整调用计入布局打包、SDK 输入/输出、独立输出拷贝与释放、官方 CPU codec。优化优先级依据实际热点，保持图的原始参考和固定数值门限。
- **验证边界：** 6 次普通调用、3 次 profile 均完成，两个 context 正常销毁。FP16 深度相对 L2 差异为 3.153% / 1.050% / 0.963%，严格 FP32 门限失败；DA3 原 CPU 模型的几何误差也仍大于 VGGT518 基线。短测不是持续吞吐或端到端验收，profile 耗时不能混入普通调用均值。此版本没有替换默认 VGGT。
- **链接：** [板端混合执行、数值和 profile](../results/da3_rknn_v24.json)、[CPU 原网络与几何比较](../results/da3_cpu_v24.json)。

## 9. 两个相机 token 的选择被移到全量转置之后

- **现象：** DA3-SMALL 相机分支只需每视图的 token 0，实际 CPU Transpose 却处理全部 433 个 token，三次 profile 为 54.867 / 54.382 / 54.062 ms。
- **根因证据：** 原源码与 ONNX 先在 token 轴 Gather，再进入相机 FC；编译日志明确显示 `swap_slice_transpose` 将 Slice 移到 Transpose 之后。FP16 逻辑载荷由所需的 3,072 字节扩大到 1,330,176 字节，恰好 433 倍。FC 本身仍只处理两个相机 token，并未扩大到全部 patch。
- **办法：** 保持 ONNX、权重和 FP16 配置不变，仅在既有两条禁用规则之外追加 `swap_slice_transpose`。新运行入口按编译器 SHA、完整祖先链和精确配置核验，计时及 SDK/codec 逻辑沿用冻结版本。
- **验证边界：** 新旧 A1/B1/B2/A2 共 24 次普通调用，另有 3 次 B profile。新 profile 的相机转置已缩为 `(2,768,1,1) → (1,768,1,2)`，耗时 0.395 / 0.394 / 0.394 ms。各组首对三次重复合并后，A 平均 1.087435 秒、B 平均 1.057424 秒，短测时长下降 2.76%。频率和散热未固定，不能将局部算子变化直接当成整网因果提速；433 倍也仅是逻辑载荷比。独立复算的 135 个 canonical 数组在新旧配置与各次重复之间均字节一致；相对原 CPU 的严格 FP32 门限仍失败，尚无实时或轨迹精度验收。
- **链接：** [静态诊断](../results/da3_layout_v25.json)、[配置与源码检查](../results/synthetic_tests_v25.json)。

## 10. 先分清位姿依赖与低频地图的预算

- **现象：** 将稠密地图降到每 2–5 秒更新，容易被误认为也能让同一网络每 50 ms 产生一次新观测位姿。
- **根因证据：** 当前固定两视图 DA3 图中，pose 不依赖 DPT，但仍依赖全部 12 个 backbone block。按三份已编译 profile 的输出依赖归属，移除 depth 独占行约省 226–235 ms，pose 与共享行仍有 853–897 ms；再将这些行的 CPU 时间置零，NPU 余项仍约 673–675 ms。
- **办法：** 分别核算新观测 pose 的依赖路径和低频 dense 分支，先验证实际 pose-only 图与原参考的数值、执行位置和完整调用时间，再讨论独立调度、可精确复用的输入计算及余下热点。
- **验证边界：** 上述是保持其他 profile 行时间不变的反事实预算，不是硬件下界，也不是新图实测。不能把取消 DPT、缓存旧 pose 或固定输入的重复调用计成 20 Hz 新观测位姿；2–5 秒地图与 20 Hz/P95≤100 ms 须分别验收。该 DA3 研究也不代表默认 VGGT 已达到目标。
- **链接：** [依赖与固定余项预算](../results/da3_budget_v25.json)（SHA `1b438ae9fc8a846a00efa7555b73f6ae618a14974ff519d10f7d4afa8392b13f`）。

## 11. 消除 CPU Resize 前，先保留它的像素坐标语义

- **现象：** 五个 DPT 双线性 Resize 实际落在 CPU，旧版 v24 三份 profile 合计平均 88.120 ms；替换成易编译的上采样方式可能更快，却改变深度的空间对应。
- **根因证据：** 官方函数与导出图均明确使用 `align_corners=True`，当前尺寸下坐标为 `u=j*(L_in-1)/(L_out-1)`。实际 CPU placement 已确认，但尚未证明仅这一属性就是不支持 NPU 的唯一条件。
- **办法：** 如进一步改写固定尺寸图，保留原坐标、边界和双线性权重；可研究分离的两轴 Gather 与加权和。先对原 CPU 参考使用既定 `rtol=1e-4`、`atol=1e-5`，再检查实际 placement 和完整时延，不能直接换成 `half_pixel`、nearest 或整数重复。
- **验证边界：** 此改写尚未实现。运算顺序会影响舍入，Gather 也可能仍在 CPU，并增加中间缓冲；保持数学表达并不自动保证字节一致或更快。
- **链接：** [原 Resize 属性与实际 profile](../results/da3_layout_v25.json)、[官方插值实现](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src/depth_anything_3/model/utils/head_utils.py)。
