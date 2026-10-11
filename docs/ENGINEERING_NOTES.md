# 可复用的工程问题记录

这里记录移植中能够复现、值得复用的故障定位方法。DA3-SMALL 和 DPVO 均是独立模型研究，原 VGGT 默认模型没有替换；ONNX 的 CPU 数值验证也不代表 NPU 或实时 SLAM 已通过。各条状态以链接中的固定源码、输入和收据为准。

记录优先保留会影响正确性、性能、部署和维护的可复用问题：触发条件与现象、根因证据、改动、验证结果、适用边界及复现链接。失败实验和未证实的判断也保留，并与已完成修复分开；整机指标使用完整回放的实际分母。

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
- **链接：** [依赖与固定余项预算](../results/da3_budget_v25.json)。

## 11. 消除 CPU Resize 前，先保留它的像素坐标语义

- **现象：** 五个 DPT 双线性 Resize 实际落在 CPU，旧版 v24 三份 profile 合计平均 88.120 ms；替换成易编译的上采样方式可能更快，却改变深度的空间对应。
- **根因证据：** 官方函数与导出图均明确使用 `align_corners=True`，当前尺寸下坐标为 `u=j*(L_in-1)/(L_out-1)`。实际 CPU placement 已确认，但尚未证明仅这一属性就是不支持 NPU 的唯一条件。
- **办法：** 如进一步改写固定尺寸图，保留原坐标、边界和双线性权重；可研究分离的两轴 Gather 与加权和。先对原 CPU 参考使用既定 `rtol=1e-4`、`atol=1e-5`，再检查实际 placement 和完整时延，不能直接换成 `half_pixel`、nearest 或整数重复。
- **验证边界：** 此改写尚未实现。运算顺序会影响舍入，Gather 也可能仍在 CPU，并增加中间缓冲；保持数学表达并不自动保证字节一致或更快。
- **链接：** [原 Resize 属性与实际 profile](../results/da3_layout_v25.json)、[官方插值实现](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src/depth_anything_3/model/utils/head_utils.py)。

## 12. 小模型也要先固定输入和完整权重契约

- **现象：** 图像与权重都能加载，不代表输入或网络已经与作者版本一致。直接复用 VGGT 的 RGB、ImageNet 归一化、336 输入，或者对遗留 checkpoint 使用 `strict=False`，都可能产生看似正常的输出。
- **根因证据：** 固定 DPVO 的 TUM reader 使用 `cv2.imread` 的 BGR uint8，先用原 K、五项畸变执行 `cv2.undistort`，再裁掉左右各 16、上下各 8 像素：原 `640×480` 变为 `608×464`，`cx=302.6, cy=247.3`。进入网络的公式是 `2*(image/255)-0.5`，范围为 `[-0.5,1.5]`，两个 encoder 输出还各除以 `4`。官方发布文件有 98 个张量，当前原始神经模块有 94 个；官方 loader 明确排除四个 `update.lmbda` 遗留键并移除 `module.` 前缀。
- **办法：** 逐项固定 BGR、K/D、裁边、一次 FP32 归一化与 `/4`，保留原始连续帧及 PNG SHA。只允许这四个已知遗留键，逐 shape/dtype/参数字节核对剩余 94 键，再 `strict=True`。本次 98 键含 3,532,549 个元素，94 个推理键含 3,384,324 个元素；不能将差额当成漏载，也不能据此接受其他缺失键。
- **验证边界：** 三个连续原始帧的 CPU capture、singleton-view 包装器与板端 CPU ORT 已形成对照链。原包装器输出字节一致，六个 ORT feature 数组在固定 `rtol=1e-4, atol=1e-5` 下均为零失败元素；独立保存数组审计 120 项通过。这里验证的是原神经 encoder 的 CPU FP32 路径，且未使用 GT 或传感器 depth；不是原完整 CUDA VO、NPU 或轨迹验收。eager/导出的差异也不能通过再归一化或放宽门限消除。
- **链接：** [源码与权重来源](../results/dpvo_sources_v26.json)、[CPU/ORT/SDK 编码器证据](../results/dpvo_encoders_v26.json)、[官方 TUM reader](https://github.com/princeton-vl/DPVO/blob/0ac95b656d1fda91c271d2a106460d19ad966fc7/evaluate_tum.py#L26-L51)、[官方归一化](https://github.com/princeton-vl/DPVO/blob/0ac95b656d1fda91c271d2a106460d19ad966fc7/dpvo/dpvo.py#L389)、[官方遗留键规则](https://github.com/princeton-vl/DPVO/blob/0ac95b656d1fda91c271d2a106460d19ad966fc7/dpvo/dpvo.py#L89-L101)。

## 13. 自定义 CUDA 算子的 shape 相同，索引和展平顺序仍可能错

- **现象：** CPU 版 correlation 也能输出 `[1,E,882]`，Update 也能正常运行，但交换 offset 或把两个 level 直接拼接，会改变已训练 FC 接收到的每个特征。
- **根因证据：** `fastba.neighbors` 的本体先在 CPU 按 patch ID 分组，再对目标帧 ID 稳定排序，最后把结果搬回 CUDA。同目标 ID 的边须保留输入顺序，组边界使用 `-1`。correlation 核先算通道点积，再做四项双线性组合；越界采样为零，没有额外通道归一化。返回轴为 `(B,E,offset_x,offset_y,patch_y,patch_x)`。原 `CorrBlock` 对 `[1,4]` 两个 level 执行 `stack(...,-1)` 再展平，因此 **level 是最内层交错维度**，不是“level 1 的完整向量 + level 4 的完整向量”。
- **办法：** 从原 C++/CUDA 和 Python wrapper 一起读取契约，用独立小张量验证稳定 ties、空组、边界、分数坐标、非方形 patch、offset 轴及 level 交错。邻居缺失时，原 Update 先将邻居输入乘零再送进有 bias 的 `c1/c2`；不能直接跳过这些 FC。分组 ID 保持 int64，不能转换为 FP32 后分组。保留原最大值平移的分组 softmax 和 scatter sum。
- **验证边界：** 当前 CPU reference 通过 38 项纯合成检查，未实际执行 CUDA extension 作逐字节对照。浮点通道归约、FP16 累加和 FMA 顺序可能不同；“数学前向语义对照通过”不能改写为 GPU bitwise 等价，也不包含训练 backward/dropout。
- **链接：** [CPU 算子与动态神经探针](../results/dpvo_motion_v26.json)、[邻居排序原实现](https://github.com/princeton-vl/DPVO/blob/0ac95b656d1fda91c271d2a106460d19ad966fc7/dpvo/fastba/ba.cpp#L61-L100)、[correlation 轴置换](https://github.com/princeton-vl/DPVO/blob/0ac95b656d1fda91c271d2a106460d19ad966fc7/dpvo/altcorr/correlation_kernel.cu#L198-L232)、[原两级展平](https://github.com/princeton-vl/DPVO/blob/0ac95b656d1fda91c271d2a106460d19ad966fc7/dpvo/net.py#L163-L175)。

## 14. 神经 Update 的二维修正不是相机位姿

- **现象：** encoder、相关性和原 Update 全部使用真实预训练权重成功运行，容易被说成已经移植了 learned VO，或把神经调用频率当作新观测 pose 频率。
- **根因证据：** 原 Update 返回边的 hidden state、二维 `delta` 和二维 sigmoid `weight`。原完整链路必须先从当前姿态与 patch 几何重投影，再算相关性/Update，构造中心像素 `target=coords+delta`，最后调用 BA 更新 SE3 与逆深度。当前动态探针使用固定 identity 图、单位逆深度和预定边集合；它没有执行 BA，也没有建立持续的帧/关键帧图。三帧结果不会自动完成原 VO 初始化。
- **办法：** 分开记录 encoder、patch 选取、correlation、分组/Update 和未来 BA 的时长与调用次数。明确当前输出是边的像素修正/权重，不输出假的 `Tcw`；固定 identity 场景是算子探针，不是轨迹。后续完整 VO 必须保留原初始化、几何重投影、关键帧/图状态、SE3 更新与失败语义；仍不能使用 ORB、LK/PnP 或补帧插值来代替要求的新观测位姿。
- **验证边界：** 当前 CPU Torch reference 的 E=96 场景记录 correlation **710.015 ms**、Update **103.582 ms**；E=864 固定图记录 **6,018.016 ms** 与 **276.762 ms**。这是真实捕获 feature 上的参考实现时长，不是优化算法或硬件下界，不能据此推断任何更快实现必然失败。当前只可记录“板端三帧神经组件成功”，不能记录“完整 DPVO 成功”“20 Hz 位姿”“稠密地图达标”或“已替换原 VGGT”。SDK/混合执行还须分别报告 CPU 与 NPU 的实际算子、同步 I/O 和所有必要动态 CPU 工作；compiled target 表不能代替实际 profile。
- **链接：** [动态神经探针与时长](../results/dpvo_motion_v26.json)、[原 Update 输出](https://github.com/princeton-vl/DPVO/blob/0ac95b656d1fda91c271d2a106460d19ad966fc7/dpvo/net.py#L74-L94)、[原重投影、Update 与 BA](https://github.com/princeton-vl/DPVO/blob/0ac95b656d1fda91c271d2a106460d19ad966fc7/dpvo/dpvo.py#L328-L354)。本地 `run-v26-dpvo-motion-board.py` 已显式设置 `BA_executed=false`、`valid_pose_output=false`、`tracked_trajectory=false`。

## 15. 小参数量仍可能被 InstanceNorm 路径的 CPU 转置与输出 I/O 拖慢

- **现象：** 两个 CNN 的 SDK 普通首帧后三次平均 **426.785 ms**，完整调用计入输入布局/提交、同步 run、两路 FP32 输出获取、拷贝及释放。CPU ORT 三帧此前约 **277–293 ms**。两个 CNN 合计仅 379,520 个参数（含 Update 的完整神经模块为 3,384,324 个）；当前短测仍没有证明混合 SDK 更快。
- **根因证据：** 三份实际 profile 均为 **82 个内部 NPU、5 个内部 CPU Transpose、3 个 CPU I/O**，共 90 行。五个 Transpose 都出现在 fnet 前五处高分辨率 InstanceNorm 路径，由 `(1,32,232,304)` 变为 `(232,304,1,32)`；三次合计分别 **216.530 / 219.718 / 216.708 ms**。普通 run API 本身约 352–357 ms，不能当完整时延。两路原 FP32 feature 输出还共 **36,110,336 字节（34.4375 MiB）**，SDK driver 实际执行了 owned copy 与 release。这些是实际布局和 I/O 证据；尚不能由节点位置推断 InstanceNorm 本身不支持 NPU 或指定唯一编译根因。
- **办法：** 优先研究保持原统计轴、eps、affine/default 语义的固定 InstanceNorm 图表示，检查能否避免这些布局转换；不能直接删 normalization 或换成有运行统计的 BatchNorm。次优先级是输出需求：原 Patchifier 对稠密 `imap` 只采样 96 个中心，FP32 context 为 **147,456 字节**，完整 `imap` 为 **27,082,752 字节**。可研究固定中心输入与仅返回采样 context，但保留完整 encoder、`/4`、坐标及双线性语义，稠密 `fmap` 仍供相关金字塔使用。先作原输出数值对照，再看实际 placement 与全调用时延；payload 减少不能自动解释成 FPS 提升。
- **验证边界：** 六次普通 SDK 调用和三次独立 profile 已完成，尚不是持久吞吐或控制温度/频率的公平 benchmark。相对原 CPU 的固定 FP32 gate 全失败：三帧 `fmap` 相对 L2 为 **0.3802 / 0.3502 / 0.4135%**，`imap` 为 **0.0902 / 0.0995 / 0.0956%**；这些是 feature 差异，不是 VO 轨迹误差。在 v26 阶段两项图/输出优化均未实施；后续 InstanceNorm 改写的实际数值失败见第 17 条，输出采样优化仍未实施。不能预设成功、全 NPU、精度通过或 20 Hz。profile 时长与普通均值分列，CNN 成功也不消除 correlation、Update、BA 和地图成本。
- **链接：** [实际 SDK、CPU/ORT 与数值对照](../results/dpvo_encoders_v26.json)、[原 feature 缩放与采样](https://github.com/princeton-vl/DPVO/blob/0ac95b656d1fda91c271d2a106460d19ad966fc7/dpvo/net.py#L114-L147)、[官方默认 96 点 RANDOM](https://github.com/princeton-vl/DPVO/blob/0ac95b656d1fda91c271d2a106460d19ad966fc7/config/default.yaml)。

## 16. 原生相关性提速要连同缓存生命周期与准备成本一起验证

- **现象：** 相同真实 feature 的板端交错短测中，原 CPU reference 的相关性后三次平均为 E=96 **691.117–756.247 ms**、E=864 **5,557.756–5,675.967 ms**。新原生 CPU 实现逐次准备、计算、关闭的完整相关性调用为 **51.053–51.163 ms** 与 **80.952–82.360 ms**；复用已准备 context 后为 **9.293–11.952 ms** 与 **48.285–48.681 ms**。这些范围分别包含固定 identity 与预定分数坐标偏移两种场景，不能混用此前 v26 独立运行的时长计算收益。
- **根因证据：** 原参考每次重新构建金字塔；新 context 持有按通道连续排列的 gmap、level 1 与独立 `kernel=stride=4` 平均池化的 level 4 值快照。缓存占 **30,102,528 字节**，一次准备实测 **18.679–46.435 ms**，此成本没有计入缓存重复查询均值。shape 或指针相同不证明内容仍属于同一 feature 版本；调用方原数组变化也不会更新已准备的快照。
- **办法：** 以明确的 feature 版本 key 绑定 context，新 feature 重新准备；每次查询核 key，close 幂等并与查询互斥，返回独立输出。保留原 FP32 坐标先除 level、通道点积后四项双线性组合、越界为零及 `offset_x → offset_y → patch_y → patch_x → level` 展平顺序。NEON 使用乘法与加法，不启用 fast-math；准备、重复查询和关闭分别记录，逐次准备模式另计完整成本。
- **验证边界：** 四种固定图、三种执行模式共 **48 次**调用，相关性与完整原 Update 的 `net/delta/weight` 均通过预先固定的 **rtol=1e-4、atol=1e-5**，所有 context 关闭后保留字节为零，输入与原权重未变。缓存模式下 Update 仍需 E=96 **80.527–89.474 ms**、E=864 **192.203–200.200 ms**；相关性加 Update 为 **92.479–98.766 ms** 与 **240.487–248.881 ms**，尚未包含 CNN、patch/图准备、几何重投影或 BA。这是原生 **CPU** 算子提速，不是 NPU、CUDA 字节等价、完整 VO 或 20 Hz 新观测位姿验收；短测也未控制温度和频率。
- **链接：** [原生相关性、完整成本与数值对照](../results/dpvo_native_corr_v27.json)、[缓存与生命周期接口](../deploy/rk3588/validation/dpvo_native/dpvo_corr.py)。

## 17. 数学等价的 InstanceNorm 改写仍可能失败于固定浮点门槛

- **现象：** 为研究第 15 条的布局成本，将十个 InstanceNorm 改成基本算术后，板端 CPU ORT 能执行，三帧 `fmap` 却分别有 **4 / 5 / 53** 个元素未通过原 CPU 的固定 FP32 门槛。收据保留为 **failed**，不能由输出有限或整体相对误差很小改写成精度通过。
- **根因证据：** 改写保留 H/W 统计轴、原 eps 与 affine：`μ=mean_HW(x)`，`y=γ*(x-μ)/sqrt(mean_HW((x-μ)^2)+eps)+β`，没有使用 `mean(x²)-μ²`。数学公式相同，分步归约与原融合算子的舍入仍可不同。相对原 CPU，三帧 `fmap` 最大绝对差约 **1.2621e-5 / 1.2606e-5 / 1.7166e-5**；相对原 ORT，则有 **10 / 2 / 69** 个失败元素，最大差 **1.7792e-5**。三帧 `imap` 与原 ORT 字节一致，不能用这一分支的通过覆盖 `fmap` 的失败。
- **办法：** 保存原图、新图、原 CPU/ORT 参考与全部真实输出；维持 **rtol=1e-4、atol=1e-5**，不按结果放宽门槛或删掉失败像素。将舍入差异作为待解释的数值反例，先核原轴、eps、常量 dtype 和归约顺序；任何后续 SDK 测试须单列为诊断，不能继承“已通过原 FP32 参考”的标签。
- **验证边界：** 三次实际 ORT 调用已完成，来源与输入未变、清理无错误，但严格 gate 失败，当时尚未编译候选或在 NPU 上 profile。后续 v32 隔离诊断见第 40／41 条，原 FP32 拒绝仍保留。小张量公式检查只能验证实现意图，不能替代真实特征的逐元素对照；本次也未证明舍入路径是唯一根因。不能将这张候选图认定为数值已验收的 NPU 部署、VO 精度通过或实时优化成功。
- **链接：** [InstanceNorm 改写与保留的严格失败证据](../results/dpvo_norm_rewrite_v27.json)。

## 18. 隔离工具链能输出版本，不代表汇编和链接链路完整

- **现象：** 在项目私有 sysroot 解包编译依赖后，`g++ --version` 正常，首个共享库 smoke 却失败于 `collect2: fatal error: cannot find 'ld'`。
- **根因证据：** 隔离目录有带 `aarch64-linux-gnu-` 前缀的工具，编译子进程仍按 `ld/as` 名称查找；版本输出没有运行这些子工具。
- **办法：** 仅在私有目录建立 `ld/as` 到对应带前缀工具的链接，保留明确的 sysroot、`-B` 搜索路径与依赖 SHA；不安装到系统或修改系统工具。再实际编译、加载共享库并调用已知结果的 tiny smoke。
- **验证边界：** 第二次 smoke 编译与加载通过，输入 2 得到预定结果 8，随后原生库另有实际构建与算子验证。这证明该隔离链路可用于本次库，不证明任意源码、ABI 或依赖均可用；后续部署仍须核库版本与真实输出。
- **链接：** [隔离构建与原生库来源](../results/dpvo_native_corr_v27.json)。

## 19. 导出动态图网络时，不能把本次边关系固化成模型

- **现象：** DPVO Update 同时依赖神经状态、相关性、同 patch 的邻居及两种分组。只让一次 trace 通过，可能把某次排序、邻接或分组固化到 ONNX；shape 不变而图关系变化时仍会静默给出错误结果。
- **办法：** 将拓扑准备移到图管理侧，导出 13 个输入：三个神经数组和十个拓扑数组，包含六个 int64 索引及四个 FP32 mask。保留原稳定排序、`ii*12345+jj` 的整数语义、逐通道分组 softmax，以及 c1→c2→agg_kk→agg_ij→GRU 的先后依赖。容量不足明确拒绝，不能截断组或边；原参数全部保留。
- **证据：** 两种固定容量 E=96/E=864 的 ONNX 均保留全部 13 个运行时输入。37 项合成检查包含同容量改变邻接及分组的 A/B/A 对照、padding 和非零隐状态；板端四组真实图的 16 个捕获状态通过原网络→包装器→CPU ORT 的固定 FP32 门限，保存数组独审通过 5,242 项。CPU ORT 的同一零状态三次 warm 均值为 E96 **6.4–7.3 ms**、E864 **48.1–52.5 ms**，包括 session.run 返回的独立输出，排除拓扑准备等外部工作。
- **状态与边界：** 固定容量模型支持本次接口中的拓扑输入，不等于任意动态图、在线图生命周期或动态 shape 已验证。ORT 每个非零状态来自原 CPU 捕获，尚未测试 ORT 自己连续反馈；数值门限通过也不能覆盖第 21 条值域问题。组件时间不包括 CNN、相关性、BA 或新帧位姿发布。
- **链接：** [完整 Update 对照链](../results/dpvo_update_cpu_v28.json)、[拓扑与导出适配](../deploy/rk3588/validation/dpvo_update_export.py)。

## 20. 混合输入 SDK 不能统一按 float32 或原生字节数打包

- **现象：** RKNN 查询的 13 个输入均标记格式 UNDEFINED；六个索引仍是 INT64，七个浮点输入的原生类型是 FP16。若复用“所有输入都是 FP32 图像”的包装器，会误传索引、布局或字节数。
- **证据与办法：** 本次实际 attrs 中，INT64 输入保留 type=8、8N 字节传输；浮点输入使用 FP32 type=0、4N 字节传输，`pass_through=0` 让 SDK 按原生 FP16 转换。输出申请 `want_float=1` 后核对 4N 字节，并在 release 前拷贝到自有数组。UNDEFINED 不能猜成 NHWC；需按 ONNX 逻辑名称、shape、dtype 和实际 attrs 逐项匹配。attrs 的 quantization_type=2 在本次为 scale=1、zero_point=0、fl=0，连整数也如此；仅凭枚举名不能推断输入已量化，也不能放行非恒等量化参数。
- **状态与边界：** 48 项 fake SDK/打包检查覆盖类型、范围、缓冲区、frame ID 与失败释放；真实 attrs 查询和随后的板端调用均完成清理。SDK 返回成功仍须独立校验输出值域和有限性，本次网络输出就被拒绝。不能将“ABI 正确”写成“模型正确”。
- **链接：** [实际 SDK 契约与失败收据](../results/dpvo_update_rknn_v28.json)、[混合输入打包](../deploy/rk3588/validation/v28_rknn_inputs.py)。

## 21. 误差门限通过，不代表输出满足下游数学值域

- **现象：** 完整 Update 的板端 CPU ORT 输出全部通过 `rtol=1e-4, atol=1e-5`，但 sigmoid 权重出现小于零的值，最小为 **−1.1920928955078125e-7**。关闭和启用 ORT 优化的两种模式都出现此现象，各自在四个零状态保存结果中有 59 个负元素。
- **证据与结论：** 绝对误差容许范围覆盖了这一级别的微负值；原 BA 使用非负权重，逐元素接近参考与满足 `[0,1]` 是两个不同条件。当前证据能确定值域违反，尚未定位到具体 sigmoid 内核或编译优化。
- **办法：** 保留原数组与固定误差门限，独立报告有限性和 `[0,1]` 检查；不悄悄裁剪后再与参考比较。本轮真实几何反馈仍用原 Torch Update 的权重，没有将这些 ORT 权重送入 BA。后续任何表达式修复或显式值域修复均须作为新候选重新对照，并验证连续反馈与轨迹影响。
- **状态：** 原 FP32 误差门限通过，ORT 权重的下游使用尚未接受；不是已完成的 CPU 加速部署。
- **链接：** [包含值域检查的独立审计](../results/dpvo_update_cpu_v28.json)、[几何反馈使用的原 Update](../results/dpvo_feedback_v28.json)。

## 22. 编译与 SDK 调用成功，也可能整组输出非有限

- **现象：** 完整 E864 Update 编译成功，首次普通 RKNN 调用在输出检查处报 `Nonfinite SDK output`。另开独立 profile 上下文，只执行一次并保存未修正输出：隐状态含 **226,977 个 NaN、104,799 个 +Inf**，位移 **1,728 个值全部 +Inf**；权重虽有限，却全部为 **0.998046875**，全部违反原参考误差门限。
- **已知证据：** 输入及其直接 FP16 转换均有限；ONNX 两个 masked-max 的未选择分支含 `-inf`，本次实际 mask 全为 1。六个 Where 和四个 exNorm 的实际 profile target 均为 NPU，但 profile 不提供中间值，不能据此找到首个污染节点。SDK 内存在历史 Where 算术改写规则，其说明明确当前禁用；不能把这个规则名当成已发生的根因。
- **办法：** 普通失败保留，诊断器只拦截两个明确的输出值域检查以保存原数组，在第二次 SDK 调用之前停止；所有来源、ABI、大小、frame ID 和释放检查继续生效。14 项 fake SDK 检查覆盖这一边界。真实单次诊断正常释放输出、销毁上下文，来源未变，候选始终标记为不接受。
- **状态与下一步：** **根因未确认。** 下一步需分别隔离 Where 非有限分支和 FP16 归一化/中间范围，任何新图先过板端原 FP32 对照。实际 profile 有 86 个内部 NPU 算子、24 个内部 CPU 算子及 16 个 CPU I/O 算子；因此也不能称为全 NPU。故障 profile 时间不是有效推理延迟或加速成果。
- **链接：** [普通失败、原输出元数据与实际 profile](../results/dpvo_update_rknn_v28.json)、[单次诊断器](../deploy/rk3588/validation/diagnose-v28-dpvo-update-rknn-board.py)。

## 23. 同名 BA 的训练实现不能直接替代在线 fastBA

- **现象：** 官方训练 BA 与在线 CUDA fastBA 使用不同的有效边规则、阻尼和逆深度更新。只迁移能在 CPU 导入的函数，会改变在线算法。
- **办法与证据：** 本次 NumPy 参考按固定版本在线方程实现：world-to-camera、xyzw 四元数、左扰动 `Exp(δ)*pose`，patch 中心逆深度、在线有效边条件、Schur 约化后的阻尼和原深度限制。另区分 correlation 的完整 patch 重投影与 BA 中心投影，以及 LieTorch 归一化四元数与 CUDA 手写运算的语义。求解失败报错，成功后才提交状态。15 项合成检查覆盖 SE3、左扰动 Jacobian 有限差分、Schur 与完整法方程、固定姿态和深度更新。
- **状态与边界：** 这是有容量限制的 dense CPU 方程参考；没有训练 backward，也不是原 CUDA 数值或字节等价证明。它已参与板端真实反馈，但后续仍需 CUDA 对照及轨迹质量验证。相关代码保留独立 MIT 归属。
- **链接：** [方程来源与验证证据](../results/dpvo_feedback_v28.json)、[CPU 几何实现](../deploy/rk3588/validation/dpvo_geometry/dpvo_cpu_geometry.py)。

## 24. 反馈残差下降，需要核对状态连续性和同一有效边集合

- **现象：** 重复 identity 坐标或重置隐状态不能验证神经—几何反馈；仅比较优化前后残差，还可能把丢弃难边误认为收敛。
- **办法与证据：** 三帧真实特征、288 个 patch、864 条固定边上执行四轮 `当前几何→重投影→原生相关性→原 Update→中心坐标+delta→两步 BA`，逐轮保存前后状态。独立审计通过 **214 项**：上一轮 pose/depth/net 精确进入下一轮，target 构造正确，预声明七条边的完整 patch 和中心投影通过独立公式检查。四轮各有 864 条有效边，未丢边；固定每轮 BA 前集合的加权残差平方和分别为 **448.735→19.2597、19.0459→8.67373、10.2212→7.60544、10.1696→8.79032**。
- **性能与状态：** 四轮共 **1.169 s**，Update 占 **56.6%**、两步 BA 占 **31.6%**、相关性占 **7.2%**。计时包含重投影、包装/输出拷贝及诊断，排除此前 CNN 捕获、一次加载/缓存准备和文件写入；没有控制温度/频率。下一步对 Update 及 BA 分别优化后，仍须重测整个步骤。缓存关闭后为零，原源码、输入和 94 个权重张量未变。
- **验证边界：** 各轮学习目标与权重会变化，以上残差不能串成统一损失曲线，更不能等同 GT 精度。初始化为 identity/unit-depth，并非官方八帧初始化；尚无新帧准入、关键帧删除、回环、在线轨迹或实时验收。峰值 RSS **341.164 MiB** 仅属本次进程，排除了另一次 CNN 捕获。
- **链接：** [四轮真实反馈、独审和性能分解](../results/dpvo_feedback_v28.json)、[保存数组审计器](../deploy/rk3588/validation/dpvo_geometry/audit-v28-dpvo-feedback.py)。

## 25. 误差门槛通过，仍须检查置信权重的值域

**现象与证据：** 原 Update 的板端 CPU ORT 输出通过 rtol=1e-4 / atol=1e-5，但 Sigmoid 权重最小为 −1.1920928955078125e-7。它在绝对误差容限内，却违反 BA 的非负权重约束，不能裁零后继续使用。

**办法与验证：** 仅将末端 Sigmoid 表示为 exp(min(x,0))/(1+exp(-abs(x)))。对有限输入，它数学等价且两个指数的自变量均非正；原权重、其余节点及 I/O 不变，未增加 output clipping。四种真实图/状态的 16 次板端 CPU ORT 比较均通过原门槛和权重值域，最小权重 1.5825583066231275e-9。

**边界与状态：** 这是已测 CPU ORT 样本上的修正，不是 RKNN 数值验收、反馈 rollout 或轨迹证明。依据：[Update CPU 对照](../results/dpvo_update_cpu_v29.json)，实际收据 SHA 98b5e47b8558cc4440945e882729940464366e0fd27c2905599de6b0358dc485；改写器 rewrite-v29-weight-sigmoid.py。

## 26. BA 批量化须保留自连接、重复索引和累加顺序

**问题与办法：** 高级索引 += 不能保证重复索引全部累加；ii==jj 的边须先合并 Ji+Jj 再计算二次项，否则丢交叉项。新版按原边顺序分块构造 Jacobian，以 np.add.at 累加重复槽位，保持 active mask、权重约束、Schur damping、左乘 SE3、逆深度约束及失败不提交语义。

**验证：** 四个真实冻结状态，旧/新各首调用和三次交错 warm 调用；另八次仪器化 BA 暴露两步系统，不纳入普通计时。八份保存结果的 224 个张量逐字节相同，六个末端输出也与 v28 实际反馈一致；独立 saved-array 审计 3,381 项通过，主机未重跑 BA。

**性能边界：** 两步 BA warm 均值由 72.309–72.569 ms 降至 13.981–14.058 ms，组件比值 5.158–5.172×。四核 affinity、OPENBLAS_NUM_THREADS=1；同进程交错、未控温，计时不含 CNN、相关性、Update、新帧或完整反馈，不能折算为 VO/20 Hz。依据：[BA 对照](../results/dpvo_geometry_v29.json)，板收据 SHA c870b31ad5b5ed96e8f178b92f3b7badb710b023eb31a845f934868e0b5fa2b4，独审 SHA b8b2c0d69fd1e47e4d660ce488d0b6a1160b2068dcc383f8a01fa4523da89d69。

## 27. 输出 tap 会改变融合，不能直接定位原图首个故障

**证据：** Update 诊断图的声明输出中，最早暴露的非有限张量是 /Exp_output_0；此前暴露的 /Where_output_0 和 /ReduceMax_output_0 已有显著数值偏差。这只是该仪器化图的观测顺序。

**办法与边界：** 同时保留原图和带 tap 图的编译/运行收据，记录逐张量 finite/NaN/Inf 及 CPU 对照，正常计时与 profile 分列。tap 可阻止融合或改变调度，不能据此断言原图对应内部 kernel 是首个故障或唯一根因；诊断图末端全 finite 也不等于数值正确。

**状态：** 原因未确认，候选未验收。依据：[RKNN 诊断](../results/dpvo_update_rknn_v29.json)，tap 独审 SHA 12a567e4225271e9507b52309c8d6cf5fa6f0c9add42ae8da1b6d28d83d910b0。

## 28. 去掉 −inf 常量，不等于消除了数值故障

**假设与办法：** 分组 ReduceMax 用 −inf 填充无效槽。固定拓扑确保真实成员在前、非空组首槽是真实成员时，用首个真实 logit 填充其 padding 不改变最大值。空组防护、mask、分母及学习权重保留，不改成任意大负数近似。

**验证与结论：** 16 个 CPU ORT 状态保持原逐元素门槛，但 Sigmoid 负权重仍在；实际 E864 RKNN 仍出现 NaN/Inf、未数值验收。有限常量只是受拓扑及有限 logit 约束约束的等价改写，失败证据不支持“−inf 是唯一根因”。编译完成、target 映射及 SDK 返回成功不能代替输出检查。

**依据：** rewrite-v29-dpvo-group-max.py 与 [失败收据](../results/dpvo_update_rknn_v29.json)。CPU SHA 7b3cbb0f5d4a17787f535bc6a4e1e60858b03a6f72545ba00d131ea25f2bb974；RKNN SHA 5cc604e09c959c620ee2dca114199091bc64401f5d9bb3fdb17df04ded052e8c。

## 29. Where 的乘加改写必须先证明两侧都有限

**新实验：** 在有限 padding 候选上，将六个 Where(c,x,y) 表示为 float(c)*x+(1-float(c))*y，保留原权重及其余节点，不改变选择约束或误差门槛。

**前提与风险：** mask 须为精确二值，运行时两侧均有限。0*Inf 或 0*NaN 会污染输出，检查图内常量有限不足以证明运行时中间张量安全；不能把带 −inf 的原图直接照搬为乘加。

**状态：** 此候选已完成 11 次正常 SDK 调用和另一次 profile，输出均 finite 且权重值域通过，但原 FP32 比较仍拒绝，不能因消除 NaN/Inf 提前采用。CPU 对照也仍有原 Sigmoid 的负权重问题。乘加改写没有因此取得数值验收、可用于 BA 或性能结论；完整审计由对应结果收据保存。 实际 profile 仍有 100 个内部 NPU 算子、24 个内部 CPU 算子及 16 个 CPU I/O，不能称为全 NPU。

**依据：** rewrite-v29-finite-where.py、[CPU 对照](../results/dpvo_update_cpu_v29.json)、[RKNN 对照](../results/dpvo_update_rknn_v29.json)。当前 CPU SHA 11a7d78d77e59ed89f3076f9bdfa78313b1cbed1b468ef7d3fcabee7a357012a；SDK SHA 9dd4ff2d8a5dfbf247632007d79b838b768c0abb6004f6eab68d2ac2f511a367。

## 30. 同当前状态的一步神经对照，不是独立的黄金轨迹

**办法与证据：** 将已验收的稳定 Sigmoid CPU ORT Update 接回真实反馈：当前 pose/depth 重投影→native correlation→保留 net 的 ORT Update→target=center+delta→两步 vectorized BA，循环四轮。每轮另用原 Torch Update 消费同一当前 corr 与 ORT 的 recurrent net，核原固定 FP32 门槛和权重值域。实际四轮、八步 BA 全部完成，清理无错误。

**边界：** 原 Torch 是相同当前输入的一步参考，不是从初始化独立运行的黄金 recurrent trajectory。pose/逆深度变化、残差降低及该门槛通过，均不能替代 GT、独立轨迹或在线 VO 验证。

**计时：** 固定三个已捕获视图的四轮反馈合计 438.817 ms，均值 109.704 ms；分项 Update 228.592 ms、BA 59.238 ms、correlation 95.842 ms。独立 Torch 对照、加载、准备、hash/保存另列，实际整个研究 driver 4.076 s。历史 v28 四轮 1.169 s 未作同期受控 AB，不宣称公平整体加速比；还不含 CNN、新帧或官方八帧初始化，不是 20 Hz。

**状态与依据：** 固定图 CPU ORT 反馈研究通过，完整 VO/轨迹/实时未验证。[反馈收据](../results/dpvo_feedback_v29.json)，实际原始收据 SHA 00b10efa4ce0cda15325dfcc73f9983f214c4ccda6d46519bf63b0810f7c4d8e。


## 31. 固定图反馈通过，不能省略原算法的新帧生命周期

**问题与源码证据：** 原 DPVO 每帧 96 个 patch，前 7 个准入帧不执行常规图 Update/BA，但仍对候选帧作运动探测；第 8 个准入帧执行 12 次 Update、每次两步 BA。初始图有 **6,144 条边**，此前 E864 的固定容量 ORT 模型不能直接用于完整初始化。顺序为旧 patch→新帧的前向边、新 patch→窗口各帧的后向边，包含 self 边；旧边隐状态继续携带。

**办法：** 图生命周期与 encoder、Update、correlation、geometry 执行分开。保留原运动阈值、damped-linear 时间间隔、建边／删帧和相对位姿链。每个候选都消费随机逆深度；初始化后再以最近 3 个保留帧的下中位数覆盖，不能用偶数样本平均中位数。当前使用原 Torch Update 处理变化的边数；压缩原生缓存只改变存储索引。本轮容量明确限为 64；超限、非有限或 BA 失败时停止，不自行发明淘汰策略或发布旧位姿。

**实际验证：** RK3588S 完成 TUM fr1_xyz 前 192 个连续 RGB 输入：186 个准入，6 个被初始运动门禁跳过；源 index 13 完成八帧初始化，产生 179 条新观测位姿。执行 169 次关键帧删除，结束保留 17 帧、23,520 条活跃边。源期 190 次图 Update，另有 13 次运动 probe 和 12 次 EOF Update，合计 215 次原神经 Update、404 步 BA。保存状态审计的 13 组检查通过，原 94 个参数张量和源／输入哈希前后不变。

**边界：** 这是 CPU FP32 learned-VO 研究，不是原 CUDA 的数值等价证明，也未替换默认 VGGT。此短序列未触发 source-age inactive 因子迁移（最终为零），该分支仍只有源码／合成检查；回环关闭。容量上限不等于 30 分钟稳定运行证明。依据：[板端及独立审计](../results/dpvo_stream_v30.json)、[实现与契约](../deploy/rk3588/validation/dpvo_stream/README.md)。

## 32. 压缩 correlation 缓存时，只重排查询索引，不改图拓扑

**问题：** 装入全部保留帧会为当前未引用的 fmap 分配空间；关键帧删除后，旧 slot 又可能指向另一图像。按 shape 或旧 slot 缓存可能得到形状合法、身份错误的结果。

**办法：** 分别取有序 source 集 `unique(kk//M)` 和 target 集 `unique(jj)`，仅装入这些不可变 gmap/fmap；native 查询改用 `kk_local=source_inverse[kk//M]*M+kk%M`、`jj_local=target_inverse[jj]`，边顺序和 patch 局部号不变。原 ii/jj/kk 继续交给 Update 和 BA。缓存键绑定有序真实 source/target frame ID 与 M，固定几何由输入契约保证。成员／顺序改变才重建；若只是 slot 压缩且这些身份不变，同一代缓存可复用。

**验证：** 五项 fake-cache 检查覆盖非连续 slot、重复边、删帧重映射和旧输出所有权。实际 215 次相关性查询产生 192 代 context；独立按每次图重建 source/target ID 与缓存 key，全部吻合，所有 context 关闭后保留字节为零。完整运行含 EOF 的相关性回调累计 **102.962 s**，包含本回调的准备／重建成本，不等于只查询缓存的时间。

**边界：** 压缩的是 feature 存储，边数和 Update 的主要矩阵计算量未减少。真实流未逐次重跑旧 correlation 参考；数值原算子对照沿用 [v27 的固定输入证据](../results/dpvo_native_corr_v27.json)，本轮独立审计验证身份与状态关系。不能据此声称 NPU 或 20 Hz。[本轮收据](../results/dpvo_stream_v30.json)。

## 33. 数据集时戳、软件入帧和位姿输出必须分开计时

**问题与办法：** 图 slot、源观测编号、数据集时间、软件入帧和输出时刻分别记录。fresh 仅表示本次准入观测通过 Update/BA；初始化等待、运动拒绝和 EOF 相对链重建不增加 fresh 计数。单目位姿仍为任意尺度，数值更新成功不是精度验收。

**实际计时：** 192 张图的原时间跨度只有 **6.368 s**，无 pacing 的板端回放却耗时 **699.765 s**，179 条新位姿平均 **0.256 次／实际秒**。首次新位姿在回放开始后 14.053 s；新观测处理时间中位数 **3.758 s**、P95 **6.655 s**，含解码、undistort/crop、encoder、patch、缓存和图更新。首三帧还包含原参考检查；加载、哈希／报告 I/O 不在逐帧时间内。EOF 另执行 **71.088 s**，整个 driver **773.075 s**。进程峰值 RSS **1,369.723 MiB** 来自 16 GB 板，不是整板峰值或 8 GB 验收。温度／频率未受控，短测不是持续吞吐证明。

**轨迹诊断：** 事后以固定全源 1 秒端点计划和 20 ms 最近 GT 关联评估。179 条 fresh 位姿在一次正 Sim3 对齐后 ATE RMSE **1.085 cm**；旋转 RPE RMSE **0.969°**，使用 162 个预定端点对中的 149 个。EOF 优化并重建的 192 条轨迹另作一次对齐，ATE RMSE **1.066 cm**，不计新位姿。GT 尺度对齐不表示在线已知米制尺度；短前缀没有独立原 CUDA／完整 VGGT 基线，不能证明相对精度损失 ≤10–20%。

**状态：** 全部真实推理在 RK3588S，未接相机、未用 NPU、未生成稠密地图；实时目标未达成。[完整时间与状态](../results/dpvo_stream_v30.json)、[fresh／EOF 分列诊断](../results/dpvo_trajectory_v30.json)。

## 34. 来源清单中的“不存在”也是一种有效记录

**现象与根因：** 首轮入口在模型加载前因 `KeyError('sha256')` 停止，完成帧数为零。固定来源清单有 25 行，其中 23 行是已获取文件，另 2 行明确记录 `available_in_pinned_tree=false`；后者本来没有 SHA。遍历时假设每行都有哈希，导致启动失败。先前六项接口预检未覆盖这个完整遍历，不能据其通过认定能启动。

**修复与证据：** 只对显式为 false 的记录核“无 SHA 且本地文件不存在”后跳过，其余项继续严格核 SHA；原 loader 所需文件及生命周期额外依赖单独固定。23 个可用文件的 bytes/SHA/Git blob 检查通过，第二轮实际完成 192 帧。原失败收据与 [失败版本源码](../deploy/rk3588/validation/dpvo_stream/startup-r1/run-v30-dpvo-stream-board.py)保留，修复前后版本可追溯。

**边界：** 清单 schema 兼容与模型数值正确是不同检查。本修复没有改模型、权重或算法，也不允许默默跳过未知缺项。[预检范围、漏检说明与修复收据](../results/dpvo_stream_v30.json)。


## 35. 图规模与温度一起变化，不能只用首尾帧推断性能根因

**观测：** 本轮读到的 thermal zone 温度从开始的 **48.076–49.000°C** 变为结束后的 **80.384–82.230°C**，同时活跃图从初始化 6,144 边增长到 23,520 边。两个 encoder 的 192 次 CPU ORT 总时间 89.495 s，末帧 ORT 为 0.667 s；整帧也随运行明显变慢。

**结论边界：** 图增长和运行状态均发生变化，不能把全部变慢归于边数，或凭两次温度／瞬时频率采样断言热降频是唯一原因。结束时的 CPU frequency 是一次快照，不等于模型执行期间的频率；目前没有连续频率轨迹，也没有相同图状态、相同热条件的对照。

**处理：** 保留只读起止温度／频率样本，记录不曾改 governor、频率或散热配置；组件计时和完整逐帧时间一起公开。后续同状态优化对照须分开检查图容量、模型执行和温度／调度变化，再判断收益；本次结果不能作为硬件计算下界或公平加速比。[原始平台样本与完整计时](../results/dpvo_stream_v30.json)。

## 36. 动态 shape 必须检查实际图和实际数值，不能只看导出参数

**问题与办法：** `dynamic_axes` 是导出请求，不能证明图内没有冻结 trace 时的 E、组数或组宽。逐一核实际 13 个输入及 3 个输出的 E/Gkk/Lkk/Gij/Lij 符号维度，保留固定 B=1、通道 384 和相关性维度 882；继续追踪 Reshape、ConstantOfShape 的 shape 来源，并在同一模型上改变形状和拓扑值做数值检查。

**静态证据：** 本轮原始图的两个 Reshape 均取待变形张量自身的运行时 Shape，六个 ConstantOfShape 也使用运行时 Shape；未发现冻结 E=96 的整数常量。50 个参数 initializer 与原静态导出的值及使用位置一致，末端稳定 Sigmoid 改写以外的节点和 I/O 不变。这些是图结构及参数证据，不是动态数值验收。

**实际失败与边界：** 板端 r1 的 13 次 E=96 调用和第一次 E=6,144 调用通过原参考门槛，第二次 E=6,144 调用的 net 与 delta 各有 1 个元素超限，入口停止，未发布 fresh 位姿。仍使用 rtol=1e-4 / atol=1e-5，未筛除元素或放宽门槛。差异根因尚未定位，不能直接归因于符号维度、分组归约或某个 kernel；CPU ORT 能执行该形状也不代表全部动态状态通过，更不代表 RKNN 动态形状支持。

**依据：** [本轮导出、静态审计和失败记录](../results/dpvo_dynamic_update_v31.json)、[导出入口](../deploy/rk3588/validation/dpvo_dynamic_update/run-v31-dpvo-update-export-board.py)。原始导出 manifest SHA `4cd7b17821ac0dd0b1b9fad33c85ee98b50b40097b8a9c4a3dd707405a6725a0`；r1 失败收据 SHA `594672ac5af2ee5ad923e7e7c869e7d2528c7224dde288fa88490fb6874ef512`。

## 37. 快速拓扑构建须保留排序、碰撞、所有权和容量

**问题与办法：** 用稳定排序替代逐组扫描时，邻居仍须在同一 kk 内按 jj 稳定排序，相同 jj 保留原边顺序；分组成员也须保留原边行顺序。ij 分组继续使用原 int64 的 `ii*12345+jj`，包括它的溢出与碰撞语义，不能悄悄改成另一种分组规则。padding、空组、inverse 和缺邻居 mask 仍由同一真实拓扑产生，不能截边来适配容量。

**所有权与边界：** 缓存 key 核完整 ii/jj/kk 值及容量，不能只核指针、shape 或哈希；调用者在调用期间须保持输入稳定。缓存私有模板与返回数组分开拥有存储，返回值被改写不能污染下一次命中。默认最多保留 2 项、缓存数组 payload 上限 32 MiB；每次两种分组表合计另受 32 MiB 分配前检查，超界拒绝。payload 计数不是 RSS，未包含 Python 记账、临时数组和调用者副本。

**证据与状态：** 45 项小型合成检查覆盖与冻结原构建器的逐值对照、稳定 ties、int64 碰撞、非连续输入、返回值篡改及容量拒绝。r1 实际 15 次请求包含 7 次命中和 8 次构建，关闭后保留项和 payload 均为零。所有权及整数拓扑检查通过不能抵消第 36 条的模型数值失败；拓扑和拷贝成本应计入 Update 对照，不能只报告 session.run 时间。

**依据：** [拓扑实现](../deploy/rk3588/validation/dpvo_dynamic_update/v31_update_topology.py)、[本轮合成检查与实际生命周期](../results/dpvo_dynamic_update_v31.json)。

## 38. 逐元素一致性、权重值域和轨迹精度是三种不同证据

**问题与办法：** 对 net/delta/weight 分别保留固定逐元素容差、超限数量及 bitwise 结果；另检查所有输出 finite 和 weight∈[0,1]。小 relative L2、有限输出或合法权重不能替代逐元素门槛；容差允许的微小负权重也不能直接进入 BA。不得裁剪数组后把原始输出标成通过。

**本轮证据：** 动态 r1 的失败调用仍有有限、域内的 weight，却未通过 net/delta 原门槛。大图对照的两个状态也都 finite 且权重域合法，但其中一个状态被逐元素门槛拒绝。来源、参数未变和清理成功只证明对应身份与生命周期检查，不应合成“模型已验收”的结论。当前差异原因未确认，保留失败样本继续定位。

**精度边界：** 同一当前输入上与原 Torch Update 对照，只能检查这个组件的计算；若此前已有候选输出反馈进 net 或几何，原参考仍读取同一候选状态，并不是一条独立原实现轨迹。完整轨迹须另核初始化、每帧准入、反馈连续性及固定 GT 评估计划。r1 未产生 fresh 位姿，大图组件对照没有执行 BA；本轮不能据此报告轨迹精度、实时吞吐或 20 Hz 验收。

**依据：** [本轮分列的数值、权重域和运行状态](../results/dpvo_dynamic_update_v31.json)；历史微负权重问题与末端修正见 [Update CPU 对照](../results/dpvo_update_cpu_v29.json)。独立保存数组复算确认 r1 两个坏元素的误差/容限比分别约为 1.00472（net）和 1.11464（delta）；该调用的 relative L2 分别约为 5.45e-7 和 1.29e-6。误差很小仍保留逐元素拒绝结论，不把它直接解释成轨迹失败，也不据此声称轨迹已通过。主机未重新运行模型。

## 39. 同状态重建和交错对照须为下一步计算重新建立金标

**问题与办法：** snapshot 已保存的是上次更新后的 net、pose/depth 和历史 target/weight。下一步须按保留帧的真实身份和 patch 中心重建 context，用快照当前几何重投影并计算 corr，再让原 Torch 和候选读取同一份 net/context/corr。历史 target/weight 不是这一步的金标；重建时不得用新 Patchifier 的默认深度覆盖快照优化后的深度。

**实际对照：** 两个 E=23,520 冻结状态分别在同一进程执行 AB/BA/AB（A=原 Torch，B=动态 CPU ORT），每种方法各 3 次，每次重置为同一状态 net。首调用与后两次 warm 分列；warm Update 总时间均值分别为 Torch 2.538/2.931 s、ORT 1.842/2.237 s，组件时长减少约 27.4%/23.7%。计时含本方法的输入转换、执行和输出拷贝，ORT 另含拓扑/缓存及检查；图像/特征重建、重投影、相关性、context gather 和文件 I/O 不包含在这个均值内。

**验收与限制：** 终止前状态在三次比较中均有 net 1 个、delta 2 个超限元素；最终快照状态全部通过原逐元素门槛，因此整体仍为 `completed_numerical_rejected`。两个状态的原始特征未在 v30 保存，虽核原图来源、中心和重建流程，不能认证重建特征与旧运行逐字节一致。交错顺序减少固定先后偏差，但未控温控频；两个固定状态不是递归新帧流，也没有 BA 或轨迹验证，不能外推整机收益、NPU 或 20 Hz。

**依据：** [大图同状态对照及重建边界](../results/dpvo_dynamic_update_v31.json)、[对照入口](../deploy/rk3588/validation/dpvo_dynamic_update/run-v31-large-update-board.py)。实际收据 SHA `25ed1a743bbda78b8fa8eeefb0ed4827ca1c3d2fdf978c0c1e2a294e97ef3520`。

## 40. 被数值门限拒绝的图，可以隔离诊断，但不能因更快而自动验收

**问题与办法：** v26 的五个大 CPU Transpose 位于 fnet InstanceNorm 展开链；v27 将十个归一化改写为 NCHW 空间均值、中心平方的总体方差与原 epsilon/affine，但板端 CPU ORT 未通过原 `rtol=1e-4, atol=1e-5`。v32 新建明确的拒绝候选研究入口，精确绑定原图、改写和失败收据，再以原 RKNN FP16/opt3/nonquant 配置编译；历史失败、`candidate_accepted=false` 和默认 VGGT 均保留。原权重、其余节点、608×464 BGR 输入和两个输出的 `/4` 不变，没有裁剪输出或放宽门限。

**实际证据：** 同进程首帧 AB/BA/AB，A 为原图、B 为空间归约图；每种方法各三次，其中后两次 warm 的完整 SDK 调用均值为 **419.830 / 207.028 ms**，短测时长减少约 **50.7%**。另外两帧各执行 AB 或 BA。独立候选 profile 的三次调用均为 **159 个内部 NPU 算子、3 个 CPU I/O 算子**，没有内部 CPU Transpose；历史原图的五个 CPU Transpose 共 216.530–219.718 ms。原 profile 是历史数据，不能视为同期硬件控制；当前收益由上述普通交错调用报告。

**验收与边界：** 所有保存输出有限、非零且完成清理，但原 FP32 门限仍失败。首帧 fmap 相对 L2 从约 **0.3802%** 变为 **0.2279%**，最大绝对误差却从 **0.006341** 增至 **0.006430**；指标改善不能抹去坏元素或当作轨迹精度提升。imap 在两种 RKNN 图之间逐字节相同，也不等于它通过原 CPU 门限。独立保存数组审计的 **997 项**通过证明收据一致，不能改写候选拒绝状态。每种方法只有两次 warm、未控温控频；这是 encoder 组件研究，没有 correlation、Update、BA、新位姿或完整 VO，不能称为 20 Hz 或原 VGGT 加速验收。

**依据：** [原编码器链](../results/dpvo_encoders_v26.json)、[保留的空间归约拒绝](../results/dpvo_norm_rewrite_v27.json)。本轮独审 `independent-review.json` SHA `0b642f4fd6db963cc81643ad6cb9a8bb182c34ce7b9f8f9f7604e70d5401ae3b`；普通调用收据 SHA `2e71afe310413149c519f21062414c09e97115118ebf362905ecf5f3c5c2fc67`。完整 [v32 实测与独审证据](../results/dpvo_encoder_layout_v32.json)。

## 41. 算子 profile 的 I/O 行，不代表完整 SDK 输出获取成本

**问题与证据：** 候选 profile 的全部 CPU I/O 行合计只有 **0.757–0.810 ms**，而同批 profile 的实际 `rknn_outputs_get` 为 **64.897–65.274 ms**。普通调用两次 warm 中，getter 均值为 **62.645 ms**，随后自有 FP32 数组复制另需 **12.217 ms**；SDK run 自身为 **121.743 ms**，完整调用为 **207.028 ms**。profile 行不是 API 墙钟时间；不能把 getter 的差值直接归因于一个 kernel、纯带宽或某次格式转换。

**办法：** 同时报告输入 native-layout packing、inputs_set、run、outputs_get、自有数组复制和 release；输出在 release 前完成拷贝。SDK 每次取回两个完整 FP32 feature，共 **36,110,336 字节**，其中 imap 占 **27,082,752 字节**。上下文初始化、误差计算、哈希和文件写入另列；预处理也不在本次组件计时内。profile 独立进程执行，不把其算子行时间加减成普通调用的实测加速或端到端时延。

**边界：** 159 个内部算子全部在 NPU，只说明这个图的实际 placement，不表示输出搬运消失、数值已验收或完整神经 VO 在 NPU。编码器短测仍约 207 ms，不能用其倒数或低频地图更新冒充 20 Hz 新观测位姿；完整处理还要计入图像预处理、patch、correlation、Update 和 BA。

**依据：** 本轮普通收据与独审同第 40 条；独立 profile 收据 SHA `40328351d7c10451b05c1a51609c5353fc7dc4d3a1c60d31ca7eef7b7fb112cf`。[原编码器完整 I/O 口径](../results/dpvo_encoders_v26.json)。

## 42. 恒真 selection 的保存数组恒等式，可以区分严重故障与微小舍入

**现象与证据：** v29 插桩 Update 的两个分组 mask 实际全为 1，因此 `Where(mask, gathered, inactive)` 应逐值返回 gathered。独立核对 SDK／CPU ORT 的两个 NPZ 与 38 个数组哈希后，CPU ORT 两处恒等式均逐字节成立。SDK 第一处 selection 却有 **331,735／331,776** 个值不同，误差最大 **19.84765625**，relative L2 为 **1.44346**；第二处 selection 全零，与对应 Gather 的全部值不同。

**判断与办法：** 第一处 Gather 相对 CPU ORT 的误差为 **0.13483%**，其输入已经包含前级误差；仅凭它不能断言外部 INT64 拓扑索引错误。随后 selection 的巨大跃变及指数出现 Inf，不能由“普通微小 FP16 舍入”概括。下一步先用可精确表示的行号／通道标签隔离 Gather 和 selection，再对同一已保存 logits 比较无 padding 的稳定 SoftAgg；不以最终输出的小误差统计掩盖值域或恒等式失败。

**边界与状态：** 这些事实只定位了**插桩图**的异常区间。额外输出可能改变 fusion、布局和缓冲区寿命，尚未证明原未插桩图的第一个错误内核。有限 padding 单独仍失败；乘加 selection 虽恢复 finite，仍被原 FP32 门槛拒绝。不能写成修复完成或已接受的 NPU 推理。

**依据：** [v29 保存数组、改写和实际 SDK profile](../results/dpvo_update_rknn_v29.json)；本轮复算与后续实验合同见 [源帧分块研究合同](../results/dpvo_update_plan_v32.json)。

## 43. 完整源帧分块能关闭依赖，但不会消除主要矩阵计算

**问题与闭合条件：** 任意 E 行、单目标帧对或部分 patch 分块，会切断原 Update 的邻居或 SoftAgg 分组。当前有界图满足 `ii=kk//96`、`ii,jj<64`；按一个源帧保留全部 96 个 patch 及全部目标边，邻居、kk 分组和 `ii*12345+jj` 分组才都留在块内。对 v30 保存的 **23,520 边**检查得到 17 个源块、每源 10–17 个目标；245 个源目标帧对均恰有完整 96 个 patch，无重复或缺失。分块前后覆盖同一边集，不删边。

**下一实现：** 按 `(jj,kk%96)` 排成 `[1,D,96,C]`，kk 聚合沿目标轴、ij 聚合沿 patch 轴，完整组不需要 padding／分组 Gather／Where。c1 的缺前邻居使用零输入，仍计算其 bias；c2 必须读取已经 c1 更新的状态，缺后邻居同理。结果逆置换回原边序，全部块校验成功后才提交 net 和进入 BA。纯 NumPy plan 的 11 项小型测试通过，但尚未执行新的神经图。

**预算与边界：** 原 Linear 静态计数为 `2699520*E + 147456*(Gkk+Gij)` MAC；该保存图为 **63.7695 GMAC**，所有源块的合计完全相同。若仅 Update 独占 50 ms，所需有效浮点运算率约 **2.55078 TFLOP/s**，还未计归一化、激活、归约、编码器、相关性、BA 和 I/O。这是工作量预算，不是 RK3588S 性能实测或绝对不可能性证明。分块可能减少泛用拓扑开销，也可能因小矩阵和额外调用变慢；改变 GEMM 形状与归约顺序还可能改变舍入，数学闭合不保证原 FP32 门槛通过，更不证明 20 Hz。

**状态与依据：** 只完成保存整数拓扑验证、layout helper 和小型合成测试；原权重与默认 VGGT 未改，未新编译或推理。[实际保存图来源](../results/dpvo_stream_v30.json)、[原 Update 与大图组件对照](../results/dpvo_dynamic_update_v31.json)、[分块合同及计划](../results/dpvo_update_plan_v32.json)。

第 43 条记录 v32 的静态阶段；其后实际 CPU 与 RKNN 验证见第 44 条。

## 44. 完整依赖分块的 CPU 一致性与 NPU 数值验收必须分别验证

**办法与实际证据：** 将原 Update 的完整源帧边排成 `[1,D,96,C]`，保留全部邻居和两类 SoftAgg 分组。对两个冻结 E=23,520 状态，各选预定最小度源块（源 0、10 个目标、960 条边），重新执行完整原 Torch 作为同输入参考。50 个 Update 参数保留，c2 读取 c1 更新后的状态。原图、分块 Torch、分块 CPU ORT 的 **18 组**输出比较全部通过固定 `rtol=1e-4, atol=1e-5`；独立保存数组审计完成 1,753 项检查。导出的 108 节点图没有分组 Gather、Where 或 padding。

**RKNN 实测：** 相同图编译为 FP16 后，板端完成 6 次普通调用和 2 次独立 profile；所有输出有限、权重在 [0,1]，同状态重复调用原始哈希一致，但仍未通过原 FP32 门限。两个状态的 delta relative L2 分别约 **0.319% / 0.357%**；不能凭小误差或有限输出直接采用。普通调用每状态两次 warm 均值为 **68.292 / 87.103 ms**，包括输入重排、set、run、get、自有输出复制及 release。

**边界：** 每次只处理 17 个源块中的一个，不能与完整 23,520 边 Update 的秒级耗时直接计算加速比。CPU 计时还是首调用，也不是同期 warm AB。独立 profile 每次有 **59 个内部 NPU、5 个内部 CPU、6 个 CPU I/O**；内部 CPU 为两处 ReduceMax、两处 Exp 和一处 Min。移除分组 Gather/Where 不等于全 NPU。未执行全图拼接、recurrent feedback、BA 或轨迹，没有实时验收。[实现与收据](../results/dpvo_source_frame_v33.json)。

## 45. SDK 内部 stride 大小不等于转换输出缓冲区的实际大小

**现象与根因：** 首次启动在任何 `rknn_run` 前被适配器拒绝。独立属性查询显示，net_out 的 FP16 逻辑大小为 **737,280 B**，内部 `size_with_stride` 为 **1,179,648 B**；delta/weight 各为 **3,840 / 24,576 B**。旧适配器要求两者相等，误把内部对齐空间当作转换后返回数组的布局。三个输入内部大小与逻辑大小相同，本次无需改变输入路径。

**修复与验证：** 保留原编译产物和失败入口，只新增输出适配器版本。仅在 `want_float=1, is_prealloc=0` 路径允许输出内部 footprint 大于逻辑大小；实际 `outputs_get` 返回仍必须为 FP32 的 **4×元素数**字节、非空指针，并在 release 前拷贝到自有连续数组。输入、shape、类型、量化及实际返回字节数检查保留。两个状态的 8 次实际调用均取得正确长度、形状和有限输出，清理成功；fake ABI 还覆盖 release 后毒化、过小 stride、错误返回大小与输入 padding 拒绝。

**边界：** SDK 头文件只说明内部 stride 和 `want_float` 的含义，不单凭这两项推定输出一定 packed。内部原生或 zero-copy 缓冲区需要单独的 stride 读取实现；本修复不授权直接把它们 reshape。编译收据继续绑定原 helper，新 runtime 另记哈希，避免将事后修复误写成实际编译来源。r1 失败收据及原源码保留，r2 的模型数值拒绝也保留。[属性、失败和修复证据](../results/dpvo_source_frame_v33.json)。

## 46. 原 FAST 配置的工作量预算需要写明图增长假设

**源码与预算：** 固定原 DPVO FAST 覆盖配置将每帧 patch 从 96 调为 48，patch lifetime 从 13 调为 11，removal window 从 22 调为 16，optimization window 从 10 调为 7。在“每帧都准入、没有几何关键帧删除”的明确条件下，原图生命周期递推得到初始化 **3,072 边**，仍需 12 次 Update、每次两步 BA；稳定窗口更新前 **14,496 边**，对应 Linear **39.2971 GMAC**。同条件默认配置为 47,712 边、129.1984 GMAC，FAST 约为其 30.4%。

**边界：** 这不是 v30 真实运行最终 23,520 边的直接比例，也不是板端加速实测；实际运动准入和几何删帧会改变图。22 个 encoder Conv 的原工作量 **11.0633 GMAC/帧**不因减少 patch 而消失。更小的图也需要独立的完整轨迹精度和失败率验证，不能因配置来自原仓库就认定满足实时或精度目标。本轮只记录源码和整数拓扑预算，未将 FAST 接入默认 VGGT 或执行新的 VO。[预算及前提](../results/dpvo_fast_budget_v33.json)。

## 47. 稀疏返回上下文可减少传输合同，但不会自动减少编码器计算

**办法：** 原 encoder 返回完整 fmap 与 imap，共 **36,110,336 B** FP32。保留全部原网络节点和参数，在 imap 的原 `/4` 后按 `y*152+x` 精确抽取 96 个整数中心，仅将上下文返回为 `[1,96,384]`；完整 fmap 仍返回。新的逻辑输出总量为 **9,175,040 B**，其中 context 为 147,456 B。内部整张 imap 仍计算，不能把输出字节减少写成 CNN MAC 或耗时减少。

**实际验证：** 在板端 CPU ORT DisableAll、4 个 A76 线程上，对原图和空间归约图分别执行 3 张真实图像的父图／稀疏图配对。预声明中心覆盖重复、逆序和边界内点，不按结果挑点；六组 sparse fmap/context 与当场父图的完整 fmap／精确 context gather **逐字节相同**。原图仍通过原 CPU 门限；空间归约图的历史 FP32 拒绝仍存在，不能因后缀抽样正确而改写成模型已通过。

**接口与边界：** patch 索引必须保留顺序和重复，形状相同不代表观测身份相同。当前中心是固定诊断规则，尚未绑定原 Patchifier 的 RNG、随机深度消费顺序和生命周期。未编译这一新后缀到 RKNN，尚不知实际 Gather 的 placement、SDK 输出大小及完整 I/O 收益。CPU 小测也不是 NPU 加速证据；须在新图实测这些条件后再接入流水线。[改写和板端对照](../results/dpvo_sparse_context_v33.json)。

第 47 条记录 v33 的 CPU 阶段；后续 SDK 实测见第 48 条。

## 48. 少返回 75% 的数据，不等于完整 SDK 调用快 75%

**实验与证据：** 对原图及空间归约图分别复用历史已编译的完整 imap 父图，新编译稀疏 context 子图。保持同一真实图像与 INT64 索引，首帧同进程 AB/BA/AB，另外两帧交错配对；普通调用每种图 5 次，profile 另开进程各 3 次。输出从 **36,110,336 B** 降到 **9,175,040 B**，但首帧后两次 warm 完整 SDK 均值仅为原图 **427.916→411.549 ms**（减少约 3.8%），空间图 **217.184→178.058 ms**（约 18.0%）。未控温控频，两次 warm 不能当作稳定长时收益。

**分解与解释：** 原图 getter 从 45.418 降到 13.826 ms，自有输出复制从 15.440 降到 3.988 ms；SDK run 却从 356.571 增到 383.328 ms。空间图 getter 为 67.921→28.253 ms，复制 17.791→5.816 ms，run 为 121.075→133.501 ms。getter/复制确实减少，但内部整张特征仍计算，新后缀也有代价；不能仅靠输出字节推算整机加速。实际 profile 的 Gather 在 CPU：原图另保留五个 CPU Transpose，空间图有 162 个内部 NPU 算子、一个内部 CPU Gather 和四个 CPU I/O 行。

**接口与验收：** 查询实际索引为 INT64、非空间端口 fmt=3（UNDEFINED），不得误认成 fmt=2 的 NC1HWC2 或悄悄转换为浮点索引。实际 getter 仍逐次核 FP32 逻辑字节，并在 release 前拥有副本。三帧两种方法的 sparse fmap/context 均与相应 RKNN 父输出的选取结果 bitwise 相同，但两种方法仍不通过原 CPU FP32 门限。主机保存的是板端选取后的 canonical 数组，未保存完整 imap，独审不冒称重新从 full imap 计算了选取；profile 数组留在板端，主机只核其收据与 raw hash 一致性。没有把候选接入 VO。[完整收据与复现入口](../results/dpvo_sparse_context_v34.json)。

## 49. 保留原算子的表达形式，可能改变编译器 placement

**问题与办法：** v33 源帧分块把分组权重写成显式 max-shift、exp、sum 和除法，RKNN 出现两处 CPU ReduceMax。v34 仅把这一段改为 `torch.softmax`，分别沿目标轴和 patch 轴；c1/c2 顺序、全部 50 个 Update 参数和末端 stable weight 表达式均保留。两个冻结 D10/M96 源块在板端重新执行父块 Torch、新块 Torch 和 CPU ORT，66 组保存数组比较全部通过原 `rtol=1e-4, atol=1e-5`。完整 E=23,520 原参考来自 v33，本次未重新执行，不能混写成新的全图回归。

**实际 placement：** 新 ONNX 为 100 节点、两处 Softmax，无显式 ReduceMax/Gather/Where；RKNN 两次独立 profile 均为 **55 个内部 NPU、3 个内部 CPU、6 个 CPU I/O**，两处 CPU ReduceMax 行消失。剩余内部 CPU 为末端权重的两处 Exp 和一处 Min，另有十个 NPU Transpose。ONNX 没有 Transpose 不代表编译后没有布局转换，静态节点数量也不能替代实际 profile。

**边界：** 两个源块普通调用的后两次 warm 均值为 **61.839 / 67.621 ms**，但没有与 v33 做同期交错对照，不能据此报告受控加速比。所有 RKNN 输出有限、权重在 [0,1]，仍被原 FP32 门限拒绝。CPU 等价检查、placement 改善、整图新位姿速度各需自己的证据；本次未执行其他源块、反馈、BA 或轨迹。[源码、CPU 审计与板端收据](../results/dpvo_source_frame_softmax_v34.json)。

## 50. FAST 配置需要完整回放验证，随机种子相同也不代表采样相同

**实现与身份：** 在冻结原 DPVO `config.py` 上加载原 `fast.yaml`，使用 M=48、lifetime=11、removal=16、optimization=7；编码器 patch 提取、相关性与图后端三个位置须使用同一 M。保留原 CPU Torch 的 x、y、深度随机数消费顺序，即使深度随后由中位数覆盖也先消费该随机数。M=48 与 M=96 每帧消耗数量不同，不能因 seed 相同就说两次用了相同 patch 子集。CPU FP32、研究容量 64 和原 CUDA mixed precision/4096 容量的区别明确保留。

**实际结果：** 同一 192 张 TUM fr1_xyz 图像和时间戳完整回放，得到 **179 个 fresh 位姿**；总回放 **348.556 s**，fresh 吞吐 **0.514 Hz**，软件逐帧 fresh 平均 **1.919 s**、P95 **2.887 s**。原默认历史回放为 699.765 s、0.256 Hz、3.880 s、6.655 s。FAST 单独 EOF 优化 24.598 s，不能计入 fresh 数量；峰值进程 RSS 为 **1,070.461 MiB**。硬件为 16 GB 板，没有因此认证 8 GB 部署或 30 分钟稳定运行。

**精度与评估：** 两次输入 manifest 字节相同，先按 fresh 标记取共同 179 个观测，再关联 GT，每条轨迹各在同一共同集合上只拟合一次正尺度 Sim3。ATE RMSE 为默认 **1.0853 cm**、FAST **1.0714 cm**；固定全源 1 秒计划 162 对中有共同有效 149 对，旋转 RPE RMSE **0.96867→0.96821°**，平移 RPE RMSE **1.66276→1.63046 cm**，方向误差中位数 **3.036→3.152°**。EOF 的 192 个重建位姿另评，不混入在线样本。它只覆盖约 6.37 秒的视频，没有原 CUDA/full VGGT 轨迹基线；GT 只用于事后评估，不用于选配置。这是完整短回放进展，仍不满足实时目标，没有增加 ORB/LK/PnP，也未改变默认 VGGT。[实现、身份和评估收据](../results/dpvo_fast_v34.json)。

## 51. Channel-first 表达、物理数组布局和 NPU placement 要分别验证

**问题：** v34 的 ONNX 没有 Transpose，实际 RKNN profile 却有十个 NPU Transpose。编译器内部的布局转换不能靠查看 ONNX 节点数量判断。

**实现与对照：** 新实验将完整源块从 `[1,D,M,C]` 改为 `[1,C,D,M]`，用同一组原参数的 1×1 Conv 表达 Linear；LayerNorm 仍只对 C 轴计算原中心化方差，保留 epsilon/affine、c1→c2 顺序、边界 bias、D/M 两类 SoftAgg 和 stable weight。两份 D10/M96 状态的原块、父块、新块和板端 CPU ORT 共 84 项比较通过原 `rtol=1e-4, atol=1e-5`，保存数组独审通过 1,421 项检查。50 个 initializer 与原 kernel/bias/affine 的精确 reshape 一一对应；CF 与 canonical 副本逐字节反置换核等。

**实际 SDK 结果：** 新图编译后完成六次普通调用、两次独立 profile。两组各两次 warm 完整 SDK 调用均值为 **54.287 / 66.368 ms**。每份 profile 为 **50 个内部 NPU、3 个内部 CPU（两个 Exp、一个 Min）、6 个 CPU I/O**，仍有 **6 个 NPU Transpose**，其 profile 行计时合计约 7.5 ms。输入查询得到 native NHWC，须把 CF 逻辑输入真正打包为 NHWC；输出按 CF 获取自有 FP32 副本，并在 release 前核验实际逻辑字节。SDK 独审通过 **1,668 项**，8 次 release 与 2 个 context destroy 均有记录；四个保存输出的 normal/profile 代表结果字节一致。

**边界：** CPU 一致不能推导 NPU 一致：**本次 RKNN 仍拒绝原 FP32 门槛**，但输出有限、weight 在合法范围。内部 Transpose 比历史 profile 少，并不等于已证明同期受控加速；没有执行新旧图交错 AB。单次只覆盖 960 条源边，既不是完整 Update，也不是 M48 轨迹所用的 Update。完整 E=23,520 原 gold 为历史参考，本轮未重跑；默认 VGGT 未改，候选未采用。[源码、原始收据和两级独审](../results/dpvo_source_frame_channel_first_v35.json)。
## 52. 完整 fresh 路径的计时必须区分外层、嵌套项和未测阶段

**口径与分母：** 重新读取 v34 原 FAST 完整 192 帧 CPU 回放的保存 JSON，按实际 fresh 标记选取 179 帧，没有重新执行图像、模型、相关性或 BA。逐帧外层恒等式为 `processing=decode/去畸变/crop+encoder外层+stream`；将该帧的 Update 和 correlation 回调总量从 stream 中扣除，余项只称 **stream 其余工作**。首个 fresh 帧含 12 次初始化 Update，保留在分母内；去掉这一帧后的 178 帧均值为 1.9017 s，并没有改变结论。

**实际路径：** 179 个 fresh 的软件处理均值 **1.9192 s**、P95 **2.8867 s**。下表均值是可相加的独立外层阶段；各项 P95 不能相加。

| 独立阶段 | fresh 均值（ms） | fresh P95（ms） | 均值占比 |
| --- | ---: | ---: | ---: |
| Update 原回调 | 933.677 | 1541.285 | 48.65% |
| encoder 外层 | 446.590 | 661.674 | 23.27% |
| correlation 外层 | 314.206 | 466.069 | 16.37% |
| stream 其余工作 | 203.229 | 281.480 | 10.59% |
| decode/去畸变/crop | 21.494 | 23.156 | 1.12% |

**嵌套边界：** Update 计时包含输入 finite 扫描、Torch view、原网络 forward、自有 NumPy 输出副本和 finite/weight 域检查，不是纯神经 kernel 时间。correlation 外层已包括 cache 关闭/重建、特征拼接、平均池化与布局准备、索引映射、native query 和外层 finite 检查；其中 cache prepare 均值 **102.682 ms**、native query **144.040 ms** 都是其子集，不能再次加到外层上。encoder 的逐 fresh ORT 分项没有保存，只有全部 192 帧累计 ORT **80.6144 s**、归一化 **0.4267 s**、采样 **2.6185 s**，不得把全源累计除以 179 冒充 fresh ORT 均值。stream 其余项含 BA、重投影、context gather、图状态复制、flow/删帧和检查；**旧收据没有独立 BA 时钟，203.229 ms 不能写成 BA 耗时**。

**目标差距与优先级：** 与串行 20 Hz 的 50 ms 服务预算相比，当前 fresh 均值需要约 **38.38 倍**缩减；与 P95≤100 ms 相比约 **28.87 倍**。这两个预算不等于物理相机时延验收。保持这条保存 trace 的其他阶段成本不变，将 encoder 与 Update 同时置零仍有均值 **538.929 ms**，再将 correlation 置零仍为 **224.723 ms**。这是固定 trace 的反事实成本拆分，不能解释成硬件理论下界或实测优化结果。单独更换 encoder 不足以证明整机目标；下一步须同时保留 encoder 完整 I/O、全图 Update、cache/相关性以及独立 BA 的计时，不能只用某个组件耗时倒数作 fresh 吞吐。

**源帧合同与限制：** 纯整数重放保存的准入/删除事件，六个 snapshot ii/jj/kk 原始数组 SHA 与归档 metadata 核等。M=48 的所有实际 Update 源块均完整覆盖目标×48 patch，观察目标 degree 为 **8–17**；最后在线 Update 前为 **10,128 边、17 源块**，删帧后的最终快照为 **9,744 active 边、16 源块**，另有 384 inactive 边。后续分块必须保留完整邻居及两类 SoftAgg 分组、c1→c2 顺序与原边序还原，所有源块成功后才整图提交和进入 BA。整数身份验证没有重建或验证 net/corr/context 数值；历史 D10/M96 的单块 SDK 时间不能当成 M48 全图速度或精度证明。

**证据与边界：** [原 FAST 完整回放](../results/dpvo_fast_v34.json)。[保存计时与拓扑拆分](../results/dpvo_fast_rknn_v35.json)绑定原运行及分析源码的 SHA。历史运行不控温控频，arrival 是软件文件准入时刻，不是物理曝光或排队年龄；EOF 12 次离线 Update 与重建另列。DPVO 是另一模型的隔离研究，默认 VGGT 未变，没有 ORB/LK/PnP；本条只分析历史 CPU trace，NPU 回放见第 53 条；不声称 20 Hz、100 ms、整段轨迹或 30 分钟稳定性验收。
## 53. 编码器 NPU 接入必须放回完整 FAST 关键路径评价

**完整回放与门槛：** 在另一模型 DPVO 的原 FAST 配置下保持 M48/lifetime11/removal16/optimization7、原权重、CPU RNG 消费顺序、原 Update/相关性/BA，复用一个 v32 dense spatial RKNN 编码器 context 完成 192 帧，实际 SDK 调用 192 次；179 个 fresh 位姿、无清理错误。默认 VGGT 不变，没有 ORB、LK 或 PnP。该编码器历史严格 FP32 门槛已失败，本次前 3 帧完整 fmap/imap 的六份输出仍全部未通过原 FP32 数值门槛，保存数组独审确认该结果；未裁剪或放宽 `rtol=1e-4, atol=1e-5`，仅显式诊断参数允许继续，`candidate_accepted=false`。

**整个路径：** 无节拍文件回放耗时 **284.058 s**，EOF 12 次离线 Update/重建另计 **23.271 s**，峰值 RSS **922.297 MiB**。完整 replay 的 fresh 吞吐为 **179/284.058=0.63015 Hz**，输入吞吐 **192/284.058=0.67592 Hz**；不能用某个组件或均值的倒数当整条 fresh 吞吐。fresh 软件处理均值 **1.54928 s**、P95 **2.37991 s**，历史全 CPU FAST 为 1.91920/2.88668 s；这不是同进程控温控频基准，更不是物理曝光到位姿的时延。

| 本次可相加的独立阶段 | fresh 均值（ms） |
| --- | ---: |
| 原 Update 回调及输入/输出检查 | 819.311 |
| correlation 外层 | 290.006 |
| encoder 外层完整 I/O/检查/采样 | 234.491 |
| 原 BA 透明计时 | 96.917 |
| stream 其余工作 | 87.121 |
| decode/去畸变/crop | 21.432 |

上述阶段均值之和为 1549.278 ms；阶段 P95 不能相加。correlation 的 prepare 100.119 ms 和 query 124.008 ms 是嵌套子项，不能再加一次。215 次真实神经/相关性回调包含 **13 次 motion probe、190 次在线 Update、12 次 EOF Update**；motion probe 不调用 BA。202 次 BA 独立计时合计 **18.9325 s**，在线 17.3482 s、EOF 1.5842 s。新 BA 计时透明委托原调用；历史 CPU 收据没有该独立时钟，其 203.229 ms stream remainder 不能称为 BA 或与新 BA 做直接阶段回归。

**轨迹与分母：** 完整输入 manifest 字节相同，原 94 项参数指纹、共享 CPU RNG 初末指纹及 192 次深度随机抽样一致。两次 replay 的实际 common-fresh 都是源观测 13..191 共 **179** 个；固定全源 1 秒计划 162 对中 **149** 对可评价，最近 GT≤20 ms。评价各用一次正 Sim3，绝不作为先验或配置选择。共同 fresh 的 ATE RMSE 为 **CPU 0.01071409 m / NPU 0.01071398 m**，旋转 RPE RMS **0.9682089° / 0.9682303°**，缩放后平移 RPE RMS **0.01630460 m / 0.01630476 m**。EOF 192 位姿及重建另列，不计 fresh。短前缀的近似轨迹不能撤销特征门槛拒绝，也不证明原 CUDA、VGGT、长轨迹精度或 30 分钟稳定性。

**下一瓶颈与证据：** Update、相关性及其缓存准备仍占主要关键路径；单独完成 encoder 接入离 20 Hz/100 ms 目标仍远，不能自动采用。保存 JSON 独审 **2014 项/0 失败**，确认逐调用归属、嵌套计时、输入/源/RNG 指纹和清理；不重新执行图像、网络、SDK、ORT 或 BA。[完整运行、轨迹评价、各项审计与原始文件 SHA](../results/dpvo_fast_rknn_v35.json)。
## 54. 量化构建含模型前向，ARM 工具链也不能照搬 x86 依赖

**容易踩的坑：** `RKNN.build(do_quantization=True)` 看似只做转换，实际校准路径会执行模型。若约束为真实神经计算全部在 RK3588S 上，不能把这一步放到电脑后再声称板端全流程。非量化图编译与校准必须区分。

**依赖证据：** 固定 RKNN Toolkit2 2.3.2 的 ARM64 CPython 3.11 requirements 要求 Torch `>=1.13.1, <=2.2.0`；已有板端 Torch 2.3.1 不在该范围，不能因为 x86 编译环境使用 2.4.0 成功就沿用。已按官方 wheel/PyPI 元数据整理隔离环境的 URL、版本、大小和 SHA。固定 `onnxoptimizer==0.3.8` 在检查的来源中没有所需 aarch64 wheel；官方 sdist 自带 ONNX、protobuf/protoc 和 pybind11 源码，但仍需本地构建。源码检查发现其内置 pybind11 为 2.9.2，准备使用独立 2.11.1 配合 CPython 3.11，不改原归档；CMake、Python.h 和项目已有 GCC/sysroot 也须核对。

**处理与边界：** 准备脚本在实际安装/PTQ 入口先检查 Linux、aarch64 和 RK3588/S device tree；使用新 venv，不覆盖现有推理环境。下载与本地构建绑定文件身份，限制构建并发，保留失败产物；INT64 patch 索引不得为了通过校准改成 float。**本轮仅完成元数据、源码检查及合成/fake SDK 验证，未实际安装、量化或生成 INT8 模型。** 混合 FP32/INT64 校准输入是否被 SDK 正确保留、校准集是否代表场景、最终算子精度与轨迹性能仍待实测。SDK 完整 wheel 的 SHA256 也要在下载后补齐，当前只有固定 Git blob 与 metadata 身份。[准备源码与明确状态](../deploy/rk3588/validation/board_ptq/README.md)、[官方出处及依赖收据](../results/board_ptq_preparation_v35.json)。

## 55. sysroot 指向目标库，但编译器子工具仍需自己的运行库

**实际故障：** ARM64 隔离安装 r1 已装完 31 个固定 wheel，构建 `onnxoptimizer 0.3.8` 时却在 CMake 编译器 smoke 阶段失败。第一条有效错误来自 GCC 调起的 assembler：`aarch64-linux-gnu-as` 找不到 `libopcodes-2.38-system.so`。只验证 gcc/cc1plus 能运行，不能覆盖整个工具链。

**原因与修复：** `--sysroot` 指定目标头文件和链接库，`-B`、PATH 选择工具路径；它们不配置子工具 ELF 加载器的共享库搜索路径。保留 r1 日志与部分 venv，另建 r2 venv，仅在构建子进程的 `LD_LIBRARY_PATH` 前置项目 sysroot 的 `usr/lib/aarch64-linux-gnu`，保留原非空路径，避免引入表示当前目录的空项。原推理环境和全局库配置不变。三项测试覆盖缺失、空值和已有路径，其他环境键值与原 installer 相同。

**验证：** r2 的 37 条实际命令均成功，完成固定依赖、板端原生 optimizer wheel、官方 ARM Toolkit2 2.3.2 安装；metadata preflight 核对 33 个版本，`pip check` 通过。原生构建用时 **371.356 s**，日志确认 GNU 11.4.0、外部 pybind11 2.11.1 和内置 `protoc-3.20.1.0`。内置构建 ONNX 为 1.13.0，与 Python 环境 ONNX 1.16.1 分别记录。保存日志独审 **364 项、0 失败**；第一次审计误把带版本的 protoc 名称判为缺失，保留失败收据并修正匹配规则，未改原日志。安装 preflight 本身不验证共享库 import；随后独立的实际 PTQ 才验证了 SDK 校准构建路径，两个阶段不混为一个结论。[修正源码](../deploy/rk3588/validation/board_ptq/install-v36-arm-toolchain-board.py)、[失败、重试和安装证据](../results/board_ptq_v36.json)。

**传输补充：** 本板的旧 ADB shell 还会在约 4 KB 的完整启动参数处直接报 `shell command too long`；另有远程 Python 失败但 ADB 退出码为零的历史情况。完整回放改为生成并保存可哈希的 `launch-board.sh`、推送后短命令执行，同时核验板端 JSON 的最终状态与清理记录。shell 退出码和“文件已推送”都不代替任务成功；第一次过长命令没有执行模型，未计入回放样本。[传输收据](../results/dpvo_frame_cache_v36.json)。

## 56. 板端 PTQ 要绑定完整输入合同，编译成功不等于量化模型验收

**固定采样：** 原稀疏编码器同时接收 FP32 BGR 图像和 INT64 patch 索引，必须保持输入顺序、dtype、索引顺序与重复项。预声明从 TUM fr1_xyz 的 798 个条目取零基序号 `256+16*i`，共 32 张、末项 752；与回放的 0–191 帧不重叠，不读 GT、不按像素或模型输出来选图。两组仍来自同一序列，不能称独立场景。板端按原 K/D 去畸变、裁到 464×608、仅一次 `2*(x/255)-0.5`；保留 BGR，不另作 RGB 交换或缩放。

**随机数与身份：** 持久 CPU Generator seed 0 仅对选中样本依次消费 x、y、原随机深度；深度只做消费审计，不输入编码器。未选帧不消费随机数，所以这不是完整在线回放的 RNG 状态。保存 centers、RNG 接缝、每份文件/raw SHA 和有序 `dataset.txt`。32 张准备耗时 **2.589 s**、记录峰值 RSS **229.547 MiB**，完整产物 **109,036,861 B**。NumPy 独审 **1,787 项、0 失败**，核对采样身份、31 个状态接缝、160 个 sampling 行和可由 centers 推导的索引哈希。主机未读取 64 个 NPY 本体，也未重做预处理或抽样；图像本体身份以板端收据绑定。[固定 policy、生成器和独审](../deploy/rk3588/validation/board_calibration/README.md)。

**实际 PTQ：** 新隔离 ARM 环境在 RK3588S CPU 上完成 `RKNN.build(do_quantization=True)`，用的是原稀疏 context 图，未使用历史被拒绝的空间归约父图。32 行 FP32 图像/INT64 索引保持原合同，SDK 配置采用 identity mean/std、无 RGB 交换；校准与 build 用时 **60.291 s**，记录峰值 RSS **1,170 MiB**，导出 **12,491,378 B** RKNN，config/load/build/export 均返回零，release 完成。模型 SHA 为 `6b11d6643a25a4bf9de1f1f2f0ecc607e20d7c0755e9262c91874509b18e9e42`，模型仅留在板端，未公开权重或二进制。

**验收边界：** INT8 请求、成功编译、实际部署 I/O、内部算子精度、原 CPU 数值门槛、轨迹与完整耗时是不同证据。校准会执行模型，本次实际校准仅在板端；不能用主机“转换”名义规避这一约束。编译收据不证明量化精度、全 INT8 placement 或实时 SLAM，也不表示替换默认 VGGT。后续 I/O 查询与部署推理须另留收据。[安装、校准、PTQ 原始证据和来源 SHA](../results/board_ptq_v36.json)。

**实际接口查询：** 随后独立完成 init→query→destroy，未调用推理。新模型的图像输入确为 NHWC INT8，两个输出也为 INT8；patch 索引仍是 INT64、UNDEFINED 格式、scale=1/zp=0。fmap 的逻辑原生大小为 2,256,896 B，带 stride 存储为 2,375,680 B，后续 `want_float=1` 输出应另核 FP32 逻辑大小，不能照搬原生存储长度。接口查询不证明内部全 INT8 placement 或数值正确。生成收据到实际 PTQ 的来源独审另有 **670 项通过**，逐项绑定 32 行、64 个文件的 SHA、大小、顺序及 index min/max；未把 metadata 核对说成重新读取了所有 NPY。

## 57. 不可变帧缓存能避免历史特征重复准备，但整机收益仍受其他阶段限制

**根因与修复：** 原 correlation 缓存以整组 source/target frame ID 及顺序为 key。新增、重排或删除帧会重建整个 context，重复拼接历史 gmap、转换 fmap 布局和计算 level-4 平均池化。新实现按稳定 frame ID 与不可变数组对象保存独立 native 快照，查询时只组合当前顺序的 handle 表；新增帧准备、删除帧释放，原 `kk/M`、`kk%M`、`jj` 映射及原边序不变。沿用同一 FP32 点积、双线性、零边界和池化实现，编译保持 `-fno-fast-math -ffp-contract=off`。query、驱逐、reset、close 共用锁，返回数组有独立 ownership，序列切换必须 reset；输入及其别名在保留期间不可修改。

**组件证据：** 板端三份真实 M96 特征做首次两帧、增帧、重排、删中间帧四个场景，每场 AB/BA/AB/BA，共 **32 次调用**。新旧八个代表输出逐字节一致，相对原 CPU 参考全元素通过 `rtol=1e-4, atol=1e-5`；最大绝对差 **2.8610e-6**。首次 callback 的旧→新耗时分别为 **13.238→12.706、46.566→32.177、62.747→42.508、35.051→19.803 ms**。累计只准备三帧，场景间新增准备为 2/1/0/0；最后三个 handle 全部释放。NumPy 独审 **1,337 项通过**，重读 16 个 NPZ 和八个代表输出；其余 24 次重复核对保存的完整数组 hash，未冒充另有 24 份 raw 数组。

**完整路径：** 另一次普通 M48 FAST 回放只替换缓存，保留原 Update、BA、RNG 和历史诊断 NPU 编码器。192 张输入产生 **179 个 fresh 位姿**，回放 **269.215 s（0.665 Hz）**；fresh 软件处理 P95 **2.170 s**，EOF 工作另计 **22.809 s**。历史 v35 为 284.058 s、0.630 Hz、P95 2.380 s。全部 192 帧中除本次时钟外的位姿、准入/删帧和 BA 诊断，以及 EOF 重建结果精确一致；source/input manifest、RNG、94 项原参数和记录的状态数组哈希保留一致。保存 JSON 独审 **1,607 项通过**，没有重读本次完整 feature/snapshot 数组或重新跑 GT，所以本条不新增原 VGGT 精度结论。

| 在线累计阶段，包含初始化与 motion probe | v35（s） | 新帧缓存（s） |
| --- | ---: | ---: |
| correlation callback 内部总量 | 52.136 | 26.794 |
| 其中 preparation | 18.076 | 1.791 |
| 其中 native query | 22.244 | 24.893 |
| 原 Update 外层 | 147.293 | 155.975 |
| 原 BA | 17.348 | 17.591 |

preparation/query 是 correlation 的嵌套项，不再次相加。两次不是同期交错且未控温控频：历史特征准备的工作确实减少，但总回放差值不能全部归因于缓存，Update 在新运行中反而更慢。新缓存累计准备 **192 帧**、命中 **2,365 次**、驱逐 175 帧，最终销毁 192 个 handle、保留零字节；峰值 native payload **166,820,864 B**，总进程 RSS **949.781 MiB**，历史 RSS 为 922.297 MiB。缓存上限只涵盖自有 native payload，不包含调用方 feature、Python metadata 和 query 输出；复用并不免费。

**下一步边界：** 主要耗时仍在完整 Update 与相关性查询。该短序列、16 GB 板端回放没有物理相机队列、30 分钟、稠密地图或 8 GB 验收；0.665 Hz 也未达到实时要求。历史编码器的原 FP32 拒绝继续保留，默认 VGGT 不变，不增加 ORB/LK/PnP。[缓存源码及复现合同](../deploy/rk3588/validation/dpvo_frame_cache/README.md)、[实际构建、AB、完整回放和独审](../results/dpvo_frame_cache_v36.json)。

## 58. 完整图分组验证需要同一次原调用的输入和输出，不能拿单块代替整图

**问题：** 历史 D10/M96 单源块 SDK 耗时约几十毫秒，但 FAST 的 M=48 完整图在增长、删帧后有不同源帧数、邻居数和边序。乘一个估计块数或复用 B=1 的端口假设，无法证明完整图速度、精度或生命周期正确。分组还必须保留 c1→c2 的状态依赖和两类 SoftAgg 的完整归约范围。

**实际捕获：** 在原 v35 完整 192 帧诊断入口上只增加观测记录，预声明取源观测 **13、127、191** 的第一个完整在线 Update，在 BA 前保存真实输入与原输出。原 callback 每次恰好执行一次，输出原样返回；motion probe 和 EOF 不能补齐缺失样本。实际共 215 次原调用，三份图分别为 **3,072 / 7,248 / 10,128 边**、8 / 13 / 17 个保留源帧；六个输入/gold NPZ 共 **170,193,896 B**，保留在实验存储，不公开特征数组。完整回放中的哈希、拷贝和存盘开销单列，捕获运行的耗时不作为普通性能结果。

**可复用合同：** 每个源帧保留全部 D 个目标与 48 个 patch；只把相同 D 的完整源帧组装成 `[B,C,D,48]`，不补零、不删邻居、不跨源归约。实际三图分别产生 **1 / 5 / 10 个桶**，初始化图可以是真实 B8/D8/M48。每桶的输出先暂存，确认快照身份未变、所有桶成功后才按原边序整体还原；失败时不提交部分 recurrent 状态。geometry/features/graph 的 revision 是捕获器根据实际内容建立的序号与哈希，不冒充原 core 没有暴露的全局地图 revision。

**验证与边界：** 保存数组独审 **242 项通过**，重读全部六个文件、核 file/raw SHA、`inp == frame_contexts[kk]`、完整拓扑与分桶后原边序 round-trip。它没有重新执行原 gold，也没有执行真实分组 Torch/ORT/SDK。三个完整图仅原 Linear 的静态预算就有 **8.359 / 19.680 / 27.492 GMAC**，分组没有减少这些乘加。末图完整 FP32 输入加输出为 **82,563,456 B**；后续必须统计全部桶的打包、完整 I/O、scatter 和原 FP32 门槛，不能把历史单块的倒数称为 fresh 位姿吞吐。新 SDK 支持 B>1 仍须单独建立实际格式、stride、输出 ownership 和释放合同。[捕获及分组源码](../deploy/rk3588/validation/dpvo_grouped_update/RUN.md)、[实际捕获、完整图预算和保存数组独审](../results/dpvo_grouped_update_capture_v36.json)。

## 59. INT8 编码器的主要耗时仍可能落在 CPU 布局转换

**现象与证据：** 原稀疏编码器实际执行普通六次、独立 profile 三次 SDK 调用。普通首帧后三次完整 I/O 均值 **429.429 ms**，包含输入打包、SDK 转换、同步 run、`want_float=1` 输出的独立拷贝与 release。三个实际 profile 各有 **95 行：80 个 NPU INT8 标签、10 个 CPU Transpose、1 个 CPU Gather、4 个 CPU I/O**。CPU 转置合计 **279.751 / 280.882 / 287.270 ms**，占各次算子表总耗时 **67.28%–67.65%**；exNorm 为 74.483–74.605 ms，卷积族合计 17.925–18.350 ms。算子表各项是 run 内部时间，不能再次加到 SDK 外层。

**办法与验证：** 不由端口 INT8 或编译成功推断内部全 NPU；保留实际 profile、真实数组与原 FP32 对照。保存数组独审 **3,017 项通过**，重读九个 NPZ、18 份 feature，并从原 CPU 完整 imap 按同一 indices 重算 context。normal/profile 同帧数组逐字节相同，九次 release、两个 context destroy 均有记录。fmap 相对原 CPU 的相对 L2 为 **11.961%–12.201%**，context 为 **2.845%–3.979%**；原 `rtol=1e-4, atol=1e-5` 门槛全部拒绝，没有裁剪或放宽。

**边界与状态：** INT8 标签不等于独立测得每个内核的算术精度，也不消除 CPU 算子。这是固定三帧编码器诊断，不是持久执行、位姿或轨迹验收。历史 FP16 411.549 ms 不是同期控温交错基准，不能声称因果加速。默认 VGGT 不变。[原 INT8 推理、profile 与独审](../results/dpvo_encoder_int8_v37.json)。

## 60. 完整源组的原参数相同，仍须保留逐元素 FP32 拒绝

**问题：** 相同参数与闭合的源帧分组能保证数学依赖未截断，不能保证改变 batch、Conv 表达与归约顺序后仍逐元素通过固定浮点门槛。用全局相对 L2 很小来覆盖少数超限元素，会改变接受定义。

**实际验证：** 原 M48 完整图 **3,072 / 7,248 / 10,128 边**按真实 degree 组成 **1 / 5 / 10 个桶**，14 张独立 ONNX 图，保留完整目标×patch 组、真实 B、不 padding、不删边。保存数组与静态 ONNX 独审 **593 项通过**，核 36 个新产物与六个原捕获 NPZ、`inp=context[kk]`、pack/scatter 原边序，以及每图全部 **50 个原 Update 参数**字节和 reshape 对应。每图静态有 21 Conv、两次 Softmax，没有 Transpose/Gather/Where/Pad；这不是 SDK placement 证明。

**拒绝证据：** 观测 13 的分组 Torch 对原 gold 的 delta 有 **1 个元素**超限，误差/允许误差为 **1.06524 倍**；观测 127 的 ORT 对原 gold 有 **3 个**，最坏 **3.00868 倍**，ORT 对分组 Torch 有 **2 个**。整体相对 L2 约 `1e-6`、finite 与 weight 值域通过，仍记为 **numerical_rejected**。gold 来自捕获时一次实际原 callback，本次未另跑全图原 Update，也没有以新结果覆盖旧 gold。

**计时与边界：** 三图完整分组 ORT 单次分别 **265.340 / 647.259 / 1,032.817 ms**，包含本次拷贝、输入验证、pack、独立结果拷贝和 scatter；callback 内部是嵌套项。每组只测一次，历史原 callback 也不是同期 AB。分组没有降低原 Linear MAC，不证明完整反馈、轨迹或 20 Hz，候选未采纳。[完整分组、原门槛失败及独审](../results/dpvo_grouped_update_v37.json)。


**真实 B>1 SDK 补充：** 初始完整图 B8/D8/M48 已单独编译并在板端执行四次普通调用、一次独立 profile，三次 warm 完整 SDK 均值 **174.289 ms**。实际输入是 NHWC，需要真正打包；输出为 CF，delta/weight 原生逻辑大小 12,288 B，内部 stride footprint 为 49,152 B，`want_float=1` 仍按实际返回的 FP32 逻辑字节核验。全部结果按原边序还原后保存。profile 有 **50 个 NPU、3 个内部 CPU（Exp×2、Min×1）及 6 个 CPU I/O**；三个输出仍拒绝原 FP32 门槛，delta 相对 L2 为 **0.18484%**，不因此采用。

该 174.289 ms 包含输入检查/打包、set/run/get、自有 CF 拷贝及 release；不含全图计划、原边序 scatter、参考比较、初始化或 BA。保存数组独审另有 **664 项通过**：十份 CF/full NPZ 按独立 lexsort 与逆置换恢复，五次普通/profile 输出逐字节相同，五次 release 和两个 context destroy 均成功。SDK 只执行了 E=3,072 的初始图，不能外推后面两份大图，更不能作为 fresh 位姿频率。新增有界 context cache 目前仅通过合成生命周期测试，本次单 context 调用不证明其真实多形状生命周期。

## 61. 原始数据清单与解析后清单的 SHA 不能互换

**现象与根因证据：** spatial PTQ 首次启动将解析后绝对路径清单的 SHA `00b58af6…`，传给原相对路径 `dataset.txt` 的校验参数。原清单实际 SHA 为 `5c4c4712…`，两份文本所指输入相同，文件字节却不同。入口报 `Calibration dataset list SHA mismatch`，**operations 为空、没有导入 RKNN 或建立 Toolkit**；这是启动参数身份错误，不是量化算法失败。r1 失败收据保留。

**办法与验证：** r2 只改启动参数为原清单 SHA，编译源码未变。七个 helper 函数 AST 与冻结 v35 相同，量化配置相同；32 行、64 个有序图像/INT64 索引文件的 SHA、大小、dtype/shape/range 元数据与 v36 原 PTQ 完全一致，前后 source pin 保留。独立复核 **383 项通过**，仅读取源码、JSON 和清单文本，**没有读取校准 NPY、复做像素处理或证明样本代表性**。原图索引保持 INT64、原一次归一化合同不变。

**结果与边界：** 板端校准/build **48.182 s**，导出 **2,250,354 B** RKNN，四个操作返回零，release 与空 cleanup 有记录。历史 spatial 父图的 FP32 拒绝始终保留；采样 suffix 的 bitwise 通过不能提升父图为通过。另一个零推理接口查询记录 init→query→destroy，**run 为零**；图像和 feature 端口是 INT8、patch 索引仍是 INT64。成功编译与端口查询不证明内部全 INT8、精度、推理速度或 VO，模型仅保留在实验存储。[PTQ、原清单错误与零推理查询证据](../results/spatial_encoder_ptq_v37.json)。

## 62. 浮点图有效的改写，量化后可能换成更慢的 CPU 算子

**实际回归：** 历史 spatial 归约改写曾减少 FP16 编码器的布局开销，但同一 spatial 图做 INT8 PTQ 后，三次 warm 完整 SDK 调用均值达到 **905.223 ms**；本轮原图 INT8 为 **429.429 ms**。两次不是控温交错 AB，不将比例写成因果速度结论。实际 profile 才揭示了执行位置：CPU Transpose 从十个变为零，取而代之的是十个大张量 CPU Div。

| 三张保存输入 | 帧 0 | 帧 1 | 帧 2 |
| --- | ---: | ---: | ---: |
| CPU Div 合计（ms） | 819.606 | 824.763 | 824.310 |
| 占 profile 算子表总耗时 | 92.18% | 92.27% | 92.27% |
| 十个 CPU Sqrt 合计（ms） | 0.124 | 0.119 | 0.118 |

**验证：** 保存数组独审 **2,481 项通过**，重读九份输出 NPZ、18 份 feature，分别比较原 CPU 与 spatial CPU ORT；normal/profile 重复字节、九次 release 和两个 context destroy 均独核。原 FP32 门限仍全部失败，historical spatial 父图拒绝保留，没有裁剪、放宽门限或接入默认 VGGT。算子标签 INT8 只记录 SDK 声明，不推定内部每步运算精度。组件结果不是新轨迹，更不是实时验收。

**下一候选的依据：** 静态查图确认十个 Div 的分母本来是 `[1,C,1,1]`，合计仅 **480 个元素**，而对应大张量共 **16,926,720 个元素**。可研究先对小分母取倒数，再广播乘法，保留原参数和其他节点。但倒数引入额外舍入，编译器还可能重新融合大 Div 或把 Mul 放到 CPU。本轮只完成这项静态提案，尚未生成或测试新候选；需重新执行板端 CPU 对照、PTQ 和实际 profile，不能提前宣称等价通过或加速。[原始执行、独审和静态提案](../results/spatial_encoder_ptq_v37.json)。


## 63. 提前转换 FP16 输入不一定减少 SDK 输入开销

**问题与对照：** B8/D8/M48 Update 的原输入提交曾耗时约 43 ms，因此在同一模型、同一 context 上比较原 FP32 typed transport 与预先转为 FP16 的 typed transport。两者都按实际 NHWC 打包，使用 `pass_through=0`；这不是 zero-copy，也没有假定 native packed layout。顺序固定为 FP32、FP16、FP16、FP32、FP32、FP16，每种首次调用单列，各余两次为 warm。

**实际结果：** warm 完整 SDK 均值为 **FP32 156.889 ms、FP16 205.686 ms**。FP16 的 NumPy 布局/类型打包约 44 ms，输入提交仍约 53 ms；原 FP32 的后两次打包约 16–19 ms、提交约 43 ms。六次 net/delta/weight 输出在 CF 与还原原边序后分别逐字节一致；原 CPU 数值门限仍全部失败。六次输出 release 成功，结束销毁 context，恢复原打包回调。

**工程结论：** 输入字节减半不等于完整调用更快。转换发生在哪里、转换实现及 SDK 是否仍作内部处理，都要用外层耗时核实；不能只看 dtype 或内部 run 时间。此次预转换更慢，保留原 FP32 输入方式。该结论只覆盖这套 NumPy typed transport、一个实际形状和少量 warm 样本，不外推到 NEON 打包或 native zero-copy。[实现、原始计时和保存数组证据](../results/dpvo_fp16_input_transport_v38.json)。

## 64. 完整图的形状缓存有真实内存成本，小块耗时不能代表全图

**实际工作：** 三份原 M48 捕获图按全部邻居及 patch 分组，在 RK3588S 执行全部 **14 种真实形状**。复用已有 B8 模型，另编译 13 个非量化 RKNN；所有模型合计 **129,598,920 B**。普通运行每图一次 cold、两次 warm，共九次完整调用、48 次 SDK；profile 独立执行三次完整图、16 次 SDK。

| 原边数 | 分组数 | cold 完整耗时（ms） | 两次 warm 均值（ms） |
| --- | ---: | ---: | ---: |
| 3,072 | 1 | 325.485 | 203.866 |
| 7,248 | 5 | 910.285 | 608.959 |
| 10,128 | 10 | 1,525.935 | 1,004.553 |

完整时钟包含不可变输入快照、输入检查、分组打包、需要时的 context 初始化、SDK I/O/copy/release、值域检查与原边序 scatter；参考误差统计、输出哈希和保存结果在外层计时之外；cold 初始化中的模型读取及源码哈希仍包含在内。各组、cache、初始化时钟互相嵌套，不得相加。此处没有 BA、反馈更新或 fresh 位姿，不能换算成实时 SLAM 帧率。

**缓存证据：** 上限为 14 contexts、256 MiB 模型缓冲、512 MiB DMA。普通运行创建 14 次、命中 34 次、没有驱逐；峰值模型缓冲 **129,598,934 B**，DMA **335,704,064 B**，进程峰值 RSS **941.594 MiB**。模型缓冲每个额外一字节来自 ctypes buffer，不是编译文件改变。profile 创建 14 次、命中两次。两组运行关闭后 contexts、模型缓冲及 DMA 都归零。仅这三种固定图的联合形状集合能全部驻留；不能据此宣称更长轨迹没有编译、驱逐或内存问题。

**执行与接受边界：** 每个实际 SDK profile 都记录 50 个内部 NPU、3 个内部 CPU（Exp×2、Min）和 6 个 CPU I/O 行，仍不是全 NPU。原逐元素 FP32 门限继续拒绝；完整图 warm 耗时也远超目标。真实14形状的普通/profile执行和缓存关闭已验证，但未接入递归 VO，未改变默认 VGGT。[源码及运行合同](../deploy/rk3588/validation/dpvo_grouped_update/V38.md)、[完整计时、数组与缓存证据](../results/dpvo_grouped_full_v38.json)。


## 65. 将大张量除法缩到小分母，必须沿 CPU、PTQ、实际 placement 分段验证

**问题与最小改写：** 上一轮十个大张量 CPU Div 合计耗时约 820 ms。父图的分母已经是 `[1,C,1,1]`，因此只将这十处 `centered / denominator` 改为 `centered * (1 / denominator)`。保留全部 65 个原 initializer 的 protobuf 字节与顺序、其他节点和 I/O，只增加一个非学习 FP32 常数 1。CPU 除法涉及的输出元素从 16,926,720 降到 480，广播乘法留给编译器调度。

**逐段验证：** 板端同一三帧、同一索引分别执行父图与新图，CPU ORT 关闭图优化；六次调用确认候选对当前父图通过原 `rtol=1e-4, atol=1e-5`，但并非 bitwise。父图对原始 CPU 网络的历史拒绝仍保留，两者对原始网络都未通过。随后在板端使用相同 32 行、64 个输入文件及量化配置完成 PTQ，校准/build **48.169 s**，模型 **2,176,498 B**。init/query/destroy 的零推理阶段确认实际端口，随后另跑六次普通、三次 profile 推理。

**实际执行位置：** 三个新 profile 各有 **133 个 NPU、25 个 CPU 行**，其中 CPU 为十个小 Div、十个 Sqrt、一个 Gather 和四个 I/O；没有 CPU Transpose。十个 Div 的输出都仅 `[1,32,1,1]` 或 `[1,64,1,1]`，未重新融合成大除法。对应十个大张量乘法实际在 NPU，九个融合 ReLU、一个独立 Mul。

| 三张保存输入 | 帧 0 | 帧 1 | 帧 2 |
| --- | ---: | ---: | ---: |
| 十个 CPU 小 Div 合计（ms） | 0.141 | 0.176 | 0.180 |
| 十个 NPU 乘法/融合乘法合计（ms） | 5.894 | 6.305 | 6.482 |
| profile 算子表总耗时（ms） | 61.702 | 65.827 | 68.230 |

普通三次 warm 完整 SDK 均值 **81.012 ms**，包含 typed 输入打包、set/run/get、自有输出拷贝与 release；相对历史 spatial INT8 的 905.223 ms 有明显下降。但两次运行非同期交错、未控温控频，不能把比值写成受控加速比，profile 内部时间也不能再加到完整调用时间。

**精度与采用边界：** fmap 对原 CPU 相对 L2 为 **15.806%–16.294%**，context 为 **2.845%–3.979%**，这些是 feature 误差，不是 ATE/RPE。原逐元素门限继续失败；没有因性能改善放宽门限或采用候选。该实验验证了消除大 CPU 除法的方法，未验证完整反馈、地图、20 Hz 或相机延迟。后续仍需解决量化误差及完整 Update 的计算成本。[源码与复现合同](../deploy/rk3588/validation/dpvo_encoder_reciprocal/README.md)、[CPU、PTQ、接口、普通/profile 与独审证据](../results/dpvo_reciprocal_encoder_v38.json)。

## 66. 缩短滑窗要检查初始化与清理顺序，几何尺寸也必须完整传播

**触发条件：** 为降低完整 Update 的工作量，预声明一个半分辨率 M16/L4/R8/O4/KF2 研究配置，不改原始权重或生命周期。只依据 `R >= L + KF` 把移除窗口设为 6 会失败：原 core 在初始化的第八个准入帧连续执行十二次 Update，尚未做旧边清理；设为 7 也会在第九个准入帧、第一次清理前失败。检查 `ii < n-R-1` 的位置说明，保持现有生命周期时 R8 才是最小可执行窗口。

**办法与证据：** 不跳过年龄检查或截掉 Update 的边，而是在执行前固定 R8。小型拓扑检查覆盖不删帧、每次删帧、交替删帧三种顺序；真实板端回放随后在观测 34 完成初始化，首个完整图为 **704 条边**，观测 127 和 191 的完整图均为 **800 条边**。本次实际调用还出现 688、784、816、832 条完整图边；motion probe 的 16 条边另计，不能当成完整 Update。

**图像与接口细节：** 保持原 640×480 去畸变和 464×608 裁剪，再用 INTER_AREA 缩到 232×304；内参应采用半像素映射 `(u+0.5)*0.5-0.5`，得到 `[258.65,258.25,151.05,123.4]`，之后只做原 RES=4 除法。58×76 feature 的原 level-4 average pool 为 **14×19**，不补齐到 15 行。原 Torch BasicEncoder 需要 `[B,N,C,H,W]`，而已导出的 ONNX 接口是 `[B,C,H,W]`；适配时显式补 singleton 帧维，不能用 ONNX 接口推断 Torch 签名。

这些是可执行性与坐标合同，不代表精度或实时性能。新分组必须包含每个源帧的全部目标与 16 个 patch，并最终恢复完整原边序；旧 M48 模型合同不能直接套用。[冻结 policy、回放入口及尺寸改写](../deploy/rk3588/validation/dpvo_half_workload/README.md)、[实际图和来源证据](../results/dpvo_half_workload_v39.json)。

## 67. 工作量下降后必须比较共同新观测，单独报告初始化覆盖和精度损失

**固定实验：** 同一原 DPVO checkpoint、同一 TUM fr1_xyz 前 192 个连续 RGB，固定半分辨率和 M16/L4/R8/O4/KF2。原 motion=2、keyframe=15 的 feature-pixel 阈值保留，因此对应的原图物理像素范围改变；这不是原 FAST 数值等价优化。参数在读取 GT 前声明，没有按轨迹误差挑选配置。

**完整结果：** 16 GB RK3588S 上完成 **158 个 fresh 位姿 / 92.874 s = 1.701 Hz**，从观测 34 起初始化完成；软件读入到 JSON 输出 flush 的 P95 为 **668.453 ms**。计时包含三处完整 Update 输入/gold 捕获，是带诊断记录的串行文件回放，不冒充普通相机延迟。EOF 十二次优化另计 **3.434 s**；进程峰值 RSS **361.332 MiB**。94 项原参数与输入保持不变，缓存关闭后归零。

**精度的共同分母：** 原 FAST CPU 为 179/192 fresh，新配置为 158/192。先取共同 158 个 fresh 观测，再关联 GT；一秒 RPE 端点先在全部 192 个原时间戳上规划，共同可用 128 对。每个轨迹只做一次正尺度 Sim3 评估对齐，结果如下：

| 共同样本指标 | 原 full-resolution FAST CPU | 半分辨率 CPU | 增幅 |
| --- | ---: | ---: | ---: |
| ATE RMSE | 1.113 cm | 1.971 cm | 77.15% |
| 平移 RPE RMSE | 1.709 cm | 2.969 cm | 73.78% |
| 旋转 RPE RMSE | 0.986° | 1.668° | 69.26% |

**采用判断：** 该配置未达到精度损失 10–20% 的目标，也未达到实时目标，保留为未采用的研究实验。这里比较的是 DPVO 配置，不是原 VGGT 精度；不同历史运行未交错控温，不能把耗时比当成受控加速比。初始化等待、motion reject、EOF 重建和相同旧位姿均不计为 fresh。

**独立数值发现：** 新尺寸下重新建立三张图的原 Torch teacher，原始 ONNX 和 reciprocal ONNX 都通过原 `rtol=1e-4, atol=1e-5` 门限；旧 full-resolution 图的拒绝仍保留。说明数值门槛需要绑定具体尺寸、输入与实现，不能把一个尺寸的通过或失败推广到另一个尺寸。量化后的误差还需独立验证。[完整回放、共同样本评估、CPU 对照及保存数组核查](../results/dpvo_half_workload_v39.json)。

## 68. 尺寸、精度与完整图合同要分别验收，不能沿用上一形状的结论

**新尺寸的分段证据：** 半分辨率编码器先建立新的原 Torch teacher，原始与 reciprocal CPU ONNX 均通过三张图的固定浮点门槛。随后在板端使用新尺寸的 32 行校准输入完成 PTQ，build 用时 **16.169 s**，RKNN 大小 **1,521,328 B**。普通六次与独立 profile 三次 SDK 调用完成，三次 warm 完整 I/O 均值 **22.024 ms**；九次输出释放、两个 context 销毁有实际记录。

**不能由速度推定精度：** INT8 的 fmap 对新 CPU teacher 相对 L2 为 **10.243%–11.177%**，context 为 **2.647%–3.258%**；对原 CPU、原 ORT、reciprocal ORT 的严格门槛全部失败。三份 profile 各记录 133 个 NPU 与 25 个 CPU 行；十个 CPU 小 Div 总计 **84/75/81 µs**，没有 CPU Transpose，但仍有 CPU Sqrt、Gather 及 I/O。22 ms 只覆盖编码器的完整 SDK 调用，不含图像预处理、Update、相关性、BA 或新位姿，不能换算成 SLAM 频率。此量化编码器尚未进入完整半分辨率轨迹。

**完整 Update 也需重新建立合同：** 三份真实 M16 图各有四个完整源帧分组，共产生八种实际 `(B,D,16)` 形状。板端原分组 Torch 与 CPU ORT 对同次原调用保存的 full-order net/delta/weight 全部通过原门槛；50 项原 Update 参数与全部边保留，分组后按原边序整体还原。不能直接复用 M48 的形状模型，也不能因为以前 M48 有数值失败就跳过本次验证。这里尚未运行新八形状的 RKNN，更没有递归反馈轨迹；本条不改变第 67 条完整配置的精度拒绝。

**可复用做法：** 让验收对象同时绑定模型参数、图表达、输入尺寸、索引/边拓扑和运行精度。保留旧尺寸失败记录，对新尺寸建立独立原始参考；组件浮点门槛、量化误差、完整反馈轨迹和端到端延迟各自报告。[源码和阶段边界](../deploy/rk3588/validation/dpvo_half_workload/README.md)、[实际 CPU/PTQ/SDK/完整 M16 图与保存数组证据](../results/dpvo_half_workload_v39.json)。

## 69. 缩小完整图后，仍要实测全部分组、传输和恢复，而不是取单块耗时的倒数

**实测对象：** 上一轮通过原 Torch/CPU ORT 对照的八种 M16 形状，分别编译为非量化 FP16 RKNN；原参数、完整源帧的目标×patch 依赖和原边序保留。主机只做 `build(do_quantization=False)`，实际 SDK 推理全部在 RK3588S。八个模型共 **59,357,800 B**，三份固定图各有四组，普通运行九次完整调用、36 次 SDK；独立 profile 三次完整调用、12 次 SDK。

| 完整图边数 / 观测 | cold 完整调用（ms） | 两次 warm 均值（ms） |
| --- | ---: | ---: |
| 704 / 34 | 286.337 | 61.294 |
| 800 / 127 | 121.829 | 68.482 |
| 800 / 191 | 256.407 | 71.344 |

**计时与缓存：** 完整时钟包含只读输入快照、检查、各组打包、缺失 context 的初始化、SDK I/O/自有输出拷贝/release、值域检查以及原 E 顺序还原。引用误差统计与结果存盘在外；cold 表示该状态第一次调用，跨状态已有 context 可以复用，不等于每次全冷启动。八个 context 同时驻留，峰值模型缓冲 **59,357,808 B**、DMA **78,249,984 B**，普通进程 RSS **221.699 MiB**；关闭后 context、模型缓冲和 DMA 均归零。这里不包含原编码器、相关性、BA、完整地图或反馈轨迹。

**接受边界：** 对原同次调用 gold，三个状态的 net 相对 L2 为 **0.149% 左右**，delta 为 **0.159%–0.175%**，但固定逐元素 `rtol=1e-4, atol=1e-5` 仍全部失败；没有放宽阈值或裁剪输出。小的组件误差也不保证递归轨迹误差小。warm Update 本身已超过 20 Hz 的 50 ms 串行周期预算，不能将单组 SDK 频率当成位姿频率；第 67 条完整配置的精度拒绝继续保留。[固定状态驱动和复现合同](../deploy/rk3588/validation/dpvo_m16_grouped/README.md)、[编译、实际 SDK 与保存数组证据](../results/dpvo_m16_grouped_v40.json)。

**实际 placement 与复核：** 八种形状的 profile 均为 **48 个内部 NPU、3 个内部 CPU（Exp×2、Min）和 6 个 I/O 行**，不能照搬旧 M48 的 50 个内部 NPU 数。保存数组独审 2,836 项通过，重读全部十二份完整输出，确认普通重复与独立 profile 逐字节一致、逆 scatter 后各组 raw SHA 正确，并保留原 135 项 CPU/ORT 通过与本次 SDK 拒绝的区别。

## 70. 静态模型目录必须覆盖在线拓扑，完整源帧拆批还存在调用开销

**容易遗漏的条件：** 三份捕获状态只有八种 `(B,D,M16)`，而同一原 CPU 回放的 **169 次完整在线 Update**（含初始化、不含 motion probe 和 EOF）出现 **13 种**。遗漏的是 B1D4、B3D4、B3D5、B3D7 和 B4D6。每种捕获图都能运行，并不能证明新增、删帧后的所有图都能运行；未知形状必须显式拒绝，不能临时截邻居、补零或静默切 CPU。

**可复用接口与代价：** 已有 `build_degree_plan(max_sources_per_batch=2)` 可只沿独立源帧轴拆批，保持每个源帧全部 D×16 依赖和最终 scatter。对这份保存轨迹，模型目录可归约为 B1/B2 × D4–7，但仍缺 B1D4、B1D6；motion probe 的 D1 要另有明确执行路径。拆批不减少 MAC，全部分组调用数反而从 **676 增至 870**，需要重新衡量完整 I/O 和 context 开销，不能只按模型种类少判断更快。

**当前状态：** 上述覆盖和拆批次数来自保存 JSON 的静态分析，尚未将拆批方案执行成新的在线 NPU 轨迹。更换数值实现还可能改变删帧决策，不能以旧 CPU 的形状集合保证未来覆盖。部署合同应声明容量、支持的形状、完整依赖和未知形状行为，单独验证生命周期与总耗时。[保存元数据分析脚本](../deploy/rk3588/validation/dpvo_m16_grouped/analyze-shape-coverage.py)、[完整形状清单与实际测试边界](../results/dpvo_m16_grouped_v40.json)。

## 71. 固定相机的去畸变映射可以复用，合并调用须保留映射字节和像素结果

**先定位剩余耗时：** 对上一轮 157 个初始化后的 fresh 观测做保存时钟分析，扣除 encoder 与 Update 后的实测剩余时间平均 **98.76 ms**、P95 **120.38 ms**；其中预处理均值 26.12 ms、相关性 30.35 ms、BA 两步 22.83 ms。嵌套时钟不能再次相加，这也不是换 NPU 后必然不变的硬件下界，但说明只优化神经网络还不足以达到目标。

**重复工作的根因：** OpenCV 4.10.0 的 `undistort` 对 640×480 图像每次构造 80 条、每条 6 行的固定点映射，并逐条调用 `remap`。本序列 K/D、图像尺寸固定，这些映射可以复用。第一版缓存完全相同的条带映射；第二版只拼接这些已生成的 map1/map2，并核对拼接前后的字节 SHA，改为一次 full-frame remap，避免以另一套浮点计算顺序重新生成全图映射。保留原 INTER_LINEAR、零边界、BGR、裁剪、INTER_AREA 缩放和一次归一化。[官方实现](https://github.com/opencv/opencv/blob/4.10.0/modules/calib3d/src/undistort.dispatch.cpp#L264-L305)。

**实际像素与计时验证：** 两版都在 RK3588S 上比较前 192 张连续真实 RGB，完整去畸变图、半分辨率图和 FP32 模型输入全部逐字节相同。每轮固定取其中 24 张，按原版/候选/候选/原版交错计时，每种方法各 48 次。下表包含去畸变、裁剪/缩放及归一化，不包含 `imread`、比较、哈希、存盘或神经网络。

| 独立对照轮次 | 同轮原版均值（ms） | 候选均值（ms） | 候选 P95（ms） |
| --- | ---: | ---: | ---: |
| 缓存原条带，保留 80 次 remap | 14.679 | 10.943 | 10.974 |
| 拼接原映射，单次 remap | 12.764 | 6.672 | 6.827 |

两轮未控温控频，原版耗时也发生变化，因此分别报告同轮对照，不以跨轮候选比值作为受控加速比。单次版本首次映射构建与拼接另需 **12.272 ms**，常驻映射数组 **1,843,200 B**，构造期间两套映射数组共 **3,686,400 B**，进程峰值 RSS **72.430 MiB**；数组容量不等于总内存上界，close 后常驻映射为零。

**启动条件也属于复现合同：** 首次测试误将未绑定 CPU 时的 OpenCV 默认 8 线程当作原回放配置，入口在任何映射/图像处理前拒绝。原流程先绑定四个 A76 核再导入 OpenCV，实测默认是 4。保留失败收据，修正为相同导入顺序、相同实际默认线程数后运行，两路不混入线程优化。

**范围与复用：** 主机只复核保存 metadata，并重读每版三张保存样本的六对实际数组；192 张完整像素检查发生在板端。固定映射缓存已获得组件等价和耗时证据，但尚未接入完整反馈轨迹，也没有新的位姿率、ATE、相机延迟或 30 分钟结论。相机 K/D 或图像尺寸改变时必须重新建立缓存。[源码与复现方法](../deploy/rk3588/validation/fixed_undistort/README.md)、[原始失败、两轮成功结果和保存数组复核](../results/fixed_undistort_v40.json)。

## 72. 精确缓存通过完整回放，不代表端到端一定更快

**问题与验证对象：** 第 71 条的单次 remap 缓存已接入原半分辨率 CPU 研究回放，只替换固定相机的去畸变映射构造/调用调度。保持同一 192 张原 RGB、原时间戳、M16 policy、94 项原权重、采样和深度 RNG、完整 Torch Update、相关性、BA 及生命周期；没有额外传统定位前端，也没有替换默认 VGGT。

**精确性证据：** 保存结果审核在只剥离预声明运行时钟字段后，核对全部 **192 个观测结果**；原 policy/RNG 相同，重读 **8 对 NPZ、55 对实际数组**全部 bitwise。两路均为 158 个 fresh 位姿，初始化、motion reject、删帧、EOF 与保存几何未因缓存改变。这里验证的是本次 CPU 路径的非时钟输出，不能提升另一实现或另一相机为通过。

**完整耗时与根因边界：** 包含解码、去畸变、裁剪和缩放的 192 帧平均阶段时钟从 **25.421938 降为 20.833521 ms**，但整段回放从历史 **92.874 s**增至本次 **107.043 s**；本次 **158 / 107.043 = 1.476 Hz**，fresh 观测读入到 JSON 输出 flush 的 P95 为 **764.036 ms**。编码器阶段均值也从 261.228 增至 314.270 ms，stream 阶段从 196.340 增至 221.651 ms。这两次没有同进程交错、控温控频，不能证明变慢由缓存、热状态或某个单独算子导致，也不能用局部节省直接预测完整频率。

**解决边界：** 固定几何缓存获得了完整 CPU 回放的输出一致性证据，仍需以完整链路计时验收性能。原半分辨率配置约 **77% 共同 fresh ATE 损失**的拒绝继续保留；这次没有新 GT 精度改善结论。文件回放不是物理相机、排队时延或 30 分钟稳定性验证，EOF 优化不计 fresh。原运行 SHA `dd9f3752…` 与缓存运行 SHA `c1f0a463…`分别保留，不混成受控 AB。[完整回放、非时钟/RNG/保存数组对照](../results/dpvo_online_update_v41.json)。

## 73. 拆独立源帧能收敛模型目录，不能裁掉源组内部依赖

**问题与办法：** 第 70 条发现三个保存状态的八种形状不足以覆盖完整在线图。固定 `max_sources_per_batch=2` 后，只沿独立源帧轴拆 true-B 批次，模型目录可使用 B1/B2×D4–7；每个源帧的全部目标×16 个 patch、邻居顺序、两个 SoftAgg 归约与最终原 E 顺序 scatter 均保留。这不是把 D/M 分成小块，不减少 MAC，也不改变边集合；调用次数及 context/I/O 成本必须另测。

**补齐两图的证据：** 六个已编译模型保持原 v40 来源，新增 B1D4/M16 与 B1D6/M16。两份真实输入固定取原观测 34、原调用 34 中 B2D4/B2D6 的第一个完整源帧，选择发生在数值比较前；gold 来自原完整 Update 的同次保存输出，未伪称重新执行完整图。板端新 Torch、CPU ORT 与原源组 gold 的三类 net/delta/weight 对照共 **18 项**全部通过固定 `rtol=1e-4, atol=1e-5`，输出有限、weight 合法，50 项原 Update 参数与 94 项原权重前后保持。新 manifest SHA 为 `2f5b3dd89b5863963bdfa52ffa213af02f08b6654a7012f5980ea3189d7ba47f`。

**运行与接受边界：** 上述导出对照本身不能证明完整在线回放；后续实际在线前缀和 EOF 形状缺口见第 74 条，也不能由此保证未知未来图覆盖。完整 Update 必须使用声明的 RKNN 图；未知形状保存完整输入并停止，不补零、删边或静默切 Torch。D1 motion probe 由冻结 core 的真实 `probe` 参数显式选择原 Torch 启动路径，不能依据 E=16 猜测，也不算完整 Update 的 CPU 回退。SDK 历史 FP32 拒绝标签继续保留，CPU 导出通过不替代实际 SDK、完整反馈和轨迹验收；EOF 仍另记离线更新。[两图原参考、固定来源和在线研究状态](../results/dpvo_online_update_v41.json)。

## 74. 在线观测全部处理完，EOF 仍可能产生模型目录未覆盖的完整源组

**问题与实际证据：** CPU 编码器＋RKNN Update、RKNN 编码器＋RKNN Update 两路都处理了原 192 个观测，发布 **158 个 fresh 位姿**；初始化和在线阶段完成 **169 次完整 Update、169 次 BA（338 步）**，另有 34 次明确的原 Torch D1 启动 motion probe。随后零起编号 **203** 的 callback 在第一次 EOF Update 因未知 `B1_D3_M16` 拒绝。两份原运行的 `status` 都是 **failed**，不能因为在线前缀完整就改写成整个回放成功。

**根因与停止边界：** 八图目录覆盖的是 B1/B2×D4–7。EOF 另有离线更新阶段，其完整图出现 D3，说明仅从在线捕获集合推导目录不足以覆盖整个生命周期。失败发生在形状预检，保存了完整失败输入，未返回部分完整图、未执行该次 BA、未完成 EOF 轨迹重优化/重建；没有补零、删边或完整 Update 的 Torch 回退。两路 callback 都记录 `full_completed=169、failed_calls=1`，关闭后 context、模型缓冲、DMA 为零，清理错误为空。下一步需要为完整 D3 源组建立原参数 CPU/ORT、编译与 SDK 的独立验证并重跑 EOF；本轮尚未补图。

**失败运行的有效评价边界：** 评价器默认仍拒绝 failed run。显式 `--allow-failed-EOF-only` 只允许核过全部 192 条观测、EOF 前一致计数、唯一最后未知形状失败及完整清理的这类记录；继续使用已经发布的 fresh 位姿评价，保留 `overall_run_success=false` 和 `EOF=failed_not_completed`，EOF 轨迹指标不可用。没有将未完成 EOF 的保存状态或历史重建计为新位姿。GT 仅在运行结束后用于数值评价，失败入口和门槛原样保留。

**实测仍未验收：** 两路在线回放时钟分别为 **111.192 s、60.732 s**，fresh/回放时长为 **1.421 Hz、2.602 Hz**；观测读入到 JSON flush 的 fresh P95 分别为 **725.280 ms、411.798 ms**。前者实际记录 892 次 Update SDK，后者为 893 次 Update SDK＋192 次编码器 SDK；这些是单次不定速文件回放，未控温控频，不是物理相机排队或 20 Hz 证明。RKNN 调用也不代表所有计算都在 NPU，相关性、BA 和显式启动路径仍有 CPU 计算；原 SDK 浮点门槛拒绝继续保留，未采用为默认路径。[失败原收据、生命周期与保存 JSON 轨迹评价](../results/dpvo_online_update_v41.json)。

## 75. 组件逐元素误差和递归轨迹误差必须分别度量

**问题与同分母对照：** 相同权重下改变 Update 或编码器的数值实现后，输出进入下一轮相关性、BA、删帧及状态更新，组件误差不能直接换算为轨迹误差。下面四路都取 **同一 158 个共同 fresh 观测、同一 128 对 1 秒端点**；端点先按全部 192 个原传感器时间戳规划（共 162 对），共同 fresh 集合在 GT/误差筛选前确定。GT 最近时间戳关联上限为 20 ms，一对一；每一路在该共同观测集合上只拟合一次正尺度 Sim3，并将其尺度复用于平移 RPE，没有逐窗口拟合。

| 固定研究路径 | ATE RMSE（m） | 1 s 旋转 RPE RMSE（°） | 1 s 平移 RPE RMSE（m，评价尺度） |
| --- | ---: | ---: | ---: |
| 原全分辨率 FAST CPU，M48 | 0.011125 | 0.985657 | 0.017087 |
| 原半分辨率缓存 CPU，M16 | 0.019708 | 1.668366 | 0.029694 |
| 半分辨率 CPU 编码器＋RKNN Update | 0.027996 | 3.176750 | 0.049296 |
| 半分辨率 RKNN 编码器＋RKNN Update | 0.019683 | 1.793720 | 0.036012 |

**证据与结论边界：** 保存 JSON 独立评价确认两份共同观测集合和端点相同；两路在线回放的 fresh 覆盖是 158/192（82.292%），本表可用端点覆盖是 128/162（79.012%）。后两路 ATE 对原全分辨率 FAST 分别增加约 **151.6%、76.9%**，旋转和平移 RPE 也未满足 10–20% 损失目标。表中的单次结果不能建立组件误差来源、误差相互抵消、因果提速或泛化精度结论；接近半分辨率 CPU 的某个指标也不改变全分辨率质量拒绝。

**后续使用边界：** 组件固定门槛、状态递推一致性、共同观测集合 GT 误差、全部观测覆盖和完整时延都要保留；两份失败运行的 EOF 指标仍不可用。本表是 DPVO 研究路径对原 DPVO FAST 的短序列比较，不是原 VGGT 518 精度基线、地图质量、30 分钟稳定性或默认模型替换验收。[共同观测集合、固定端点、单次 Sim3 与完整误差证据](../results/dpvo_online_update_v41.json)。

## 76. 读取一个活动标志，不应深复制全部 context 来源与属性

**触发与证据：** 完整 RKNN Update 已返回后，冻结 v41 callback 每次调用 `cache.metrics()`，只从 history/live session 中提取 `npu_used is True`。真实 bothNPU 回放结束前有八个 live context、零次 eviction；该 reporting query 经 `_usage()` 和 `sessions` 至少深复制 16 份 session metadata，随后又复制完整 cache，其中包含 attrs、bindings、内存记录和来源 SHA。157 次稳态完整 Update 的 callback 均值为 **169.551 ms**，完整 staged graph 为 **136.546 ms**；差额 **33.006 ms**还包含外层快照、plan 与输出检查，不能全部归因于 metadata。

**可测方案：** 只替换这一个 reporting block，按原 cache→session 锁顺序读取相同的 history/live 布尔值，保留历史最后一项、live 覆盖历史和严格 `is True` 的规则。标志仍由原 SDK session 在实际 run API 调用处设置；不根据模型文件名猜测 NPU。公开 `metrics()`、source recheck、context close、内存清理检查、双重输入所有权快照、finite/domain 检查、完整源组、revision 检查与失败存盘保持。原 cache 每次 hit 为读取一个 `closed` 字段再深复制 session，也是可单独测量的冗余，本候选尚未改变它。

**验证边界：** 七个 generated/fake 检查通过，包括精确源码差异、历史/live 覆盖、hot metrics 调用由一次降为零、输出 bitwise 与独占所有权、晚组失败、未知 D3、stale revision、六输入失败保存和 close/cleanup 检查。它们验证契约，不是板端提速；候选仍使用 v41 八形状目录，不能据此称 EOF 已修复。两份真实 v41 run 的 failed EOF 与严格 FP32 拒绝保持，未更改默认 VGGT、数学或接受门槛。[保存时钟与静态分析](../results/dpvo_update_performance_v42.json)、[最小候选和 fake 检查](../deploy/rk3588/validation/dpvo_update_performance/README.md)、[原运行边界](../results/dpvo_online_update_v41.json)。

**后续板端同状态对照：** 在三个真实保存输入（观测34/127/191，704/800/752条边）上，原版A和只改reporting block的B使用两套独立cache、相同八模型目录。每状态执行ABBA两轮；各方法各状态排除首次调用，余3次为暖样本。完整callback墙钟均值A→B分别为 **93.686→81.816、146.065→124.458、122.133→105.570 ms**。24次调用的net/delta/weight与各状态首份A输出字节相同，两路各56次SDK均完成输出release，各6个实际创建的context关闭、cache模型/DMA归零。保存、哈希、输入预检和外部统计在计时外，callback本身的正常记录复制仍在墙钟内。

这些是本次固定状态的组件结果，每项只有3个暖样本，未控温控频；完整图内部时钟也有波动，不能把全部差额都归于一个复制函数。输入来自已有NPU反馈轨迹，没有新的CPUteacher、BA或递归状态提交；候选尚未接入v42完整在线回放，不能换算新鲜位姿Hz、改变原FP32拒绝或称为20Hz通过。[板端交错对照和保存输出审核](../results/dpvo_metadata_callback_ab_v42.json)。

## 77. 限制独立源 batch 可缩小模型目录，却增加完整图的调用次数

**触发与证据：** 第 73 条按独立源轴设置 `max_sources_per_batch=2`，保留每个源的全部 D×M 依赖。读取实际 v41 callback 的 `source_degrees`，逐次核对源数、每源 D×16 边数和已保存 degree plan：bothNPU 的 **169 次成功完整 Update**共有 **893 个** B1/B2 组；同 degree 的完整独立源不设 B2 上限时为 **668 个**。其中157次稳态为 **845→620**，调用组数减少 **26.63%**；CPUencoder 的对应数为892→668、844→620。

**可测方案：** 为真实前缀建立精确静态目录并比较完整链路：M16 的 B1/B2/B3×D4、B1/B2/B3×D5、B2/B3/B4/B5×D6、B1/B2/B3×D7，共13种。第一次 EOF 的 noncap 图另为 B1D3+B1D4+B3D5+B3D6，须单独覆盖。只合并等 degree 的完整独立源，保持源组内部邻居、归约、所有边和最终原边序；不同 batch 的 SDK 数值与 complete graph 时钟仍需实测，不把独立性当成已验证的浮点 bitwise 等价。

**验证边界：** 以上是固定两份保存拓扑的静态重分组，未执行大 batch 模型，也不能保证后续 EOF 迭代或新反馈轨迹的形状。输入元素、MAC 和输出元素不减少；组数比例只适用于假设的固定每次调用成本，不能直接套到 inputset 或 run。该 bothNPU 稳态 SDK run API 合计均值已为 **64.910 ms**，inputset 为 **30.241 ms**；保留原 run 时钟而将其余开销都置零，也仍超过50ms。这只是保持原调用时钟的极端预算，不是大 batch 的硬件下界或提速预测。原 run 仍 failed EOF，候选未采用，20Hz 目标未验证。[逐次拓扑核对、目录与嵌套时钟](../results/dpvo_update_performance_v42.json)、[可复用只读分析器](../deploy/rk3588/validation/dpvo_update_performance/audit-v41-update-performance.py)。

## 78. 逻辑 FP16 属性不能代替 native 查询，预转换反例也不能外推到 NEON

**触发与证据：** 第 63 条发现 NumPy FP16 typed transport 并未减少完整 SDK 成本。同 context 的固定 AB/BA/AB 中，每种去掉首次调用后的两次 warm 均值为：FP32 pack/inputset/run/forward **17.265/42.707/84.507/156.889 ms**，FP16 为 **44.276/52.727/91.243/205.686 ms**；两路都是 `pass_through=0`，没有零拷贝。另一方面，冻结 v41 session 实际只查询普通输入/输出属性（command1/2）；保存的 FP16/NHWC logical attrs 不能证明 command8/10 的实际 native 布局。

**可测方案：** 先对每个实际模型的三个输入查询 native8 与 nativeNHWC10，保留返回码、dims、format/type、quantization、字节数和全部 stride/padding。只支持由这些查询证明的布局，模糊或不支持的布局显式拒绝。随后可单独测试一个小 NEON 内核，将原 FP32 CF 输入一次转置并转换到已确认的 native FP16 自有 buffer，再使用 `inputs_set` 的 pass-through1；保持 finite/FP16 overflow、字节边界和同步所有权检查，原输出申请、独立拷贝及 release 路径保持。若 native 是 NC1HWC2，不猜通道分块或尾部 padding，也不能直接套 dense NHWC 内核。

**验证边界：** 此记录仅为本地固定 header、源码和保存 JSON 的静态可行性分析，native 查询与 NEON 方案未在本收据中执行。后续须同当前状态做原 SDK 对照，分别报告 pack、inputset、run、完整图和 callback，并保留严格原 Torch 门槛的拒绝；v38 的六次输出字节一致不授权新 native 路径。pass-through1 不自动消除 `inputs_set` 的拷贝，因此不能称 zero-copy。SDK 管理的 `create_mem/set_io_mem/mem_sync/destroy_mem` 是另一实验，须另核缓存一致性、DMA预算与销毁顺序。旧 NumPy 反例不证明 NEON 或 native 一定更慢，也不支持其一定更快。[固定 source/query 证据与后续验证条件](../results/dpvo_native_input_feasibility_v42.json)、[原 FP16 反例](../results/dpvo_fp16_input_transport_v38.json)。

**后续实际 native 查询：** 九个固定 M16 模型已由板端独立执行 init、coremask7、command1/8/10 查询及立即 destroy，共9次初始化、9次成功销毁、零次 SDK forward；保存收据 SHA 为 `89f0cead99df024b25ed3d2638e412a075f91d6736ecdb2b35ace2304eef0642`。独立保存 JSON 审查逐个核了81份 attrs：27个输入的 command8与command10属性完全相同，均为 **FP16、NC1HWC2、C2=8**；普通 command1 为 NHWC/rank4。net/inp 的 native shape 是 `[B,48,D,16,8]`，corr 为 `[B,111,D,16,8]`，对应882个有效通道扩为888个存储通道，末块六个 padding 通道。全部 w_stride=16、h_stride=0、size_with_stride=size，没有额外空间 stride 字节；qnt_type=2 但 scale=1、zp=0、fl=0，不能因此称输入已做 INT8 量化。

**由查询收敛后的方案与边界：** `nativeNHWC` 的命令名称没有返回 dense NHWC，因此当前查询不支持将 dense NHWC FP16 直接当 native buffer。可测下一步应收敛到实际 **C2=8** 的自有打包：`dst[b,c//8,d,m,c%8]=FP16(src[b,c,d,m])`，corr 尾部六通道初始化为零，并核精确字节数、finite/FP16 overflow、索引、所有权和销毁。查询到的 pass_through字段仍为0；它没有提交任何输入，不能证明 `inputs_set/pass_through1`、padding消费、舍入或 `set_io_mem` 的接受与输出等价性。这些须同当前保存状态对原 SDK 做板端 AB，再比较完整 pack/inputset/run/graph 时钟；查询成功不是传输验收、零拷贝、精度通过或提速结果。原 SDK FP32拒绝和不采用边界保持。[真实查询与独立保存属性审查](../results/dpvo_native_input_query_v42.json)。

## 79. EOF 补图需要同输入参考，并单独验证在线前缀未改变

**问题与修复：** 第 74 条的完整在线前缀在 EOF 首次请求 `B1_D3_M16` 时失败。现在只补这一种真实形状：读取原失败的完整640边输入，在板端执行一次原参数完整 Torch Update 产生新的同输入参考，再取源帧11、目标9/10/11的全部48边。该参考绑定的是含 NPU 反馈的失败输入，不是此前纯 CPU 轨迹的旧 gold；三类 Torch/ORT 对照共9项均通过原 `rtol=1e-4, atol=1e-5`。原94项权重和50项 Update 参数保持，旧八份模型及编译来源保持，目录仅增加独立D3项，没有截邻居、补零或完整 Update 的 CPU 回退。

**完整生命周期证据：** 新运行 SHA `c0546eb004e79fe9d953901a882d51ff60a620cc124cad7c9eed42365a339fcf` 为 completed：192个观测、158个 fresh、169次在线完整 Update，以及随后12次 EOF Update/24步 BA。EOF独立耗时 **2.970749 s**，历史位姿重建不计 fresh。共965次 Update SDK和192次编码器SDK；九个 Update context、编码器context全部关闭，模型缓冲/DMA归零、清理错误为空。先前两次 failed run 原样保留，未知新形状仍拒绝，本次覆盖不保证任意未来拓扑。

**不改变在线输出的证据：** 保存数据独审核对192条非时钟结果、EOF前 metadata，以及7对NPZ中的41对实际数组，全部一致；初始seed/RNG与94项权重metadata一致。旧failed运行未保存最终RNG，因此没有宣称最终RNG SHA可直接比较。新增图只在EOF首次使用，在线位姿和保存状态保持一致。所有真实Torch/ORT/SDK调用在RK3588S，主机仅非量化编译与保存数据审核。

**性能和质量仍须分别验收：** 本次在线回放 **58.554 s，158/58.554=2.698 Hz**；从观测读入到事件JSON flush的fresh P95为 **386.036 ms**，16GB板峰值RSS **510.852 MiB**。不同轮次未控温控频，不将它与v41的2.602Hz之差归因于EOF补图。共同158个fresh、128对1秒端点的ATE/旋转RPE/平移RPE仍为 **0.019683m / 1.793720° / 0.036012m**，相对原全分辨率FAST分别增加 **76.92% / 81.98% / 110.76%**。EOF完成不改善在线精度，也不使原SDK逐元素拒绝、20Hz、相机排队、稠密地图或30分钟目标变为通过。默认VGGT与模型权重许可边界保持。[补图和完整回放源码](../deploy/rk3588/validation/dpvo_online_replay/V42.md)、[实际导出/编译、前缀一致性与独立轨迹评价](../results/dpvo_eof_update_v42.json)。

## 80. 元数据优化通过完整回放，仍要分开报告组件收益和整链路时延

**改动与验收方法：** 将第76条已做同状态AB的活动标志查询移植到九形状callback。源码检查确认仅替换reporting block；固定模型目录、输入快照、完整源组、失败保存、D1探测、EOF、数值计算和清理逻辑保持。板端重新执行192个真实数据集观测及完整EOF，并与v42成功运行比较，不能用单组件的耗时差额推算位姿频率。

**实际一致性：** 保存数据独审核对192条非时钟结果、8对NPZ中的55对实际数组，全部逐字节一致，包含14对最终状态数组；EOF结果也一致。初始及最终RNG的metadata和摘要相同，均为192次深度随机采样。RNG没有单独保存原始字节，不能把摘要核对表述为重读原始RNG。181次完整Update、34次显式Torch探测、965次Update SDK和192次编码器SDK均保留，全部context关闭、DMA归零，清理无错误。

**性能与边界：** 本次158个fresh位姿用时 **57.548625秒，2.745504Hz**，fresh读入到发布P95 **383.419ms**，峰值RSS **513.148MiB**。EOF额外 **2.632466秒**，不计入fresh。该轮与v42的2.698Hz处于不同热状态，不能将差值直接归因于元数据优化；第76条交错组件AB与本轮完整回放解决的是不同验证问题。原严格FP32拒绝、半分辨率质量拒绝、20Hz/相机/地图/30分钟未通过的状态均保持。默认VGGT没有更换。[完整回放和逐项保存数据核对](../results/dpvo_metadata_online_v43.json)、[最小集成源码](../deploy/rk3588/validation/dpvo_online_replay/V43.md)。

## 81. Native FP16 打包必须同时验证通道尾部、舍入和缓冲所有权

**实现：** 第78条实际查询确认的是 `[B,ceil(C/8),D,16,8]` 的NC1HWC2。新的AArch64 NEON内核按8通道×4空间位置tile，一次完成CF转置、FP32→FP16转换和数值检查；corr的882个有效通道扩为888，最后六通道写入正零。接口区分输入元素数与输出容量字节数，拒绝别名、错误形状、容量不足、NaN/Inf和转换后溢出；失败时不返回可提交的部分buffer。FPCR只读，非默认舍入、flush-to-zero等不支持状态显式拒绝，不修改调用线程浮点环境。

**边界与实际验证：** 在板端已有隔离工具链编译，不使用fast-math。实际库8056字节，加载时FPCR=0。九个已查询形状的generated数据和一份真实B2D6完整源组，全部与独立NumPy FP16参考的布局及位模式相同；验证正负零、ties-to-even、最小subnormal、normal边界和最大有限值。65504上方仍可舍入回65504的有限值允许通过，65520溢出tie、NaN/Inf必须拒绝，没有裁剪。容量与别名拒绝、目标两侧canary也通过。

**为何要拆开验证：** 转换正确只证明字节内容和所有权，不证明SDK接受pass-through1或网络输出相同。该conversion-only程序未加载SDK或执行模型，不能用其打包时间代替完整网络耗时；真实输入来自已保存NPU反馈状态，没有新Torchteacher。宿主机只重读保存数组和做参考分析。真实forward仍在RK3588S。[内核、边界测试与复现方法](../deploy/rk3588/validation/dpvo_native_input/README.md)、[板端构建与转换证据](../results/dpvo_native_input_v43.json)。

## 82. Native 输入提交减少转换开销，收益须用同状态完整调用测量

**同组对照：** 使用真实观测34中两个完整源的B2D6M16输入，固定原模型字节。A保留原FP32/NHWC/pass-through0；B在自己的context重新查询command8/10，核实际NC1HWC2/C2=8/FP16属性后，以自有FP16 buffer和pass-through1提交。原输出get、独立FP32拷贝、release、日志、锁及销毁路径保留。两套独立context执行ABBA两轮，8份输出的net_out/delta/weight逐字节相同，两context关闭，8次输出release成功、DMA和模型buffer归零。

| 每路排除首次后的3个暖样本均值 | 原FP32路径A | Native FP16路径B |
| --- | ---: | ---: |
| 输入打包 | 0.900 ms | 0.568 ms |
| SDK inputs_set | 2.797 ms | 0.072 ms |
| SDK run | 8.154 ms | 8.768 ms |
| 完整SDK forward | 12.197 ms | 9.747 ms |
| execute实际墙钟 | 12.654 ms | 10.186 ms |

**可得结论与限制：** 这证明了该模型、该输入与本次SDK的native路径实际接受及同状态输出一致，修正了“查询成功即可当传输成功”的错误推断。上表时钟互相嵌套，不能相加；A/B各仅3个暖样本，未控温控频，run本身有波动。pass-through1仍可能涉及拷贝，不能称zero-copy；同SDK输出一致也不能消除原Torch严格门槛的拒绝。单个192边组不是704边完整Update，更不是端到端位姿频率；需进一步完整图和递归回放验证。[原始调用记录及保存输出审核](../results/dpvo_native_input_v43.json)。


**进一步完整图对照：** 两路都使用第80条同一metadata callback和同一九模型目录，保留输入快照、完整源组与原边序，仅B改native transport。在观测34/127/191的704/800/752边真实保存输入上，各执行ABBA两轮，24份完整输出逐字节一致；两路各56次SDK与release、各6个context销毁，模型和DMA归零，清理无错误。各状态各路排除首次后剩3个暖样本：

| 保存观测 | callback墙钟 A→B | 完整图 A→B | 全部inputs_set之和 A→B |
| --- | ---: | ---: | ---: |
| 34 | 94.828→71.467 ms | 88.286→64.542 ms | 20.578→0.427 ms |
| 127 | 105.123→94.093 ms | 97.505→81.920 ms | 23.711→0.558 ms |
| 191 | 116.358→91.406 ms | 108.530→82.014 ms | 26.226→0.598 ms |

这些结果支持在上述三份完整状态上继续验证native路径。完整图仍包含SDK之外的快照、调度、metadata与输出恢复成本；native路径还保存了更多查询metadata，相关复制耗时尚未单独测量，不能把inputs_set差额全部转成端到端收益。本轮尚未让native输出进入新鲜观测的递归回放，没有新增Torchteacher、BA或相机输入。原SDK精度拒绝和半分辨率质量拒绝继续保持；这组独立状态实验不改变第80条真实回放的2.746Hz。[完整AB、生命周期及独立保存数据审核](../results/dpvo_native_input_v43.json)。


## 83. Native 传输必须验证递归状态，而不止固定输入输出

**问题与集成：** 第82条的同状态AB只能证明三份输入上的SDK输出一致。网络输出进入相关性、BA、删帧及后续网络后，仍需完整递归验证。v44保持v43的metadata callback、九模型目录、原94项参数、半分辨率策略、采样、D1及EOF，只替换空cache的输入传输；本轮没有混入新的closed标志优化。

**实测证据：** 板端完成192个连续真实观测、158个fresh位姿与12次独立EOF Update。重读8对NPZ、55对实际数组（含14对最终状态），并比较192条非时钟结果及EOF，均与v43逐字节一致。初始/最终RNG摘要及metadata一致，原始RNG字节没有单独保存。965次Update SDK、192次编码器SDK完成，九个Update context关闭、模型/DMA归零、清理无错误。

**整链路表现：** 活跃回放54.094486秒，**2.920815Hz**；软件读入到事件flush的fresh P95 **366.793772ms**，16GB板峰值RSS **515.507812MiB**。EOF另需2.499685秒、不计fresh。相同共同观测上的ATE/旋转RPE/平移RPE仍为0.019683m/1.793720°/0.036012m，没有因为打包优化而改善精度。不同轮次的温度与频率历史不同，不把与v43的差额全部归为因果收益。

**可复用结论：** 输送同一数值的优化可按“转换位模式→单组SDK→完整图→递归回放”逐层验证，各层保留独立边界。pass-through不等于zero-copy；相同SDK输出不等于原Torch严格FP32通过；约2.92Hz文件回放也不是20Hz、相机排队、地图或30分钟验收。[可复现集成](../deploy/rk3588/validation/dpvo_online_replay/V44.md)、[原始收据摘要和独立核对](../results/dpvo_native_online_v44.json)。


## 84. 降采样后应检查像素单位阈值，但单位修正不保证轨迹更准

**问题与证据：** 冻结的上游DPVO及板端几何实现先投影到特征网格，再对XY差求双向平均flow。keyframe_threshold的单位是特征图像素，K只按原RES=4除一次；它不是米或角度。在相同ray、SE3和depth的受控几何输入上，图像和K缩为一半时，该flow也缩为一半。原半分辨率研究策略仍沿用15，等效原图运动门槛因而翻倍；固定3×3patch、相关性半径和初始化门槛也各有尺度含义，不能一次混改后宣称已定位原因。

**单变量试验：** 保持原94项权重、CPU神经计算、半分辨率输入、M16/L4/R8/O4和motion probe=2，仅把关键帧删除阈值15改为7.5。数值7.5由投影单位预先选定，没有按GT误差调参。保存的原轨迹在157次删帧检查中有62次flow落在[7.5,15)，提示该变量确实会改变状态；它们不能预言新轨迹，因为第一次删帧差异会反馈到后续计算。

**实际分叉：** 两次回放均完成192个观测、158个fresh及完整EOF。观测索引54的flow同为8.138959407806396，发布的当前位姿相同，但原阈值会删除观测53，新阈值保留它；随后图拓扑与数值递推变化。结束前保留帧数由10变为17、inactive边由144变为800，不能只看active边数下降就推断总状态或计算减少。新回放用时114.157487秒，峰值RSS395.414MiB；单次未控温控频的CPU对照不用于宣称因果提速。

**评价规则：** 新策略是独立研究候选，原策略、原baseline和接受门槛均保留。使用共同fresh观测、全部192个原时间戳规划的一秒端点、一次正尺度Sim3，并同时报告各自fresh/端点覆盖。初始化motion probe仍是2，首次fresh仍为观测34；此次试验不能解决初始化覆盖损失。原模型替换、20Hz和VGGT精度均须另验。[源码与复现](../deploy/rk3588/validation/dpvo_half_keyframe/README.md)、[预声明策略及实际质量评价](../results/dpvo_half_keyframe_v44.json)。

**实际质量结果：** 共同158个fresh、128对一秒端点上，新策略ATE为 **0.017149m**、旋转RPE为 **1.586857°**、平移RPE为 **0.026901m**。相对原半分辨率CPU分别下降 **12.98%、4.89%、9.41%**；相对原全分辨率FAST仍增加 **54.15%、60.99%、57.43%**，10%及20%目标均未通过。额外记录的平移方向RMSE相对半分辨率CPU增加24.05%，没有因三个主要指标改善而隐藏它。新旧half的fresh覆盖均158/192，使用端点128/162；full为179/192、149/162。逐帧核查确认0..53全部非时钟字段相同，验证了预声明的首次分叉位置。该修正有局部收益，尚不能修复半分辨率、patch数量/支持范围及窗口共同带来的全部质量损失。


## 85. 热路径读取一个标志，应保留锁语义而避免整份元数据复制

**问题与最小改动：** 第76条还留下cache hit读取closed的deepcopy。新候选只将冻结`BoundedContextCache._get`中的`session.metrics()['closed']`换成持原session锁读取`session._metrics['closed']`，外层仍由继承的execute持cache锁，锁序保持cache→session。其余_get AST完全一致，其余方法继承原函数；冷初始化、预算、LRU、history、公开metrics和close仍用完整metadata。通过独立子类绑定，不改原运行时全局，不迁移live context。

**公平性细节：** 新native运行时增加了query、属性和来源metadata。若只在B加入helper来源pin，两路复制的metadata负担也会改变；本次在建立两套空cache前合并全部来源，严格要求A/B每个实际context的pin字典相同。两路都用同一native session/packer、callback、九模型目录和相同源组；唯一执行差异是closed查询。

**板端结果：** 在三份真实保存状态上各跑ABBA两轮，24份输出的72个实际数组全部逐字节一致。两路各56次SDK/run/release、各6个context关闭，模型/DMA归零、清理无错误；保存审核逐项核对两路72次native查询与336条输入打包记录。各状态每路排除首次调用后，余3个暖样本：

| 观测索引 | 完整callback墙钟 A→B | 完整图 A→B |
| --- | ---: | ---: |
| 34 | 83.106→70.597 ms | 75.082→62.826 ms |
| 127 | 105.655→97.252 ms | 95.521→85.215 ms |
| 191 | 107.248→94.043 ms | 96.882→83.613 ms |

**边界：** 这支持移除热路径冗余复制，但完整SDK run时钟也有波动，不能把全部差额归于该读取。每路每状态仅3个暖样本，未控温控频；嵌套时钟不能相加。该候选尚未接入第83条连续回放，不改变其中2.920815Hz，也没有执行新的BA、轨迹或相机输入。权重、输出所有权、严格数值门槛和原质量拒绝保留。[实现与复现](../deploy/rk3588/validation/dpvo_cache_closed_flag/README.md)、[真实交错AB和独立保存审核](../results/dpvo_cache_closed_flag_v44.json)。


## 86. 单标志缓存优化仍须验证完整递归与资源生命周期

**问题与处理：** 第85条的固定输入AB保留了输出，但只有接入连续观测后，才能核对该缓存替换是否影响状态、EOF或context销毁。v45只在首次context建立前，将v44的空cache换成已验证的closed查询子类；原native session、九模型目录、callback、半分辨率策略和94项权重保持。19个generated/fake检查覆盖显式入口、已使用cache拒绝、异常清理，以及frame/update/RNG/EOF/finally源码结构。

**完整实测：** RK3588S完成192个观测、158个fresh位姿和12次独立EOF Update。与v44比较，192条非时钟结果及8对NPZ的55对实际数组逐字节一致，包含14对最终状态；RNG仅核已保存metadata/摘要，未声称重读原始RNG。965次Update SDK及release、192次编码器SDK、九个Update context清理均完成，模型缓冲与DMA归零。独立轨迹评价与上轮一致。

**时间边界：** 活跃回放52.561283秒，fresh吞吐为**3.006015Hz**；软件读入到事件flush的fresh P95为**338.112205ms**，16GB板峰值RSS515.586MiB。EOF另需2.421033秒，不计fresh。与v44不同轮次的热状态不受控，不能把全部耗时差归于closed查询；原FP32与轨迹质量拒绝保留，20Hz、相机队列、稠密地图、30分钟及8GB目标尚未通过。

**可复用结论：** 生命周期或统计代码也可能位于热路径。减少复制时，既要保持锁序和对象所有权，也要按“同状态AB→完整递归→清理审查”验证，不能仅由微基准断言整机提速。[完整回放与保存数据证据](../results/dpvo_closed_online_v45.json)、[源码与复现](../deploy/rk3588/validation/dpvo_online_replay/V45.md)。


## 87. 降采样后的初始化门槛可修复覆盖，但覆盖增加不保证共同轨迹更准

**问题与单位：** 原DPVO Update的delta加到投影的特征网格XY上，motion probe取delta范数的中位数，单位是特征图像素；K只按RES=4除一次。半分辨率仍用motion2，其名义原图像素门槛翻倍。由单位预先选择1，不使用GT选参，也不假定学习网络的响应随resize精确缩放。第84条关键帧7.5试验保持motion2，未修复观测34才初始化的覆盖损失，因此本次单独测motion1。

**单变量与装载：** 从原half策略重新开始，保持KF15、M16/L4/R8/O4、原94项state-dict、K/图像处理、共享采样/depth RNG、完整图Update/BA、数值失败边界和EOF。独立stream只改`probe < 2.0`为`probe < 1.0`及对应reason文字；原文件、原SOURCE_PINS和verify_sources不改。新driver同时锁定原/候选stream的SHA并核精确两行变换，候选SHA不能由任意CLI值批准；原权重/源码装载守卫保留，不叠加关键帧7.5。

**实际分叉与覆盖：** RK3588S CPU实际完成192个观测。逐字段保存审查确认观测0去时钟后相同，首次admission分叉在1：两路probe均为1.0345186591148376，原门槛拒绝、新门槛接受，此时均无fresh位姿。后续状态来自各自实际反馈，不能套用旧probe或gold。首次fresh由34提前到11；新fresh为181/192、固定全部192原时间戳的一秒端点使用151/162，原half为158/192、128/162，full FAST为179/192、149/162。

**实际质量：** 对原half的共同158个fresh、128对端点，ATE/旋转RPE/平移RPE RMSE分别变化+7.2214%、-21.4205%、+13.1601%，三主指标仅通过相对half的20%限值，10%失败。对full FAST的共同179个fresh、149对端点，分别增加119.9226%、30.7296%、105.4787%，10%及20%限值均失败。共同集合在GT和误差过滤前确定，每个总体用一次正尺度Sim3；增加初始化覆盖不能替代共同轨迹误差门槛。另报方向RMSE相对half下降12.0706%、相对full增加51.3653%，不把它新增为接受标准。

**时间与证据边界：** 活跃文件回放119.443248043秒，181fresh/该分母为1.515364Hz；fresh软件读入到事件flush的P95为780.780036ms，峰值RSS368.9375MiB。12次EOF另用3.580708026秒且不算fresh。不同轮次热状态不受控，不据此归因提速；20Hz/P95≤100ms与full基线质量目标尚未通过。重读obs11/127/191的6份NPZ、27个数组，SHA/形状/dtype/finite/weight范围与本次callback身份均一致；新gold只表示本次同一次原Torch输出，不对应旧obs34。RNG、94权重、源码稳定和context清理仅核已保存metadata，无神经/native/BA重跑，不采纳该研究候选、不改默认VGGT。[源码与复现](../deploy/rk3588/validation/dpvo_half_motion/README.md)、[预声明策略及实际评价](../results/dpvo_half_motion_v45.json)。


## 88. 20Hz 预算应按完整 fresh 观测分解，内嵌时钟不能重复相加

**问题与测量范围：** 单个SDK调用、相关性kernel或固定状态AB都不能代表新鲜位姿的端到端频率。保存JSON分析预先固定初始化后的观测35..191，共157个fresh，排除首次初始化位姿34和EOF；保留其中的冷context与捕获I/O异常值。按实际串行调用索引绑定同一观测的相关性、完整Update与两步BA，没有用热样本筛掉慢帧。

**已测证据：** v44 native回放中，该群体读入到发布平均300.350ms、P95 360.601ms。后续v45加入closed读取优化的完整回放中，相同群体平均290.135ms、P95 334.433ms。v45全158个fresh的P95是338.112ms，分母与上述157个稳态观测不同。不同实际回放的温频状态未受控，不能把差额直接归因于closed读取。

| v45稳态阶段 | 平均墙钟 |
| --- | ---: |
| decode/undistort/crop/resize | 25.377 ms |
| 编码器完整调用 | 36.886 ms |
| stream完整处理 | 227.548 ms |
| 发布/flush | 0.323 ms |
| stream内相关性完整调用 | 53.087 ms |
| stream内完整Update | 108.719 ms |
| stream内两步BA | 36.150 ms |
| stream内捕获/metadata | 0.856 ms |
| stream剩余未单独测量部分 | 28.736 ms |

**为何不能相加：** 相关性kernel48.344ms包含在相关性完整调用内；SDK run之和56.072ms包含在SDK forward68.452ms、完整图94.956ms、callback100.577ms与Update108.719ms内。这些内嵌数值不能再加到227.548ms上。stream剩余项包含重投影、原context gather、图/删帧处理、拷贝及部分报告开销，不应全部归作BA。回放分母还包括各次发布之间的进度写入和健康采样，故它也不等于157个单帧延迟的均值。

**优化判断与边界：** 相关性kernel占其完整调用约91.1%，值得优化真实计算；仅继续削减metadata不足以跨越50ms预算。在“其余阶段保持现状的串行回放”条件下，算术上完全扣除48.344ms相关性kernel，平均延迟仍约241.791ms；预处理与编码器均值合计62.264ms，完整Update内SDK run也有56.072ms。这个扣除是条件性预算估算，既非优化实测，也不是未来流水线或异步架构的硬下界。原SDK严格FP32拒绝、半分辨率质量拒绝与20Hz未验收状态继续保留。[原始时钟、固定群体与可复查分析](../results/dpvo_realtime_budget_v45.json)。

## 89. 相邻相关性 dot 可并行累加，但每个输出的 FP32 顺序必须保留

**热点与最小改动：** 冻结v36已经缓存了owned NHWC帧特征；原raw8×8采样对每个位置单独调用NEON dot，同一patch被反复加载，C128的单个四lane累加器有32轮依赖。新候选只把完全有效的四个横向邻点同时计算：每轮加载一次patch、加载四个target，各自做独立vmul/vadd。每个输出仍按原通道次序累加，最后仍用`((p0+p1)+p2)+p3`和原尾部标量顺序；不使用FMA、不重关联。边界不完整的四点继续使用原dot或原正零。准备/池化、双线性权重、offset/level输出顺序、线程、所有权、字节预算与失败守卫保持。

**板端证据：** 原缓存与候选缓存使用同一冻结Python helper、三份历史M96真实特征和原保存图，四种case各ABBA两轮，共32次reported调用。每个case保存首次A/B共8份corr输出，加4份图与4份原CPU参考，共16份NPZ；其余重复调用以原始raw metadata绑定。不是32份实际输出数组全部另存。入口逐次报告A/B逐字节相同且原CPU固定FP32门槛通过，两路各16次成功查询、3次帧准备/销毁，关闭后cache字节归零，源一致、清理无错误。独立保存数据审查重读16份NPZ，重算首A/B输出的位模式与原CPU门槛，并将32条调用的数组摘要与对应保存输出逐项核对；没有把未另存的重复输出表述成重新读取的原始数组。

| case与边数 | 暖callback A→B | 内嵌kernel A→B |
| --- | ---: | ---: |
| 两帧首motion，E96 | 2.423→2.208 ms | 2.015→1.780 ms |
| 增为三帧，E864 | 29.643→20.439 ms | 29.135→19.943 ms |
| 同三帧换slot顺序，E864 | 42.856→33.376 ms | 42.266→32.771 ms |
| 删除中间slot，E384 | 24.622→19.238 ms | 23.767→18.539 ms |

**验证边界：** 各路每case排除首次后仅3个暖样本，表内时钟嵌套；不同case也不是受控相同性能负载。该结果支持在这些保存特征上继续评估运算调度，没有删除边或修改相关性数学。目前尚未用于半分辨率M16、完整递归回放或新观测，不能把E864的9ms差额直接从第88条48ms在线kernel中扣除，更不能称已达20Hz。默认VGGT未改变，模型权重不随此源码发布。[冻结候选、实际AB及保存数据核对](../results/dpvo_realtime_budget_v45.json)。
