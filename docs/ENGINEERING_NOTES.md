# 可复用的工程问题记录

这里记录移植中能够复现、值得复用的故障定位方法。DA3-SMALL 和 DPVO 均是独立模型研究，原 VGGT 默认模型没有替换；ONNX 的 CPU 数值验证也不代表 NPU 或实时 SLAM 已通过。各条状态以链接中的固定源码、输入和收据为准。

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
- **验证边界：** 三次实际 ORT 调用已完成，来源与输入未变、清理无错误，但严格 gate 失败，候选未编译或在 NPU 上 profile。小张量公式检查只能验证实现意图，不能替代真实特征的逐元素对照；本次也未证明舍入路径是唯一根因。不能将这张候选图认定为数值已验收的 NPU 部署、VO 精度通过或实时优化成功。
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
