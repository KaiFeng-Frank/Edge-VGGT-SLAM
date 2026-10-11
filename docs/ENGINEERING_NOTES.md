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
